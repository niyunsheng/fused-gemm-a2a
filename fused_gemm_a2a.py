"""Ulysses projections with BF16 interfaces and BF16 or experimental MXFP8 GEMMs.

Sequence side: [M, H]. Head side: [P*M, heads/P, D]. QKV weights use per-head
[q, k, v, (gate)] rows. All CP ranks use equal M and matching call order.
Create FusedGemmA2AWorkspace before execution/capture/PP receives, then pass it to
every forward. Returned activations own their storage.
"""
from dataclasses import dataclass, field
import os

import cuda.bindings.driver as cuda
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from torch import nn
from torch.profiler import record_function

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import Int32, Int64

from fused_gemm_a2a_kernels import (
    CLUSTER_MN, CTA_M, MMA_N, MMA_N_OUT_PUSH, MMA_N_RECV, MMA_TILER_MN,
    PUSHER_STEADY_WARPS, PUSHER_WARPS, PUSH_SPLIT, SWIZZLE,
    UlyssesGemm, barrier, padded_rows, push_split, reduce_bias_grad, resize_rank_rows,
    tile_swizzle, validate_peer_count, validate_tile_n,
    mxfp8_wgrad, mxfp8_quantize, compile_mxfp8,
)

__all__ = ['FusedGemmA2AWorkspace', 'FusedQKVProjection', 'FusedOutputProjection']

WGRAD_BACKEND = os.environ.get('FUSED_WGRAD_BACKEND', 'torch')
if WGRAD_BACKEND not in ('torch', 'fused_accum'):
    raise ValueError('FUSED_WGRAD_BACKEND must be torch or fused_accum')
if WGRAD_BACKEND == 'fused_accum':
    # Megatron's extension writes the cuBLAS FP32 accumulator directly into
    # main_grad, avoiding a full temporary dW and a separate elementwise add.
    import fused_weight_gradient_mlp_cuda as _wgrad_accum

_CUTE_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16, torch.float32: cutlass.Float32,
                torch.int32: cutlass.Int32, torch.int64: cutlass.Int64,
                torch.float8_e4m3fn: cutlass.Float8E4M3FN, torch.uint8: cutlass.Float8E8M0FNU}


def _ptr(t: torch.Tensor, align=16) -> cute.Pointer:
    """Device pointer of a contiguous tensor. The GEMM takes pointers, not DLPack tensors: a DLPack
    view keeps the tensor alive, and caching such views by address had pinned every output tensor,
    so each call allocated a fresh block (one cudaMalloc, about a millisecond of host time)."""
    if not t.is_contiguous():
        raise ValueError('Fused kernels need contiguous tensors')
    return make_ptr(_CUTE_DTYPES[t.dtype], t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _ptrs(t: torch.Tensor) -> cute.Tensor:
    """DLPack view of a persistent buffer (barrier flags and epoch); keeps the tensor alive."""
    return from_dlpack(t.detach(), assumed_align=8)


class _RecvBuffer:
    """A pooled receive side: the symmetric buffer and row-block counters with their peer pointer arrays,
    and the device-side reuse epoch. One per maximum (rows, columns) shape per group,
    shared by every module and active-length view of that allocation."""

    def __init__(self, recv, peer_recv, counters, peer_counters, epoch):
        self.recv, self.peer_recv, self.counters, self.peer_counters, self.epoch = (
            recv, peer_recv, counters, peer_counters, epoch)


class _Comm:
    """Symmetric-memory state shared by the fused modules of one process group: flags, barrier, compile
    cache and a pool of symmetric buffers keyed by shape. Modules of the same shape in different layers
    share their buffers (sequential layers never hold them at the same time), so the workspace does not
    grow with depth. Q/K/V/gate outputs, sequence-side outputs and the saved O_seq
    own ordinary storage, retaining their values across subsequent workspace reuse.
    Output-projection backward also returns an owned dO gradient."""

    _instances = {}

    @classmethod
    def get(cls, group):
        """The shared state of a process group (created on first use)."""
        key = (group, torch.cuda.current_device())
        if key not in cls._instances:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Create FusedGemmA2AWorkspace before capture to initialize communication')
            cls._instances[key] = cls(group)
        return cls._instances[key]

    def __init__(self, group):
        self.group = dist.group.WORLD if group is None else group
        self.rank, self.p = dist.get_rank(self.group), dist.get_world_size(self.group)
        validate_peer_count(self.p)  # Reject unsupported groups before allocating symmetric buffers.
        if torch.cuda.get_device_capability()[0] != 10:
            raise ValueError('These kernels target SM100/Blackwell GPUs')
        self.device = torch.device('cuda', torch.cuda.current_device())
        cluster_size = CLUSTER_MN[0] * CLUSTER_MN[1]
        self.max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        symm.set_backend('NVSHMEM')
        self.flags, self.peer_flags = self.symmetric((self.p,), torch.int64)
        self.flags.zero_()
        self.epoch = torch.zeros(1, dtype=torch.int32, device=self.device)  # recv invocations so far
        torch.cuda.synchronize()
        dist.barrier(self.group)
        self.compiled = {}
        self.pool = {}
        self.projection_buffers = {}
        self._barrier_args = (_ptrs(self.flags), _ptrs(self.peer_flags), _ptrs(self.epoch))
        self._barrier = {bump: cute.compile(barrier, *self._barrier_args, self.rank, self.p, bump,
                                          _ptr(self.epoch, 8), Int32(0), self.stream())
                         for bump in (False, True)}

    @staticmethod
    def stream(torch_stream=None):
        return cuda.CUstream((torch_stream or torch.cuda.current_stream()).cuda_stream)

    def symmetric(self, shape, dtype):
        tensor = symm.empty(shape, device=self.device, dtype=dtype)
        tensor.zero_()
        pointers = symm.rendezvous(tensor, self.group).buffer_ptrs
        valid = torch.tensor(int(all(pointers) and all(p % 16 == 0 for p in pointers)), device=self.device)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=self.group)
        if not valid.item():
            raise RuntimeError('Peer buffers must be aligned and GPU-addressable')
        peers = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        return tensor, peers

    def barrier(self, bump=False, epoch=None, counters=None):
        """Group barrier; bump=True also advances the device-side recv epoch of `epoch` (a receive buffer's
        invocation counter)."""
        epoch = self.epoch if epoch is None else epoch
        count = 0 if counters is None else counters.numel()
        self._barrier[bump](self._barrier_args[0], self._barrier_args[1], _ptrs(epoch),
                            _ptr(epoch if counters is None else counters, 8), Int32(count), self.stream())

    def pooled(self, key, shape, dtype):
        """A symmetric buffer and its peer pointers shared by every module asking for the same key and shape."""
        k = (key, tuple(shape), dtype)
        if k not in self.pool:
            self.pool[k] = self.symmetric(shape, dtype)
        return self.pool[k]

    def recv_buffer(self, m, k):
        """The pooled receive side for an [m, k] receive buffer (see _RecvBuffer)."""
        key = ('recv', m, k)
        if key not in self.pool:
            recv, peer_recv = self.symmetric((m, k), torch.bfloat16)
            counters, peer_counters = self.symmetric((m // CTA_M,), torch.int32)
            counters.zero_()
            epoch = torch.zeros(1, dtype=torch.int32, device=self.device)
            torch.cuda.synchronize()
            dist.barrier(self.group)
            self.pool[key] = _RecvBuffer(recv, peer_recv, counters, peer_counters, epoch)
        return self.pool[key]

    @staticmethod
    def push_split(a, num_in, p):
        """Row sub-blocks per (row block, destination) pusher job; see fused_gemm_a2a_kernels.push_split."""
        return push_split(a.shape[1], p, num_in, PUSH_SPLIT)

    def gemm(self, mode, a, b, b_is_kn, c, aux, counters, num_in, n_segment, num_segments, srcs=(0, 0, 0, 0),
             a_rows=None, head_dim=0, bias=None, src_rows=None, scales=(), mx_sources=()):
        """Run the Ulysses GEMM. a [M, K]; b [N, K] (b_is_kn=False) or [K, N] (True); see UlyssesGemm.
        Receive counters reset at the reuse boundary; each row waits for P * num_in * split arrivals.
        srcs are the head-side inputs distributed by the GEMM's own pusher warps.
        a_rows (push mode) is the tile-aligned row count the GEMM enumerates; rows of a beyond its logical shape
        read as zero through TMA and land as zero rows in the destinations."""
        a_rows = a.shape[0] if a_rows is None else a_rows
        src_rows = a.shape[0] if src_rows is None else src_rows
        if src_rows < 1 or src_rows > a.shape[0]:
            raise ValueError('Receive source rows must be positive and fit the receive buffer')
        pad_recv = mode == 'recv' and src_rows != a.shape[0]
        if a_rows < a.shape[0] or a_rows % MMA_TILER_MN[0] or (mode == 'recv' and a_rows != a.shape[0]):
            raise ValueError('a_rows must be a tile multiple >= the rows of a, and equal to them in recv mode')
        # A singleton M dimension triggers an invalid TMA instruction in the
        # two-CTA GEMM. Materialize its physical row tile for both QKV forward
        # and output dgrad; callers retain logical tokens and saved activations.
        if mode == 'push' and a.shape[0] == 1:
            a = torch.nn.functional.pad(a, (0, 0, 0, a_rows - 1))
        in_gemm_pushers = mode == 'recv'
        clusters = self.max_active_clusters
        aux = list(aux) + [aux[0]] * (4 - len(aux))      # the kernel always takes four peer arrays
        srcs = list(srcs) + [srcs[0]] * (4 - len(srcs))  # and four head-side sources
        tile_n = 128 if scales else (MMA_N_RECV if mode == 'recv' else (MMA_N_OUT_PUSH if num_segments == 1 else MMA_N))
        output_width = b.shape[1] if b_is_kn else b.shape[0]
        projections = num_segments if mode == 'push' else num_in
        validate_tile_n(mode, tile_n, output_width, self.p, n_segment, projections, head_dim)
        if bias is not None and (mode != 'push' or tuple(bias.shape) != (output_width,)
                                 or bias.dtype != torch.bfloat16):
            raise ValueError('GEMM bias requires push mode and a BF16 vector of output-width entries')
        key = (mode, a.shape, a_rows, b.shape, b_is_kn, c.shape, a.dtype, num_in, n_segment, num_segments,
               in_gemm_pushers, clusters, head_dim, tile_n, bias is not None, pad_recv, bool(scales))
        args = (_ptr(a), _ptr(b), _ptr(c), _ptr(b if bias is None else bias),
                *(_ptr(x, 8) for x in aux), _ptr(counters, 8),
                *(Int64(x) for x in srcs), Int32(src_rows),
                tuple(_ptr(x) for x in (scales or (b, b))),
                tuple(Int64(x) for x in (mx_sources or (0,) * 8)))
        if key not in self.compiled:
            op = UlyssesGemm(mode, self.p, self.rank, n_segment, num_segments, head_dim, tile_n)
            op.mxfp8 = bool(scales)
            op.pad_recv = pad_recv
            op.b_is_kn = b_is_kn
            op.has_bias = bias is not None
            op.num_in = num_in
            op.a_shape, op.b_shape, op.c_shape = tuple(a.shape), tuple(b.shape), tuple(c.shape)
            op.a_rows = a_rows
            if any(len(x) != self.p for x in aux) or (mode == 'recv' and counters.numel() != c.shape[0] // CTA_M):
                raise ValueError('Peer pointer arrays must have P entries and counters one per row block')
            if mode == 'recv':
                op.push_split = self.push_split(a, num_in, self.p)
            if in_gemm_pushers:
                # Extra pusher warps only pay off while the GEMM waits for its first row group, and
                # only if that group has enough jobs to keep them all busy; otherwise they just slow
                # the GEMM (more threads per program, more pipeline contention).
                clusters_m = a.shape[0] // CTA_M // CLUSTER_MN[0]
                first_blocks = tile_swizzle(clusters_m, SWIZZLE) * MMA_TILER_MN[0] // CTA_M
                first_jobs = first_blocks * num_in * self.p * op.push_split
                programs = clusters * CLUSTER_MN[0] * CLUSTER_MN[1]
                op.pusher_warps = max(PUSHER_STEADY_WARPS, min(PUSHER_WARPS, first_jobs // programs))
                op.threads_per_cta = 32 * (6 + op.pusher_warps)
            else:
                op.pusher_warps = 0
                op.threads_per_cta = 32 * 6
            compile_gemm = compile_mxfp8 if scales else cute.compile
            self.compiled[key] = compile_gemm(op, *args, clusters, self.stream())
        self.compiled[key](*args, self.stream())

    def recv_gemm(self, a, b, b_is_kn, c, peer_recv, peer_counters, counters, num_in, srcs,
                  head_dim=0, src_rows=None):
        """Push head-side inputs to peers while GEMM consumes completed receive row blocks."""
        aux = [peer_recv, peer_counters]
        self.gemm('recv', a, b, b_is_kn, c, aux, counters, num_in, 0, 1, srcs=srcs,
                  head_dim=head_dim, src_rows=src_rows)


def _check(t, shape, label):
    if tuple(t.shape) != tuple(shape) or t.dtype != torch.bfloat16 or not t.is_contiguous():
        raise ValueError(f'{label} must be a contiguous BF16 tensor of shape {tuple(shape)}, got {tuple(t.shape)}')


def _weight_grad(weight, a, b, precision="bf16"):
    """dW = a^T b. With Megatron's gradient-accumulation fusion the parameter carries an fp32
    `main_grad`: accumulate into it with an fp32-output GEMM and return a dummy gradient (see
    _accumulated); otherwise return the BF16 gradient for autograd."""
    main_grad = getattr(weight, 'main_grad', None)
    if precision == 'mxfp8':
        if main_grad is not None and (main_grad.dtype != torch.float32 or not main_grad.is_contiguous()):
            raise ValueError('MXFP8 accumulation requires contiguous FP32 main_grad')
        out = torch.empty_like(weight) if main_grad is None else main_grad
        mxfp8_wgrad(a, b, out, accumulate=main_grad is not None)
        return out if main_grad is None else _accumulated(weight)
    if main_grad is None:
        return a.t() @ b
    if WGRAD_BACKEND == 'fused_accum':
        if main_grad.dtype != torch.float32:
            raise ValueError('fused_accum requires FP32 main_grad')
        _wgrad_accum.wgrad_gemm_accum_fp32(b, a, main_grad)
        return _accumulated(weight)
    try:
        main_grad.add_(torch.mm(a.t(), b, out_dtype=torch.float32))
    except TypeError:  # torch without out_dtype on mm
        main_grad.add_((a.t() @ b).float())
    return _accumulated(weight)


_dummy_wgrads = {}


def _accumulated(param):
    """After accumulating into param.main_grad. Megatron's DDP registers its gradient-ready hook on the
    parameter's accumulation node, which autograd only runs when a gradient arrives, so return a dummy
    of the right shape (zeros when the parameter asks for zero_out_wgrad). As in Transformer Engine,
    cache these placeholders by shape instead of retaining one allocation per captured backward.
    Actual gradients live in main_grad; the DDP hook discards the placeholder."""
    param.grad_added_to_main_grad = True
    key = (param.device, param.dtype, tuple(param.shape))
    if key not in _dummy_wgrads:
        _dummy_wgrads[key] = torch.empty(param.shape, device=param.device, dtype=param.dtype)
    dummy = _dummy_wgrads[key]
    if getattr(param, 'zero_out_wgrad', False):
        dummy.zero_()
    return dummy.detach()


def _bias_grad(bias, dy_rows, accumulate=True, *, use_cute=False):
    """Reduce logical rows in FP32; optionally write main_grad directly.

    accumulate=False returns parameter-dtype dbias for autograd/DDP, matching
    TE's bias-gradient cast even when the parameter carries an FP32 main_grad.
    """
    if bias is None:
        return None
    main_grad = getattr(bias, 'main_grad', None)
    if main_grad is None or not accumulate:
        return reduce_bias_grad(dy_rows, torch.empty_like(bias), use_cute=use_cute)
    reduce_bias_grad(dy_rows, main_grad, accumulate=True, use_cute=use_cute)
    return _accumulated(bias)


class _FusedQKV(torch.autograd.Function):
    @staticmethod
    @record_function('fused_qkv_forward')
    def forward(ctx, x, weight, bias, module, workspace):
        tokens = x.shape[0]
        comm = workspace.comm
        comm.barrier()  # Prior consumers complete before buffer reuse.
        workspace.run_push(x, weight, False, module.segments,
                           bias=bias if module.bias_in_gemm else None)
        comm.barrier()  # Incoming tiles are complete.
        ctx.module = module
        ctx.workspace, ctx.tokens = workspace, tokens
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight, *(() if bias is None else (bias,)))
        # Return owned logical rows before another call reuses the peer buffers.
        outputs = resize_rank_rows(
            workspace.heads, comm.p, tokens, copy=True,
            bias=bias if module.apply_bias and not module.bias_in_gemm else None,
            head_dim=module.head_dim, head_offset=comm.rank * module.num_heads // comm.p, use_cute=workspace.precision == 'mxfp8')
        return tuple(buf.view(-1, buf.shape[1] // module.head_dim, module.head_dim)
                     for buf in outputs)

    @staticmethod
    @record_function('fused_qkv_backward')
    def backward(ctx, *head_grads):
        x, weight, *rest = ctx.saved_tensors
        bias = rest[0] if ctx.has_bias else None
        workspace, tokens = ctx.workspace, ctx.tokens
        module, comm = ctx.module, workspace.comm
        grads = [g.contiguous().flatten(1) for g in head_grads]
        workspace.begin_recv()
        dx = torch.empty(workspace.capacity, x.shape[1], device=x.device, dtype=x.dtype)
        workspace.run_recv(grads, weight, True, dx)
        # dY is now materialized locally in the weight's row order; rows beyond the logical count are
        # not consumed.
        dy = workspace.recv[:tokens]
        # Saved-tensor hooks may unpack a Parameter as a plain Tensor. Use
        # the original parameter for main_grad and DDP metadata; the unpacked
        # weight above still supplies the saved numerical value for dgrad.
        dw = _weight_grad(module.weight, dy, x, workspace.precision)
        db = _bias_grad(module.bias if ctx.has_bias else None, dy, module.accumulate_bias_grad, use_cute=workspace.precision == 'mxfp8')
        return dx[:tokens], dw, db, None, None


class _FusedOutput(torch.autograd.Function):
    @staticmethod
    @record_function('fused_out_forward')
    def forward(ctx, o, weight, bias, module, workspace, tokens):
        comm = workspace.comm
        op = o.flatten(1)
        workspace.begin_recv()
        y = torch.empty(workspace.capacity, weight.shape[0], device=o.device, dtype=o.dtype)
        workspace.run_recv([op], weight, False, y)
        ctx.module = module
        ctx.workspace, ctx.tokens = workspace, tokens
        # O_seq sits in the shared receive buffer, which the next forward of this shape overwrites; the
        # weight gradient needs it in backward, so autograd keeps its own copy (the activation a
        # row-parallel linear saves anyway). Only the logical rows are kept.
        saved = workspace.recv[:tokens].clone()
        ctx.save_for_backward(weight, saved[:tokens])
        ctx.has_bias = bias is not None
        y = y[:tokens]
        if bias is not None and module.apply_bias:
            y.add_(bias.view(1, -1))  # y is this call's own tensor
        return y

    @staticmethod
    @record_function('fused_out_backward')
    def backward(ctx, dy):
        weight, o_seq = ctx.saved_tensors
        workspace, tokens = ctx.workspace, ctx.tokens
        module, comm = ctx.module, workspace.comm
        dy = dy.contiguous()
        comm.barrier()
        workspace.run_push(dy, weight, True, 1)
        comm.barrier()
        dw = _weight_grad(module.weight, dy, o_seq, workspace.precision)  # Metadata belongs to the original Parameter.
        db = _bias_grad(module.bias, dy, module.accumulate_bias_grad, use_cute=workspace.precision == 'mxfp8') if ctx.has_bias else None
        # Strip padding and own dO in one copy: other backwards may reuse the workspace.
        do, = resize_rank_rows(workspace.heads, comm.p, tokens, copy=True, use_cute=workspace.precision == 'mxfp8')
        return do.view(-1, do.shape[1] // module.head_dim, module.head_dim), dw, db, None, None, None


class _ProjectionBuffers:
    """Fixed maximum allocation shared by serialized layers and micro-batches."""

    def __init__(self, comm, capacity, hidden_size, head_dim, segments):
        self.comm, self.capacity, self.head_dim = comm, capacity, head_dim
        self.workspaces = {}
        self.mx_recv = None
        self.heads, self.peer_heads = [], []
        for i in range(segments):
            key = ('dhead' if segments == 1 else 'head', i)
            buf, ptrs = comm.pooled(key, (comm.p * capacity, hidden_size // comm.p), torch.bfloat16)
            self.heads.append(buf)
            self.peer_heads.append(ptrs)
        rb = comm.recv_buffer(capacity, segments * hidden_size)
        self.recv, self.peer_recv, self.counters, self.peer_counters, self.epoch = (
            rb.recv, rb.peer_recv, rb.counters, rb.peer_counters, rb.epoch)
        torch.cuda.synchronize()
        dist.barrier(comm.group)

    def init_mxfp8(self):
        if self.mx_recv is not None:
            return
        rows, k = self.recv.shape
        self.mx_recv, self.peer_mx_recv = self.comm.symmetric((rows, k), torch.float8_e4m3fn)
        self.mx_scale, self.peer_mx_scale = self.comm.symmetric((rows * k // 32,), torch.uint8)


class _ProjectionWorkspace:
    """Views of a fixed allocation, sized for this invocation's active row capacity."""

    def __init__(self, buffers, capacity, precision):
        self.comm, self.capacity, self.head_dim = buffers.comm, capacity, buffers.head_dim
        self.precision = precision
        self.heads = [buf[:self.comm.p * capacity] for buf in buffers.heads]
        self.peer_heads = buffers.peer_heads
        self.recv, self.peer_recv = buffers.recv[:capacity], buffers.peer_recv
        self.counters, self.peer_counters = buffers.counters[:capacity // CTA_M], buffers.peer_counters
        self.epoch = buffers.epoch
        if precision == 'mxfp8':
            self.mx_recv = buffers.mx_recv[:capacity]
            self.mx_scale = buffers.mx_scale[:capacity * self.recv.shape[1] // 32]
            self.peer_mx_recv, self.peer_mx_scale = buffers.peer_mx_recv, buffers.peer_mx_scale

    def run_push(self, a, weight, b_is_kn, segments, bias=None):
        scales = ()
        width = weight.shape[1] if b_is_kn else weight.shape[0]
        if self.precision == 'mxfp8':
            a, sa = mxfp8_quantize(a)
            weight, sb = mxfp8_quantize(weight.T if b_is_kn else weight)
            scales, b_is_kn = (sa, sb), False
        self.comm.gemm('push', a, weight, b_is_kn, self.heads[0], self.peer_heads,
                       self.counters, 0, width // segments, segments,
                       a_rows=self.capacity, head_dim=self.head_dim, bias=bias, scales=scales)

    def begin_recv(self):
        self.comm.barrier(bump=True, epoch=self.epoch, counters=self.counters)

    def run_recv(self, inputs, weight, b_is_kn, out):
        """Pusher warps publish BF16 saved values, MX payloads and scales before
        releasing each row counter. TMA then consumes that completed row tile."""
        shape = inputs[0].shape
        if any(t.shape != shape for t in inputs):
            raise ValueError('All head-side inputs must have the same shape')
        if shape[0] % self.comm.p:
            raise ValueError('Head-side rows must divide evenly over CP ranks')
        srcs = [t.data_ptr() for t in inputs]
        if self.precision == 'mxfp8':
            quantized = [mxfp8_quantize(t, k_multiple=128) for t in inputs]
            weight, sb = mxfp8_quantize(weight.T if b_is_kn else weight)
            qsrc = [q.data_ptr() for q, _ in quantized]
            ssrc = [s.data_ptr() for _, s in quantized]
            qsrc += [qsrc[0]] * (4 - len(qsrc))
            ssrc += [ssrc[0]] * (4 - len(ssrc))
            self.comm.gemm('recv', self.mx_recv, weight, False, out,
                           [self.peer_recv, self.peer_counters, self.peer_mx_recv, self.peer_mx_scale],
                           self.counters, len(inputs), 0, 1, srcs=srcs, head_dim=self.head_dim,
                           src_rows=shape[0] // self.comm.p, scales=(self.mx_scale, sb),
                           mx_sources=tuple(qsrc + ssrc))
        else:
            self.comm.recv_gemm(self.recv, weight, b_is_kn, out, self.peer_recv, self.peer_counters, self.counters,
                               len(inputs), srcs, self.head_dim, src_rows=shape[0] // self.comm.p)

@dataclass(frozen=True, eq=False)
class FusedGemmA2AWorkspace:
    """Caller-owned fixed-capacity communication workspace for QKV/G and output.

    Create after distributed initialization and before PP receives or capture.
    Pass the same object to serialized layers with matching geometry. Each
    invocation retains its own active-length view for backward.
    """

    hidden_size: int
    num_heads: int
    max_local_tokens: int
    cp_group: object = None
    gate: bool = False
    precision: str = "bf16"
    comm: object = field(init=False, repr=False)
    _buffers: dict = field(init=False, repr=False)

    def __post_init__(self):
        if self.precision not in ('bf16', 'mxfp8'):
            raise ValueError('precision must be bf16 or mxfp8')
        if self.precision == 'mxfp8' and self.hidden_size != 128 * self.num_heads:
            raise ValueError('Experimental MXFP8 requires head_dim=128')
        capacity = padded_rows(self.max_local_tokens, MMA_TILER_MN[0])
        if self.num_heads <= 0 or self.hidden_size <= 0 or self.hidden_size % self.num_heads:
            raise ValueError('hidden_size must be positive and divisible by num_heads')
        if not dist.is_initialized():
            raise RuntimeError('Initialize the CP process group before creating FusedGemmA2AWorkspace')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Create FusedGemmA2AWorkspace before CUDA Graph capture')
        group = dist.group.WORLD if self.cp_group is None else self.cp_group
        peers = dist.get_world_size(group)
        validate_peer_count(peers)
        if self.num_heads % peers or any(
                tile <= 0 or (self.hidden_size // peers) % tile
                for tile in (MMA_N, MMA_N_OUT_PUSH)):
            raise ValueError('Heads must split over CP with tile-aligned per-rank width')
        head_dim = self.hidden_size // self.num_heads
        if MMA_N != head_dim:
            raise ValueError(f'QKV needs one N tile per head: set FUSED_MMA_N={head_dim}')
        comm = _Comm.get(group)
        buffers = {}
        for segments in (3 + bool(self.gate), 1):
            key = (capacity, self.hidden_size, head_dim, segments)
            if key not in comm.projection_buffers:
                comm.projection_buffers[key] = _ProjectionBuffers(comm, *key)
            buffers[segments] = comm.projection_buffers[key]
            if self.precision == 'mxfp8':
                buffers[segments].init_mxfp8()
        object.__setattr__(self, 'cp_group', group)
        object.__setattr__(self, 'comm', comm)
        object.__setattr__(self, '_buffers', buffers)

    def _validate(self, module, tensor):
        if self.precision == 'mxfp8' and type(module.weight) not in (torch.Tensor, nn.Parameter):
            raise TypeError('MXFP8 currently takes BF16 master weights, not quantized Parameter subclasses')
        if (module.weight.shape[1] != self.hidden_size or module.num_heads != self.num_heads
                or module.segments not in self._buffers):
            raise ValueError('Workspace hidden size, head count and QKV/QKVG mode must match the projection')
        if module.weight.device != self.comm.device or torch.cuda.current_device() != self.comm.device.index:
            raise ValueError('Workspace and projection must use the current CUDA device')
        if tensor.device != self.comm.device:
            raise ValueError('Input and workspace must use the same CUDA device')
        if module.bias is not None and (module.bias.device != self.comm.device or not module.bias.is_contiguous()):
            raise ValueError('Bias must be contiguous and on the workspace CUDA device')

    def _view(self, local_tokens, segments):
        """Select active views without allocating or resizing GPU buffers."""
        capacity = padded_rows(local_tokens, MMA_TILER_MN[0])
        if local_tokens > self.max_local_tokens:
            raise ValueError(f'local_tokens={local_tokens} exceeds max_local_tokens={self.max_local_tokens}')
        buffers = self._buffers[segments]
        key = (capacity, self.precision)
        if key not in buffers.workspaces:
            buffers.workspaces[key] = _ProjectionWorkspace(buffers, capacity, self.precision)
        return buffers.workspaces[key]


def _check_workspace(workspace, module, tensor):
    if not isinstance(workspace, FusedGemmA2AWorkspace):
        raise TypeError('Pass an explicitly created FusedGemmA2AWorkspace to forward')
    workspace._validate(module, tensor)


class FusedQKVProjection(nn.Module):
    """[T/P, H] -> q [T, heads/P, D], k and v [T, kv_heads/P, D] (and a gate [T, heads/P, D] with gate=True).
    Weight is [(3 + gate) H, H] in per-head [q; k; v; optional gate] row order.
    Every returned tensor owns its storage and contains no internal tile padding."""

    weight_layout = 'head_qkvg'

    def __init__(self, hidden_size, num_heads, num_kv_heads=None, *, weight,
                 gate=False, bias=None, apply_bias=True,
                 bias_in_gemm=False, accumulate_bias_grad=True):
        """weight/bias are caller-owned tensors, retained without copying or initialization.
        Weight is required; bias=None disables bias. apply_bias=False
        leaves the forward add to the consumer (the module still produces dbias). bias_in_gemm=True
        requires apply_bias=True and adds bias before rounding the GEMM output to BF16.
        accumulate_bias_grad=False delegates parameter-dtype dbias to autograd/DDP."""
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            raise TypeError('weight must be a caller-provided tensor')
        if bias is not None and not isinstance(bias, torch.Tensor):
            raise TypeError('bias must be a caller-provided tensor or None')
        if bias_in_gemm and not apply_bias:
            raise ValueError('bias_in_gemm requires apply_bias=True; the consumer must not add bias again')
        self.bias_in_gemm = bias_in_gemm
        self.accumulate_bias_grad = accumulate_bias_grad
        self.apply_bias = apply_bias
        self.segments = 3 + bool(gate)
        num_kv_heads = num_kv_heads or num_heads
        self.num_heads = num_heads
        if num_heads <= 0 or hidden_size <= 0 or hidden_size % num_heads:
            raise ValueError('hidden_size must be positive and divisible by num_heads')
        self.head_dim = hidden_size // num_heads
        self.n_q, self.n_kv = num_heads * self.head_dim, num_kv_heads * self.head_dim
        bn = MMA_TILER_MN[1]
        if self.n_kv != self.n_q:
            raise NotImplementedError('GQA is not supported yet; Q, K and V must have equal widths')
        if bn != self.head_dim:
            raise ValueError(f'QKV needs one N tile per head: set FUSED_MMA_N={self.head_dim}')
        rows = self.segments * self.n_q
        _check(weight.data, (rows, hidden_size), 'weight')
        self.weight = weight
        if bias is not None and (tuple(bias.shape) != (rows,) or bias.dtype != torch.bfloat16):
            raise ValueError(f'bias must be a BF16 vector of {rows} entries')
        self.bias = bias

    def forward(self, x, workspace):
        _check_workspace(workspace, self, x)
        _check(x, (x.shape[0], self.weight.shape[1]), 'x')
        active = workspace._view(x.shape[0], self.segments)
        return _FusedQKV.apply(x, self.weight, self.bias, self, active)


class FusedOutputProjection(nn.Module):
    """[T, heads/P, D] -> [T/P, H]; weight is [H, H]."""

    def __init__(self, hidden_size, num_heads, *, weight,
                 bias=None, apply_bias=True, accumulate_bias_grad=True):
        """weight/bias are caller-owned tensors; weight is required and bias=None disables bias.
        apply_bias=False returns y without the bias (Megatron's skip-bias-add convention: the caller adds it
        in its fused residual) while the module still produces dbias. Set accumulate_bias_grad=False
        to return parameter-dtype dbias through autograd instead of directly writing main_grad."""
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            raise TypeError('weight must be a caller-provided tensor')
        if bias is not None and not isinstance(bias, torch.Tensor):
            raise TypeError('bias must be a caller-provided tensor or None')
        self.apply_bias = apply_bias
        self.accumulate_bias_grad = accumulate_bias_grad
        self.segments = 1
        self.num_heads = num_heads
        if num_heads <= 0 or hidden_size <= 0 or hidden_size % num_heads:
            raise ValueError('hidden_size must be positive and divisible by num_heads')
        if MMA_N_OUT_PUSH <= 0:
            raise ValueError('FUSED_MMA_N_OUT_PUSH must be positive')
        self.head_dim = hidden_size // num_heads
        _check(weight.data, (hidden_size, hidden_size), 'weight')
        self.weight = weight
        if bias is not None and (tuple(bias.shape) != (hidden_size,) or bias.dtype != torch.bfloat16):
            raise ValueError(f'bias must be a BF16 vector of {hidden_size} entries')
        self.bias = bias

    def forward(self, o, workspace):
        """Logical head-side rows [T, heads/P, D] -> sequence-side rows [T/P, H]."""
        _check_workspace(workspace, self, o)
        h, p = self.weight.shape[0], workspace.comm.p
        _check(o, (o.shape[0], h // p // self.head_dim, self.head_dim), 'o')
        if o.shape[0] % p:
            raise ValueError('Token count must be divisible by CP')
        tokens = o.shape[0] // p
        active = workspace._view(tokens, 1)
        return _FusedOutput.apply(o, self.weight, self.bias, self, active, tokens)
