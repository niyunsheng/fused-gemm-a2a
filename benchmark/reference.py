# Adapted from VeOmni (https://github.com/ByteDance-Seed/VeOmni), files
# veomni/distributed/sequence_parallel/{ulysses,backward,async_ulysses}.py.
# Copyright 2025 Bytedance Ltd. and/or its affiliates. Licensed under the Apache
# License, Version 2.0; see ../THIRD_PARTY_LICENSES.
# Modifications: THD-only tensors, RoPE/QK-norm/padding/KV-head repetition removed,
# optional gate projection and per-stage torch.profiler ranges added.
"""Unfused Ulysses projections for THD inputs: separate q, k, v GEMMs with async all-to-all.

Tokens of all packed sequences form one dimension of length T, padded by the caller
to a multiple of the CP size P. Rank r owns the contiguous tokens [rT/P, (r+1)T/P);
sequence-side tensors are [T/P, H] and head-side tensors are [T, heads/P, D]. The
all-to-all concatenates rank chunks in order, so token order and cu_seqlens are
unchanged; only attention needs them.

As in VeOmni, the QKV forward launches each projection's all-to-all asynchronously
so it overlaps the next projection's GEMM, and the backward overlaps each
gradient's all-to-all with the previous projection's weight and input gradients.

Weights are replicated across the CP group; their gradients are per-rank partial
sums and the CP all-reduce is left to the trainer. Construct the modules with
identical RNG state on every rank, or load a checkpoint; nothing here
synchronizes parameters.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function


def all_to_all_tensor(x, scatter_dim, gather_dim, group, async_op=False):
    """All-to-all on dim 0 or 1. With async_op, returns a wait() closure yielding the result."""
    sp_world_size = dist.get_world_size(group)
    assert scatter_dim <= 1 and gather_dim <= 1, "scatter_dim and gather_dim must be 0 or 1"
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape([gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
            .transpose(0, 1)
            .reshape([gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
        )
    x = x.contiguous()
    output = torch.empty_like(x)
    comm = dist.all_to_all_single(output, x, group=group, async_op=async_op)

    def wait():
        if comm is not None:
            comm.wait()
        if scatter_dim == 0:
            return torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
        return output

    return wait if async_op else wait()


def linear_input_backward(grad_output, input_tensor, weight):
    return grad_output @ weight


def linear_parameter_backward(grad_output, input_tensor, weight, has_bias):
    grad_weight = grad_output.transpose(0, 1) @ input_tensor
    grad_bias = grad_output.sum(dim=0) if has_bias else None
    return grad_weight, grad_bias


def linear_backward(grad_output, input_tensor, weight, has_bias):
    grad_input = linear_input_backward(grad_output, input_tensor, weight)
    grad_weight, grad_bias = linear_parameter_backward(grad_output, input_tensor, weight, has_bias)
    return grad_input, grad_weight, grad_bias


class AsyncUlyssesQKVProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
                head_dim, group, gate_weight=None, gate_bias=None):
        ctx.has_gate = gate_weight is not None
        ctx.group = group
        ctx.save_for_backward(hidden_states, q_weight, q_bias, k_weight, k_bias,
                              v_weight, v_bias, gate_weight, gate_bias)
        with record_function('qkvg_forward' if ctx.has_gate else 'qkv_forward'):
            projections = [(q_weight, q_bias), (k_weight, k_bias), (v_weight, v_bias)]
            if ctx.has_gate:
                projections.append((gate_weight, gate_bias))
            pending = []
            for weight, bias in projections:
                # Launch each A2A before the next GEMM, including V -> gate.
                y = F.linear(hidden_states, weight, bias).view(hidden_states.shape[0], -1, head_dim)
                pending.append(all_to_all_tensor(y, 1, 0, group, async_op=True))
            return tuple(wait() for wait in pending)

    @staticmethod
    def backward(ctx, *head_grads):
        with record_function('qkvg_backward' if ctx.has_gate else 'qkv_backward'):
            hidden_states, *params = ctx.saved_tensors
            # Positions in forward's arguments; head_dim/group separate V from gate.
            weight_indices = [1, 3, 5] + ([9] if ctx.has_gate else [])
            grad_inputs = [None] * len(ctx.needs_input_grad)
            dxs = [None] * len(head_grads)
            pending = all_to_all_tensor(head_grads[-1], 0, 1, ctx.group, async_op=True)
            for i in reversed(range(len(head_grads))):
                grad = pending().flatten(1)
                if i:
                    # Receive V while computing gate dgrad/wgrad, then K during V, etc.
                    pending = all_to_all_tensor(head_grads[i - 1], 0, 1, ctx.group, async_op=True)
                weight, bias = params[2 * i:2 * i + 2]
                wi = weight_indices[i]
                dxs[i], grad_inputs[wi], grad_inputs[wi + 1] = linear_backward(
                    grad, hidden_states, weight,
                    has_bias=bias is not None and ctx.needs_input_grad[wi + 1])
            # Preserve Q/K/V's accumulation order; append gate's contribution last.
            grad_inputs[0] = dxs[0]
            for dx in dxs[1:]:
                grad_inputs[0] = grad_inputs[0] + dx
            return tuple(grad_inputs)


class AsyncUlyssesOutputProjection(torch.autograd.Function):
    @staticmethod
    @record_function('out_forward')
    def forward(ctx, hidden_states, proj_weight, proj_bias, group):
        ctx.head_dim = hidden_states.shape[-1]
        hidden_states = all_to_all_tensor(hidden_states, scatter_dim=0, gather_dim=1, group=group).flatten(1)
        ctx.group = group
        ctx.save_for_backward(hidden_states, proj_weight, proj_bias)
        return F.linear(hidden_states, proj_weight, proj_bias)

    @staticmethod
    @record_function('out_backward')
    def backward(ctx, grad_output):
        hidden_states, proj_weight, proj_bias = ctx.saved_tensors

        # output grad, then launch its communication
        grad_o = linear_input_backward(grad_output, hidden_states, proj_weight)
        grad_o = grad_o.view(grad_o.shape[0], -1, ctx.head_dim)
        grad_out_res = all_to_all_tensor(grad_o, scatter_dim=1, gather_dim=0, group=ctx.group, async_op=True)

        # parameter grads overlap the communication
        grad_proj_weight, grad_proj_bias = linear_parameter_backward(
            grad_output, hidden_states, proj_weight, has_bias=proj_bias is not None and ctx.needs_input_grad[2])

        return grad_out_res(), grad_proj_weight, grad_proj_bias, None


class UlyssesQKVProjection(nn.Module):
    """Separate q, k, v (optionally gate) GEMMs with overlapped all-to-alls.

    [T/P, H] -> q [T, heads/P, D], k and v [T, kv_heads/P, D].
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads=None, bias=False, cp_group=None,
                 device=None, dtype=None, gate=False):
        super().__init__()
        self.group = dist.group.WORLD if cp_group is None else cp_group
        num_kv_heads = num_kv_heads or num_heads
        p = dist.get_world_size(self.group)
        if hidden_size % num_heads or num_heads % p or num_kv_heads % p:
            raise ValueError('hidden_size % num_heads, num_heads % CP, and num_kv_heads % CP must be 0')
        self.head_dim = hidden_size // num_heads
        kw = dict(bias=bias, device=device, dtype=dtype)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, **kw)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, **kw)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, **kw)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, **kw) if gate else None

    @property
    def projections(self):
        return (self.q_proj, self.k_proj, self.v_proj) + (
            (self.gate_proj,) if self.gate_proj is not None else ())

    def forward(self, x):
        return AsyncUlyssesQKVProjection.apply(
            x, self.q_proj.weight, self.q_proj.bias, self.k_proj.weight, self.k_proj.bias,
            self.v_proj.weight, self.v_proj.bias, self.head_dim, self.group,
            self.gate_proj.weight if self.gate_proj is not None else None,
            self.gate_proj.bias if self.gate_proj is not None else None)


class UlyssesOutputProjection(nn.Module):
    """Head-to-token all-to-all then output GEMM: [T, heads/P, D] -> [T/P, H]."""

    def __init__(self, hidden_size, num_heads, bias=False, cp_group=None, device=None, dtype=None):
        super().__init__()
        self.group = dist.group.WORLD if cp_group is None else cp_group
        if hidden_size % num_heads or num_heads % dist.get_world_size(self.group):
            raise ValueError('hidden_size % num_heads and num_heads % CP must be 0')
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

    def forward(self, o):
        return AsyncUlyssesOutputProjection.apply(o, self.o_proj.weight, self.o_proj.bias, self.group)


def interleave_qkv_rows(segment_major, heads, head_dim):
    """Weight rows (or weight-gradient rows) from the segment-major order [Q; K; V(; gate)] to
    Per-head order [q_h; k_h; v_h(; gate_h)]. Works
    on any tensor whose first dimension holds the rows."""
    h = heads * head_dim
    if segment_major.shape[0] not in (3 * h, 4 * h):
        raise ValueError(f'{segment_major.shape[0]} rows are not 3 or 4 segments of {h}')
    segments = segment_major.shape[0] // h
    shape = segment_major.shape[1:]
    per_head = segment_major.reshape((segments, heads, head_dim) + shape).transpose(0, 1)
    return per_head.reshape(segment_major.shape)


def mxfp8_mm(a, b):
    """Native PyTorch MXFP8 GEMM on the same quantized operands as CuTe.

    This checks GEMM parity, not the shared quantizer's scale-selection policy.
    Kept outside benchmark timing; no dequantized matmul fallback.
    """
    from fused_gemm_a2a_kernels import mxfp8_quantize
    aq, sa = mxfp8_quantize(a)
    bq, sb = mxfp8_quantize(b.T)
    out = F.scaled_mm(
        aq, bq.T, sa.view(torch.float8_e8m0fnu), F.ScalingType.BlockWise1x32,
        sb.view(torch.float8_e8m0fnu), F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4, output_dtype=torch.float32)
    return out[:a.shape[0], :b.shape[1]]


@torch.no_grad()
def mxfp8_projections(x, grads, o, dy, qkv_weight, out_weight, qkv_bias=None, out_bias=None,
                     *, heads, bias_in_gemm=False):
    """Native PyTorch MXFP8 GEMMs plus gathered BF16 transport.

    Wgrad requantizes each rank's reduction axis before gradient reduction.
    Returned weight/bias gradients are FP32 rank-local contributions.
    """
    p, rank = dist.get_world_size(), dist.get_rank()
    m, h = x.shape
    d, segments = h // heads, len(grads)
    head_slice = slice(rank * (heads // p), (rank + 1) * (heads // p))
    rows = slice(rank * m, (rank + 1) * m)

    def gather(tensor, dim):
        parts = [torch.empty_like(tensor) for _ in range(p)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.cat(parts, dim=dim)

    raw = mxfp8_mm(x, qkv_weight.T)
    if qkv_bias is not None:
        raw = (raw if bias_in_gemm else raw.to(x.dtype).float()) + qkv_bias.float()
    packed = gather(raw.to(x.dtype), 0).view(p * m, heads, segments, d)
    qs = tuple(packed[:, head_slice, s].contiguous() for s in range(segments))
    grad = gather(torch.stack(grads, dim=2), 1).reshape(p * m, segments * h)[rows]
    dx, dwq = mxfp8_mm(grad, qkv_weight).to(x.dtype), mxfp8_mm(grad.T, x)
    oseq = gather(o, 1).flatten(1)[rows]
    y = mxfp8_mm(oseq, out_weight.T).to(o.dtype)
    if out_bias is not None:
        y = y + out_bias
    full_do = gather(mxfp8_mm(dy, out_weight).to(o.dtype), 0).view(p * m, heads, d)
    return dict(qkv=qs, dx=dx, dwq=dwq, dbq=grad.float().sum(0), y=y,
                do=full_do[:, head_slice].contiguous(), dwo=mxfp8_mm(dy.T, oseq),
                dbo=dy.float().sum(0))
