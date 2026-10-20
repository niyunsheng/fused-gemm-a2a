# Derived from NVIDIA CUTLASS Blackwell persistent GEMM examples.
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# BSD-3-Clause terms: THIRD_PARTY_LICENSES.
# Modifications: Ulysses peer transport, synchronization, projection layout and autograd.
"""CuTe GEMM/peer transport and Triton copies for fused_gemm_a2a.

Push: the GEMM epilogue writes directly into each peer's head-side buffer.
Receive: pusher warps publish row counters after peer writes; TMA waits before
consuming a row block. The same transport zero-fills internal padding.
Device barriers protect shared-buffer reuse, including CUDA Graph replay.
"""
import os
from math import gcd

import cuda.bindings.driver as cuda
import torch
import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Boolean, Constexpr, Int32, Int64, T, const_expr, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import distributed
from cutlass.utils.gemm.sm100 import (epilogue_smem_copy_and_partition, epilogue_tma_store,
                                      epilogue_tmem_copy_and_partition, transform_partitioned_tensor_layout)

CTA_M = 128  # rows per CTA and per receive counter; a pusher job covers CTA_M // split rows


def validate_peer_count(peers):
    """The device barrier assigns one warp lane to each peer."""
    if not 1 <= peers <= 32:
        raise ValueError('The single-warp device barrier supports CP sizes from 1 to 32')


def validate_tile_n(mode, tile_n, output_width, peers, segment_width=0, num_segments=1, head_dim=0):
    """Do not truncate scheduler tiles or let a push tile cross a peer/head boundary.

    Interleaved receive traffic permutes the GEMM's K dimension, independently
    of output N; only the push epilogue maps one N tile to one head.
    """
    if mode not in ('push', 'recv') or tile_n <= 0 or output_width <= 0 or output_width % tile_n:
        raise ValueError('GEMM mode must be push/recv and tile N must divide output width')
    if num_segments not in (1, 3, 4):
        raise ValueError('Expected one output projection or three/four QKV projections')
    if num_segments > 1 and head_dim <= 0:
        raise ValueError('QKV needs a positive head dimension')
    if mode == 'push':
        if peers <= 0 or segment_width <= 0 or segment_width % (peers * tile_n):
            raise ValueError('Push segment width must be a multiple of CP * tile N')
        if num_segments > 1 and tile_n != head_dim:
            raise ValueError('QKV push needs one N tile per head')


def tile_swizzle(grouped_clusters, requested):
    """Swizzle group size for the persistent tile scheduler. CUTLASS pads the grouped dimension to a
    multiple of the swizzle and marks the padding tiles valid; a local TMA epilogue clips them, but
    remote stores and counter waits cannot, so the group must divide the cluster count exactly
    (CP4 with 1024 tokens per rank hung before this)."""
    if grouped_clusters < 1 or requested < 1:
        raise ValueError('Cluster extent and FUSED_SWIZZLE must be positive')
    return gcd(grouped_clusters, requested)


def push_split(k, peers, segments, requested=0):
    """Row sub-blocks per (row block, destination) pusher job. Jobs of about 64 KB: enough rows to
    amortize the signal, few enough jobs in flight that row blocks complete early and in order. The
    split must be a power of two so the sub-blocks cover all CTA_M rows (a 1536-column segment would
    otherwise give 6 sub-blocks of 21 rows and drop two rows) and each job keeps the four rows the
    copy layout needs."""
    if peers < 1 or segments < 1 or k % (peers * segments):
        raise ValueError('Receive K must split evenly over peers and input segments')
    width = k // (peers * segments)
    if width % 64:
        raise ValueError('The per-peer segment width must be a multiple of 64 BF16 elements')
    split = requested or max(1, min(CTA_M // 8, CTA_M * width * 2 // (64 * 1024)))
    if not requested:
        split = 1 << (split.bit_length() - 1)
    if split < 1 or CTA_M % split or (CTA_M // split) % 4:
        raise ValueError('FUSED_PUSH_SPLIT must divide 128 and leave at least four rows per job')
    return split


def padded_rows(valid, tile_m):
    """Rows the per-rank buffers hold for `valid` logical rows: the next multiple of `tile_m`."""
    if valid < 1 or tile_m < 1:
        raise ValueError('Token count and tile height must be positive')
    return -(-valid // tile_m) * tile_m

USE_2CTA = bool(int(os.environ.get('FUSED_2CTA', 1)))   # 256x256 MMA across a CTA pair, as cuBLAS uses
MMA_N = int(os.environ.get('FUSED_MMA_N', 128))        # MMA tile N; per-rank head width must be a multiple
MMA_N_RECV = int(os.environ.get('FUSED_MMA_N_RECV', 256))  # recv output N is independent of interleaved K
MMA_N_OUT_PUSH = int(os.environ.get('FUSED_MMA_N_OUT_PUSH', MMA_N))  # output dgrad has no interleaved rows
MMA_TILER_MN = (256 if USE_2CTA else 128, MMA_N)
CLUSTER_N = int(os.environ.get('FUSED_CLUSTER_N', 1))    # CTAs along N per cluster: 2 multicasts A tiles across the pair
CLUSTER_MN = (2 if USE_2CTA else 1, CLUSTER_N)
RASTER_ALONG_M = os.environ.get('FUSED_RASTER', 'n') == 'm'   # tile order: 'n' visits the N tiles of a row group first
SWIZZLE = int(os.environ.get('FUSED_SWIZZLE', 4))       # clusters per row group when rastering along N (recv GEMM)
SWIZZLE_PUSH = int(os.environ.get('FUSED_SWIZZLE_PUSH', 16))  # same for the push GEMM: 16 measured 5% faster than 8
PUSHER_WARPS = int(os.environ.get('FUSED_PUSHER_WARPS', 4))  # max pusher warps per recv GEMM program while the first row group is pushed
PUSHER_STEADY_WARPS = int(os.environ.get('FUSED_PUSHER_STEADY_WARPS', 1))  # warps left after the first row group
if not 1 <= PUSHER_STEADY_WARPS <= max(PUSHER_WARPS, 1):
    raise ValueError('FUSED_PUSHER_STEADY_WARPS must be between 1 and FUSED_PUSHER_WARPS')
PUSH_SPLIT = int(os.environ.get('FUSED_PUSH_SPLIT', 0))      # row sub-blocks per (row block, destination) job; 0 = ~64 KB jobs
PUSH_CHUNK = 8  # 16-byte vectors in flight per thread between a batch of loads and a batch of stores
PUSHER_CHUNK = int(os.environ.get('FUSED_PUSHER_CHUNK', 16))  # same, for the pusher warps (probe: 16 > 8)
PUSHER_STREAMING = os.environ.get('FUSED_PUSHER_CS', '1') == '1'  # ld/st.global.cs in the pushers (evict-first)


if hasattr(distributed, "red_release_sys_add1"):
    red_release_sys_add1 = distributed.red_release_sys_add1
else:
    @dsl_user_op
    def red_release_sys_add1(ptr, *, loc=None, ip=None):
        """Publish a completed remote copy with system-scope release ordering."""
        # Some 4.5 builds expose red_add1 but omit its cute.arch.red dependency.
        # Emit the same PTX operation directly, as for our acquire/release loads.
        llvm.inline_asm(
            None,
            [ptr.toint().ir_value(loc=loc, ip=ip), Int32(1).ir_value(loc=loc, ip=ip)],
            "red.release.sys.global.add.s32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def ld_acquire_sys_u32(ptr, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [ptr.toint().ir_value(loc=loc, ip=ip)],
                                 "ld.acquire.sys.global.u32 $0, [$1];", "=r,l",
                                 has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def ld_acquire_sys_u64(ptr, *, loc=None, ip=None) -> Int64:
    return Int64(llvm.inline_asm(T.i64(), [ptr.toint().ir_value(loc=loc, ip=ip)],
                                 "ld.acquire.sys.global.u64 $0, [$1];", "=l,l",
                                 has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None) -> None:
    """Back off a spinning warp so that polling does not contend with the traffic it waits for."""
    llvm.inline_asm(None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r",
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def st_release_sys_u64(ptr, value: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint().ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
                    "st.release.sys.global.u64 [$0], $1;", "l,l",
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def ld_cs_128(ptr, *, loc=None, ip=None):
    """16-byte load with the streaming cache operator: the line is allocated evict-first in L1 and
    L2, so one-pass pusher traffic does not displace the GEMM's operand tiles."""
    regs = llvm.inline_asm(ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"), [ptr.toint().ir_value(loc=loc, ip=ip)],
                           "ld.global.cs.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l",
                           has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return tuple(llvm.extractvalue(T.i32(), regs, [i]) for i in range(4))


@dsl_user_op
def ld_cs_128_zfill(ptr, valid, *, loc=None, ip=None):
    """A predicated vector load: invalid rows return zero without reading memory."""
    regs = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [ptr.toint().ir_value(loc=loc, ip=ip), Int32(valid).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.u32 p, $5, 0; "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; "
        "@p ld.global.cs.v4.b32 {$0, $1, $2, $3}, [$4]; }", "=r,=r,=r,=r,l,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return tuple(llvm.extractvalue(T.i32(), regs, [i]) for i in range(4))


@dsl_user_op
def st_cs_128(ptr, x, y, z, w, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint().ir_value(loc=loc, ip=ip), x, y, z, w],
                    "st.global.cs.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@cute.kernel
def _barrier_kernel(flags: cute.Tensor, peer_flags: cute.Tensor, epoch_out: cute.Tensor,
                    counters: cute.Pointer, counter_count: Int32, RANK: Constexpr, P: Constexpr, BUMP: Constexpr):
    """Epoch barrier over the CP group. With BUMP, reset row counters before publishing entry
    and advance the receive invocation epoch. All state updates also run on graph replay."""
    tidx, _, _ = cute.arch.thread_idx()
    if const_expr(BUMP):
        # Reset this invocation's active row counters before publishing entry.
        # A previous shorter invocation need not have touched the same rows.
        rows = cute.make_tensor(counters, cute.make_layout((counter_count,), stride=(1,)))
        for i in range(tidx, counter_count, 32):
            rows[i] = 0
        cute.arch.sync_warp()
    epoch = ld_acquire_sys_u64(flags.iterator + RANK) + 1
    if tidx < P:
        peer_slot = cute.make_ptr(cutlass.Int64, peer_flags[tidx] + RANK * 8, cute.AddressSpace.gmem, assumed_align=8)
        st_release_sys_u64(peer_slot, epoch)
        seen = ld_acquire_sys_u64(flags.iterator + tidx)
        while seen < epoch:
            nanosleep(256)
            seen = ld_acquire_sys_u64(flags.iterator + tidx)
    if const_expr(BUMP):
        if tidx == 0:
            epoch_out[0] = epoch_out[0] + 1


@cute.jit
def barrier(flags: cute.Tensor, peer_flags: cute.Tensor, epoch_out: cute.Tensor, RANK: Constexpr, P: Constexpr,
            BUMP: Constexpr, counters: cute.Pointer, counter_count: Int32, stream: cuda.CUstream):
    _barrier_kernel(flags, peer_flags, epoch_out, counters, counter_count, RANK, P, BUMP).launch(
        grid=[1, 1, 1], block=[32, 1, 1], stream=stream)


# ----------------------------------------------------------------------------
# Block copy helper shared by the push epilogue and the pusher warps
# ----------------------------------------------------------------------------

def _copy_in_chunks(tiled_copy, tS, tD, chunk=PUSH_CHUNK):
    """Copy a thread-partitioned (V, I, J) tile in batches of `chunk` vectors: all loads of a batch
    are issued before its stores, so the load latency is paid once per batch instead of once per
    vector (global stores may alias, which stops the compiler from reordering them itself)."""
    pairs = [(i, j) for i in range(cute.size(tS, mode=[1])) for j in range(cute.size(tS, mode=[2]))]
    for start in range(0, len(pairs), chunk):
        batch = pairs[start:start + chunk]
        frags = [cute.make_fragment_like(tS[(None, i, j)]) for i, j in batch]
        for (i, j), frag in zip(batch, frags):
            cute.copy(tiled_copy, tS[(None, i, j)], frag)
        for (i, j), frag in zip(batch, frags):
            cute.copy(tiled_copy, frag, tD[(None, i, j)])


def _copy_streaming(tS, tD, chunk, coords=None, row0=0, src_rows=0):
    """_copy_in_chunks with ld/st.global.cs: the pusher reads its input once and writes each
    destination once, so neither should be allowed to displace the GEMM's A and B tiles in L2."""
    pairs = [(i, j) for i in range(cute.size(tS, mode=[1])) for j in range(cute.size(tS, mode=[2]))]
    for start in range(0, len(pairs), chunk):
        batch = pairs[start:start + chunk]
        regs = ([ld_cs_128(tS[(None, i, j)].iterator) for i, j in batch] if coords is None else
                [ld_cs_128_zfill(tS[(None, i, j)].iterator,
                                 row0 + coords[(0, i, j)][0] < src_rows) for i, j in batch])
        for (i, j), r in zip(batch, regs):
            st_cs_128(tD[(None, i, j)].iterator, *r)


@cute.jit
def _copy_recv_tile(tiled_copy: cute.TiledCopy, copy: cute.ThrCopy,
                    src: cute.Tensor, dst: cute.Tensor, row0: Int32,
                    src_rows: Int32, padded: Constexpr):
    if const_expr(padded):
        if row0 + cute.size(src, mode=[0]) > src_rows:
            # Every 16-byte vector stays in one row. Read logical rows directly
            # and zero-fill the receive buffer's tail, including fully empty tiles.
            tS, tD = copy.partition_S(src), copy.partition_D(dst)
            coords = copy.partition_S(cute.make_identity_tensor(src.shape))
            _copy_streaming(tS, tD, PUSHER_CHUNK, coords, row0, src_rows)
        else:
            if const_expr(PUSHER_STREAMING):
                _copy_streaming(copy.partition_S(src), copy.partition_D(dst), PUSHER_CHUNK)
            else:
                _copy_in_chunks(tiled_copy, copy.partition_S(src), copy.partition_D(dst), PUSHER_CHUNK)
    else:
        if const_expr(PUSHER_STREAMING):
            _copy_streaming(copy.partition_S(src), copy.partition_D(dst), PUSHER_CHUNK)
        else:
            _copy_in_chunks(tiled_copy, copy.partition_S(src), copy.partition_D(dst), PUSHER_CHUNK)


@cute.jit
def _push_one(job: Int32, lane: Int32, p: Constexpr, rank: Constexpr, m_local: Constexpr, k_total: Constexpr,
              num_in: Constexpr, split: Constexpr, head_dim: Constexpr,
              dst_ptrs: cute.Tensor, cnt_ptrs: cute.Tensor,
              src0: Int64, src1: Int64, src2: Int64, src3: Int64,
              src_rows: Int32, padded: Constexpr):
    """One pusher job of one warp. Job = (row block, segment, destination, row sub-block): copy the
    sub-block of this rank's head-side input into the destination's receive buffer, then release the
    destination's counter for that row block. QKV receive columns follow Megatron's
    [head][q|k|v|optional gate][D] order. Output-projection inputs
    have ordinary contiguous head columns."""
    dtype = cutlass.BFloat16
    seg_width = k_total // (num_in * p)   # columns of one segment from one rank
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
    thr_layout = cute.make_layout((4, 8), stride=(8, 1))
    val_layout = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(lane)
    sub_rows = CTA_M // split
    src_layout = cute.make_layout((sub_rows, seg_width), stride=(seg_width, 1))
    dst_layout = cute.make_layout((sub_rows, seg_width), stride=(k_total, 1))
    jobs_per_block = num_in * p * split
    mi = job // jobs_per_block
    rest = job % jobs_per_block
    t = rest // (p * split)
    d = ((rest // split) % p + rank) % p
    q = rest % split
    row0 = mi * CTA_M + q * sub_rows
    base = src0
    if t == 1:
        base = src1
    if t == 2:
        base = src2
    if t == 3:
        base = src3
    if const_expr(padded):
        src_base = base + (Int64(d) * src_rows + row0) * seg_width * 2
    else:
        src_base = base + (d * m_local + row0) * seg_width * 2
    if const_expr(num_in > 1):
        # One head at a time: its D columns are contiguous on both sides, the heads are not.
        heads_per_rank = seg_width // head_dim
        head_src_layout = cute.make_layout((sub_rows, head_dim), stride=(seg_width, 1))
        head_dst_layout = cute.make_layout((sub_rows, head_dim), stride=(k_total, 1))
        for j in range(heads_per_rank):
            head = rank * heads_per_rank + j
            col = (head * num_in + t) * head_dim
            src = cute.make_tensor(cute.make_ptr(dtype, src_base + j * head_dim * 2, cute.AddressSpace.gmem,
                                                 assumed_align=16), head_src_layout)
            dst = cute.make_tensor(cute.make_ptr(dtype, dst_ptrs[d] + (row0 * k_total + col) * 2,
                                                 cute.AddressSpace.gmem, assumed_align=16), head_dst_layout)
            _copy_recv_tile(tiled_copy, thr_copy, src, dst, row0, src_rows, padded)
    else:
        src = cute.make_tensor(cute.make_ptr(dtype, src_base, cute.AddressSpace.gmem, assumed_align=16), src_layout)
        dst = cute.make_tensor(cute.make_ptr(
            dtype, dst_ptrs[d] + (row0 * k_total + rank * seg_width) * 2,
            cute.AddressSpace.gmem, assumed_align=16), dst_layout)
        _copy_recv_tile(tiled_copy, thr_copy, src, dst, row0, src_rows, padded)
    # The warp barrier orders every lane's stores before lane 0's release, and a release is
    # cumulative, so no separate fence.acq_rel.sys is needed; the probe measured the fence at
    # about 15% of pusher bandwidth with 64 KB jobs.
    cute.arch.sync_warp()
    if lane == 0:
        red_release_sys_add1(cute.make_ptr(cutlass.Int32, cnt_ptrs[d] + mi * 4, cute.AddressSpace.gmem,
                                           assumed_align=4))


@cute.jit
def _push_jobs(lane: Int32, program: Int32, pusher: Int32, num_programs: Int32, warps: Constexpr,
               steady_warps: Constexpr, first_blocks: Constexpr, p: Constexpr, rank: Constexpr,
               m_local: Constexpr, k_total: Constexpr, num_in: Constexpr, split: Constexpr,
               head_dim: Constexpr,
               dst_ptrs: cute.Tensor, cnt_ptrs: cute.Tensor,
               src0: Int64, src1: Int64, src2: Int64, src3: Int64, src_rows: Int32, padded: Constexpr):
    """Pusher job loop of warp `pusher` (0..warps-1) inside recv GEMM program `program`.
    Jobs are ordered by row block and
    spread over the warps, so row blocks complete early and in order. Two phases: all warps push the
    first `first_blocks` row blocks (the GEMM has nothing to do until they arrive, so their
    contention is free), then only the first `steady_warps` warps of each program continue, which
    keeps the GEMM fast once it is running."""
    jobs_per_block = num_in * p * split
    num_jobs = (m_local // CTA_M) * jobs_per_block
    first_jobs = min(first_blocks, m_local // CTA_M) * jobs_per_block
    for job in range(program * warps + pusher, first_jobs, num_programs * warps):
        _push_one(job, lane, p, rank, m_local, k_total, num_in, split, head_dim, dst_ptrs, cnt_ptrs,
                  src0, src1, src2, src3, src_rows, padded)
    if pusher < steady_warps:
        for job in range(first_jobs + program * steady_warps + pusher, num_jobs, num_programs * steady_warps):
            _push_one(job, lane, p, rank, m_local, k_total, num_in, split, head_dim, dst_ptrs,
                      cnt_ptrs, src0, src1, src2, src3, src_rows, padded)


# ----------------------------------------------------------------------------
# Persistent GEMM with push epilogue or receive-side waits
# ----------------------------------------------------------------------------


def _compute_stages(tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, c_dtype, smem_capacity, c_smem_layout):
    """(ACC stages, A/B stages, C stages), as in the CUTLASS persistent GEMM example."""
    num_acc_stage, num_c_stage = 2, 2
    a_one = utils.sm100.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
    b_one = utils.sm100.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
    ab_bytes = cute.size_in_bytes(a_dtype, a_one) + cute.size_in_bytes(b_dtype, b_one)
    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    num_ab_stage = (smem_capacity - (1024 + c_bytes_per_stage * num_c_stage)) // ab_bytes
    num_c_stage += (smem_capacity - ab_bytes * num_ab_stage - (1024 + c_bytes_per_stage * num_c_stage)) // c_bytes_per_stage
    return num_acc_stage, num_ab_stage, num_c_stage


class UlyssesGemm:
    """Blackwell persistent TMA/tcgen05 GEMM C[M, N] = A[M, K] @ B[N, K]^T with Ulysses-specific
    scheduling. mode 'push': tiles are stored into peer head-side buffers. mode 'recv': the TMA
    warp waits for each row block's counter before loading it."""

    def __init__(self, mode: str, p: int, rank: int, n_segment: int, num_segments: int,
                 head_dim: int = 0, mma_n: int = None):
        self.mode = mode
        self.pad_recv = False
        # QKV uses per-head [q | k | v | optional gate] rows.
        # Its push epilogue needs one N tile per head; recv permutes K only.
        self.head_dim = head_dim
        self.b_is_kn = False
        self.has_bias = False
        self.p, self.rank = p, rank
        self.n_segment = n_segment          # weight rows per [Q|K|V] segment (push mode)
        self.a_rows = None                  # tile rows of A (set by the host; >= logical rows)
        self.num_segments = num_segments
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = USE_2CTA
        self.cluster_shape_mn = CLUSTER_MN
        self.mma_tiler_mn = (MMA_TILER_MN[0], MMA_N if mma_n is None else mma_n)
        self.mma_tiler = (*self.mma_tiler_mn, 1)
        self.use_tma_store = True
        self.arch = "sm_100"
        self.cta_group = tcgen05.CtaGroup.TWO if USE_2CTA else tcgen05.CtaGroup.ONE
        self.num_in = 1
        self.push_split = 1
        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        # recv mode adds pusher warps after the TMA warp; they copy this rank's head-side input into
        # the peers' receive buffers while the other warps compute, so the two always share the SM.
        self.pusher_warps = PUSHER_WARPS if mode == 'recv' else 0
        self.threads_per_cta = 32 * (6 + self.pusher_warps)
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(self.a_dtype, self.a_major_mode, self.b_major_mode,
                                                  self.acc_dtype, self.cta_group, self.mma_tiler[:2])

    def _setup_attributes(self):
        tiled_mma = self._create_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * 4)
        self.cta_tile_shape_mnk = (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
                                   self.mma_tiler[1], self.mma_tiler[2])
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)),
                                                     (tiled_mma.thr_id.shape,))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(self.cta_tile_shape_mnk, self.use_2cta_instrs,
                                                                self.c_layout, self.c_dtype)
        c_smem_layout = utils.sm100.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, 1)
        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma, self.mma_tiler, self.a_dtype, self.b_dtype, self.c_dtype, self.smem_capacity, c_smem_layout)
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile,
                                                                     self.num_c_stage)
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype,
                                                                   self.num_ab_stage)
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype,
                                                                   self.num_ab_stage)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch=self.arch)

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, c: cute.Pointer, bias: cute.Pointer, aux0: cute.Pointer, aux1: cute.Pointer,
                 aux2: cute.Pointer, aux3: cute.Pointer, counters: cute.Pointer,
                 src0: Int64, src1: Int64, src2: Int64, src3: Int64, src_rows: Int32,
                 max_active_clusters: Constexpr, stream: cuda.CUstream):
        """Arguments are device pointers; the (static) shapes live in a_shape, b_shape and c_shape, so
        the host passes bare addresses and never keeps a tensor alive. a [M, K]; b [N, K] or [K, N]
        (b_is_kn); c [M', N'] is the real output in recv mode and the local head-side buffer in push
        mode (same layout as every peer's). Push mode: aux0..3 are the Int64 peer pointer arrays of the
        q, k, v and optional gate head-side buffers. Recv
        mode: aux0 holds the peers' receive-buffer addresses, aux1 their counter addresses, counters is
        this rank's Int32 row-block counter array, and
        src0..3 the addresses of the head-side inputs [T, C] of each segment that the pusher warps
        distribute."""
        (m, k), (c_m, c_n) = self.a_shape, self.c_shape
        # A keeps its logical row count: TMA zero-fills the rows of the last tile beyond m, so a
        # push GEMM over a non-aligned token count needs no zero-padded copy of its input. Tiles are
        # enumerated over a_rows (a tile multiple, equal to m when the shape is aligned).
        a = cute.make_tensor(a, cute.make_layout((m, k, 1), stride=(k, 1, m * k)))
        if const_expr(self.b_is_kn):
            k_b, n = self.b_shape
            b = cute.make_tensor(b, cute.make_layout((n, k_b, 1), stride=(1, n, k_b * n)))
        else:
            n, k_b = self.b_shape
            b = cute.make_tensor(b, cute.make_layout((n, k_b, 1), stride=(k_b, 1, k_b * n)))
        c = cute.make_tensor(c, cute.make_layout((c_m, c_n, 1), stride=(c_n, 1, c_m * c_n)))
        peers = cute.make_layout((self.p,), stride=(1,))
        aux0, aux1, aux2, aux3 = (cute.make_tensor(x, peers) for x in (aux0, aux1, aux2, aux3))
        counter_count = max(1, c_m // CTA_M)
        counters = cute.make_tensor(counters, cute.make_layout((counter_count,), stride=(1,)))
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.num_tma_load_bytes = (cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout)) * atom_thr_size
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile)

        self.recv_m = cute.size(a, mode=[0])
        self.recv_k = cute.size(a, mode=[1])
        # Tiles are enumerated over the logical output [M, N]: M from A, N from B.
        num_tiles_mnl = (self.a_rows // self.cta_tile_shape_mnk[0],
                         cute.size(b, mode=[0]) // self.cta_tile_shape_mnk[1], 1)
        # Rastering along N keeps a wave of programs within a few row blocks: the recv GEMM can start on
        # the first blocks while later ones are still arriving, and the push GEMM spreads a wave's
        # stores over all destinations instead of one. The swizzle groups SWIZZLE clusters of rows so
        # each weight tile is still reused across the group.
        # The swizzle group must divide the grouped cluster count exactly (see tile_swizzle).
        grouped_dim = 1 if RASTER_ALONG_M else 0
        swizzle = tile_swizzle(num_tiles_mnl[grouped_dim] // self.cluster_shape_mn[grouped_dim],
                               SWIZZLE_PUSH if self.mode == 'push' else SWIZZLE)
        self.swizzle = swizzle
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_tiles_mnl, (*self.cluster_shape_mn, 1), swizzle_size=swizzle, raster_along_m=RASTER_ALONG_M)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        self.kernel(tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_c, tma_tensor_c, c,
                    bias, aux0, aux1, aux2, aux3, counters, src0, src1, src2, src3, src_rows,
                    self.cluster_layout_vmnk,
                    self.a_smem_layout_staged, self.b_smem_layout_staged, self.c_smem_layout_staged,
                    self.epi_tile, tile_sched_params).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1), stream=stream,
            min_blocks_per_mp=1)  # bounds registers to the 10-warp program

    def _tile_mapping(self, m_idx, n_idx):
        """Logical tile (m, n) -> (A row tile, B row tile, destination rank, column tile within the
        destination, segment). Push mode rotates destinations by rank; recv mode is the identity."""
        if const_expr(self.mode == 'push'):
            bn = self.cta_tile_shape_mnk[1]
            s0 = self.n_segment // bn                 # tiles per segment
            per_peer = self.n_segment // self.p // bn  # tiles per destination within a segment
            segment = n_idx // s0
            local = n_idx % s0
            dest = (local // per_peer + self.rank) % self.p
            within = local % per_peer
            if const_expr(self.num_segments > 1):
                head = dest * per_peer + within       # one tile per head (bn == head_dim)
                # One contiguous Q/K/V/(gate) group per head.
                b_tile = head * self.num_segments + segment
            else:
                b_tile = dest * per_peer + within
            return m_idx, b_tile, dest, within, segment
        return m_idx, n_idx, Int32(0), n_idx, Int32(0)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
               tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, tma_atom_c: cute.CopyAtom, mC_mnl: cute.Tensor,
               mC_real: cute.Tensor, bias: cute.Pointer, aux0: cute.Tensor, aux1: cute.Tensor, aux2: cute.Tensor, aux3: cute.Tensor,
               counters: cute.Tensor, src0: Int64, src1: Int64,
               src2: Int64, src3: Int64, src_rows: Int32,
               cluster_layout_vmnk: cute.Layout,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               c_smem_layout_staged: cute.ComposedLayout, epi_tile: cute.Tile,
               tile_sched_params: utils.PersistentTileSchedulerParams):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if const_expr(self.mode == 'recv'):
                cpasync.prefetch_descriptor(tma_atom_c)
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_x, grid_y, grid_z = cute.arch.grid_dim()
        cta_linear = bidx + bidy * grid_x + bidz * (grid_x * grid_y)   # the grid is (cluster_m, cluster_n, clusters)
        num_ctas = grid_x * grid_y * grid_z
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(), num_stages=self.num_ab_stage,
            producer_group=ab_producer_group, consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
        ).make_participants()
        acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len(self.epilogue_warp_id) * (2 if self.use_2cta_instrs else 1))
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(), num_stages=self.num_acc_stage,
            producer_group=acc_producer_group, consumer_group=acc_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)))
        tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier,
                                   allocator_warp_id=self.epilogue_warp_id[0], is_two_cta=self.use_2cta_instrs,
                                   two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr)
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(element_type=self.a_dtype, layout=a_smem_layout_staged.outer,
                                  byte_alignment=128, swizzle=a_smem_layout_staged.inner)
        sB = smem.allocate_tensor(element_type=self.b_dtype, layout=b_smem_layout_staged.outer,
                                  byte_alignment=128, swizzle=b_smem_layout_staged.inner)
        sC = smem.allocate_tensor(element_type=self.c_dtype, layout=c_smem_layout_staged.outer,
                                  byte_alignment=128, swizzle=c_smem_layout_staged.inner)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if const_expr(self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk,
                                                                  mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk,
                                                                  mcast_mode=1)

        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gC_real = cute.local_tile(mC_real, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)        # TMA coordinate space
        tCgC_real = thr_mma.partition_C(gC_real)  # memory pointers, for re-basing onto peers
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
                                           cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
                                           cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(),
                                                                cute.arch.grid_dim())
        work_tile = tile_sched.initial_work_tile_info()


        #
        # Pusher warps (recv mode): job = (row block, segment, destination, row sub-block); copy the
        # sub-block into the peer's receive buffer, then release the peer's counter for that row
        # block. Jobs are ordered by row block and spread over all pusher warps of all programs, and
        # the split keeps only a few row blocks in flight, so blocks complete early and in order
        # instead of all sharing the link bandwidth and finishing together.
        #
        if const_expr(self.mode == 'recv' and self.pusher_warps > 0):
            if warp_idx > self.tma_warp_id:
                pusher_idx = warp_idx - (self.tma_warp_id + 1)
                # The first row group of the N raster covers `swizzle` clusters of rows.
                first_blocks = self.swizzle * self.mma_tiler[0] // CTA_M
                _push_jobs(tidx % 32, cta_linear, pusher_idx, num_ctas, self.pusher_warps, PUSHER_STEADY_WARPS,
                           first_blocks, self.p, self.rank, self.recv_m, self.recv_k, self.num_in, self.push_split,
                           self.head_dim, aux0, aux1,
                           src0, src1, src2, src3, src_rows, self.pad_recv)

        #
        # TMA load warp
        #
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        if warp_idx == self.tma_warp_id:
            target = Int32(0)
            if const_expr(self.mode == 'recv' and self.pusher_warps > 0):
                target = Int32(self.p * self.num_in * self.push_split)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                m_cta = cur_tile_coord[0]                 # 128-row block owned by this CTA
                m_tile, b_tile, dest, within, segment = self._tile_mapping(m_cta // atom_thr_size, cur_tile_coord[1])
                if const_expr(self.mode == 'recv'):
                    # Acquire this CTA's row block from every source, then order the TMA loads after it.
                    seen = ld_acquire_sys_u32(counters.iterator + m_cta)
                    while seen < target:
                        nanosleep(512)
                        seen = ld_acquire_sys_u32(counters.iterator + m_cta)
                    cute.arch.fence_proxy("async.global")
                tAgA_slice = tAgA[(None, m_tile, None, 0)]
                tBgB_slice = tBgB[(None, b_tile, None, 0)]
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(tma_atom_a, tAgA_slice[(None, handle.count)], tAsA[(None, handle.index)],
                              tma_bar_ptr=handle.barrier, mcast_mask=a_full_mcast_mask)
                    cute.copy(tma_atom_b, tBgB_slice[(None, handle.count)], tBsB[(None, handle.index)],
                              tma_bar_ptr=handle.barrier, mcast_mask=b_full_mcast_mask)
                    peek_ab_empty_status = Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer,
                                                              self.num_acc_stage)
            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                ab_consumer.reset()
                peek_ab_full_status = Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                        peek_ab_full_status = Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            fake_base = mC_real.iterator.toint()
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer,
                                                              self.num_acc_stage)
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id))
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                m_tile, b_tile, dest, within, segment = self._tile_mapping(cur_tile_coord[0] // atom_thr_size,
                                                                           cur_tile_coord[1])
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed
                if const_expr(self.mode == 'push'):
                    peer_base = aux0[dest]
                    if segment == 1:
                        peer_base = aux1[dest]
                    if segment == 2:
                        peer_base = aux2[dest]
                    if segment == 3:
                        peer_base = aux3[dest]
                    nm = cute.size(mC_real, mode=[0]) // (self.p * self.mma_tiler[0])  # MMA tiles per rank
                    fake_coord = (self.rank * nm + m_tile, within, Int32(0))
                    acc_consumer_state = _epilogue_push_simt(
                        self, tidx, tCtAcc_base, sC, tCgC_real, epi_tile, num_tiles_executed, fake_base,
                        peer_base, fake_coord, acc_consumer_state, acc_pipeline,
                        bias, b_tile * self.cta_tile_shape_mnk[1])
                else:
                    acc_consumer_state = epilogue_tma_store(
                        self, tidx, warp_idx, tma_atom_c, tCtAcc_base, sC, tCgC, epi_tile, num_tiles_executed,
                        lambda x: x, (m_tile, b_tile, Int32(0)), acc_consumer_state, acc_pipeline, c_pipeline)
            c_pipeline.producer_tail()
            if const_expr(self.mode == 'push'):
                # Remote stores are complete; make them visible before the completion barrier.
                cute.arch.fence_proxy("async.global")
                cute.arch.fence_acq_rel_sys()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


@cute.jit
def _add_push_bias(gemm_kernel, epi_tidx: Int32, tiled_copy_t2r: cute.TiledCopy,
                   epi_tile: cute.Tile, subtile_idx: Int32, accumulator: cute.Tensor,
                   bias: cute.Pointer, bias_column: Int32):
    """Add weight-row bias in FP32 before the push epilogue rounds to BF16.

    The zero row stride broadcasts one bias vector. Partition it exactly as
    the accumulator's register destination, including every epilogue subtile.
    bias_column is in original weight-row order, not peer-head column order.
    """
    tile = cute.make_tensor(
        bias + bias_column,
        cute.make_layout(gemm_kernel.cta_tile_shape_mnk[:2], stride=(0, 1)))
    partitioned = tiled_copy_t2r.get_slice(epi_tidx).partition_D(cute.flat_divide(tile, epi_tile))
    partitioned = cute.group_modes(partitioned, 3, cute.rank(partitioned))
    values = partitioned[(None, None, None, subtile_idx)]
    for i in cutlass.range(cute.size(accumulator), unroll_full=True):
        accumulator[i] = accumulator[i] + values[i].to(cutlass.Float32)


@cute.jit
def _epilogue_push_simt(gemm_kernel, epi_tidx: Int32, tCtAcc_base: cute.Tensor, sC: cute.Tensor,
                        tCgC_base: cute.Tensor, epi_tile: cute.Tile, num_tiles_executed: Int32, fake_base: Int64,
                        peer_base: Int64, fake_coord, acc_consumer_state, acc_pipeline,
                        bias: cute.Pointer, bias_column: Int32):
    """Epilogue that stages each subtile through shared memory (as the TMA path does) and then
    writes it to the destination peer with 16-byte vector stores laid out along rows, so every
    8 threads cover one 128-byte row segment. tCgC_base is the real-pointer partition of this
    rank's head-side buffer; the subtile offset relative to fake_base is valid on the peer."""
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs)
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC)
    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    epi_n = cute.size(epi_tile[1])
    threads_per_row = epi_n // 8
    atom_s2g = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gemm_kernel.c_dtype, num_bits_per_copy=128)
    thr_layout = cute.make_layout((128 // threads_per_row, threads_per_row), stride=(threads_per_row, 1))
    val_layout = cute.make_layout((1, 8), stride=(8, 1))
    tiled_s2g = cute.make_tiled_copy_tv(atom_s2g, thr_layout, val_layout)
    thr_s2g = tiled_s2g.get_slice(epi_tidx)
    epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=gemm_kernel.epilog_sync_bar_id,
                                                num_threads=32 * len(gemm_kernel.epilogue_warp_id))
    gC_tile = tCgC_epi[(None, None, None, None, *fake_coord)]
    gC_tile = cute.group_modes(gC_tile, 2, cute.rank(gC_tile))
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
    acc_pipeline.consumer_wait(acc_consumer_state)
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    num_prev_subtiles = num_tiles_executed * subtile_cnt
    for subtile_idx in range(subtile_cnt):
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
        if const_expr(gemm_kernel.has_bias):
            _add_push_bias(gemm_kernel, epi_tidx, tiled_copy_t2r, epi_tile, subtile_idx,
                           tTR_rAcc, bias, bias_column)
        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
        tRS_rC.store(acc_vec.to(gemm_kernel.c_dtype))
        c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
        epilog_sync_barrier.arrive_and_wait()
        gD_fake = gC_tile[(None, None, subtile_idx)]
        gD = cute.make_tensor(cute.make_ptr(gemm_kernel.c_dtype, peer_base + (gD_fake.iterator.toint() - fake_base),
                                            cute.AddressSpace.gmem, assumed_align=16), gD_fake.layout)
        tS = thr_s2g.partition_S(sC[(None, None, c_buffer)])
        tD = thr_s2g.partition_D(gD)
        _copy_in_chunks(tiled_s2g, tS, tD)
        epilog_sync_barrier.arrive_and_wait()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state


@triton.jit(do_not_specialize=['rows'])
def _bias_grad_partial(dy, partial, rows, WIDTH: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       FINAL: tl.constexpr, ACCUMULATE: tl.constexpr):
    # Vary columns first so adjacent CTAs spread wide rows across memory partitions.
    r = (tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    c = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    values = tl.load(dy + r[:, None] * WIDTH + c[None, :],
                     (r[:, None] < rows) & (c[None, :] < WIDTH), other=0).to(tl.float32)
    total = tl.sum(values, axis=0)
    target = partial + c if FINAL else partial + tl.program_id(1) * WIDTH + c
    if FINAL and ACCUMULATE:
        total += tl.load(target, c < WIDTH, other=0).to(tl.float32)
    tl.store(target, total, c < WIDTH)


@triton.jit
def _bias_grad_finish(partial, out, WIDTH: tl.constexpr, PARTS: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      ACCUMULATE: tl.constexpr):
    c = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    total = tl.full((BLOCK_N,), 0, tl.float32)
    # Bound register use for very long inputs; ordinary CP8/CP16 shapes use one tile.
    for first in range(0, PARTS, BLOCK_M):
        r = first + tl.arange(0, BLOCK_M)
        values = tl.load(partial + r[:, None] * WIDTH + c[None, :],
                         (r[:, None] < PARTS) & (c[None, :] < WIDTH), other=0)
        total += tl.sum(values, axis=0)
    if ACCUMULATE:
        total += tl.load(out + c, c < WIDTH, other=0).to(tl.float32)
    tl.store(out + c, total, c < WIDTH)


def reduce_bias_grad(dy, out, *, accumulate=False):
    """Sum logical BF16 rows in FP32, then cast or accumulate once at the output."""
    rows, width = dy.shape
    if (rows < 1 or not dy.is_cuda or not dy.is_contiguous() or dy.dtype != torch.bfloat16
            or out.shape != (width,) or out.device != dy.device or not out.is_contiguous()
            or out.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError('Bias reduction requires contiguous BF16 rows and a BF16/FP32 output vector')
    block_m, block_n = min(256, triton.next_power_of_2(rows)), min(64, triton.next_power_of_2(width))
    parts = triton.cdiv(rows, block_m)
    partial = out if parts == 1 else torch.empty((parts, width), device=dy.device, dtype=torch.float32)
    _bias_grad_partial[(triton.cdiv(width, block_n), parts)](
        dy, partial, rows, width, block_m, block_n, parts == 1, accumulate, num_warps=4)
    if parts > 1:
        _bias_grad_finish[(triton.cdiv(width, block_n),)](
            partial, out, width, parts, min(128, triton.next_power_of_2(parts)), block_n, accumulate, num_warps=4)
    return out


@triton.jit(do_not_specialize=['input_rows', 'output_rows'])
def _copy_rank_rows(inputs, outputs, input_rows, output_rows,
                    WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    rank = tl.program_id(1).to(tl.int64)
    segment = tl.program_id(2)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < output_rows * WIDTH
    input_mask = output_mask & (offsets < input_rows * WIDTH)
    for i in tl.static_range(len(inputs)):
        if segment == i:
            values = tl.load(inputs[i] + rank * input_rows * WIDTH + offsets,
                             input_mask, other=0)
            tl.store(outputs[i] + rank * output_rows * WIDTH + offsets, values, output_mask)


@triton.jit(do_not_specialize=['input_rows', 'output_rows'])
def _copy_biased_rank_rows(inputs, outputs, input_rows, output_rows, bias,
                          WIDTH: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
                          HEAD_DIM: tl.constexpr, HEAD_OFFSET: tl.constexpr):
    rank = tl.program_id(1).to(tl.int64)
    segment = tl.program_id(2)
    column_tiles: tl.constexpr = triton.cdiv(WIDTH, COLS)
    rows = tl.program_id(0) // column_tiles * ROWS + tl.arange(0, ROWS)
    columns = tl.program_id(0) % column_tiles * COLS + tl.arange(0, COLS)
    output_mask = (rows[:, None] < output_rows) & (columns[None, :] < WIDTH)
    for i in tl.static_range(len(inputs)):
        if segment == i:
            index = ((HEAD_OFFSET + columns // HEAD_DIM) * len(inputs) + i) * HEAD_DIM
            # Load each bias column once and broadcast across rows.
            b = tl.load(bias + index + columns % HEAD_DIM, columns < WIDTH, other=0)
            values = tl.load(inputs[i] + (rank * input_rows + rows[:, None]) * WIDTH + columns[None, :],
                             output_mask & (rows[:, None] < input_rows), other=0)
            # Preserve GEMM -> BF16 -> bias -> BF16 rounding. Added padding stays zero.
            values = tl.where(rows[:, None] < input_rows,
                              values.to(tl.float32) + b[None, :].to(tl.float32), 0)
            tl.store(outputs[i] + (rank * output_rows + rows[:, None]) * WIDTH + columns[None, :],
                     values, output_mask)


def resize_rank_rows(tensors, peers, rows, *, copy=False, bias=None,
                     head_dim=128, head_offset=0):
    """Resize each rank's row chunk, zeroing added rows and dropping tail rows.

    All tensors are contiguous CUDA matrices with identical shape and dtype.
    Return owned storage when resizing, adding bias or copy=True; otherwise reuse the inputs.
    Optional interleaved bias is added while copying this rank's heads.
    """
    tensors = tuple(tensors)
    first = tensors[0]
    if peers < 1 or rows < 1 or first.shape[0] % peers:
        raise ValueError('Positive row counts must split evenly across peers')
    if first.ndim != 2 or not first.is_cuda:
        raise ValueError('Rank copies require CUDA matrices')
    if any(t.shape != first.shape or t.dtype != first.dtype or t.device != first.device
           or not t.is_contiguous() for t in tensors):
        raise ValueError('Rank copies require matching contiguous tensors')
    input_rows, width = first.shape[0] // peers, first.shape[1]
    if bias is not None:
        if (head_dim < 1 or width % head_dim or head_offset < 0
                or bias.ndim != 1 or not bias.is_contiguous()
                or bias.device != first.device or bias.dtype != first.dtype
                or bias.numel() < (head_offset * head_dim + width) * len(tensors)):
            raise ValueError('Bias must cover the interleaved heads being copied')
    if input_rows == rows and bias is None:
        if not copy:
            return tensors
        if len(tensors) == 1:
            return (first.clone(),)
        # Copy Q/K/V(/gate) in one launch even when no padding is present.
    outputs = tuple(torch.empty((peers * rows, width), device=t.device, dtype=t.dtype)
                    for t in tensors)
    if bias is None:
        _copy_rank_rows[(triton.cdiv(rows * width, 4096), peers, len(tensors))](
            tensors, outputs, input_rows, rows, width, 4096, num_warps=4)
    else:
        columns = min(512, triton.next_power_of_2(width))
        tile_rows = 2048 // columns
        grid = (triton.cdiv(rows, tile_rows) * triton.cdiv(width, columns), peers, len(tensors))
        _copy_biased_rank_rows[grid](tensors, outputs, input_rows, rows, bias, width,
                                   tile_rows, columns, head_dim, head_offset, num_warps=4)
    return outputs


# Experimental MXFP8 path; independently quantize each GEMM's reduction axis.
@triton.jit
def _mx_operand(x, R: tl.constexpr, K: tl.constexpr):
    groups = x.to(tl.float32).reshape(R, K // 32, 32)
    amax = tl.max(tl.abs(groups), 2)
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 2.0 ** -120) / 448.0))
    exponent = tl.where(amax == 0, 0.0, tl.minimum(tl.maximum(exponent, -126.0), 127.0))
    scaled = groups * tl.exp2(-exponent)[:, :, None]
    payload = tl.minimum(tl.maximum(scaled, -448.0), 448.0).to(tl.float8e4nv)
    return payload.reshape(R, K), (exponent + 127).to(tl.uint8)


@triton.jit
def _mx_quant(A, Q, S, ROWS: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
              SR: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr = 32, BK: tl.constexpr = 128):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    x = tl.load(A + r[:, None] * SR + k[None, :] * SK,
                (r[:, None] < ROWS) & (k[None, :] < K), 0)
    q, s = _mx_operand(x, BM, BK)
    tl.store(Q + r[:, None] * KP + k[None, :], q, r[:, None] < ROWS)
    sk = tl.program_id(1) * (BK // 32) + tl.arange(0, BK // 32)
    tl.store(S + r[:, None] * (KP // 32) + sk[None, :], s, r[:, None] < ROWS)


@triton.jit
def _mx_tile(M, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr = 8):
    # Keep a small row group resident while traversing weight-column tiles.
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    first = pid // (GROUP_M * nn) * GROUP_M
    group = tl.minimum(nm - first, GROUP_M)
    within = pid % (GROUP_M * nn)
    return first + within % group, within // group


@triton.jit
def _mx_push(A, AS, B, BS, Bias, Peer0, Peer1, Peer2, Peer3,
             M, N: tl.constexpr, K: tl.constexpr, CAP: tl.constexpr,
             P: tl.constexpr, RANK: tl.constexpr, D: tl.constexpr,
             SEGMENTS: tl.constexpr, HAS_BIAS: tl.constexpr,
             BM: tl.constexpr = 128, BN: tl.constexpr = 128, BK: tl.constexpr = 128):
    pm, pn = _mx_tile(M, N, BM, BN)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        aq = tl.load(A + rm[:, None] * K + k[None, :], rm[:, None] < M, 0.0)
        bq = tl.load(B + rn[:, None] * K + k[None, :], rn[:, None] < N, 0.0)
        ks = start * (BK // 32) + tl.arange(0, BK // 32)
        sa = tl.load(AS + rm[:, None] * (K // 32) + ks[None, :], rm[:, None] < M, 127)
        sb = tl.load(BS + rn[:, None] * (K // 32) + ks[None, :], rn[:, None] < N, 127)
        acc = tl.dot_scaled(aq, sa, 'e4m3', bq.T, sb, 'e4m3', acc)
    if HAS_BIAS:
        acc += tl.load(Bias + rn, rn < N, 0)[None, :]
    # A tile stays within a single head/segment and therefore one destination.
    n0 = pn * BN
    width = N // SEGMENTS // P
    head = n0 // (SEGMENTS * D)
    peer = head // (width // D)
    segment = n0 // D % SEGMENTS
    col = (rn // (SEGMENTS * D) % (width // D)) * D + rn % D
    if SEGMENTS == 1:
        peer = n0 // width
        col = rn % width
    address = tl.load(Peer0 + peer)
    if SEGMENTS > 1:
        if segment == 1:
            address = tl.load(Peer1 + peer)
        if segment == 2:
            address = tl.load(Peer2 + peer)
        if SEGMENTS == 4:
            if segment == 3:
                address = tl.load(Peer3 + peer)
    out = address.to(tl.pointer_type(tl.bfloat16))
    tl.store(out + (RANK * CAP + rm[:, None]) * width + col[None, :],
             acc.to(tl.bfloat16), (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _mx_publish_tile(A, AS, Saved, Ready, Peers, Payloads, Scales,
                     M, K: tl.constexpr, CAP: tl.constexpr, P: tl.constexpr,
                     RANK: tl.constexpr, D: tl.constexpr, SEGMENTS: tl.constexpr,
                     pm, BM: tl.constexpr, BK: tl.constexpr):
    # The first *running* CTA elects itself as producer. No consumer waits for
    # a CTA that has not yet been scheduled, even on a fully occupied GPU.
    owner = tl.atomic_cas(Ready + pm, 0, 1, sem="acquire", scope="gpu") == 0
    if owner:
        for row_chunk in range(BM // 32):
            rm = pm * BM + row_chunk * 32 + tl.arange(0, 32)
            rk = tl.arange(0, BK)
            ks = tl.arange(0, BK // 32)
            width = K // P // SEGMENTS
            for start in range(tl.cdiv(K, BK)):
                k0 = ((start + RANK * (K // P // BK)) % (K // BK)) * BK
                k = k0 + rk
                peer = k0 // (width * SEGMENTS)
                segment = k0 // D % SEGMENTS
                col = (k // (SEGMENTS * D) % (width // D)) * D + k % D
                scol = (k0 // (SEGMENTS * D) % (width // D)) * (D // 32) + k0 % D // 32 + ks
                if SEGMENTS == 1:
                    col = k % width
                    scol = k0 % width // 32 + ks
                address = tl.full((), 0, tl.int64)
                qa = tl.full((), 0, tl.int64)
                ss = tl.full((), 0, tl.int64)
                for i in tl.static_range(SEGMENTS):
                    if segment == i:
                        address = tl.load(Peers[i] + peer)
                        qa = tl.load(Payloads[i] + peer)
                        ss = tl.load(Scales[i] + peer)
                offsets = (RANK * CAP + rm[:, None]) * width + col[None, :]
                aq = tl.load(qa.to(tl.pointer_type(tl.float8e4nv)) + offsets, rm[:, None] < M, 0.0)
                av = tl.load(address.to(tl.pointer_type(tl.bfloat16)) + offsets, rm[:, None] < M, 0)
                sa = tl.load(ss.to(tl.pointer_type(tl.uint8)) +
                             (RANK * CAP + rm[:, None]) * (width // 32) + scol[None, :], rm[:, None] < M, 127)
                local = rm[:, None] * K + k[None, :]
                tl.store(A + local, aq, rm[:, None] < M)
                tl.store(Saved + local, av, rm[:, None] < M)
                tl.store(AS + rm[:, None] * (K // 32) + (k0 // 32 + ks)[None, :], sa, rm[:, None] < M)
        tl.debug_barrier()
        tl.atomic_xchg(Ready + pm, 2, sem="release", scope="gpu")
    else:
        while tl.atomic_add(Ready + pm, 0, sem="acquire", scope="gpu") != 2:
            pass


@triton.jit
def _mx_recv(B, BS, C, Saved, A, AS, Ready, Peers, Payloads, Scales,
             M, N: tl.constexpr, K: tl.constexpr, CAP: tl.constexpr,
             P: tl.constexpr, RANK: tl.constexpr, D: tl.constexpr,
             SEGMENTS: tl.constexpr,
             BM: tl.constexpr = 128, BN: tl.constexpr = 128, BK: tl.constexpr = 128):
    pm, pn = _mx_tile(M, N, BM, BN, 128)
    _mx_publish_tile(A, AS, Saved, Ready, Peers, Payloads, Scales,
                     M, K, CAP, P, RANK, D, SEGMENTS, pm, BM, BK)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        ks = start * (BK // 32) + tl.arange(0, BK // 32)
        aq = tl.load(A + rm[:, None] * K + k[None, :], rm[:, None] < M, 0.0, cache_modifier=".cg")
        sa = tl.load(AS + rm[:, None] * (K // 32) + ks[None, :], rm[:, None] < M, 127, cache_modifier=".cg")
        bq = tl.load(B + rn[:, None] * K + k[None, :], rn[:, None] < N, 0.0)
        sb = tl.load(BS + rn[:, None] * (K // 32) + ks[None, :], rn[:, None] < N, 127)
        acc = tl.dot_scaled(aq, sa, 'e4m3', bq.T, sb, 'e4m3', acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _mx_wgrad(DY, DS, X, XS, DW, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
              ACCUMULATE: tl.constexpr,
              BM: tl.constexpr = 128, BN: tl.constexpr = 128, BK: tl.constexpr = 128):
    pm, pn = _mx_tile(N, H, BM, BN)
    rn = pm * BM + tl.arange(0, BM)
    rh = pn * BN + tl.arange(0, BN)
    rt = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(T, BK)):
        t = start * BK + rt
        aq = tl.load(DY + rn[:, None] * T + t[None, :], rn[:, None] < N, 0.0)
        bq = tl.load(X + rh[:, None] * T + t[None, :], rh[:, None] < H, 0.0)
        ts = start * (BK // 32) + tl.arange(0, BK // 32)
        sa = tl.load(DS + rn[:, None] * (T // 32) + ts[None, :], rn[:, None] < N, 127)
        sb = tl.load(XS + rh[:, None] * (T // 32) + ts[None, :], rh[:, None] < H, 127)
        acc = tl.dot_scaled(aq, sa, 'e4m3', bq.T, sb, 'e4m3', acc)
    offsets = rn[:, None] * H + rh[None, :]
    mask = (rn[:, None] < N) & (rh[None, :] < H)
    if ACCUMULATE:
        acc += tl.load(DW + offsets, mask, 0)
    tl.store(DW + offsets, acc, mask)


_native_mxfp8_kernels = set()


def _require_native_mxfp8(kernel):
    # dot_scaled can silently decompose unsupported tiles into ordinary GEMMs.
    # Validate each compiled specialization once, before accepting its results.
    if kernel not in _native_mxfp8_kernels:
        ptx = kernel.asm['ptx']
        if 'tcgen05.mma' not in ptx or 'block_scale' not in ptx:
            raise RuntimeError('MXFP8 requires native SM100 block-scaled MMA; check Triton and tile shapes')
        _native_mxfp8_kernels.add(kernel)
    return kernel


def mxfp8_quantize(a, out=None):
    rows, k = a.shape
    kp = triton.cdiv(k, 128) * 128
    if out is None:
        out = (torch.empty((rows, kp), device=a.device, dtype=torch.float8_e4m3fn),
               torch.empty((rows, kp // 32), device=a.device, dtype=torch.uint8))
    _mx_quant[(triton.cdiv(rows, 32), kp // 128)](
        a, *out, rows, k, kp, *a.stride(), num_warps=4)
    return out


def mxfp8_push(a, b, peers, *, rank, capacity, head_dim, segments, b_is_kn=False, bias=None):
    n = b.shape[1] if b_is_kn else b.shape[0]
    ptrs = list(peers) + [peers[0]] * (4 - len(peers))
    aq, sa = mxfp8_quantize(a)
    bq, sb = mxfp8_quantize(b.T if b_is_kn else b)
    bm = 128  # SM100 block-scaled MMA needs a 128-row tile, even for short inputs.
    kernel = _mx_push[(triton.cdiv(a.shape[0], bm) * triton.cdiv(n, 128),)](
        aq, sa, bq, sb, b if bias is None else bias, *ptrs, a.shape[0], n, aq.shape[1], capacity,
        len(peers[0]), rank, head_dim, segments, bias is not None, BM=bm, num_warps=4)
    return _require_native_mxfp8(kernel)


def mxfp8_recv(b, out, saved, recv_quantized, ready, peers, payloads, scales, *, rank, capacity, tokens, head_dim, segments, b_is_kn=False):
    n = b.shape[1] if b_is_kn else b.shape[0]
    k = b.shape[0] if b_is_kn else b.shape[1]
    bq, sb = mxfp8_quantize(b.T if b_is_kn else b)
    bm = 128
    kernel = _mx_recv[(triton.cdiv(tokens, bm) * triton.cdiv(n, 128),)](
        bq, sb, out, saved, *recv_quantized, ready, tuple(peers), tuple(payloads), tuple(scales), tokens, n, k, capacity, len(peers[0]), rank,
        head_dim, segments, BM=bm, num_warps=4)
    return _require_native_mxfp8(kernel)


def mxfp8_wgrad(dy, x, out, *, accumulate=False):
    dq, ds = mxfp8_quantize(dy.T)
    xq, xs = mxfp8_quantize(x.T)
    kernel = _mx_wgrad[(triton.cdiv(dy.shape[1], 128) * triton.cdiv(x.shape[1], 128),)](
        dq, ds, xq, xs, out, dq.shape[1], dy.shape[1], x.shape[1], accumulate, BM=128, num_warps=4)
    return _require_native_mxfp8(kernel)


def mxfp8_stage(inputs, outputs, peers, capacity):
    """Publish head-side sources in symmetric buffers without an allocation."""
    rows, width = inputs[0].shape
    _copy_rank_rows[(triton.cdiv(capacity * width, 4096), peers, len(inputs))](
        tuple(inputs), tuple(outputs), rows // peers, capacity, width, 4096, num_warps=4)
