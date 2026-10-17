"""Two-layer training with public varlen FA4 and no caller-side QKV copies.

Two microbatches finish forward before either backward. Compare complete outputs,
dX and CP-reduced dW against independent FP32 PyTorch attention, rounding linear
and attention outputs to BF16. Run with torchrun; FA4 is a test dependency only.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn.cute.interface import flash_attn_varlen_func
from fused_gemm_a2a import FusedGemmA2AWorkspace, FusedOutputProjection, FusedQKVProjection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=10))
    cp, rank = dist.get_world_size(), dist.get_rank()
    if args.tokens % cp or 32 % cp:
        raise ValueError('Tokens and heads must divide over CP')
    local = args.tokens // cp
    rows = slice(rank * local, (rank + 1) * local)
    torch.manual_seed(1234)
    def weight(rows):
        return torch.nn.Parameter(torch.empty(rows, 4096, device=device, dtype=torch.bfloat16)
                                  .normal_(std=4096 ** -0.5))

    layers = [(FusedQKVProjection(4096, 32, weight=weight(3 * 4096)),
               FusedOutputProjection(4096, 32, weight=weight(4096))) for _ in range(2)]
    workspace = FusedGemmA2AWorkspace(4096, 32, local)
    parameters = tuple(module.weight for layer in layers for module in layer)
    reference_weights = [w.detach().float().requires_grad_() for w in parameters]
    torch.manual_seed(5678 + rank)
    xs = [torch.randn(local, 4096, device=device, dtype=torch.bfloat16,
                      requires_grad=True) for _ in range(2)]
    dys = [torch.randn_like(x) for x in xs]
    cu = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)

    def fused_step():
        ys = []
        for x in xs:
            for qkv, proj in layers:
                q, k, v = qkv(x, workspace)
                for tensor, buffer in zip((q, k, v), workspace._view(local, 3).heads):
                    assert tensor.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()
                o = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=args.tokens, max_seqlen_k=args.tokens,
                    causal=False, num_splits=1)
                if isinstance(o, (tuple, list)):
                    o = o[0]
                x = x + proj(o, workspace)
            ys.append(x)
        gradients = torch.autograd.grad(ys, (*xs, *parameters), dys)
        return (*ys, *gradients)

    def gather(x):
        parts = [torch.empty_like(x) for _ in range(cp)]
        dist.all_gather(parts, x.detach())
        return torch.cat(parts)

    def reference_step():
        inputs = [gather(x).requires_grad_() for x in xs]
        gradients = [gather(dy) for dy in dys]
        ys = []
        with sdpa_kernel(SDPBackend.MATH):
            for x in inputs:
                for wqkv, wo in zip(reference_weights[::2], reference_weights[1::2]):
                    packed = F.linear(x.float(), wqkv).to(torch.bfloat16)
                    q, k, v = packed.view(args.tokens, 32, 3, 128).unbind(2)
                    o = F.scaled_dot_product_attention(
                        q.transpose(0, 1).float(), k.transpose(0, 1).float(),
                        v.transpose(0, 1).float()).transpose(0, 1).to(torch.bfloat16)
                    x = x + F.linear(o.flatten(1).float(), wo).to(torch.bfloat16)
                ys.append(x)
        grads = torch.autograd.grad(ys, (*inputs, *reference_weights), gradients)
        return (*(y[rows] for y in ys), *(g[rows] for g in grads[:2]), *grads[2:])

    errors = {}
    def check(actuals, expected, label):
        labels = ('mb0/y', 'mb1/y', 'mb0/dx', 'mb1/dx',
                  'layer0/dw_qkv', 'layer0/dw_o', 'layer1/dw_qkv', 'layer1/dw_o')
        for i, (actual, ref, name) in enumerate(zip(actuals, expected, labels)):
            if i >= 4:
                actual = actual.float().clone()
                dist.all_reduce(actual)
            actual, ref = actual.float(), ref.float()
            delta = (actual - ref).abs()
            peak = ref.abs().max().clamp_min(1e-12)
            rms = delta.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12)
            good = torch.isfinite(actual).all() & (delta <= 0.02 * (peak + ref.abs())).all() & (rms <= 0.02)
            metrics = torch.stack((delta.max() / peak, rms, (~good).float()))
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            errors[f'{label}/{name}'] = dict(max_error_over_peak=metrics[0].item(), relative_rms=metrics[1].item())
            if metrics[2].item():
                raise AssertionError(f'{label}/{name}: {errors[f"{label}/{name}"]}')

    check(fused_step(), reference_step(), 'eager')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_step()
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fused_step()
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(3):
        with torch.no_grad():
            for x in (*xs, *dys):
                x.normal_()
        graph.replay()
        check(captured, reference_step(), f'graph{replay}')
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(
            tokens=args.tokens, cp=cp, heads=32, head_dim=128, layers=2, microbatches=2,
            cuda_graph=True, caller_clones=False, errors=errors,
            source_sha256={name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                for name in ('fused_gemm_a2a.py', 'fused_gemm_a2a_kernels.py')},
        ), indent=2) + '\n')
        print(f'Two-layer FA4 training: {len(errors)} checks passed with owned QKV outputs', flush=True)
    del graph, captured
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
