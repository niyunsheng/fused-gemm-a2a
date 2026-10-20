"""Compare experimental MXFP8 with the BF16 operator; run parity tests first."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from benchmark.common import BenchmarkRuntime
from fused_gemm_a2a import FusedGemmA2AWorkspace, FusedQKVProjection, FusedOutputProjection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cp', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=131072)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--graph-batch', type=int, default=2)
    parser.add_argument('--cuda-graphs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.profile and not args.cuda_graphs:
        parser.error('--profile requires CUDA Graph timing')
    runtime = BenchmarkRuntime(args, parser)
    h, m, segments = args.heads * 128, args.tokens // args.cp, 3 + args.gate
    torch.manual_seed(123)
    def parameter(shape):
        return torch.nn.Parameter(torch.randn(shape, device=runtime.device, dtype=torch.bfloat16) / h**.5)
    wq, wo = parameter((segments*h, h)), parameter((h, h))
    bq, bo = (parameter((segments*h,)), parameter((h,))) if args.bias else (None, None)
    qkv = FusedQKVProjection(h, args.heads, weight=wq, bias=bq, gate=args.gate)
    proj = FusedOutputProjection(h, args.heads, weight=wo, bias=bo)
    torch.manual_seed(1000 + runtime.rank)
    x = torch.randn(m, h, device=runtime.device, dtype=torch.bfloat16, requires_grad=True)
    o = torch.randn(args.tokens, args.heads//args.cp, 128, device=runtime.device,
                    dtype=torch.bfloat16, requires_grad=True)
    gs = [torch.randn_like(o) for _ in range(segments)]
    dy = torch.randn_like(x)
    timings = {}
    for precision in ('bf16', 'mxfp8'):
        workspace = FusedGemmA2AWorkspace(h, args.heads, m, gate=args.gate, precision=precision)
        qs = qkv(x, workspace)
        y = proj(o, workspace)
        qparams = [x, wq] + ([] if bq is None else [bq])
        oparams = [o, wo] + ([] if bo is None else [bo])
        stages = {
            'qkv_forward': lambda: qkv(x, workspace),
            'qkv_backward': lambda: torch.autograd.grad(qs, qparams, gs, retain_graph=True),
            'output_forward': lambda: proj(o, workspace),
            'output_backward': lambda: torch.autograd.grad(y, oparams, dy, retain_graph=True),
        }
        for name, fn in stages.items():
            us = runtime.measure(fn, args.cuda_graphs, name=f'{precision}/{name}')
            flops = 2 * m * h * h * (segments if name.startswith('qkv') else 1)
            flops *= 2 if name.endswith('backward') else 1
            tflops = flops / us / 1e6
            timings[f'{precision}/{name}'] = dict(us=us, tflops=tflops,
                                                  bf16_equivalent_mfu=tflops/2500)
            runtime.log(precision, name, json.dumps(timings[f'{precision}/{name}']))
        if args.profile:
            from torch.profiler import profile, ProfilerActivity, record_function
            captures = []
            for name, fn in stages.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=runtime.capture_stream):
                    outputs = fn()
                captures.append((name, graph, outputs))
            dist.barrier()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for name, graph, outputs in captures:
                    with record_function(f'{precision}/{name}'):
                        graph.replay()
                    torch.cuda.synchronize()
                prof.step()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(args.output.with_suffix(f'.{precision}.rank{runtime.rank}.json')))
            del captures, graph, outputs
            dist.barrier()
    torch.cuda.synchronize()
    if runtime.rank == 0:
        from fused_gemm_a2a_kernels import _native_mxfp8_kernels
        kernel_metadata = [dict(name=k.name, registers=k.n_regs, spills=k.n_spills,
                                shared=k.metadata.shared) for k in _native_mxfp8_kernels]
        source = Path(__file__).resolve().parents[1]
        hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                  for name in ('fused_gemm_a2a.py', 'fused_gemm_a2a_kernels.py')}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(config={**vars(args), 'output': str(args.output)},
                                              source_sha256=hashes, timings=timings,
                                              kernel_metadata=kernel_metadata,
                                              trials_us=runtime.timing_samples), indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
