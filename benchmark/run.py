"""Validate and time the fused Ulysses projections (fused_gemm_a2a.py) against reference.py; launch with torchrun.

All implementations share the same weight values. The QKV projection and the output
projection are tested independently with their own random inputs and output
gradients; every forward output and gradient of both implementations is compared
with the same computation on all T tokens. Timing reports forward and backward
MFU of each part for the reference and the fused modules, writes JSON to --output,
and records a torch profiler trace per rank next to it.
"""
import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule

from benchmark.common import BenchmarkRuntime, capture_profile_stage, profile_stage
from fused_gemm_a2a import FusedGemmA2AWorkspace, FusedOutputProjection, FusedQKVProjection, MMA_TILER_MN
from benchmark.reference import UlyssesOutputProjection, UlyssesQKVProjection
from benchmark.reference import interleave_qkv_rows
from fused_gemm_a2a_kernels import padded_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, default=131072,
                   help='Total packed tokens T; divisible by --cp. The per-rank count need not be a tile '
                        'multiple: the fused modules pad it internally (e.g. 119296 at CP16 gives 7456 rows)')
    p.add_argument('--hidden-size', type=int, default=4096)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--cp', type=int, default=16)
    p.add_argument('--max-local-tokens', type=int,
                   help='Fixed workspace limit per rank; defaults to the benchmark local length')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--trials', type=int, default=5, help='Median of max-rank CUDA-event measurements')
    p.add_argument('--cuda-graphs', action='store_true', help='Capture each timed stage in a CUDA graph')
    p.add_argument('--graph-batch', type=int, default=16, help='Invocations per graph replay')
    p.add_argument('--output', default='results/fused.json')
    p.add_argument('--profile', action='store_true',
                   help='Trace each operator in its selected eager/CUDA Graph timing mode')
    p.add_argument('--peak-tflops', type=float, default=2500)
    p.add_argument('--gate', action='store_true',
                   help='Fourth projection of width H (the DiT VSA coarse gate): the fused module runs four '
                        'segments, the reference overlaps four separate GEMMs and asynchronous all-to-alls')
    p.add_argument('--layers', type=int, default=2,
                   help='Lifetime check: this many module pairs share the pooled workspace, run all forwards '
                        'then the backwards in reverse, and are compared layer by layer (1 disables)')
    p.add_argument('--bias', action='store_true', help='Give every projection a bias (checked incl. dbias)')
    p.add_argument('--bias-in-gemm', action='store_true',
                   help='Add QKV bias to FP32 accumulators before the BF16 push output cast')
    p.add_argument('--validation-rounds', type=int, default=3,
                   help='Extra checks with changed inputs; in graph mode they replay one captured fused step')
    a = p.parse_args()
    if a.bias_in_gemm and not a.bias:
        p.error('--bias-in-gemm requires --bias')
    runtime = BenchmarkRuntime(a, p, collect_failures=True)
    device, capture_stream = runtime.device, runtime.capture_stream
    dtype, ref_dtype, tol = torch.bfloat16, runtime.ref_dtype, runtime.tolerance
    rank, P = runtime.rank, a.cp
    T, H, heads, kv_heads = a.tokens, a.hidden_size, a.heads, a.heads
    D, M = H // heads, T // P
    hq = slice(rank * (heads // P), (rank + 1) * (heads // P))
    hkv = slice(rank * (kv_heads // P), (rank + 1) * (kv_heads // P))
    rows = slice(rank * M, (rank + 1) * M)
    errors, timings, failures = runtime.errors, {}, runtime.failures
    log, gather, check = runtime.log, runtime.gather, runtime.check
    measure = runtime.measure
    reduced = runtime.reduced

    profile_graphs = {}
    try:
        torch.manual_seed(1234)  # Same seed on every rank: the replicated weights must agree.
        ref_qkv = UlyssesQKVProjection(H, heads, kv_heads, bias=a.bias, device=device, dtype=dtype,
                                      gate=a.gate)
        ref_out = UlyssesOutputProjection(H, heads, bias=a.bias, device=device, dtype=dtype)
        segments = 3 + a.gate
        workspace = FusedGemmA2AWorkspace(H, heads, M if a.max_local_tokens is None else a.max_local_tokens, gate=a.gate)
        def to_fused_rows(seg_major):
            return interleave_qkv_rows(seg_major, heads, D)

        def make_fused(rq, ro):
            # Independent parameters with the reference values, allocated by the caller.
            weight = torch.nn.Parameter(to_fused_rows(torch.cat([p.weight for p in rq.projections])))
            qkv_bias = torch.nn.Parameter(torch.zeros(segments * H, device=device, dtype=dtype)) if a.bias else None
            out_bias = torch.nn.Parameter(torch.zeros(H, device=device, dtype=dtype)) if a.bias else None
            modules = (
                FusedQKVProjection(H, heads, kv_heads, weight=weight, bias=qkv_bias,
                                   gate=a.gate, bias_in_gemm=a.bias_in_gemm),
                FusedOutputProjection(H, heads, weight=torch.nn.Parameter(ro.o_proj.weight.detach().clone()),
                                      bias=out_bias),
            )
            return modules

        fused_qkv, fused_out = make_fused(ref_qkv, ref_out)
        ref_qkv_weights = [proj.weight for proj in ref_qkv.projections]
        ref_qkv_biases = [proj.bias for proj in ref_qkv.projections] if a.bias else []
        ref_qkv_params = ref_qkv_weights + ref_qkv_biases
        ref_out_params = [ref_out.o_proj.weight] + ([ref_out.o_proj.bias] if a.bias else [])
        fused_qkv_params = [fused_qkv.weight] + ([fused_qkv.bias] if a.bias else [])
        fused_out_params = [fused_out.weight] + ([fused_out.bias] if a.bias else [])
        with torch.no_grad():
            if a.bias:
                for b in ref_qkv_biases + [ref_out.o_proj.bias]:
                    b.normal_()  # nonzero, so a dropped bias cannot hide
                fused_qkv.bias.copy_(to_fused_rows(torch.cat(ref_qkv_biases)))
                fused_out.bias.copy_(ref_out.o_proj.bias)
        torch.manual_seed(1234 + rank)  # Per-rank activations and gradients.
        x = torch.randn(M, H, device=device, dtype=dtype)
        gq = torch.randn(T, heads // P, D, device=device, dtype=dtype)
        gk = torch.randn(T, kv_heads // P, D, device=device, dtype=dtype)
        gv = torch.randn(T, kv_heads // P, D, device=device, dtype=dtype)
        gg = torch.randn(T, heads // P, D, device=device, dtype=dtype) if a.gate else None
        o = torch.randn(T, heads // P, D, device=device, dtype=dtype)
        gy = torch.randn(M, H, device=device, dtype=dtype)
        # Public inputs and outputs contain only logical rows. Any internal
        # padding and ownership copies are included in the measured operators.
        Mc = padded_rows(M, MMA_TILER_MN[0])
        grads_ref = [gq, gk, gv] + ([gg] if a.gate else [])
        grads_f = grads_ref
        log(f'CP={P} | {dtype} | x={tuple(x.shape)} | heads={heads} kv_heads={kv_heads} head_dim={D} | '
            f'internal rows per rank={Mc} | segments={segments} '
            '| rows=Megatron')

        # QKV projection: full-token reference, then both implementations.
        X = gather(x, 0).requires_grad_()
        Ws = [w.detach().to(ref_dtype).requires_grad_() for w in ref_qkv_weights]
        Bs = [b.detach().to(ref_dtype).requires_grad_() for b in ref_qkv_biases] or [None] * len(Ws)
        full = [F.linear(X, W, B).unflatten(1, (-1, D)) for W, B in zip(Ws, Bs)]  # Q, K, V (, gate), all tokens
        torch.autograd.backward(full, [gather(g, 1) for g in grads_ref])
        dW_qkv = torch.cat([W.grad for W in Ws])  # segment-major rows
        dB_qkv = torch.cat([B.grad for B in Bs]) if a.bias else None
        xr = x.clone().requires_grad_()
        ref_outs = ref_qkv(xr)
        q, k, v = ref_outs[:3]
        torch.autograd.backward(list(ref_outs), grads_ref, retain_graph=True)  # Graphs reused for timing.
        xf = x.clone().requires_grad_()
        fused_outs = fused_qkv(xf, workspace)
        torch.autograd.backward(list(fused_outs), grads_f, retain_graph=True)
        labels = ('q', 'k', 'v', 'gate')[:segments]
        for name, (outs, xr_, dw, expected_dw) in dict(
                reference=(ref_outs, xr, torch.cat([reduced(w.grad) for w in ref_qkv_weights]), dW_qkv),
                fused=(fused_outs, xf, reduced(fused_qkv.weight.grad), to_fused_rows(dW_qkv))).items():
            for label, actual, expected in zip(labels, outs, full):
                check(actual, expected[:, hq], f'{name} {label}')
            check(xr_.grad, X.grad[rows], f'{name} dx')
            check(dw, expected_dw, f'{name} dW_qkv (CP-reduced)')
        if a.bias:
            check(torch.cat([reduced(b.grad) for b in ref_qkv_biases]), dB_qkv, 'reference dbias_qkv (CP-reduced)')
            check(reduced(fused_qkv.bias.grad), to_fused_rows(dB_qkv), 'fused dbias_qkv (CP-reduced)')
        del X, full, Ws, Bs

        # Output projection.
        O = gather(o, 1).requires_grad_()
        Wo = ref_out.o_proj.weight.detach().to(ref_dtype).requires_grad_()
        Bo = ref_out.o_proj.bias.detach().to(ref_dtype).requires_grad_() if a.bias else None
        Y = F.linear(O.flatten(1), Wo, Bo)
        torch.autograd.backward(Y, gather(gy, 0))
        orr = o.clone().requires_grad_()
        y = ref_out(orr)
        torch.autograd.backward(y, gy, retain_graph=True)
        of = o.clone().requires_grad_()
        yf = fused_out(of, workspace)
        torch.autograd.backward(yf, gy, retain_graph=True)
        for name, (y_, do, dw) in dict(reference=(y, orr.grad, reduced(ref_out.o_proj.weight.grad)),
                                     fused=(yf, of.grad.clone(), reduced(fused_out.weight.grad))).items():
            check(y_, Y[rows], f'{name} y')
            check(do, O.grad[:, hq], f'{name} do')
            check(dw, Wo.grad, f'{name} dW_o (CP-reduced)')
        if a.bias:
            check(reduced(ref_out.o_proj.bias.grad), Bo.grad, 'reference dbias_o (CP-reduced)')
            check(reduced(fused_out.bias.grad), Bo.grad, 'fused dbias_o (CP-reduced)')
        del O, Y, Wo
        if failures:
            raise RuntimeError(f'Validation failed for: {failures}')

        # Identical data on every call can hide a stale receive (a counter running ahead of its epoch
        # would let the GEMM read last call's buffer unnoticed). Change activations and output
        # gradients, in graph mode by replaying one captured fused step, and compare against the
        # reference modules on the same data.
        sx, so = x.clone().requires_grad_(), o.clone().requires_grad_()
        sg = [g.clone() for g in grads_ref] + [gy.clone()]
        ng = len(grads_ref)

        def full_step(qkv_module, out_module, fused=False):
            if fused:
                qs = qkv_module(sx, workspace)
                params = [qkv_module.weight]
            else:
                qs, params = qkv_module(sx), ref_qkv_weights  # biases are checked once above
            gs = sg[:ng]
            dx, *dws = torch.autograd.grad(qs, [sx, *params], gs)
            dw = dws[0] if fused else to_fused_rows(torch.cat(dws))
            so_in = so.detach().requires_grad_() if fused else so
            ys = out_module(so_in, workspace) if fused else out_module(so_in)
            do, dwo = torch.autograd.grad(ys, [so_in, out_module.weight if fused else out_module.o_proj.weight], sg[ng])
            return (*qs, dx, dw, ys, do, dwo)

        replay_graph = None
        if a.cuda_graphs:
            full_step(fused_qkv, fused_out, fused=True)  # Warm every kernel and allocation the capture will use.
            torch.cuda.synchronize()
            dist.barrier()
            replay_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(replay_graph, stream=capture_stream):
                replay_outputs = full_step(fused_qkv, fused_out, fused=True)
            dist.barrier()
            torch.cuda.synchronize()
        for round_ in range(a.validation_rounds):
            with torch.no_grad():
                for tensor in (sx, so, *sg):
                    tensor.copy_(torch.randn_like(tensor))
            if replay_graph is None:
                actuals = full_step(fused_qkv, fused_out, fused=True)
            else:
                replay_graph.replay()
                actuals = replay_outputs
            expecteds = full_step(ref_qkv, ref_out)
            for label, actual, expected in zip((*labels, 'dx', 'dW_qkv', 'y', 'do', 'dW_o'), actuals, expecteds):
                check(actual, expected, f'changed inputs round {round_} {label}')
        if failures:
            raise RuntimeError(f'Validation failed for: {failures}')
        del replay_graph

        if a.layers > 1:
            # Buffer lifetime: several layers share the group's pooled workspace (same shapes -> same
            # symmetric buffers). Run every layer's forward, then the backwards in reverse order like a
            # pipeline schedule, and compare each layer with its own reference modules. The forward of
            # layer i+1 overwrites the receive buffer that held layer i's O_seq, so this fails unless the
            # output projection saved O_seq and QKV returned owned head-side outputs.
            layers = []
            for li in range(a.layers):
                torch.manual_seed(100 + li)
                rq = UlyssesQKVProjection(H, heads, kv_heads, device=device, dtype=dtype,
                                         gate=a.gate)  # no bias: checked above
                ro = UlyssesOutputProjection(H, heads, device=device, dtype=dtype)
                fq, fo = make_fused(rq, ro)
                ws = [proj.weight for proj in rq.projections]
                layers.append((rq, ro, fq, fo, ws))

            def layer_forward(li, fused):
                rq, ro, fq, fo, ws = layers[li]
                xi = x.clone().requires_grad_()
                if fused:
                    outs = fq(xi, workspace)  # Retain the public outputs directly across later forwards.
                    oi = o.clone().requires_grad_()
                    yi = fo(oi, workspace)
                else:
                    outs = list(rq(xi))
                    oi = o.clone().requires_grad_()
                    yi = ro(oi)
                return xi, outs, oi, yi

            runs = {fused: [layer_forward(li, fused) for li in range(a.layers)] for fused in (False, True)}
            for li in reversed(range(a.layers)):
                for fused in (False, True):
                    xi, outs, oi, yi = runs[fused][li]
                    torch.autograd.backward(outs, grads_f if fused else grads_ref)
                    torch.autograd.backward(yi, gy)
            for li in range(a.layers):
                rq, ro, fq, fo, ws = layers[li]
                (xr_, outs_r, or_, yr_), (xf_, outs_f, of_, yf_) = runs[False][li], runs[True][li]
                for label, af, ar in zip(labels, outs_f, outs_r):
                    check(af, ar, f'layers L{li} {label}')
                check(xf_.grad, xr_.grad, f'layers L{li} dx')
                check(reduced(fq.weight.grad), to_fused_rows(torch.cat([reduced(w.grad) for w in ws])),
                      f'layers L{li} dW_qkv')
                check(yf_, yr_, f'layers L{li} y')
                check(of_.grad, or_.grad, f'layers L{li} do')
                check(reduced(fo.weight.grad), reduced(ro.o_proj.weight.grad), f'layers L{li} dW_o')
            if failures:
                raise RuntimeError(f'Validation failed for: {failures}')
            del layers, runs
        # Put the data of the timed graphs back into the modules' buffers.
        torch.autograd.grad(fused_qkv(xf, workspace), [xf, *fused_qkv_params], grads_f)
        fused_out(of, workspace)
        log('Validation passed | max relative errors:', json.dumps({k: round(e, 5) for k, e in errors.items()}))

        # Time complete projections, including communication and owned activations.
        stages = [
            ('reference_qkv_forward_us', lambda: ref_qkv(x)),
            ('reference_qkv_backward_us', lambda: torch.autograd.grad(list(ref_outs), [xr, *ref_qkv_params],
                                                                      grads_ref, retain_graph=True)),
            ('fused_qkv_forward_us', lambda: fused_qkv(x, workspace)),
            ('fused_qkv_backward_us', lambda: torch.autograd.grad(list(fused_outs), [xf, *fused_qkv_params],
                                                                  grads_f, retain_graph=True)),
            ('reference_out_forward_us', lambda: ref_out(o)),
            ('reference_out_backward_us', lambda: torch.autograd.grad(y, [orr, *ref_out_params], gy, retain_graph=True)),
            ('fused_out_forward_us', lambda: fused_out(o, workspace)),
            ('fused_out_backward_us', lambda: torch.autograd.grad(yf, [of, *fused_out_params], gy, retain_graph=True)),
        ]
        for name, fn in stages:
            use_graph = a.cuda_graphs
            timings[name] = measure(fn, use_graph, name=name)
            log(f'{name:28s} {timings[name]:10.2f} us')

        N = H + 2 * kv_heads * D + (H if a.gate else 0)
        flops = dict(qkv=2 * M * H * N, out=2 * M * H * H)  # One GEMM; backward runs dgrad + wgrad.
        import fused_gemm_a2a_kernels
        config = {key: getattr(fused_gemm_a2a_kernels, key) for key in (
            'USE_2CTA', 'MMA_N', 'MMA_N_RECV', 'MMA_N_OUT_PUSH', 'CLUSTER_N', 'RASTER_ALONG_M', 'SWIZZLE', 'PUSHER_WARPS',
            'PUSH_SPLIT', 'PUSHER_CHUNK', 'PUSHER_STREAMING', 'SWIZZLE_PUSH')}
        hosts = [None] * P
        dist.all_gather_object(hosts, socket.gethostname())
        result = dict(args=vars(a), torch=torch.__version__, cuda=torch.version.cuda,
                      cutlass=version('nvidia-cutlass-dsl'), triton=version('triton'),
                      gpu=torch.cuda.get_device_name(), hosts=hosts, fused_config=config,
                      timing_samples_us=runtime.timing_samples,
                      source_sha256={name: hashlib.sha256(
                          (Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                          for name in ('fused_gemm_a2a.py', 'fused_gemm_a2a_kernels.py')},
                      allow_bf16_reduced_precision_reduction=False,
                      timed_invocations=(math.ceil(a.iterations / a.graph_batch) * a.graph_batch
                                         if a.cuda_graphs else a.iterations),
                      validation_max_relative_error=errors, tolerance=tol, **timings)
        result['timing_execution'] = {}
        for name, _ in stages:
            use_graph = a.cuda_graphs
            result['timing_execution'][name] = dict(
                mode='cuda_graph' if use_graph else 'eager',
                invocations=math.ceil(a.iterations / a.graph_batch) * a.graph_batch
                if use_graph else a.iterations)
        table = ['| part | async reference | fused | speedup |',
                 '| --- | ---: | ---: | ---: |']
        for part in ('qkv', 'out'):
            for direction, factor in (('forward', 1), ('backward', 2)):
                cells = []
                for impl in ('reference', 'fused'):
                    us = timings[f'{impl}_{part}_{direction}_us']
                    tflops = factor * flops[part] / (us * 1e6)
                    mfu = tflops / a.peak_tflops
                    result[f'{impl}_{part}_{direction}_tflops'] = tflops
                    result[f'{impl}_{part}_{direction}_mfu'] = mfu
                    cells.append(f'{us:.1f} us, {tflops:.1f} TFLOPS, MFU {100 * mfu:.1f}%')
                speedup = timings[f'reference_{part}_{direction}_us'] / timings[f'fused_{part}_{direction}_us']
                result[f'{part}_{direction}_speedup'] = speedup
                label = 'qkvg' if part == 'qkv' and a.gate else part
                table.append(f'| {label} {direction} | {" | ".join(cells)} | {speedup:.2f}x |')
        log('\n'.join(table))

        # Match the requested timing mode in the trace too. Capture outside
        # the profiler; each stage validates its graph outputs before replay.
        if a.profile and a.cuda_graphs:
            for name, fn in stages:
                stage = name.removesuffix('_us')
                profile_graphs[stage] = capture_profile_stage(fn, stage, capture_stream, check)

        # Eager Functions label themselves; graph replay has an explicit *_cg range.
        def profiled_step():
            for impl, module_qkv, module_out, params_qkv, params_out, grads, o_in in (
                    ('reference', ref_qkv, ref_out, ref_qkv_params, ref_out_params, grads_ref, o),
                    ('fused', fused_qkv, fused_out, fused_qkv_params, fused_out_params, grads_f, o)):
                if f'{impl}_qkv_forward' in profile_graphs:
                    for part in ('qkv', 'out'):
                        for direction in ('forward', 'backward'):
                            stage = f'{impl}_{part}_{direction}'
                            with profile_stage(stage):
                                profile_graphs[stage]()
                    continue
                xp = x.clone().requires_grad_()
                with profile_stage(f'{impl}_qkv_forward'):
                    outs_ = module_qkv(xp, workspace) if impl == 'fused' else module_qkv(xp)
                with profile_stage(f'{impl}_qkv_backward'):
                    torch.autograd.grad(list(outs_), [xp, *params_qkv], list(grads))
                op = o_in.clone().requires_grad_()
                with profile_stage(f'{impl}_out_forward'):
                    y_ = module_out(op, workspace) if impl == 'fused' else module_out(op)
                with profile_stage(f'{impl}_out_backward'):
                    torch.autograd.grad(y_, [op, *params_out], gy)

        if a.profile:
            trace_dir = Path(a.output).with_suffix('')
            trace_dir.mkdir(parents=True, exist_ok=True)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         schedule=schedule(wait=1, warmup=1, active=3, repeat=1)) as prof:
                for _ in range(5):
                    profiled_step()
                    prof.step()
            torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace_dir / f'rank{rank}.json'))
            result['trace_dir'] = str(trace_dir)
            result['profile_mode'] = 'isolated_cuda_graph' if a.cuda_graphs else 'isolated_eager'
            result['profile_execution'] = dict(reference='cuda_graph' if profile_graphs else 'eager',
                                               fused='cuda_graph' if a.cuda_graphs else 'eager',
                                               invocations_per_stage=1)
        if rank == 0:
            Path(a.output).parent.mkdir(parents=True, exist_ok=True)
            Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        # NCCL finalization waits for graph-owned communication resources.
        profile_graphs.clear()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
