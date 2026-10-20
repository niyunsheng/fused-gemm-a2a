"""Independent quantized-math parity, workspace lifetime and graph replay.

Reference: OCP block-scaled product used by CUTLASS MXF8 GEMM:
https://github.com/NVIDIA/cutlass/blob/v4.5.0/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py
Use E4M3, 32 reduction elements per E8M0 scale, ceiling scale selection.
Backward requantizes each operand along the new reduction axis; it does not
transpose an already quantized payload or differentiate quantization itself.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from fused_gemm_a2a import FusedGemmA2AWorkspace, FusedQKVProjection, FusedOutputProjection
from fused_gemm_a2a_kernels import mxfp8_wgrad


def quantized(x):
    shape = x.shape
    x = F.pad(x.float(), (0, -shape[-1] % 32))
    groups = x.reshape(x.shape[0], -1, 32)
    peak = groups.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(peak.clamp_min(2.0 ** -120) / 448.0)).clamp(-126, 127)
    exponent = torch.where(peak == 0, 0.0, exponent)
    scale = torch.exp2(exponent)
    payload = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (payload * scale).reshape(x.shape)[:, :shape[-1]]


def mm(a, b):
    return quantized(a) @ quantized(b.T).T


def gather(x):
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', default=[33, 257])
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-graphs', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda', torch.cuda.current_device())
    dist.init_process_group('nccl', timeout=timedelta(minutes=15), device_id=device)
    # Keep parameter AccumulateGrad nodes on the capture stream from the start.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    # The MXFP8 path must not dispatch to the BF16-only Triton auxiliaries.
    import fused_gemm_a2a_kernels as kernels
    class ForbiddenTritonKernel:
        def __getitem__(self, grid):
            raise AssertionError('MXFP8 dispatched a Triton kernel')
    for name in ('_copy_rank_rows', '_copy_biased_rank_rows', '_bias_grad_partial', '_bias_grad_finish'):
        setattr(kernels, name, ForbiddenTritonKernel())
    p, rank = dist.get_world_size(), dist.get_rank()
    h, d = args.heads * 128, 128
    torch.manual_seed(123)
    errors = {}
    def check(actual, expected, label):
        a, b = actual.detach().float().flatten(), expected.float().flatten()
        cosine = F.cosine_similarity(a, b, dim=0)
        denom = a.square().sum() + b.square().sum()
        similarity = torch.where(denom > 0, 2 * (a * b).sum() / denom, 1.0)
        relative = (a - b).norm() / b.norm().clamp_min(1e-12)
        invalid = (~torch.isfinite(a).all()) | (cosine < .999) | (similarity < .999) | (relative > .005)
        metrics = torch.stack((1 - cosine, 1 - similarity, relative, invalid.float()))
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        errors[label] = metrics[:3].tolist()
        assert metrics[-1].item() == 0, (label, errors[label])

    for segments, bias_mode in [(3, 'none'), (3, 'post'), (4, 'post'), (4, 'pre')]:
        workspace = FusedGemmA2AWorkspace(h, args.heads, max(args.lengths), gate=segments == 4, precision='mxfp8')
        # Replicated parameters must agree even after rank-local input RNG use.
        torch.manual_seed(123)
        wq = torch.nn.Parameter(torch.randn(segments * h, h, device=device, dtype=torch.bfloat16) / h ** .5)
        wo = torch.nn.Parameter(torch.randn(h, h, device=device, dtype=torch.bfloat16) / h ** .5)
        bq = None if bias_mode == 'none' else torch.nn.Parameter(torch.randn(segments * h, device=device, dtype=torch.bfloat16) * .1)
        bo = None if bias_mode == 'none' else torch.nn.Parameter(torch.randn(h, device=device, dtype=torch.bfloat16) * .1)
        qkv = FusedQKVProjection(h, args.heads, weight=wq, bias=bq, gate=segments == 4, bias_in_gemm=bias_mode == 'pre')
        proj = FusedOutputProjection(h, args.heads, weight=wo, bias=bo)
        for m in args.lengths:
            torch.manual_seed(1000 + rank + m)
            xs = [torch.randn(m, h, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
            gs = [[torch.randn(p*m, args.heads//p, d, device=device, dtype=torch.bfloat16) for _ in range(segments)] for _ in xs]
            os_ = [torch.randn_like(gs[0][0], requires_grad=True) for _ in xs]
            dys = [torch.randn_like(x) for x in xs]
            for accumulate in (False, True):
                parameters = [wq, wo] + ([] if bq is None else [bq, bo])
                if accumulate:
                    for param in parameters:
                        param.main_grad = torch.zeros_like(param, dtype=torch.float32)
                def step():
                    if accumulate:
                        for param in parameters:
                            param.main_grad.zero_()
                    # Both live forwards share buffers. Backward in reverse order.
                    heads = [qkv(x, workspace) for x in xs]
                    ys = [proj(o, workspace) for o in os_]
                    qs = [t for group in reversed(heads) for t in group]
                    qgs = [t for group in reversed(gs) for t in group]
                    qparams = [wq] + ([] if bq is None else [bq])
                    oparams = [wo] + ([] if bo is None else [bo])
                    qgrad = torch.autograd.grad(qs, xs + qparams, qgs)
                    ograd = torch.autograd.grad(list(reversed(ys)), os_ + oparams, list(reversed(dys)))
                    if accumulate:
                        qgrad = (*qgrad[:2], *(t.main_grad.clone() for t in qparams))
                        ograd = (*ograd[:2], *(t.main_grad.clone() for t in oparams))
                    return heads, ys, qgrad, ograd
                @torch.no_grad()
                def reference():
                    heads, ys, dxs, dos, dwqs, dwos, dbqs, dbos = [], [], [], [], [], [], [], []
                    for x, g, o, dy in zip(xs, gs, os_, dys):
                        raw = mm(x.detach(), wq.detach().T)
                        if bq is not None:
                            raw = (raw if bias_mode == 'pre' else raw.to(torch.bfloat16).float()) + bq.float()
                        packed = torch.cat(gather(raw.to(torch.bfloat16))).view(p*m, args.heads, segments, d)
                        heads.append(tuple(packed[:, rank*(args.heads//p):(rank+1)*(args.heads//p), s].contiguous() for s in range(segments)))
                        parts = gather(torch.stack(g, dim=2))
                        full_dy = torch.cat(parts, dim=1).reshape(p*m, segments*h)[rank*m:(rank+1)*m]
                        dxs.append(mm(full_dy, wq.detach()).to(torch.bfloat16))
                        dwqs.append(mm(full_dy.T, x.detach()))
                        dbqs.append(full_dy.float().sum(0))
                        oseq = torch.cat(gather(o.detach()), dim=1).flatten(1)[rank*m:(rank+1)*m]
                        y = mm(oseq, wo.detach().T).to(torch.bfloat16)
                        if bo is not None:
                            y = y + bo
                        ys.append(y)
                        full_dx = torch.cat(gather(mm(dy, wo.detach()).to(torch.bfloat16)))
                        dos.append(full_dx.view(p*m, args.heads, d)[:, rank*(args.heads//p):(rank+1)*(args.heads//p)].contiguous())
                        dwos.append(mm(dy.T, oseq))
                        dbos.append(dy.float().sum(0))
                    def total(values):
                        if accumulate:
                            return values[1] + values[0]
                        return values[1].to(torch.bfloat16) + values[0].to(torch.bfloat16)
                    return heads, ys, (*dxs, total(dwqs), *([] if bq is None else [total(dbqs)])), (*dos, total(dwos), *([] if bo is None else [total(dbos)]))
                def compare(actual, label):
                    expected = reference()
                    def flatten(value):
                        if isinstance(value, (list, tuple)):
                            return [x for item in value for x in flatten(item)]
                        return [value]
                    actual, expected = flatten(actual), flatten(expected)
                    assert len(actual) == len(expected)
                    for i, (a, b) in enumerate(zip(actual, expected)):
                        check(a, b, f'{segments}/{bias_mode}/{m}/{accumulate}/{label}/{i}')
                # Identity saved-tensor hooks exercise metadata ownership.
                with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                    compare(step(), 'eager')
                if not args.skip_graphs and m == args.lengths[-1]:
                    with torch.cuda.stream(stream):
                        step()
                    stream.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = step()
                    torch.cuda.current_stream().wait_stream(stream)
                    for replay in range(2):
                        with torch.no_grad():
                            for value in xs + os_ + dys + [g for group in gs for g in group]:
                                value.normal_()
                            for weight in (wq, wo):
                                weight.mul_(1.03125)
                        graph.replay()
                        compare(captured, f'graph{replay}')
                    del graph, captured
                if accumulate:
                    for param in parameters:
                        del param.main_grad
            if rank == 0:
                print('PARITY', segments, bias_mode, m, flush=True)
    dy = torch.randn(128, h, device=device, dtype=torch.bfloat16)
    out = torch.empty(h, h, device=device, dtype=torch.float32)
    compiled = mxfp8_wgrad(dy, dy, out)
    ptx = compiled.__ptx__
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.ptx').write_text(ptx)
    assert 'tcgen05.mma' in ptx and 'block_scale' in ptx, 'Native block-scaled MMA missing'
    torch.cuda.synchronize()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(cp=p, heads=args.heads, lengths=args.lengths,
                                              native_block_scaled_mma=True, errors=errors), indent=2)+'\n')
        args.output.with_suffix('.ptx').write_text(ptx)
        print('MXFP8_PASS', len(errors), 'native_block_scaled_mma=True', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    from torch.distributed.elastic.multiprocessing.errors import record
    record(main)()
