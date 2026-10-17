"""Different micro-batch lengths on the same projections, with delayed backward.

Run with torchrun. Independent FP32 PyTorch linears use the documented head-QKVG
weight order and BF16 operator boundaries. Check outputs, dX/dO and CP-reduced
parameter gradients in eager execution and changed-input CUDA Graph replays.
"""
import argparse
from dataclasses import FrozenInstanceError
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fused_gemm_a2a import FusedGemmA2AWorkspace, FusedOutputProjection, FusedQKVProjection, MMA_TILER_MN, _Comm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-tokens', nargs='+', type=int, default=[512, 128, 513, 255, 257, 1, 512])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delay-push', action='store_true',
                        help='Rotate a delayed producer after the QKV entry barrier, including graph replay')
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    # Construction must not consult a process group or allocate communication
    # resources. WORLD is resolved only after distributed initialization.
    with patch.object(_Comm, 'get', side_effect=AssertionError('eager communication setup')):
        probe_qkv = FusedQKVProjection(256, 2, weight=torch.empty(768, 256, device=device, dtype=torch.bfloat16))
        probe_out = FusedOutputProjection(256, 2, weight=torch.empty(256, 256, device=device, dtype=torch.bfloat16))
    try:
        FusedGemmA2AWorkspace(256, 2, 128)
    except RuntimeError as error:
        assert 'Initialize the CP process group' in str(error)
    else:
        raise AssertionError('Workspace lookup accepted an uninitialized process group')
    del probe_qkv, probe_out
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=15))
    rank, cp = dist.get_rank(), dist.get_world_size()
    hidden, heads, dim = 4096, 32, 128
    if heads % cp:
        raise ValueError('32 heads must divide over CP')
    head_slice = slice(rank * heads // cp, (rank + 1) * heads // cp)
    errors = {}

    def gather(x, axis):
        parts = [torch.empty_like(x) for _ in range(cp)]
        dist.all_gather(parts, x.detach().contiguous())
        return torch.cat(parts, dim=axis)

    def check(actual, expected, label, reduce=False):
        actual, expected = actual.detach().float(), expected.detach().float()
        if reduce:
            actual = actual.clone()
            dist.all_reduce(actual)
        delta = (actual - expected).abs()
        peak = expected.abs().max().clamp_min(1e-12)
        rms = delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)
        a, b = actual.flatten().double(), expected.flatten().double()
        dot = (a * b).sum()
        norm_a, norm_b = a.square().sum(), b.square().sum()
        cosine = dot / (norm_a * norm_b).sqrt().clamp_min(1e-24)
        similarity = 2 * dot / (norm_a + norm_b).clamp_min(1e-24)
        good = (torch.isfinite(actual).all() & (delta <= 0.02 * (peak + expected.abs())).all()
                & (rms <= 0.02) & (cosine >= 0.999) & (similarity >= 0.999))
        metrics = torch.stack((delta.max() / peak, rms, 1 - cosine, 1 - similarity, (~good).float()))
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        errors[label] = dict(max_error_over_peak=metrics[0].item(), relative_rms=metrics[1].item(),
                             cosine=1 - metrics[2].item(), similarity=1 - metrics[3].item())
        if metrics[4].item():
            raise AssertionError(f'{label}: {errors[label]}')

    # Bias-free QKV and both biased QKVG paths share one logical-row interface.
    for gate, bias_in_gemm in ((False, False), (True, True), (True, False)):
        torch.manual_seed(1234)
        segments = 3 + gate
        def weight(rows):
            return torch.nn.Parameter(torch.empty(rows, hidden, device=device, dtype=torch.bfloat16)
                                      .normal_(std=hidden ** -0.5))

        wq, wo = weight(segments * hidden), weight(hidden)
        bq = torch.nn.Parameter(torch.randn(segments * hidden, device=device, dtype=torch.bfloat16)) if gate else None
        bo = torch.nn.Parameter(torch.randn(hidden, device=device, dtype=torch.bfloat16)) if gate else None
        qkv = FusedQKVProjection(hidden, heads, weight=wq, bias=bq, gate=gate,
                                 bias_in_gemm=bias_in_gemm, accumulate_bias_grad=False)
        proj = FusedOutputProjection(hidden, heads, weight=wo, bias=bo,
                                    accumulate_bias_grad=False)
        params = (wq, wo, bq, bo) if gate else (wq, wo)
        refs = tuple(p.detach().float().requires_grad_() for p in params)
        maximum = max(args.local_tokens)
        workspace = FusedGemmA2AWorkspace(hidden, heads, maximum, gate=gate)
        for field_name, value in (('max_local_tokens', maximum + 1), ('cp_group', None)):
            try:
                setattr(workspace, field_name, value)
            except FrozenInstanceError:
                pass
            else:
                raise AssertionError('Workspace configuration must be immutable')
        for segments_ in (segments, 1):
            try:
                workspace._view(maximum + 1, segments_)
            except ValueError as error:
                assert 'max_local_tokens' in str(error)
            else:
                raise AssertionError('Maximum workspace capacity was not enforced')
        other_qkv = FusedQKVProjection(hidden, heads, weight=wq, bias=bq, gate=gate,
                                      bias_in_gemm=bias_in_gemm, accumulate_bias_grad=False)
        other_out = FusedOutputProjection(hidden, heads, weight=wo, bias=bo,
                                         accumulate_bias_grad=False)
        probe = torch.empty(maximum + 1, hidden, device=device, dtype=torch.bfloat16)
        for action, message in (
                (lambda: qkv(probe, workspace), 'max_local_tokens'),
                (lambda: qkv(probe[:1], None), 'FusedGemmA2AWorkspace'),
                (lambda: qkv(probe[:1].cpu(), workspace), 'same CUDA device'),
                (lambda: FusedQKVProjection(hidden, heads // 2, weight=wq, gate=gate), 'one N tile per head')):
            try:
                action()
            except (ValueError, TypeError) as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError(f'Invalid call accepted: {message}')
        wrong_mode = FusedQKVProjection(hidden, heads, weight=weight((4 if not gate else 3) * hidden), gate=not gate)
        try:
            wrong_mode(probe[:1], workspace)
        except ValueError as error:
            assert 'QKV/QKVG mode' in str(error)
        else:
            raise AssertionError('Mismatched QKV/QKVG workspace accepted')
        del probe, wrong_mode
        torch.manual_seed(5678 + rank)
        batches = []
        for length in args.local_tokens:
            x = torch.randn(length, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
            o = torch.randn(cp * length, heads // cp, dim, device=device, dtype=torch.bfloat16)
            o.requires_grad_()
            gs = [torch.randn(cp * length, heads // cp, dim, device=device, dtype=torch.bfloat16)
                  for _ in range(segments)]
            dy = torch.randn_like(x)
            batches.append((length, x, o, gs, dy))

        gemm = workspace.comm.gemm
        push_call = 0

        def delayed_gemm(mode, *positional, **keywords):
            nonlocal push_call
            if mode == 'push':
                if rank == push_call % cp:
                    torch.cuda._sleep(2_000_000)
                push_call += 1
            return gemm(mode, *positional, **keywords)

        def fused_step():
            if args.delay_push:
                with patch.object(workspace.comm, 'gemm', delayed_gemm):
                    return run_step()
            return run_step()

        def run_step():
            outputs = []
            for i, (length, x, o, gs, dy) in enumerate(batches):
                q_module, o_module = (qkv, proj) if i % 2 == 0 else (other_qkv, other_out)
                outputs.append((*q_module(x, workspace), o_module(o, workspace)))
            # All forwards have finished on the same module instances. Reverse
            # backward must use the earlier call's capacity and logical length.
            gradients = [None] * len(batches)
            for i in reversed(range(len(batches))):
                length, x, o, gs, dy = batches[i]
                gradients[i] = torch.autograd.grad(outputs[i], (x, o, *params), (*gs, dy))
            return outputs, gradients

        def reference_step():
            results = []
            for length, x, o, gs, dy in batches:
                rows = slice(rank * length, (rank + 1) * length)
                xg = gather(x, 0).requires_grad_()
                og = gather(o, 1).requires_grad_()
                rg = [gather(g, 1) for g in gs]
                dyg = gather(dy, 0)
                rq, ro = refs[:2]
                rbq, rbo = refs[2:] if gate else (None, None)
                packed = F.linear(xg.float(), rq, rbq if qkv.bias_in_gemm else None).to(torch.bfloat16)
                if gate and not qkv.bias_in_gemm:
                    packed = packed + rbq.to(torch.bfloat16)
                outs = list(packed.view(cp * length, heads, segments, dim).unbind(2))
                y = F.linear(og.flatten(1).float(), ro).to(torch.bfloat16)
                if gate:
                    y = y + rbo.to(torch.bfloat16)
                grads = torch.autograd.grad((*outs, y), (xg, og, *refs), (*rg, dyg))
                results.append(((*(v[:, head_slice] for v in outs), y[rows]),
                                (grads[0][rows], grads[1][:, head_slice], *grads[2:])))
            return results

        def validate(actuals, label):
            expected = reference_step()
            outs, grads = actuals
            names = ('q', 'k', 'v', 'gate')[:segments]
            for i, (length, *_) in enumerate(batches):
                for j, name in enumerate((*names, 'y')):
                    actual = outs[i][j]
                    check(actual, expected[i][0][j], f'{label}/mb{i}/{name}')
                for j, name in enumerate(('dx', 'do', 'dw_qkv', 'dw_o', 'db_qkv', 'db_o')[:len(grads[i])]):
                    actual = grads[i][j]
                    check(actual, expected[i][1][j], f'{label}/mb{i}/{name}', reduce=j >= 2)

        name = f'{"qkvg_bias" if gate else "qkv"}/bias_in_gemm={bias_in_gemm}'
        # Setup happens once for the maximum. New active lengths must not allocate
        # symmetric memory; all backwards run after every forward finishes.
        with patch.object(_Comm, 'symmetric', side_effect=AssertionError('allocation after setup')):
            validate(fused_step(), name + '/eager')
        workspaces = {}
        # Another handle with matching geometry reuses the same allocation.
        with patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('cached device sync')), \
                patch.object(dist, 'barrier', side_effect=AssertionError('cached group barrier')):
            shared = FusedGemmA2AWorkspace(hidden, heads, maximum, gate=gate)
            allocated = torch.cuda.memory_allocated()
            for length in args.local_tokens:
                qws, ows = workspace._view(length, segments), workspace._view(length, 1)
                if qws.capacity in workspaces:
                    assert workspaces[qws.capacity] == (qws, ows)
                workspaces[qws.capacity] = (qws, ows)
                assert shared._view(length, segments) is qws
                assert shared._view(length, 1) is ows
                for view, buffers in ((qws, workspace._buffers[segments]), (ows, workspace._buffers[1])):
                    assert view.recv.data_ptr() == buffers.recv.data_ptr()
                    assert view.recv.shape[0] == view.capacity
                    assert view.counters.data_ptr() == buffers.counters.data_ptr()
                    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(view.heads, buffers.heads))
            assert torch.cuda.memory_allocated() == allocated
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused_step()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fused_step()
        torch.cuda.current_stream().wait_stream(stream)
        for replay in range(3):
            with torch.no_grad():
                if gate:
                    bq.normal_()
                    bo.normal_()
                    # CP shares parameters; only activations/gradients differ by rank.
                    dist.broadcast(bq, src=0)
                    dist.broadcast(bo, src=0)
                    refs[2].copy_(bq)
                    refs[3].copy_(bo)
                for _, x, o, gs, dy in batches:
                    for tensor in (x, o, *gs, dy):
                        tensor.normal_()
            # A replay must not depend on the most recently selected host state.
            workspace._view(args.local_tokens[replay % len(batches)], segments)
            workspace._view(args.local_tokens[(replay + 1) % len(batches)], 1)
            graph.replay()
            validate(captured, f'{name}/graph{replay}')
        del graph, captured
        if rank == 0:
            print(f'PASS {name}: lengths={args.local_tokens}', flush=True)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(
            cp=cp, hidden_size=hidden, heads=heads, head_dim=dim, local_tokens=args.local_tokens,
            tile_m=MMA_TILER_MN[0], cuda_graph=True, errors=errors,
            delayed_push=args.delay_push,
            source_sha256={name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                for name in ('fused_gemm_a2a.py', 'fused_gemm_a2a_kernels.py')},
        ), indent=2) + '\n')
        print(f'Variable-length projections: {len(errors)} checks passed', flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
