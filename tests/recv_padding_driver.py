"""GPU regression: logical receive rows, exact transport, tail zeros and graph reuse."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from fused_gemm_a2a import FusedGemmA2AWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device)
    rank, cp = dist.get_rank(), dist.get_world_size()
    lengths = [1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 255]
    checks = 0
    for segments in (1, 3, 4):
        torch.manual_seed(1234)
        weight = torch.randn(segments * 4096, 4096, device=device, dtype=torch.bfloat16) / 64
        shared = FusedGemmA2AWorkspace(4096, 32, max(lengths), gate=segments == 4)
        torch.manual_seed(5678 + rank)
        inputs = [tuple(torch.randn(cp * n, 4096 // cp, device=device, dtype=torch.bfloat16)
                        for _ in range(segments)) for n in lengths]
        seen = set()

        def calls():
            received, outputs = [], []
            for n, values in zip(lengths, inputs):
                workspace = shared._view(n, segments)
                geometry = (workspace.capacity, n != workspace.capacity)
                compiled = len(shared.comm.compiled)
                workspace.begin_recv()
                out = torch.empty(workspace.capacity, 4096, device=device, dtype=torch.bfloat16)
                workspace.run_recv(values, weight, segments > 1, out)
                if geometry in seen:
                    assert len(shared.comm.compiled) == compiled, 'Logical length recompiled the receive kernel'
                seen.add(geometry)
                received.append(workspace.recv.clone())
                outputs.append(out)
            return received, outputs

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            calls()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            received, outputs = calls()
        torch.cuda.current_stream().wait_stream(stream)
        for replay in range(3):
            expected = []
            for n, values in zip(lengths, inputs):
                parts = []
                for x in values:
                    x.normal_()
                    gathered = [torch.empty_like(x) for _ in range(cp)]
                    dist.all_gather(gathered, x)
                    parts.append(torch.cat([v[rank * n:(rank + 1) * n] for v in gathered], dim=1))
                # Independent PyTorch mapping to [row, head, Q/K/V/G, channel].
                logical = torch.stack([x.view(n, 32, 128) for x in parts], dim=2).flatten(1)
                ref = torch.zeros_like(shared._view(n, segments).recv)
                ref[:n] = logical
                expected.append(ref)
            graph.replay()
            ok = torch.ones((), dtype=torch.int32, device=device)
            for ref, actual, out in zip(expected, received, outputs):
                projected = ref.float() @ (weight.float() if segments > 1 else weight.float().T)
                ok &= torch.all(actual == ref).int()  # Also checks exact zeros in every padding row.
                ok &= torch.all((out.float() - projected).abs() <=
                                0.02 * (projected.abs().max() + projected.abs())).int()
                checks += 2
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if not ok.item():
                raise AssertionError(f'Receive padding failed: segments={segments}, replay={replay}')
        del graph, received, outputs
        if rank == 0:
            print(f'PASS receive padding: segments={segments}, lengths={lengths}', flush=True)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(cp=cp, local_tokens=lengths, checks=checks,
                                              cuda_graph=True), indent=2) + '\n')
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
