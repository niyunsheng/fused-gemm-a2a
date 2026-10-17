"""Regression: prepare symmetric buffers before pipeline receives are posted.

Run with four ranks (CP2 stages) and --prepare.
Stage 0 must initialize its buffers before stage 1 can send the return token.
Without --prepare, symmetric allocation deadlocks this legal PP ordering.
"""
import argparse
import datetime
import faulthandler
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--prepare', action='store_true')
args = parser.parse_args()
faulthandler.dump_traceback_later(40, repeat=True)
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=75))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 4
stage_size = world // 2
cp_groups = [dist.new_group(list(range(stage_size))),
             dist.new_group(list(range(stage_size, world)))]
# Independent directions avoid serializing a send behind its return receive.
pp_forward = [dist.new_group([i, i + stage_size]) for i in range(stage_size)]
pp_return = [dist.new_group([i, i + stage_size]) for i in range(stage_size)]
cp = cp_groups[rank // stage_size]
forward, backward = pp_forward[rank % stage_size], pp_return[rank % stage_size]
recv_group = backward if rank < stage_size else forward
send_group = forward if rank < stage_size else backward
from fused_gemm_a2a import FusedGemmA2AWorkspace, FusedQKVProjection, FusedOutputProjection
hidden, heads = 256 * stage_size, 2 * stage_size
def make_modules():
    def weight(rows):
        return torch.nn.Parameter(torch.empty(rows, hidden, device='cuda', dtype=torch.bfloat16)
                                  .normal_(std=hidden ** -0.5))

    qkv = FusedQKVProjection(hidden, heads, gate=True,
                             weight=weight(4 * hidden))
    out = FusedOutputProjection(hidden, heads, weight=weight(hidden))
    return qkv, out

qkv, out = make_modules()
workspace = None
if args.prepare:
    workspace = FusedGemmA2AWorkspace(hidden, heads, 128, cp_group=cp, gate=True)
signal = torch.tensor([rank + 1.0], device='cuda')
received = torch.empty_like(signal)
peer = (rank + stage_size) % world
# Warm the unbatched P2P communicators too: NCCL creates separate communicators
# for unbatched P2P even if a collective on the group has already completed.
for group in (forward, backward):
    if rank < stage_size:
        dist.isend(signal, dst=peer, group=group).wait()
        dist.irecv(received, src=peer, group=group).wait()
    else:
        dist.irecv(received, src=peer, group=group).wait()
        dist.isend(signal, dst=peer, group=group).wait()
    torch.cuda.current_stream().synchronize()
torch.cuda.current_stream().synchronize()
dist.barrier()
print(f'rank={rank} prepared; posting independent PP receive', flush=True)
peer = (rank + stage_size) % world
pending = dist.irecv(received, src=peer, group=recv_group)
if rank >= stage_size:
    pending.wait()
    torch.cuda.current_stream().synchronize()
print(f'rank={rank} starting projection setup', flush=True)
if workspace is None:
    workspace = FusedGemmA2AWorkspace(hidden, heads, 128, cp_group=cp, gate=True)
print(f'rank={rank} projection setup returned', flush=True)
dist.isend(signal, dst=peer, group=send_group).wait()
if rank < stage_size:
    pending.wait()
torch.cuda.current_stream().synchronize()
assert received.item() == peer + 1
x = torch.randn(128, hidden, device='cuda', dtype=torch.bfloat16, requires_grad=True)
q,k,v,g = qkv(x, workspace)
y = out(q + k + v + g, workspace)
y.float().square().mean().backward()
assert torch.isfinite(x.grad).all()
dist.barrier()
print(f'PIPELINE_SETUP_PASS rank={rank}', flush=True)
faulthandler.cancel_dump_traceback_later()
dist.destroy_process_group()
