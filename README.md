# Fused GEMM + All-to-All

Fused projection GEMMs and Ulysses all-to-all, with backward, for BF16 MHA
training on NVLink-connected GB200 (SM100) GPUs.
**GQA is not supported yet.**

The `codex/mxfp8` branch also provides an experimental MXFP8 path. It is
not part of the v0.1.0 release; BF16 remains the default.

## How it works

QKV/QKVG GEMMs send completed tiles directly to peer GPUs over NVLink.
The output GEMM consumes input tiles as they arrive. Communication overlaps
computation inside the GEMM kernel.

## Tensor layouts

All shapes below are **per rank**. `P` is the CP group size, `M` the local
token count, `N` the total head count, and `H = N * D` the hidden size.
The default head dimension `D` is 128.

| Projection | Input | Weight | Output |
| --- | --- | --- | --- |
| QKV | `x: [M, H]` | `[3*H, H]` | `q, k, v`, each `[P*M, N/P, D]` |
| QKVG for VSA | `x: [M, H]` | `[4*H, H]` | `q, k, v, gate`, each `[P*M, N/P, D]` |
| Output | `o: [P*M, N/P, D]` | `[H, H]` | `y: [M, H]` |

Weights and biases use **Megatron's per-head interleaving**. Each `q_i`,
`k_i`, `v_i` or `g_i` spans `D` consecutive weight rows or bias entries.

```text
QKV:  [q0, k0, v0,     q1, k1, v1,     ...]
QKVG: [q0, k0, v0, g0, q1, k1, v1, g1, ...]
```

Q/K/V/gate are separate, contiguous THD tensors. All returned activations
own their storage and contain no internal padding.

## Installation

Use CUDA-enabled PyTorch with Triton and NVSHMEM symmetric memory support.

```bash
python -m pip install -r requirements.txt
export NVSHMEM_DISABLE_NCCL=1 NVSHMEM_DISABLE_NVLS=1 NVSHMEM_MAX_TEAMS=512
export NCCL_GRAPH_REGISTER=0 CUDA_DEVICE_MAX_CONNECTIONS=32
```

Import from the checkout, or copy these two files into your project.

| File | Purpose |
| --- | --- |
| `fused_gemm_a2a.py` | Public workspace, projections and autograd |
| `fused_gemm_a2a_kernels.py` | CuTe GEMMs, peer transport, Triton copies/reductions and tile geometry |

## API

Initialize distributed and select the CUDA device before creating a workspace.
Supply your own attention function, weights and optional biases (`bias=None`
by default). Use contiguous BF16 tensors; examples assume trainable weights
and inputs.

### QKV

```python
from fused_gemm_a2a import FusedGemmA2AWorkspace, FusedQKVProjection, FusedOutputProjection

workspace = FusedGemmA2AWorkspace(H, N, max_local_tokens=M_max, cp_group=cp_group)
qkv = FusedQKVProjection(H, N, weight=Wqkv, bias=None)
proj = FusedOutputProjection(H, N, weight=Wo, bias=None)

q, k, v = qkv(x, workspace)
o = attention(q, k, v)
y = proj(o, workspace)
y.backward(grad_y)
```

### QKVG for VSA

For VSA, set `gate=True` on both the workspace and QKV projection.

```python
workspace = FusedGemmA2AWorkspace(H, N, max_local_tokens=M_max, cp_group=cp_group, gate=True)
qkvg = FusedQKVProjection(H, N, weight=Wqkvg, gate=True)
proj = FusedOutputProjection(H, N, weight=Wo)

q, k, v, gate = qkvg(x, workspace)
o = vsa(q, k, v, gate)
y = proj(o, workspace)
y.backward(grad_y)
```

Parameter gradients are rank-local; the trainer handles CP/DP gradient reduction.

### Experimental MXFP8

Select the GEMM precision on the workspace; projection calls stay the same.

```python
workspace = FusedGemmA2AWorkspace(
    H, N, max_local_tokens=M_max, cp_group=cp_group,
    gate=True, precision="mxfp8",
)
q, k, v, gate = qkvg(x, workspace)
y = proj(vsa(q, k, v, gate), workspace)
y.backward(grad_y)
```

Forward, input-gradient and weight-gradient GEMMs use native SM100 MXFP8
instructions with FP32 accumulation. Each operand is independently quantized
to E4M3 along its reduction axis, with an E8M0 scale per 32 values. Scales use
`ceil(log2(amax / 448))`; this policy is not a bitwise TE compatibility claim.

Inputs, outputs and caller-owned master weights remain BF16.
The path requires `D=128` and supports optional bias and FP32 `main_grad`
accumulation. TE quantized Parameter subclasses are not accepted yet.
Quantization runs on every invocation, including CUDA Graph replay after
weight updates. QKV/output-gradient tiles are sent as BF16. Receive GEMMs
elect one producer CTA per row tile to gather peer payloads/scales and save
the BF16 operand for backward; other column CTAs reuse those local rows.
This prototype uses a different pipeline from the BF16 receive-pusher path.

```bash
torchrun --standalone --nproc-per-node=4 tests/mxfp8_driver.py \
  --lengths 1 33 1153 --output results/mxfp8-parity.json
torchrun --standalone --nproc-per-node=4 benchmark/mxfp8.py \
  --cp 4 --gate --tokens 131072 --output results/mxfp8-timing.json
```

The timing comparison includes quantization and reports BF16-equivalent MFU
using the same 2500 TFLOPS denominator for both paths.

## Execution constraints

- **CP ranks:** Use equal `M` and `M_max`, and the same call order.
  Require `N % P == 0` and `P <= 32`; pad total tokens before CP partitioning if needed.
- **Capacity:** Create the workspace before execution, pipeline receives or
  CUDA Graph capture. Group, geometry and `M_max` are fixed; `M` may be odd
  and vary between micro-batches, with `M <= M_max`.
- **Sharing:** Pass the same workspace to serialized layers with matching
  geometry. Matching workspace objects also share buffers; serialize their calls.
- **CUDA Graphs:** Warm each shape before capture. Replay with the captured
  shape, or pad inputs to a matching bucket.

## Benchmarks

The **Async Ulysses** reference adapts
[VeOmni](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/distributed/sequence_parallel/async_ulysses.py):
projection GEMMs overlap NCCL all-to-all from earlier projections.
This implementation overlaps GEMM and all-to-all per tile within the same kernel.

### Results

GB200 NVLink · PyTorch 2.12 (26.04) · CUDA 13.2 · CUTLASS 4.8.1 · BF16,
no bias · 131,072 total tokens · H=4096 · 32 heads × 128 · CUDA Graphs.

| CP | Projection | Pass | Async: ms / TFLOPS / MFU | Fused: ms / TFLOPS / MFU | Speedup |
| ---: | --- | --- | ---: | ---: | ---: |
| 8 | QKV | forward | 1.972 / 836 / 33.5% | 1.431 / 1153 / 46.1% | 1.38x |
| 8 | QKV | backward | 3.036 / 1087 / 43.5% | 2.741 / 1203 / 48.1% | 1.11x |
| 8 | QKVG | forward | 2.551 / 862 / 34.5% | 1.896 / 1160 / 46.4% | 1.34x |
| 8 | QKVG | backward | 4.105 / 1071 / 42.9% | 3.646 / 1206 / 48.3% | 1.13x |
| 8 | Output | forward | 0.620 / 887 / 35.5% | 0.485 / 1134 / 45.3% | 1.28x |
| 8 | Output | backward | 1.060 / 1037 / 41.5% | 0.902 / 1219 / 48.7% | 1.18x |
| 16 | QKV | forward | 1.076 / 766 / 30.6% | 0.719 / 1146 / 45.9% | 1.50x |
| 16 | QKV | backward | 1.637 / 1008 / 40.3% | 1.425 / 1157 / 46.3% | 1.15x |
| 16 | QKVG | forward | 1.390 / 791 / 31.6% | 0.953 / 1153 / 46.1% | 1.46x |
| 16 | QKVG | backward | 2.102 / 1046 / 41.8% | 1.893 / 1162 / 46.5% | 1.11x |
| 16 | Output | forward | 0.349 / 787 / 31.5% | 0.275 / 999 / 39.9% | 1.27x |
| 16 | Output | backward | 0.622 / 884 / 35.4% | 0.499 / 1101 / 44.0% | 1.25x |

TFLOPS is per GPU; MFU uses a BF16 dense peak of 2500 TFLOPS.

### Reproduce

```bash
torchrun --standalone --nproc-per-node=4 benchmark/run.py \
  --cp 4 --tokens 131072 --heads 32 --cuda-graphs \
  --warmup 20 --iterations 128 --graph-batch 16 --trials 5 \
  --output results/qkv.json
```

Add `--gate` for QKVG and `--profile` for Torch traces. For CP8/CP16, use
multi-node torchrun on 2/4 four-GPU nodes with matching `--cp`.
CPU checks: `python -m unittest discover -s tests`.

### Experimental MXFP8 status

The MXFP8 path is numerically validated but currently **slower than BF16**.
It remains a development implementation, not a throughput replacement.

GB200 NVLink, 131,072 tokens, 32 heads × 128, QKVG + output projections,
bias enabled, CUDA Graphs. Same-job comparison; times are forward / backward.

| CP | BF16 (ms) | MXFP8 (ms) |
| ---: | ---: | ---: |
| 8 | 2.980 / 5.135 | 3.835 / 10.675 |
| 16 | 1.263 / 2.301 | 2.196 / 6.501 |

CP8 and CP16 each pass 660 independent quantized-reference checks, including
changed-weight graph replay and FP32 gradient accumulation. Maximum relative
L2 error is below 2.84e-4. Native block-scaled MMA is checked for every GEMM
specialization. DiT integration and end-to-end training validation remain
separate work.

## License

Original code is released under [MIT](LICENSE). This project builds on:

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass): Blackwell GEMM examples,
  [BSD-3-Clause](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt).
- [ByteDance VeOmni](https://github.com/ByteDance-Seed/VeOmni): async Ulysses
  benchmark reference, [Apache-2.0](https://github.com/ByteDance-Seed/VeOmni/blob/main/LICENSE).

The CuTe DSL runtime has separate
[NVIDIA license terms](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html).
Notices for adapted source are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
