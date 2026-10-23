# MXFP8 projection design and validation

`FusedGemmA2AWorkspace(..., precision="mxfp8")` selects native SM100
block-scaled GEMMs for forward, input gradients and weight gradients. BF16
remains the default. The production implementation is still two Python files.

## Numerical and storage contract

- Public activations, outputs and caller-owned master weights are BF16.
- Independently quantize each GEMM operand along its reduction axis to E4M3,
  with one E8M0 power-of-two scale per 32 elements and FP32 accumulation.
- Use `ceil(log2(amax / 448))`, clamped to [-126, 127]; zero groups use scale 1.
  This is a specified quantization policy, not a bitwise TE-recipe guarantee.
- Groups containing Inf use the E8M0 NaN scale (byte 255), and NaN inputs
  retain NaN payloads. Nonfinite inputs therefore remain observable through
  native MMA rather than being silently saturated to finite gradients.
- Quantize on each invocation, including graph replay after weight updates.
  Transposing an existing quantized representation does not implement a
  different reduction-axis quantization.
- All activation communication remains BF16. Receive warps quantize completed
  BF16 row blocks into local payload/scale buffers before TMA consumes them.
  Pusher and quantization warps are separate so waits cannot stop peer writes.
- Saved BF16 operands support independently quantized wgrad and exact-input
  bias reductions. Outputs and saved activations own their storage across
  shared-workspace reuse and overlapping microbatch lifetimes.
- Gradients remain rank-local; the trainer owns distributed gradient reduction
  and optimizer updates. TE quantized Parameter subclasses are not supported.

## Scale calculation

The standalone and receive quantizers share a scale helper that derives the
exponent from the exact BF16 maximum. Since 448 equals `1.75 * 2**8`, the
exponent and a significand comparison implement the ceiling without
`log2`/`exp2`. Finite BF16 subnormals saturate at the minimum scale exponent.

The local validation driver compares both payload bytes and decoded E8M0
scales to a float64 PyTorch reference, independently of the shared helper.
It exhausts all finite BF16 magnitudes and both signs, including signed zero,
and covers mixed-magnitude groups, transposed storage and padding.
The reference corrects the floating-point `log2` estimate using the defining
power-of-two inequalities before rounding; byte parity is required even at
exact scale boundaries. Vector loads require a 16-byte-aligned data pointer
and leading stride, with unit minor stride. Logical tails are supported in
aligned storage; unsupported layouts raise `ValueError` before launch.

## FP32 gradient accumulation

When a weight carries FP32 `main_grad`, the local wgrad epilogue adds its
completed FP32 tile directly to that buffer. There is no full-size temporary
dW or separate `add_` launch. Each output tile has a unique writer; ordinary
FP32 loads, addition and stores avoid atomic reduction and preserve subnormals.
Loads and stores predicate padded M rows against the logical output boundary.
The compilation cache distinguishes overwrite and accumulate specializations.
The caller's `main_grad` must have the weight's exact shape and device,
contiguous FP32 storage and a 16-byte-aligned pointer. Invalid buffers raise
`ValueError` before launching wgrad; their shape must never determine a
different GEMM output extent.

For H4096 QKVG plus output projections, eliminating the temporary write and
read removes `8 * (4 + 1) * H**2 = 640 MiB` of logical memory traffic per
backward call. This byte count is not a measured HBM bandwidth or speedup.

Tests start from nonzero gradients, accumulate several changed-input graph
replays, alternate overwrite and accumulation, protect output boundaries with
canaries and exercise rectangular widths 128/384 and FP32 subnormals.
Nonfinite regressions use tiny finite counterpart operands, so downstream
overflow cannot hide incorrect finite saturation. They cover both standalone
quantizer orientations, either wgrad operand, overwrite/accumulate, receive
quantization, and changed-input CUDA Graph replay followed by finite recovery.

## Reproduce validation

Use the CUDA-enabled environment described in the root README.

```bash
python -m unittest discover -s tests
python tests/mxfp8_local_driver.py --output results/mxfp8-local.json
torchrun --standalone --nproc-per-node=3 tests/mxfp8_driver.py \
  --heads 3 --lengths 1 33 1153 --output results/mxfp8-cp3.json
```

The distributed driver also checks independent quantized PyTorch math,
QKV/QKVG/output projections, bias before/after BF16 rounding, backward,
FP32 main_grad, saved-tensor hooks, changed-weight graph replay and two live
microbatches with reversed backward. Run it at CP8 and CP16 with 32 heads.
Run `tests/variable_length_driver.py` for BF16 regression.

## Performance measurement

The historical benchmark uses `--gate --bias`. To include training-style
FP32 gradient accumulation and DiT's QKVG-only, pre-rounding bias, use:

```bash
torchrun --standalone --nproc-per-node=4 benchmark/mxfp8.py \
  --cp 4 --gate --main-grad --qkv-bias-only --bias-in-gemm \
  --tokens 131072 --warmup 10 --iterations 128 --trials 7 --graph-batch 8 \
  --output results/mxfp8-main-grad.json
```

Use multi-node torchrun for CP8/CP16 and repeat with 119296 tokens and
`--precision-order mxfp8-first`. Timing includes conversion, communication,
padding, owned copies and gradient accumulation. Gradient zeroing is outside
timing. The BF16 wgrad backend is selected by `FUSED_WGRAD_BACKEND` as for the
normal API; the default is `torch`.

Report native MXFP8 MFU against 5000 TFLOPS/GPU and BF16-equivalent MFU against
2500 TFLOPS/GPU separately. These projection benchmarks do not measure full
attention, training iterations or convergence. Compare revisions on the same
allocation with the same harness, retain BF16 control timings, and use
uninstrumented multi-trial timings for performance conclusions.

## Validation results (2026-10-22)

The production source in `6b09fbf` passes 21 unit tests, eight exhaustive
payload/scale byte-parity cases, and 76 exact FP32 accumulation comparisons.
The accumulation checks include nonzero initial gradients, repeated
changed-input CUDA Graph replay, rectangular output shapes, boundary
canaries, overwrite/accumulate cache separation and FP32 subnormals.

At CP3, CP8 and CP16, the distributed driver passes 678 independent
quantized-reference comparisons per configuration, plus byte-exact receive
quantization and finite-BF16 scale checks. BF16 regression passes 480
comparisons at each of CP8 and CP16. PTX checks require native block-scaled
MMA. These checks validate the declared quantized math and storage/lifetime
contracts; they do not establish full-model convergence.

## Paired performance results (2026-10-22)

GB200, PyTorch 26.04 / CUDA 13.2, CUTLASS DSL 4.6.2, H4096, 32 heads.
Pairs are **forward / backward**, summing QKVG and output projections.
Compare implementation `6b09fbf` with `85d1ecc` using the same harness and
allocation per CP size. Average both precision-order medians; each stage
uses seven trials, 128 iterations, ten warmups and graph batch eight.

| Mode | CP | Tokens | BF16 ms | MXFP8 ms | Speedup / BF16 | Speedup / 85d1ecc | Native MXFP8 MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| historical | 8 | 131,072 | 2.535 / 4.653 | 1.971 / 3.461 | 1.286x / 1.344x | 1.002x / 1.030x | 27.89% / 31.76% |
| historical | 8 | 119,296 | 2.672 / 4.376 | 1.855 / 3.231 | 1.440x / 1.354x | 1.000x / 1.020x | 26.97% / 30.97% |
| historical | 16 | 131,072 | 1.298 / 2.454 | 1.098 / 1.860 | 1.182x / 1.319x | 1.009x / 1.021x | 25.02% / 29.55% |
| historical | 16 | 119,296 | 1.342 / 2.331 | 1.055 / 1.748 | 1.271x / 1.333x | 1.013x / 1.024x | 23.71% / 28.63% |
| training | 8 | 131,072 | 2.375 / 4.780 | 1.826 / 3.608 | 1.301x / 1.325x | 1.014x / 1.017x | 30.11% / 30.47% |
| training | 8 | 119,296 | 2.525 / 4.479 | 1.706 / 3.320 | 1.480x / 1.349x | 1.016x / 1.011x | 29.33% / 30.14% |
| training | 16 | 131,072 | 1.224 / 2.566 | 1.016 / 2.087 | 1.205x / 1.229x | 1.046x / 1.009x | 27.06% / 26.34% |
| training | 16 | 119,296 | 1.269 / 2.450 | 0.981 / 1.904 | 1.294x / 1.286x | 1.017x / 1.015x | 25.51% / 26.27% |

Historical mode uses `--gate --bias`; training-style mode uses
`--gate --main-grad --qkv-bias-only --bias-in-gemm`. BF16 wgrad uses
the default `torch` backend in both modes. Gradient zeroing is outside
timing. These are projection measurements, not full-model training MFU.

The largest BF16 aggregate control drift between revisions is 0.84%.
The small revision-to-revision gains should be read with that variability.
Native MXFP8 MFU uses 5000 TFLOPS/GPU; BF16-equivalent MFU is twice
the displayed percentage. Timings are uninstrumented. A separate CP8
trace confirms that the MXFP8 wgrad path has no separate dW-add kernel.

[Machine-readable results and source hashes](mxfp8-results-20261022.json).

## Review corrections (2026-10-22)

Review of `f795546` reproduced two correctness issues, fixed in `48eb2de`:

- A 384x128 parameter accepted a 128x128 `main_grad`, silently computing a
  truncated gradient while returning a full-size dummy. Exact shape/device,
  dtype, stride and alignment validation now rejects malformed buffers.
- Inf quantized to finite E4M3 +/-448 with E8M0 byte 247. With the other
  wgrad operand equal to 2^-64, the result was finite +/-3.22818e19. Inf
  groups now use E8M0 NaN byte 255 in both standalone and receive paths.

The review also restored the README's original license and attribution
section, which had been accidentally replaced by the benchmark table.

GB200 validation of the corrected production source passes 25 unit tests,
eight exhaustive finite-BF16 byte cases, 76 exact FP32 accumulation checks
and 22 standalone nonfinite checks. CP3/CP8/CP16 each pass 678 independent
quantized-reference comparisons and eight nonfinite receive checks, including
changed-input graph replay and finite recovery. Receive payload/scale bytes
match the standalone quantizer for finite and nonfinite cases. CP8/CP16 each
also pass 480 BF16 regression comparisons. Native block-scaled MMA remains
required. These tests do not establish full-model convergence.

[Review validation and source hashes](mxfp8-review-20261022.json).

Paired training-style timing compares `48eb2de` with pre-review `f795546`
on the same allocation per CP size. The environment, shapes and timing
settings match the training rows above: both precision orders, seven trials,
128 iterations, ten warmups and graph batch eight. All 16 timing files pass
source-hash, configuration and native-MMA checks. Pairs are forward / backward.

| CP | Tokens | Corrected MXFP8 ms | Latency change vs f795546 | Speedup / BF16 | Native MXFP8 MFU |
|---:|---:|---:|---:|---:|---:|
| 8 | 131,072 | 1.839 / 3.617 | +0.16% / -0.13% | 1.31x / 1.33x | 29.90% / 30.40% |
| 8 | 119,296 | 1.712 / 3.345 | -0.19% / +0.15% | 1.48x / 1.35x | 29.23% / 29.91% |
| 16 | 131,072 | 1.021 / 2.079 | +0.13% / -0.55% | 1.20x / 1.24x | 26.93% / 26.45% |
| 16 | 119,296 | 0.984 / 1.910 | +0.16% / -0.40% | 1.30x / 1.28x | 25.42% / 26.19% |

Maximum absolute BF16 aggregate control drift is 0.27%. The largest MXFP8
latency increase is 0.16%, within that variability; these measurements show
no material regression from the correctness fixes. Native MXFP8 MFU uses
5000 TFLOPS/GPU and covers projections only. Historical-mode timing was not
repeated for this correction; its earlier measurements remain above.
