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

For H4096 QKVG plus output projections, eliminating the temporary write and
read removes `8 * (4 + 1) * H**2 = 640 MiB` of logical memory traffic per
backward call. This byte count is not a measured HBM bandwidth or speedup.

Tests start from nonzero gradients, accumulate several changed-input graph
replays, alternate overwrite and accumulation, protect output boundaries with
canaries and exercise rectangular widths 128/384 and FP32 subnormals.

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
