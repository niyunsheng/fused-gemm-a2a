"""Check local bias gradients, FP32 accumulation and changed-input graph replay.

Run on one GPU. The independent oracle is a PyTorch FP64 row reduction.
No process group or parameter-gradient collective is used.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fused_gemm_a2a import _bias_grad
from fused_gemm_a2a_kernels import reduce_bias_grad


def validate():
    torch.manual_seed(701)
    results = []
    # CP8/CP16 QKV/QKVG/output widths and representative local video lengths.
    shapes = [(1, 128), (17, 513), (127, 4096), (128, 4096), (129, 4096),
              (513, 12288), (7456, 16384), (8192, 4096), (8192, 12288),
              (8192, 16384), (16384, 4096), (16384, 12288), (16384, 16384), (65537, 513)]
    for rows, width in shapes:
        x = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
        # Check accumulation before rounding, including near-cancelling columns.
        x[1::2, :16] = -x[:2 * (rows // 2):2, :16]
        exact = x.double().sum(0)
        mass = x.double().abs().sum(0)
        bound = (4 * torch.finfo(torch.float32).eps * mass).clamp_min(1e-7)
        for dtype in (torch.float32, torch.bfloat16):
            for accumulate in (False, True):
                initial = torch.randn(width, device='cuda', dtype=dtype)
                out = initial.clone()
                reduce_bias_grad(x, out, accumulate=accumulate)
                expected = exact + (initial.double() if accumulate else 0)
                error = (out.double() - expected).abs()
                # BF16 output is rounded only once after the FP32 reduction/add.
                allowance = bound + (expected.abs() / 256 if dtype == torch.bfloat16 else
                                     expected.abs() * torch.finfo(torch.float32).eps)
                assert torch.isfinite(out).all() and (error <= allowance + 1e-7).all(), (
                    rows, width, dtype, accumulate, error.max().item())
                results.append(dict(rows=rows, width=width, dtype=str(dtype),
                                    accumulate=accumulate, max_abs_error=error.max().item()))

    # Exact integer sums catch omitted rows/columns without a loose relative tolerance.
    x = torch.randint(-4, 5, (513, 257), device='cuda').to(torch.bfloat16)
    out = torch.empty(257, device='cuda', dtype=torch.bfloat16)
    reduce_bias_grad(x, out)
    torch.testing.assert_close(out, x.double().sum(0).to(out.dtype), rtol=0, atol=0)

    # The last row starts beyond signed 32-bit element offsets (4 GiB of BF16).
    x = torch.zeros(131073, 16384, device='cuda', dtype=torch.bfloat16)
    x[-1] = 1
    out = torch.empty(16384, device='cuda', dtype=torch.float32)
    reduce_bias_grad(x, out)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    del x

    for rows in (17, 7456):
        x = torch.empty(rows, 16384, device='cuda', dtype=torch.bfloat16)
        bias = torch.nn.Parameter(torch.zeros(16384, device='cuda', dtype=torch.bfloat16))
        bias.main_grad = torch.zeros_like(bias, dtype=torch.float32)
        for accumulate in (False, True):
            x.normal_()
            _bias_grad(bias, x, accumulate)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = _bias_grad(bias, x, accumulate)
            expected_main = torch.zeros_like(bias.main_grad)
            bias.main_grad.zero_()
            for repeat in range(3):
                # Integral gradients have exact FP32 sums, exposing stale replay buffers.
                x.copy_(torch.randint(-2, 3, x.shape, device='cuda'))
                expected = x.double().sum(0).float()
                graph.replay()
                if accumulate:
                    expected_main.add_(expected)
                    torch.testing.assert_close(bias.main_grad, expected_main, rtol=0, atol=0)
                    assert bias.grad_added_to_main_grad
                else:
                    assert result.dtype == torch.bfloat16
                    torch.testing.assert_close(result, expected.to(result.dtype), rtol=0, atol=0)
                    torch.testing.assert_close(bias.main_grad, expected_main, rtol=0, atol=0)
            del graph
    assert _bias_grad(None, x) is None
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = validate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: {len(results)} numerical cases plus exact sums and changed-input graph accumulation')
