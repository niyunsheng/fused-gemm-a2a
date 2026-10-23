"""Single-GPU MXFP8 byte parity and FP32 accumulation regressions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from fused_gemm_a2a_kernels import mxfp8_quantize, mxfp8_wgrad


def quantizer_reference(x, padded_shape):
    """Independent float64 scale calculation, including finite BF16 extremes."""
    rows, cols = padded_shape
    groups = F.pad(x.double(), (0, cols - x.shape[1], 0, rows - x.shape[0])).reshape(rows, -1, 32)
    peak = groups.abs().amax(-1)
    exponent = torch.ceil(torch.log2(peak / 448))
    # CUDA log2(double) can round an exact integer slightly upward. Repair
    # the estimate by checking the defining inequalities with exact binary
    # scale boundaries, rather than relaxing byte parity at those boundaries.
    exponent = torch.where(peak <= 224 * torch.exp2(exponent), exponent - 1, exponent)
    exponent = torch.where(peak > 448 * torch.exp2(exponent), exponent + 1, exponent)
    exponent = torch.where(peak == 0, 0, exponent).clamp(-126, 127)
    payload = (groups / torch.exp2(exponent).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(rows, cols)
    # Decode the public SM100 block-scale layout into logical [row, K-group].
    r = torch.arange(rows, device=x.device)[:, None]
    g = torch.arange(cols // 32, device=x.device)[None, :]
    offsets = ((r // 128) * (cols // 128) + g // 4) * 512 + r % 32 * 16 + r % 128 // 32 * 4 + g % 4
    return payload, (exponent + 127).to(torch.uint8), offsets


def validate_quantization():
    magnitudes = torch.arange(0x7f80, device='cuda', dtype=torch.int32).to(torch.int16)
    cases = 0
    for sign in (0, -32768):
        values = (magnitudes ^ sign).view(torch.bfloat16)
        logical = values[:, None].expand(-1, 32).contiguous()
        # Also test groups containing mixed magnitudes and an odd token tail.
        # Keep the physical leading dimension vector aligned while testing a
        # logical K tail, just as the projection operands' layouts require.
        mixed = F.pad(values[:128 * 255].reshape(128, 255), (0, 1))[:, :255]
        for source in (logical, mixed):
            for transpose in (False, True):
                x = source.T.contiguous().T if transpose else source
                payload, scales = mxfp8_quantize(x)
                expected, expected_scales, offsets = quantizer_reference(x, payload.shape)
                if not torch.equal(payload.view(torch.uint8), expected.view(torch.uint8)):
                    positions = (payload.view(torch.uint8) != expected.view(torch.uint8)).nonzero()[:8]
                    details = [dict(row=r, col=c, source=float(x[r, c]) if r < x.shape[0] and c < x.shape[1] else 0,
                                    actual=int(payload.view(torch.uint8)[r, c]), expected=int(expected.view(torch.uint8)[r, c]),
                                    scale=int(scales[offsets[r, c // 32]]), expected_scale=int(expected_scales[r, c // 32]))
                               for r, c in positions.tolist()]
                    raise AssertionError(('payload', sign, transpose, details))
                assert torch.equal(scales[offsets], expected_scales), ('scale', sign, transpose)
                cases += 1
    for bad in (torch.zeros(128, 255, device='cuda', dtype=torch.bfloat16),
                torch.zeros(128, 256, device='cuda', dtype=torch.bfloat16)[:, 1:129]):
        try:
            mxfp8_quantize(bad)
        except ValueError as error:
            assert 'aligned' in str(error)
        else:
            raise AssertionError('Unaligned quantizer input was accepted')
    return cases


def validate_accumulation():
    cases = 0
    errors = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for rows, cols in ((128, 128), (384, 128), (128, 384), (384, 256), (512, 384)):
            for tokens in (1, 33, 1153):
                dy = torch.randn(tokens, rows, device='cuda', dtype=torch.bfloat16)
                x = torch.randn(tokens, cols, device='cuda', dtype=torch.bfloat16)
                storage = torch.full((rows * cols + 256,), 123.25, device='cuda')
                out = storage[128:-128].view(rows, cols)
                seed = torch.randn_like(out)
                delta = torch.empty_like(out)
                mxfp8_wgrad(dy, x, delta)
                out.copy_(seed)
                mxfp8_wgrad(dy, x, out, accumulate=True)
                torch.testing.assert_close(out, seed + delta, rtol=0, atol=0)
                assert torch.all(storage[:128] == 123.25) and torch.all(storage[-128:] == 123.25)
                cases += 1
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    mxfp8_wgrad(dy, x, out, accumulate=True)
                expected = out.clone()
                for _ in range(3):
                    dy.normal_()
                    x.normal_()
                    mxfp8_wgrad(dy, x, delta)
                    expected.add_(delta)
                    graph.replay()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    assert torch.all(storage[:128] == 123.25) and torch.all(storage[-128:] == 123.25)
                    cases += 1
                # The overwrite specialization must stay independent of the
                # accumulation specialization in the compilation cache.
                mxfp8_wgrad(dy, x, out)
                torch.testing.assert_close(out, delta, rtol=0, atol=0)
                cases += 1
                graph.reset()
                errors.append(float((out - delta).abs().max()))
        # FP32 subnormals in an existing main_grad must survive adding zero.
        zero = torch.zeros(33, 128, device='cuda', dtype=torch.bfloat16)
        out = torch.full((128, 128), 2.0 ** -140, device='cuda')
        mxfp8_wgrad(zero, zero, out, accumulate=True)
        torch.testing.assert_close(out, torch.full_like(out, 2.0 ** -140), rtol=0, atol=0)
        cases += 1
    stream.synchronize()
    return dict(comparisons=cases, maximum_absolute_error=max(errors), fp32_subnormals=True)


def validate_nonfinite():
    """Overflow must remain observable, including after changed-input replay."""
    cases = 0
    for value in (float('inf'), -float('inf'), float('nan')):
        for transpose in (False, True):
            source = torch.ones(128, 128, device='cuda', dtype=torch.bfloat16)[:, :33]
            source[0, 0] = source[1, 31] = source[2, 32] = value
            if transpose:
                source = source.T.contiguous().T
            payload, scales = mxfp8_quantize(source)
            assert torch.isnan(payload.float()[[0, 1, 2], [0, 31, 32]]).all()
            if value == value:  # Inf groups must carry the E8M0 NaN scale.
                assert torch.all(scales[torch.tensor([0, 16, 33], device='cuda')] == 255)
            assert torch.isfinite(payload.float()[3:]).all()
            cases += 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for operand in ('dy', 'x'):
            for accumulate in (False, True):
                dy = torch.ones(33, 128, device='cuda', dtype=torch.bfloat16)
                x = torch.ones_like(dy)
                (x if operand == 'dy' else dy).fill_(2.0 ** -64)
                out = torch.ones(128, 128, device='cuda', dtype=torch.float32)
                mxfp8_wgrad(dy, x, out, accumulate=accumulate)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    mxfp8_wgrad(dy, x, out, accumulate=accumulate)
                for value in (float('inf'), -float('inf'), float('nan'), 1.0):
                    target = dy if operand == 'dy' else x
                    target[0, 0] = value
                    out.fill_(1)
                    graph.replay()
                    affected = out[0, :] if operand == 'dy' else out[:, 0]
                    if value == 1.0:
                        assert torch.isfinite(out).all(), (operand, accumulate, 'finite recovery')
                    else:
                        assert (~torch.isfinite(affected)).all(), (operand, accumulate, value)
                        assert torch.isfinite(out[1:, 1:]).all()
                    cases += 1
                graph.reset()
    stream.synchronize()
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(123)
    result = dict(quantizer_byte_comparisons=validate_quantization(), accumulation=validate_accumulation(),
                  nonfinite_comparisons=validate_nonfinite())
    root = Path(__file__).resolve().parents[1]
    result['source_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('fused_gemm_a2a.py', 'fused_gemm_a2a_kernels.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('MXFP8_LOCAL_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
