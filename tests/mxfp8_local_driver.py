"""Native PyTorch MXFP8 parity and FP32 accumulation regressions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fused_gemm_a2a_kernels import mxfp8_quantize, mxfp8_wgrad
from benchmark.reference import mxfp8_mm


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
                delta = mxfp8_mm(dy.T, x)
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
                    delta = mxfp8_mm(dy.T, x)
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
    result = dict(torch=torch.__version__, validation_gemm='torch.nn.functional.scaled_mm',
                  accumulation=validate_accumulation(), nonfinite_comparisons=validate_nonfinite())
    root = Path(__file__).resolve().parents[1]
    result['source_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('fused_gemm_a2a.py', 'fused_gemm_a2a_kernels.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('MXFP8_LOCAL_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
