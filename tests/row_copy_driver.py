"""Exact CUDA row-copy checks against independent PyTorch slicing and padding."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fused_gemm_a2a_kernels import resize_rank_rows


def main():
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    checks = 0
    for peers, width in ((1, 128), (8, 384), (8, 512), (4, 640), (4, 1024), (16, 256)):
        for rows in (1, 127, 255, 256, 257, 7103, 14206):
            capacity = ((rows + 255) // 256) * 256
            for segments in (1, 3, 4):
                sources = tuple(torch.randn(peers * rows, width, device='cuda', dtype=torch.bfloat16)
                                for _ in range(segments))
                padded = resize_rank_rows(sources, peers, capacity)
                for actual, source in zip(padded, sources):
                    expected = torch.zeros(peers, capacity, width, device='cuda', dtype=source.dtype)
                    expected[:, :rows] = source.view(peers, rows, width)
                    torch.testing.assert_close(actual, expected.flatten(0, 1), rtol=0, atol=0)
                outputs = resize_rank_rows(padded, peers, rows, copy=True)
                for actual, source, workspace in zip(outputs, sources, padded):
                    assert actual.data_ptr() != workspace.data_ptr()
                    assert actual.is_contiguous()
                    torch.testing.assert_close(actual, source, rtol=0, atol=0)
                expected = tuple(t.clone() for t in outputs)
                for workspace in padded:
                    workspace.zero_()
                for actual, wanted in zip(outputs, expected):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                checks += segments
                if segments > 1:
                    # Independent head/segment view catches rank and interleave mistakes.
                    sources = tuple(torch.randn_like(t) for t in sources)
                    bias = torch.randn(peers * width // 128, segments, 128,
                                       device='cuda', dtype=sources[0].dtype)
                    padded_sources = resize_rank_rows(sources, peers, capacity)
                    for head_offset in sorted({0, (peers - 1) * width // 128}):
                        local_bias = bias[head_offset:head_offset + width // 128]
                        for inputs, output_rows in ((sources, rows), (sources, capacity), (padded_sources, rows)):
                            biased = resize_rank_rows(
                                inputs, peers, output_rows, copy=True, bias=bias.flatten(),
                                head_dim=128, head_offset=head_offset)
                            for i, (actual, source) in enumerate(zip(biased, sources)):
                                expected = torch.zeros(peers, output_rows, width,
                                                       device='cuda', dtype=source.dtype)
                                expected[:, :rows] = (source.view(peers, rows, width)
                                                     + local_bias[:, i].reshape(1, 1, width))
                                assert actual.data_ptr() != source.data_ptr()
                                torch.testing.assert_close(actual, expected.flatten(0, 1), rtol=0, atol=0)
                                checks += 1
    if int(os.environ.get('RANK', 0)) == 0:
        print(f'Row copies: {checks} exact pad/unpad/bias/ownership checks passed', flush=True)


if __name__ == '__main__':
    main()
