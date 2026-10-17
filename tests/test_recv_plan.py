"""Geometry invariants of the fused kernels; no GPU execution: python3 -m unittest tests/test_recv_plan.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fused_gemm_a2a_kernels import CTA_M, push_split, tile_swizzle, validate_peer_count, validate_tile_n  # noqa: E402


class RecvPlanTest(unittest.TestCase):
    def test_device_barrier_peer_limit(self):
        for peers in (1, 4, 8, 16, 32):
            validate_peer_count(peers)
        for peers in (0, -1, 33, 64):
            with self.assertRaisesRegex(ValueError, 'single-warp'):
                validate_peer_count(peers)



    def test_interleaved_recv_does_not_constrain_output_tile_to_head_width(self):
        validate_tile_n('recv', 256, 4096, 16, num_segments=4, head_dim=128)
        validate_tile_n('push', 128, 16384, 16, 4096, 4, 128)
        with self.assertRaisesRegex(ValueError, 'one N tile per head'):
            validate_tile_n('push', 256, 16384, 16, 4096, 4, 128)

    def test_output_push_cannot_cross_peer_boundary(self):
        validate_tile_n('push', 256, 4096, 16, 4096)
        with self.assertRaisesRegex(ValueError, r'CP \* tile N'):
            validate_tile_n('push', 256, 4096, 32, 4096)
        validate_tile_n('push', 128, 4096, 32, 4096)

    def test_output_tiles_must_cover_exact_width(self):
        for width in (4095, 4097):
            with self.assertRaises(ValueError):
                validate_tile_n('recv', 256, width, 16)
        for tile in (0, -128):
            with self.assertRaises(ValueError):
                validate_tile_n('recv', tile, 4096, 16)

    def test_swizzle_never_creates_padding_tiles(self):
        for clusters in range(1, 129):
            for requested in (1, 2, 3, 4, 8, 16):
                swizzle = tile_swizzle(clusters, requested)
                self.assertEqual(-(-clusters // swizzle) * swizzle, clusters)
                self.assertLessEqual(swizzle, requested)
        self.assertEqual(tile_swizzle(32, 8), 8)  # the benchmark shape keeps the full group
        self.assertEqual(tile_swizzle(4, 8), 4)   # 1024 tokens per rank with the two-CTA MMA
        self.assertEqual(tile_swizzle(6, 8), 2)
        for clusters, requested in ((0, 8), (4, 0), (4, -1)):
            with self.assertRaises(ValueError):
                tile_swizzle(clusters, requested)

    def test_split_covers_every_row(self):
        for width in range(64, 8193, 64):  # per-peer segment widths, including H=6144 at CP4 (1536)
            for segments in (1, 3):
                split = push_split(width * 4 * segments, 4, segments)
                rows = CTA_M // split
                self.assertEqual(rows * split, CTA_M, width)
                self.assertEqual(rows % 4, 0, width)
        self.assertEqual(push_split(4096, 16, 1), 1)      # 64 KB jobs at CP16
        self.assertEqual(push_split(12288, 16, 3), 1)
        self.assertEqual(push_split(12288, 4, 3), 4)      # 1536 columns: 384 KB blocks -> 96 KB jobs
        self.assertEqual(push_split(4096, 16, 1, requested=8), 8)

    def test_invalid_geometry_is_rejected(self):
        for requested in (-1, 3, 6, 17, 64, 129):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                push_split(4096, 16, 1, requested)
        for args in ((4096, 0, 1), (4000, 16, 1), (32 * 16, 16, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                push_split(*args)


if __name__ == '__main__':
    unittest.main()
