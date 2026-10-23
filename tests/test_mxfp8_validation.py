"""Reject malformed main_grad buffers before entering a raw-pointer GEMM."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import fused_gemm_a2a as fused


class MXFP8MainGradTest(unittest.TestCase):
    def setUp(self):
        self.weight = torch.nn.Parameter(torch.empty(384, 128, dtype=torch.bfloat16))
        self.dy = torch.empty(33, 384, dtype=torch.bfloat16)
        self.x = torch.empty(33, 128, dtype=torch.bfloat16)

    def reject(self, buffer):
        self.weight.main_grad = buffer
        with patch.object(fused, 'mxfp8_wgrad') as gemm:
            with self.assertRaisesRegex(ValueError, 'main_grad'):
                fused._weight_grad(self.weight, self.dy, self.x, 'mxfp8')
            gemm.assert_not_called()

    def test_wrong_shape_cannot_silently_truncate_wgrad(self):
        for shape in ((128, 128), (384, 256), (384 * 128,)):
            with self.subTest(shape=shape):
                self.reject(torch.empty(shape, dtype=torch.float32))

    def test_wrong_device_is_rejected(self):
        self.reject(torch.empty(self.weight.shape, device='meta', dtype=torch.float32))

    def test_misaligned_contiguous_view_is_rejected(self):
        buffer = torch.empty(self.weight.numel() + 1, dtype=torch.float32)[1:].view(self.weight.shape)
        self.assertTrue(buffer.is_contiguous())
        self.assertNotEqual(buffer.data_ptr() % 16, 0)
        self.reject(buffer)

    def test_dtype_and_strides_are_checked(self):
        self.reject(torch.empty_like(self.weight))
        self.reject(torch.empty(128, 384, dtype=torch.float32).T)


if __name__ == '__main__':
    unittest.main()
