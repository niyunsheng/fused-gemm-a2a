"""Independent affine gradients and asynchronous reference scheduling (CPU)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from benchmark import reference


def local_a2a(x, scatter_dim, gather_dim, group, async_op=False):
    # CP1 identity transport isolates the projection math; GPU tests cover NCCL/layout.
    return (lambda: x) if async_op else x


class ReferenceTest(unittest.TestCase):
    def test_mxfp8_scale_boundary_and_tail(self):
        x = torch.zeros(3, 33, dtype=torch.bfloat16)
        x[0, 0], x[1, 0] = 448, 450
        x[:2, 1] = 2.0 ** -9
        x[:, 32] = 2.0 ** -133
        expected = x.float()
        expected[1, 0], expected[1, 1] = 448, 0
        # Crossing 448 doubles the scale: a minimum E4M3 subnormal now
        # rounds to zero. The final column belongs to its own padded group.
        torch.testing.assert_close(reference.quantize_mxfp8(x), expected, rtol=0, atol=0)

    def test_mxfp8_oracle_preserves_nonfinite_inputs(self):
        for value in (float('inf'), -float('inf'), float('nan')):
            x = torch.ones(2, 32, dtype=torch.bfloat16)
            x[0, 0] = value
            got = reference.quantize_mxfp8(x)
            self.assertFalse(torch.isfinite(got[0, 0]))
            torch.testing.assert_close(got[1], torch.ones(32), rtol=0, atol=0)

    def make_module(self, gate, bias, kv_heads=4):
        with patch.object(reference.dist, 'get_world_size', return_value=1):
            return reference.UlyssesQKVProjection(
                16, 4, kv_heads, bias=bias, gate=gate, device='cpu', dtype=torch.float64)

    def test_affine_outputs_and_all_gradients(self):
        for gate in (False, True):
            for bias in (False, True):
                for kv_heads in (2, 4):
                    with self.subTest(gate=gate, bias=bias, kv_heads=kv_heads):
                        torch.manual_seed(93)
                        module = self.make_module(gate, bias, kv_heads)
                        x = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
                        params = list(module.parameters())
                        oracle_x = x.detach().clone().requires_grad_()
                        oracle_params = [p.detach().clone().requires_grad_() for p in params]
                        pairs = iter(oracle_params)
                        expected = [F.linear(oracle_x, next(pairs), next(pairs) if bias else None)
                                    .view(7, -1, 4) for _ in module.projections]
                        with patch.object(reference, 'all_to_all_tensor', side_effect=local_a2a):
                            actual = module(x)
                            grads = [torch.randn_like(y) for y in expected]
                            got = torch.autograd.grad(actual, [x, *params], grads)
                        want = torch.autograd.grad(expected, [oracle_x, *oracle_params], grads)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
                        for a, b in zip(got, want):
                            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_gate_bias_can_be_frozen(self):
        module = self.make_module(True, True)
        module.gate_proj.bias.requires_grad_(False)
        x = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
        with patch.object(reference, 'all_to_all_tensor', side_effect=local_a2a):
            outputs = module(x)
            sum(t.sum() for t in outputs).backward()
        self.assertIsNone(module.gate_proj.bias.grad)
        self.assertIsNotNone(module.gate_proj.weight.grad)
        self.assertIsNotNone(x.grad)

    def test_gate_overlaps_v_transport_in_both_directions(self):
        module = self.make_module(True, False)
        x = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
        events = []
        names = {p.weight.data_ptr(): n for n, p in zip('qkvg', module.projections)}
        linear = F.linear
        backward = reference.linear_backward
        launches = []

        def affine(x, w, b=None):
            events.append('gemm_' + names[w.data_ptr()])
            return linear(x, w, b)

        def transport(x, scatter_dim, gather_dim, group, async_op=False):
            self.assertTrue(async_op)
            name = ('qkvg' if scatter_dim == 1 else 'gvkq')[len(launches)]
            launches.append(name)
            events.append('launch_' + name)

            def wait():
                events.append('wait_' + name)
                return x

            return wait

        def affine_backward(g, x, w, has_bias):
            events.append('grad_' + names[w.data_ptr()])
            return backward(g, x, w, has_bias)

        with patch.object(reference, 'all_to_all_tensor', side_effect=transport), \
                patch.object(reference.F, 'linear', side_effect=affine), \
                patch.object(reference, 'linear_backward', side_effect=affine_backward):
            outputs = module(x)
            self.assertLess(events.index('launch_v'), events.index('gemm_g'))
            self.assertLess(events.index('gemm_g'), events.index('wait_v'))
            self.assertLess(events.index('launch_g'), events.index('wait_q'))
            events.clear()
            launches.clear()
            torch.autograd.grad(outputs, [x, *module.parameters()], [torch.ones_like(y) for y in outputs])
            for current, following in zip('gvk', 'vkq'):
                self.assertLess(events.index('launch_' + following), events.index('grad_' + current))
                self.assertLess(events.index('grad_' + current), events.index('wait_' + following))



if __name__ == '__main__':
    unittest.main()
