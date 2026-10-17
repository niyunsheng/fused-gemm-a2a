"""Row bookkeeping for non-aligned local token counts; no GPU execution:
python3 -m unittest tests/test_row_blocks.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchmark.reference import interleave_qkv_rows
from fused_gemm_a2a_kernels import padded_rows


def pad_rows(x, capacity):
    """Sequence side [valid, ...] -> [capacity, ...] with zero rows appended; `x` itself when equal."""
    valid = x.shape[0]
    if valid == capacity:
        return x
    if valid > capacity:
        raise ValueError(f'{valid} rows exceed the capacity {capacity}')
    out = x.new_zeros((capacity,) + tuple(x.shape[1:]))
    out[:valid] = x
    return out


def pad_rank_blocks(x, peers, capacity):
    """Head side, rank-major rows: [peers * valid, ...] -> [peers * capacity, ...] with zero rows after
    each rank's valid rows; `x` itself when nothing is padded."""
    if x.shape[0] % peers:
        raise ValueError(f'{x.shape[0]} rows do not split over {peers} ranks')
    valid = x.shape[0] // peers
    if valid == capacity:
        return x
    if valid > capacity:
        raise ValueError(f'{valid} rows per rank exceed the capacity {capacity}')
    tail = tuple(x.shape[1:])
    out = x.new_zeros((peers, capacity) + tail)
    out[:, :valid] = x.view((peers, valid) + tail)
    return out.view((peers * capacity,) + tail)


def unpad_rank_blocks(x, peers, valid, *, copy=False):
    """Inverse of pad_rank_blocks: the first `valid` rows of each rank's block as one contiguous copy,
    which may alias x when no compaction is needed. copy=True always returns owned,
    contiguous storage, combining compaction and the ownership copy in one operation."""
    if x.shape[0] % peers:
        raise ValueError(f'{x.shape[0]} rows do not split over {peers} ranks')
    capacity = x.shape[0] // peers
    if valid == capacity and not copy:
        return x
    if valid > capacity:
        raise ValueError(f'{valid} valid rows per rank exceed the buffer rows {capacity}')
    tail = tuple(x.shape[1:])
    rows = x.view((peers, capacity) + tail)[:, :valid]
    if copy:
        rows = rows.clone(memory_format=torch.contiguous_format)
    return rows.reshape((peers * valid,) + tail)


class RowBlocksTest(unittest.TestCase):
    def test_capacity(self):
        self.assertEqual(padded_rows(7456, 256), 7680)  # 119296-token DiT pack at CP16, two-CTA MMA
        self.assertEqual(padded_rows(7456, 128), 7552)
        self.assertEqual(padded_rows(8192, 256), 8192)
        self.assertEqual(padded_rows(1, 256), 256)
        for args in ((0, 256), (7456, 0), (-1, 128)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                padded_rows(*args)

    def test_sequence_side_padding(self):
        x = torch.randn(37, 8)
        self.assertIs(pad_rows(x, 37), x)
        xp = pad_rows(x, 48)
        self.assertEqual(tuple(xp.shape), (48, 8))
        self.assertTrue(torch.equal(xp[:37], x))
        self.assertFalse(xp[37:].any())
        with self.assertRaises(ValueError):
            pad_rows(x, 36)

    def test_head_side_padding_roundtrip(self):
        peers, valid, capacity = 4, 37, 48
        x = torch.randn(peers * valid, 2, 8)
        self.assertIs(pad_rank_blocks(x, peers, valid), x)
        self.assertIs(unpad_rank_blocks(x, peers, valid), x)
        xp = pad_rank_blocks(x, peers, capacity)
        self.assertEqual(tuple(xp.shape), (peers * capacity, 2, 8))
        blocks = xp.view(peers, capacity, 2, 8)
        self.assertTrue(torch.equal(blocks[:, :valid].reshape_as(x), x))
        self.assertFalse(blocks[:, valid:].any())
        back = unpad_rank_blocks(xp, peers, valid)
        self.assertTrue(torch.equal(back, x))
        self.assertTrue(back.is_contiguous())
        self.assertNotEqual(back.data_ptr(), xp.data_ptr())  # the compaction is a copy, not an alias
        with self.assertRaises(ValueError):
            pad_rank_blocks(torch.randn(peers * valid + 1, 2), peers, capacity)
        with self.assertRaises(ValueError):
            pad_rank_blocks(x, peers, valid - 1)
        with self.assertRaises(ValueError):
            unpad_rank_blocks(xp, peers, capacity + 1)

    def test_owned_head_rows_survive_workspace_reuse(self):
        for peers in (1, 4):
            for valid, capacity in ((1, 16), (16, 16), (37, 48)):
                with self.subTest(peers=peers, valid=valid, capacity=capacity):
                    workspace = torch.randn(peers * capacity, 2, 8)
                    expected = workspace.view(peers, capacity, 2, 8)[:, :valid].clone()
                    owned = unpad_rank_blocks(workspace, peers, valid, copy=True)
                    self.assertTrue(owned.is_contiguous())
                    self.assertNotEqual(owned.untyped_storage().data_ptr(),
                                        workspace.untyped_storage().data_ptr())
                    workspace.zero_()
                    torch.testing.assert_close(owned, expected.reshape(peers * valid, 2, 8))

    def test_padded_flow_matches_reference(self):
        """CPU model of the fused data flow on padded buffers: QKV forward and backward and the output
        projection forward and backward reproduce the unpadded Ulysses reference on P ranks."""
        torch.manual_seed(0)
        P, m, tile, heads, D = 4, 37, 16, 8, 4
        H = heads * D
        C = H // P  # head-side columns per rank
        capacity = padded_rows(m, tile)
        self.assertGreater(capacity, m)
        dtype = torch.float64

        # --- QKV projection (one weight segment is enough for the row bookkeeping) ---
        Wq = torch.randn(H, H, dtype=dtype, requires_grad=True)
        xs = [torch.randn(m, H, dtype=dtype, requires_grad=True) for _ in range(P)]
        ys = [x @ Wq.t() for x in xs]
        q_ref = [torch.cat([y[:, d * C:(d + 1) * C] for y in ys]) for d in range(P)]  # rank-major rows
        heads_pad = [torch.zeros(P * capacity, C, dtype=dtype) for _ in range(P)]
        for r, x in enumerate(xs):
            yp = pad_rows(x.detach(), capacity) @ Wq.detach().t()  # whole tiles, including zero rows
            for d in range(P):
                heads_pad[d][r * capacity:(r + 1) * capacity] = yp[:, d * C:(d + 1) * C]
        for d in range(P):
            torch.testing.assert_close(unpad_rank_blocks(heads_pad[d], P, m), q_ref[d].detach())
        gq = [torch.randn_like(q) for q in q_ref]
        torch.autograd.backward([(q * g).sum() for q, g in zip(q_ref, gq)])
        dw_fused = torch.zeros_like(Wq)
        for r in range(P):
            recv = torch.zeros(capacity, H, dtype=dtype)  # dY of this rank's tokens, one K chunk per peer
            for d in range(P):
                recv[:, d * C:(d + 1) * C] = pad_rank_blocks(gq[d], P, capacity)[r * capacity:(r + 1) * capacity]
            self.assertFalse(recv[m:].any())
            torch.testing.assert_close((recv @ Wq.detach())[:m], xs[r].grad)
            dw_fused += recv[:m].t() @ xs[r].detach()
        torch.testing.assert_close(dw_fused, Wq.grad)

        # --- output projection ---
        Wo = torch.randn(H, H, dtype=dtype, requires_grad=True)
        os_ = [torch.randn(P * m, C, dtype=dtype, requires_grad=True) for _ in range(P)]
        O_ref = [torch.cat([o[r * m:(r + 1) * m] for o in os_], dim=1) for r in range(P)]
        y_ref = [O @ Wo.t() for O in O_ref]
        recvs, y_fused = [], []
        for r in range(P):
            recv = torch.zeros(capacity, H, dtype=dtype)
            for d in range(P):
                recv[:, d * C:(d + 1) * C] = pad_rank_blocks(os_[d].detach(), P, capacity)[r * capacity:(r + 1) * capacity]
            recvs.append(recv)
            y_fused.append((recv @ Wo.detach().t())[:m])
            torch.testing.assert_close(y_fused[r], y_ref[r].detach())
        gy = [torch.randn_like(y) for y in y_ref]
        torch.autograd.backward([(y * g).sum() for y, g in zip(y_ref, gy)])
        head_pad = [torch.zeros(P * capacity, C, dtype=dtype) for _ in range(P)]
        dwo_fused = torch.zeros_like(Wo)
        for r in range(P):
            dyp = pad_rows(gy[r], capacity)
            dO = dyp @ Wo.detach()  # [capacity, H], then token->head push
            for d in range(P):
                head_pad[d][r * capacity:(r + 1) * capacity] = dO[:, d * C:(d + 1) * C]
            dwo_fused += dyp[:m].t() @ recvs[r][:m]
        for d in range(P):
            torch.testing.assert_close(unpad_rank_blocks(head_pad[d], P, m), os_[d].grad)
        torch.testing.assert_close(dwo_fused, Wo.grad)

    def test_qkv_row_orders(self):
        heads, D, H = 4, 8, 32
        Wq, Wk, Wv, Wg = (torch.randn(H, 16) for _ in range(4))
        for weights in ([Wq, Wk, Wv], [Wq, Wk, Wv, Wg]):
            seg = torch.cat(weights)
            inter = interleave_qkv_rows(seg, heads, D)
            self.assertEqual(tuple(inter.shape), tuple(seg.shape))
            for h in range(heads):
                rows = slice(h * len(weights) * D, (h + 1) * len(weights) * D)
                torch.testing.assert_close(inter[rows], torch.cat([w[h * D:(h + 1) * D] for w in weights]))
        with self.assertRaises(ValueError):
            interleave_qkv_rows(torch.randn(5 * H, 16), heads, D)


if __name__ == '__main__':
    unittest.main()
