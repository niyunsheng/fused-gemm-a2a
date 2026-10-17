"""Distributed validation must reject errors on any participating rank."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from benchmark.common import BenchmarkRuntime


class BenchmarkValidationTest(unittest.TestCase):
    def setUp(self):
        self.runtime = BenchmarkRuntime.__new__(BenchmarkRuntime)
        self.runtime.device = torch.device('cpu')
        self.runtime.rank = 1
        self.runtime.ref_dtype = torch.float32
        self.runtime.tolerance = 0.02
        self.runtime.errors, self.runtime.failures = {}, []
        self.runtime.collect_failures = False

    def test_remote_failure_rejects_locally_matching_tensors(self):
        def remote_result(tensor, op):
            if op == dist.ReduceOp.MIN:
                tensor.zero_()
        with patch.object(dist, 'all_reduce', side_effect=remote_result):
            with self.assertRaisesRegex(RuntimeError, 'remote mismatch'):
                self.runtime.check(torch.ones(2), torch.ones(2), 'remote')

    def test_shape_mismatch_does_not_broadcast(self):
        with patch.object(dist, 'all_reduce') as reduce:
            with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
                self.runtime.check(torch.ones(2, 1), torch.ones(2), 'shape')
        self.assertEqual(reduce.call_count, 2)
        self.assertEqual(self.runtime.errors['shape'], float('inf'))

    def test_continue_is_opt_in_and_records_failure(self):
        with patch.dict(os.environ, {'CHECK_CONTINUE': '1'}), patch.object(dist, 'all_reduce'):
            with self.assertRaises(RuntimeError):
                self.runtime.check(torch.zeros(2), torch.ones(2), 'strict')
            self.runtime.collect_failures = True
            self.runtime.check(torch.zeros(2), torch.ones(2), 'collected')
        self.assertEqual(self.runtime.failures, ['collected'])

    def test_invalid_graph_batch_rejected_before_gpu_setup(self):
        class Parser(argparse.ArgumentParser):
            def error(self, message):
                raise ValueError(message)
        with patch.dict(os.environ, {'WORLD_SIZE': '4'}), patch.object(torch.cuda, 'set_device') as setup:
            for batch in (0, -1):
                args = SimpleNamespace(cp=4, tokens=16, graph_batch=batch)
                with self.assertRaisesRegex(ValueError, 'graph-batch must be positive'):
                    BenchmarkRuntime(args, Parser())
        setup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
