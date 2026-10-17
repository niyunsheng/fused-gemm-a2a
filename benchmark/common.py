"""Shared distributed validation, timing and isolated profiling for benchmarks."""
from contextlib import contextmanager
from datetime import timedelta
import math
import os
from statistics import median

import torch
import torch.distributed as dist
from torch.profiler import record_function


class BenchmarkRuntime:
    """Own benchmark state without importing CuTe or a particular projection."""

    def __init__(self, args, parser, *, collect_failures=False):
        if int(os.environ.get('WORLD_SIZE', 0)) != args.cp:
            parser.error('torchrun world size must equal --cp')
        if args.tokens % args.cp:
            parser.error('--tokens must be divisible by --cp')
        if args.graph_batch <= 0:
            parser.error('--graph-batch must be positive')
        if args.iterations <= 0 or args.trials <= 0 or args.warmup < 0:
            parser.error('iterations/trials must be positive and warmup nonnegative')
        self.args = args
        self.device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        dist.init_process_group('nccl', device_id=self.device, timeout=timedelta(minutes=5))
        self.capture_stream = None
        if args.cuda_graphs:
            # Backward uses its forward's stream: select it before any forwards.
            self.capture_stream = torch.cuda.Stream()
            torch.cuda.set_stream(self.capture_stream)
        self.rank = dist.get_rank()
        self.ref_dtype, self.tolerance = torch.float32, 0.02
        self.errors, self.failures = {}, []
        self.timing_samples = {}
        self.collect_failures = collect_failures

    def log(self, *args):
        if self.rank == 0:
            print(*args, flush=True)

    def gather(self, tensor, dim):
        parts = [torch.empty_like(tensor) for _ in range(self.args.cp)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.cat(parts, dim=dim).to(self.ref_dtype)

    def check(self, actual, expected, label):
        actual, expected = actual.detach().to(self.ref_dtype), expected.detach().to(self.ref_dtype)
        scale = expected.abs().max().clamp_min(1e-6)
        same_shape = actual.shape == expected.shape
        error = (actual - expected).abs().max() / scale if same_shape else torch.tensor(math.inf, device=self.device)
        ok = torch.tensor(float(same_shape and torch.allclose(
            actual, expected, atol=self.tolerance * scale.item(), rtol=self.tolerance)), device=self.device)
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        self.errors[label] = error.item()
        if not ok.item():
            if self.collect_failures and os.environ.get('CHECK_CONTINUE'):
                self.failures.append(label)
                self.log(f'MISMATCH {label}: max relative error {self.errors[label]:.3g}')
            else:
                raise RuntimeError(f'{label} mismatch: max relative error {self.errors[label]:.3g}')

    @staticmethod
    def reduced(grad):
        grad = grad.detach().clone()
        dist.all_reduce(grad)
        return grad

    def measure(self, fn, use_graph=False, *, name):
        a, device, capture_stream = self.args, self.device, self.capture_stream
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        iterations = count = a.iterations
        if use_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                for _ in range(a.graph_batch):
                    fn()
            dist.barrier()  # Every rank has finished capture before any rank starts replay.
            torch.cuda.synchronize()
            for _ in range(max(1, math.ceil(a.warmup / a.graph_batch))):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            fn = graph.replay
            iterations = math.ceil(a.iterations / a.graph_batch)
            count = iterations * a.graph_batch
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        samples = []
        for _ in range(a.trials):
            dist.barrier()
            torch.cuda.synchronize()
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end) * 1000 / count, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
        self.timing_samples[name] = samples
        return median(samples)


def _profile_barrier(name):
    # Drain every local stream before aligning ranks, then wait for NCCL too.
    # Keep harness synchronization outside the operators' own profiler ranges.
    with record_function(f'benchmark_sync/{name}'):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()


@contextmanager
def profile_stage(name):
    """All ranks must enter stages in the same order; use outside CUDA capture."""
    _profile_barrier(f'{name}/before')
    yield
    _profile_barrier(f'{name}/after')


def capture_profile_stage(fn, name, stream, check):
    """Capture one operator invocation before profiling and validate its replay.

    Backward callables retain their original autograd graph, as in the timing
    loop. Keep captured outputs alive along with the graph's private pool.
    """
    for _ in range(3):
        expected = fn()
    _profile_barrier(f'capture/{name}/before')
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            outputs = fn()
        _profile_barrier(f'capture/{name}/after')
        for _ in range(3):
            graph.replay()
        _profile_barrier(f'capture/{name}/replayed')
        actual_tensors = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
        expected_tensors = (expected,) if isinstance(expected, torch.Tensor) else expected
        if len(actual_tensors) != len(expected_tensors):
            raise RuntimeError(f'{name}: captured output count changed')
        for index, (actual, wanted) in enumerate(zip(actual_tensors, expected_tensors)):
            check(actual, wanted, f'profile graph {name} output {index}')
    except BaseException:
        # Tracebacks retain locals: release NCCL graph resources before PG cleanup.
        graph.reset()
        raise

    def replay():
        # Python/autograd annotations inside capture do not run during replay.
        with record_function(f'{name}_cg'):
            graph.replay()
        return outputs

    return replay
