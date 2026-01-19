"""
Sample Code: Collective and Compute Monitoring Hooks for PyTorch/ROCm Training
==============================================================================

This provides instrumentation hooks to monitor backend metrics during training jobs.
Use these to debug performance drops at scale.

Action Item: Rao, Manoj - Share sample code for Fremont training job instrumentation
"""

import os
import time
import torch
import torch.distributed as dist
from contextlib import contextmanager
from typing import Optional, Dict, Any
import json
from datetime import datetime

# =============================================================================
# Environment Setup
# =============================================================================

def setup_monitoring_env():
    """Set environment variables for enhanced monitoring."""
    # Enable RCCL debug info (use sparingly - can impact performance)
    # os.environ["RCCL_DEBUG"] = "INFO"  # OPTIONS: WARN, INFO, TRACE
    # os.environ["RCCL_DEBUG_SUBSYS"] = "ALL"  # OPTIONS: INIT, COLL, P2P, SHM, NET, ALL

    # Enable HIP error tracking
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "")

    # Lightweight logging that doesn't impact repro rates
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # OPTIONS: OFF, INFO, DETAIL


# =============================================================================
# Collective Operation Monitoring
# =============================================================================

class CollectiveMonitor:
    """
    Lightweight monitoring for RCCL collective operations.
    Tracks latency and throughput without significantly impacting performance.
    """

    def __init__(self, rank: int, world_size: int, log_interval: int = 100):
        self.rank = rank
        self.world_size = world_size
        self.log_interval = log_interval
        self.metrics: Dict[str, list] = {
            "allreduce": [],
            "allgather": [],
            "reduce_scatter": [],
            "broadcast": [],
        }
        self.call_count = 0
        self.start_time = time.time()

    def _log_metrics(self, op_name: str, duration_ms: float, size_bytes: int):
        """Store metrics for later analysis."""
        self.metrics[op_name].append({
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "size_bytes": size_bytes,
            "rank": self.rank,
        })

        self.call_count += 1
        if self.call_count % self.log_interval == 0:
            self._print_summary()

    def _print_summary(self):
        """Print periodic summary of collective performance."""
        if self.rank != 0:
            return

        elapsed = time.time() - self.start_time
        print(f"\n[CollectiveMonitor] Summary after {self.call_count} operations ({elapsed:.1f}s)")

        for op_name, measurements in self.metrics.items():
            if not measurements:
                continue
            durations = [m["duration_ms"] for m in measurements[-self.log_interval:]]
            avg = sum(durations) / len(durations)
            max_d = max(durations)
            min_d = min(durations)
            print(f"  {op_name}: avg={avg:.2f}ms, min={min_d:.2f}ms, max={max_d:.2f}ms")
        print()

    @contextmanager
    def monitor_allreduce(self, tensor: torch.Tensor):
        """Context manager to monitor allreduce operations."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - start) * 1000
        self._log_metrics("allreduce", duration_ms, tensor.numel() * tensor.element_size())

    @contextmanager
    def monitor_allgather(self, tensor: torch.Tensor):
        """Context manager to monitor allgather operations."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - start) * 1000
        self._log_metrics("allgather", duration_ms, tensor.numel() * tensor.element_size())

    def export_metrics(self, filepath: str):
        """Export collected metrics to JSON file."""
        output = {
            "rank": self.rank,
            "world_size": self.world_size,
            "total_calls": self.call_count,
            "duration_seconds": time.time() - self.start_time,
            "metrics": self.metrics,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)


# =============================================================================
# Compute Monitoring Hooks
# =============================================================================

class ComputeMonitor:
    """
    Monitor GPU compute operations and memory usage.
    Helps identify performance degradation over time.
    """

    def __init__(self, device_id: int, log_interval: int = 100):
        self.device_id = device_id
        self.log_interval = log_interval
        self.iteration = 0
        self.metrics = []

    def record_iteration(self, loss: Optional[float] = None, extra: Optional[Dict] = None):
        """Record metrics for current iteration."""
        self.iteration += 1

        # Collect GPU metrics
        mem_allocated = torch.cuda.memory_allocated(self.device_id)
        mem_reserved = torch.cuda.memory_reserved(self.device_id)
        max_mem = torch.cuda.max_memory_allocated(self.device_id)

        record = {
            "iteration": self.iteration,
            "timestamp": time.time(),
            "memory_allocated_gb": mem_allocated / (1024**3),
            "memory_reserved_gb": mem_reserved / (1024**3),
            "max_memory_gb": max_mem / (1024**3),
            "loss": loss,
        }
        if extra:
            record.update(extra)

        self.metrics.append(record)

        # Periodic logging
        if self.iteration % self.log_interval == 0:
            self._print_status(record)

    def _print_status(self, record: Dict):
        """Print current status."""
        print(f"[Iter {record['iteration']}] "
              f"Mem: {record['memory_allocated_gb']:.2f}GB / {record['memory_reserved_gb']:.2f}GB reserved, "
              f"Peak: {record['max_memory_gb']:.2f}GB"
              + (f", Loss: {record['loss']:.4f}" if record['loss'] else ""))

    def check_memory_growth(self, threshold_gb: float = 1.0) -> bool:
        """
        Check for unexpected memory growth over time.
        Returns True if memory growth exceeds threshold.
        """
        if len(self.metrics) < 100:
            return False

        early_mem = sum(m["memory_allocated_gb"] for m in self.metrics[:10]) / 10
        recent_mem = sum(m["memory_allocated_gb"] for m in self.metrics[-10:]) / 10

        growth = recent_mem - early_mem
        if growth > threshold_gb:
            print(f"[WARNING] Memory growth detected: {growth:.2f}GB increase")
            return True
        return False

    def export_metrics(self, filepath: str):
        """Export metrics to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)


# =============================================================================
# Example Usage in Training Loop
# =============================================================================

def example_training_loop():
    """
    Example showing how to integrate monitoring into a training loop.
    """
    # Initialize distributed
    dist.init_process_group(backend="nccl")  # RCCL on AMD
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    # Initialize monitors
    collective_monitor = CollectiveMonitor(rank, world_size, log_interval=100)
    compute_monitor = ComputeMonitor(local_rank, log_interval=100)

    # Dummy model and optimizer for example
    model = torch.nn.Linear(1024, 1024).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for iteration in range(10000):
        # Forward pass
        x = torch.randn(32, 1024, device="cuda")
        y = model(x)
        loss = y.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Monitored allreduce for gradient sync (DDP does this internally, shown for illustration)
        # In practice, you'd hook into DDP's communication

        optimizer.step()

        # Record compute metrics
        compute_monitor.record_iteration(loss=loss.item())

        # Periodic memory growth check
        if iteration % 1000 == 0 and iteration > 0:
            if compute_monitor.check_memory_growth():
                print(f"[Rank {rank}] Potential memory leak detected at iteration {iteration}")

    # Export metrics at end of training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collective_monitor.export_metrics(f"collective_metrics_rank{rank}_{timestamp}.json")
    compute_monitor.export_metrics(f"compute_metrics_rank{rank}_{timestamp}.json")

    dist.destroy_process_group()


if __name__ == "__main__":
    setup_monitoring_env()
    example_training_loop()
