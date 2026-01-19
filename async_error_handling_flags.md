# Async Error Handling Flags for PyTorch/ROCm

**Action Item**: Rao, Manoj - Share async error handling flags (`torch_sync_async_handling`)

This document covers environment variables and flags for debugging illegal memory access errors and other async GPU errors at scale.

---

## Problem Context

- Illegal memory access errors occurring ~1 in 8 jobs at 32N+ scale
- Enabling `AMD:LOGLVL=4` does not reproduce the issue (too heavyweight)
- Need lightweight async error detection that doesn't slow down production

---

## Recommended Environment Variables

### 1. PyTorch Async Error Handling

```bash
# Enable synchronous error checking in PyTorch
# This makes CUDA/HIP errors surface immediately rather than asynchronously
export TORCH_USE_CUDA_DSA=1

# Force synchronization after each kernel (HEAVY - use only for debugging)
# export CUDA_LAUNCH_BLOCKING=1  # NVIDIA
export HIP_LAUNCH_BLOCKING=1     # AMD/ROCm equivalent
```

### 2. RCCL/NCCL Async Error Handling

```bash
# Enable async error handling in NCCL/RCCL
export NCCL_ASYNC_ERROR_HANDLING=1

# Set timeout for collective operations (seconds)
# Helps detect hangs vs actual errors
export NCCL_TIMEOUT=1800  # 30 minutes, adjust based on workload

# For RCCL specifically on AMD
export RCCL_TIMEOUT=1800
```

### 3. PyTorch Distributed Error Handling

```bash
# Enable detailed distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Options: OFF, INFO, DETAIL

# Set timeout for distributed operations
export TORCH_DIST_INIT_BARRIER_TIMEOUT=1800

# Enable synchronous error checking for distributed ops
export TORCH_SHOW_CPP_STACKTRACES=1
```

### 4. HIP/ROCm Specific Flags

```bash
# Enable HIP error checking (lightweight)
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Explicit device list

# Memory debugging (moderate overhead)
export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2

# Enable kernel serialization for debugging (HEAVY)
# export HIP_LAUNCH_BLOCKING=1

# GPU memory pool settings (can help with fragmentation issues)
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
```

---

## Lightweight Production Configuration

For production jobs where you want to catch errors without significant overhead:

```bash
#!/bin/bash
# production_debug_env.sh - Lightweight error detection

# Async error handling (minimal overhead)
export NCCL_ASYNC_ERROR_HANDLING=1
export RCCL_TIMEOUT=3600

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

# Don't enable these in production (too slow):
# export HIP_LAUNCH_BLOCKING=1
# export CUDA_LAUNCH_BLOCKING=1
# export AMD_LOG_LEVEL=4
```

---

## Debug Configuration (When Reproducing Issues)

For dedicated debug runs when trying to reproduce the 1-in-8 failure:

```bash
#!/bin/bash
# debug_env.sh - Full debugging (will slow down execution)

# Force synchronous execution
export HIP_LAUNCH_BLOCKING=1

# Maximum error reporting
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# RCCL debugging
export RCCL_DEBUG=INFO
export RCCL_DEBUG_SUBSYS=ALL
export NCCL_ASYNC_ERROR_HANDLING=1

# Memory debugging
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256

# Log to file
export RCCL_DEBUG_FILE=/tmp/rccl_debug_%h_%p.log
```

---

## PyTorch Code-Level Error Handling

Add this to your training script for better error capture:

```python
import torch
import torch.distributed as dist
import os
import traceback

def setup_async_error_handling():
    """Configure PyTorch for better async error detection."""

    # Enable anomaly detection (catches NaN/Inf in autograd)
    torch.autograd.set_detect_anomaly(True)

    # Set default tensor type error checking
    if hasattr(torch.cuda, 'set_sync_debug_mode'):
        # Available in newer PyTorch versions
        torch.cuda.set_sync_debug_mode(1)  # 0=off, 1=warn, 2=error


def safe_collective_wrapper(collective_fn, *args, timeout=300, **kwargs):
    """
    Wrapper for collective operations with timeout and error handling.

    Usage:
        safe_collective_wrapper(dist.all_reduce, tensor, op=dist.ReduceOp.SUM)
    """
    import threading

    result = [None]
    error = [None]

    def run_collective():
        try:
            # Sync before collective
            torch.cuda.synchronize()
            collective_fn(*args, **kwargs)
            torch.cuda.synchronize()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=run_collective)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"Collective operation timed out after {timeout}s")

    if error[0] is not None:
        raise error[0]


def periodic_health_check(iteration: int, check_interval: int = 100):
    """
    Run periodic health checks during training.
    Call this in your training loop.
    """
    if iteration % check_interval != 0:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0

    try:
        # Check for CUDA/HIP errors
        torch.cuda.synchronize()

        # Check memory state
        mem_allocated = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()

        # Simple allreduce health check
        if dist.is_initialized():
            health_tensor = torch.ones(1, device='cuda')
            dist.all_reduce(health_tensor)
            expected = dist.get_world_size()
            if health_tensor.item() != expected:
                print(f"[Rank {rank}] WARNING: Health check allreduce mismatch: "
                      f"got {health_tensor.item()}, expected {expected}")

    except Exception as e:
        print(f"[Rank {rank}] ERROR in health check at iteration {iteration}: {e}")
        traceback.print_exc()
        raise
```

---

## Interpreting Errors

### Common Error Patterns

| Error Message | Likely Cause | Action |
|--------------|--------------|--------|
| `illegal memory access` | Buffer overrun, use-after-free | Enable `HIP_LAUNCH_BLOCKING`, check tensor shapes |
| `RCCL timeout` | Network issue, GPU hang, deadlock | Check XGMI links, review collective order |
| `device-side assert` | Kernel assertion failure | Enable debug build, check input validity |
| `out of memory` | Memory leak, fragmentation | Monitor memory growth, adjust batch size |

### Collecting Debug Info on Failure

```python
def on_training_failure(exception, iteration, rank):
    """Call this in your exception handler to collect debug info."""
    import subprocess
    import json

    debug_info = {
        "exception": str(exception),
        "iteration": iteration,
        "rank": rank,
        "timestamp": time.time(),
    }

    # GPU state
    try:
        result = subprocess.run(
            ["rocm-smi", "--json"],
            capture_output=True, text=True, timeout=10
        )
        debug_info["rocm_smi"] = json.loads(result.stdout)
    except:
        pass

    # Memory state
    debug_info["memory"] = {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }

    # Save debug info
    with open(f"/tmp/training_failure_rank{rank}_{iteration}.json", "w") as f:
        json.dump(debug_info, f, indent=2)

    print(f"Debug info saved to /tmp/training_failure_rank{rank}_{iteration}.json")
```

---

## Quick Reference

**Minimal production debugging** (low overhead):
```bash
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
```

**Full debugging** (high overhead, for repro):
```bash
export HIP_LAUNCH_BLOCKING=1
export RCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

*Contact: Manoj Rao - AMD AI Group*
