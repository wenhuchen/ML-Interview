"""
Problem 20: Distributed Training with FSDP - Memory Profiling Demo

This script demonstrates how GPU memory consumption decreases as fsdp_size increases.

Usage:
  # fsdp_size=1 (no sharding within each replica group)
  torchrun --nnodes 1 --nproc-per-node 4 problem20.py --fsdp_size 1

  # fsdp_size=2 (sharding across 2 GPUs)
  torchrun --nnodes 1 --nproc-per-node 4 problem20.py --fsdp_size 2

  # fsdp_size=4 (sharding across all 4 GPUs)
  torchrun --nnodes 1 --nproc-per-node 4 problem20.py --fsdp_size 4
"""
import sys
import argparse
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import os
import torch
from problem5 import Transformer
from torch.distributed.fsdp import ShardingStrategy

def setup():
    """Initialize the distributed process group"""
    # Get rank and world size from environment
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", -1))


def get_rank():
    return int(os.environ.get("RANK", -1))


def cleanup():
    """Clean up the distributed process group"""
    dist.destroy_process_group()


def get_rank_specific_data(rank, batch_size, seq_len, vocab_size):
    """Generate different data for each rank by setting different random seeds"""
    # Set a different seed for each rank to ensure different data
    torch.manual_seed(rank + 42)
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    return data


def get_gpu_memory_stats(device):
    """Get GPU memory statistics in MB"""
    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024    # MB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
    return {
        'allocated': allocated,
        'reserved': reserved,
        'max_allocated': max_allocated
    }


def print_memory_stats(rank, stage, device):
    """Print GPU memory statistics for a given stage"""
    stats = get_gpu_memory_stats(device)
    print(f"[Rank {rank}] {stage:30s} | "
          f"Allocated: {stats['allocated']:8.2f} MB | "
          f"Reserved: {stats['reserved']:8.2f} MB | "
          f"Peak: {stats['max_allocated']:8.2f} MB")
    return stats


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FSDP Memory Profiling Demo')
    parser.add_argument('--fsdp_size', type=int, default=2, 
                       help='Number of GPUs to shard across (must divide world_size evenly)')
    args = parser.parse_args()

    # Initialize process group
    setup()
    rank = get_rank()
    world_size = get_world_size()

    # Validate fsdp_size
    if world_size % args.fsdp_size != 0:
        if rank == 0:
            print(f"ERROR: fsdp_size ({args.fsdp_size}) must divide world_size ({world_size}) evenly")
        cleanup()
        sys.exit(1)

    # Setup device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Print configuration
    if rank == 0:
        print("=" * 100)
        print("FSDP Memory Profiling - Configuration:")
        print(f"  World Size: {world_size}")
        print(f"  FSDP Size: {args.fsdp_size}")
        print(f"  DDP Replicas: {world_size // args.fsdp_size}")
        print("  Sharding Strategy: HYBRID_SHARD")
        print("=" * 100)
        print()

    # Stage 1: Initial state
    dist.barrier()
    print_memory_stats(rank, "Stage 1: Initial", device)

    # Stage 2: Create model (before FSDP)
    model = Transformer(
        vocab=3000, 
        dim=1024, 
        head=8, 
        layers=24, 
        max_length=20
    )

    dist.barrier()
    print_memory_stats(rank, "Stage 2: Model Created", device)

    # Stage 3: Create device mesh and wrap with FSDP
    device_mesh = init_device_mesh(
        "cuda", 
        mesh_shape=(world_size // args.fsdp_size, args.fsdp_size),
        mesh_dim_names=["ddp", "fsdp"]
    )

    if rank == 0:
        print(f"\nDevice mesh created: {device_mesh}\n")

    model = FSDP(
        model, 
        device_mesh=device_mesh,
        device_id=rank,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    )

    if rank == 0:
        print("--------------------------------")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Theoretical memory usage: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024} MB")
        print("--------------------------------")

    dist.barrier()
    print_memory_stats(rank, "Stage 3: FSDP Wrapped", device)

    # Stage 4: Create input data
    batch_size = 2
    seq_len = 18
    vocab_size = 3000
    input_data = get_rank_specific_data(rank, batch_size, seq_len, vocab_size).to(device)
    dist.barrier()
    print_memory_stats(rank, "Stage 4: Input Data Loaded", device)

    # Stage 5: Forward pass
    y = model(input_data)
    dist.barrier()
    print_memory_stats(rank, "Stage 5: After Forward Pass", device)

    # Stage 6: Compute loss
    loss = y.sum()
    dist.barrier()
    print_memory_stats(rank, "Stage 6: After Loss Compute", device)

    # Stage 7: Backward pass
    loss.backward()
    dist.barrier()
    print_memory_stats(rank, "Stage 7: After Backward Pass", device)

    # Gather peak memory from all ranks
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
    all_peak_memory = [torch.zeros(1, device=device) for _ in range(world_size)]

    dist.all_gather(all_peak_memory, torch.tensor([peak_memory], device=device))

    if rank == 0:
        print("\n" + "=" * 100)
        print(f"SUMMARY - FSDP Size: {args.fsdp_size}")
        print("=" * 100)
        for i, mem in enumerate(all_peak_memory):
            print(f"  Rank {i}: Peak Memory = {mem.item():.2f} MB")
        avg_peak = sum([m.item() for m in all_peak_memory]) / len(all_peak_memory)
        print(f"\n  Average Peak Memory: {avg_peak:.2f} MB")
        print("=" * 100)
        print("\nNote: Larger fsdp_size = More sharding = Lower memory per GPU")
        print("=" * 100)

    cleanup()
