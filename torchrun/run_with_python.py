import torch
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",  # Use "gloo" for CPU
        rank=rank,
        world_size=world_size
    )

def cleanup():
    torch.distributed.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    # Set device for this process
    print(rank, '#', world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

def main():
    world_size = torch.cuda.device_count()  # Number of GPUs
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size
    )

if __name__ == "__main__":
    main()
