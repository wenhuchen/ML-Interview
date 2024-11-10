import torch
import os
import torch.distributed as dist
import socket
import sys

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    try:
        dist.init_process_group("nccl")
    except Exception as e:
        print(f"Error in setup_distributed: {str(e)}")
        sys.exit(1)

    torch.cuda.set_device(local_rank)

    hostname = socket.gethostname()
    print(f'{hostname}$Process {local_rank}:', local_rank, '#', world_size)
    print(f"{hostname}$Process {local_rank}: Default device is set to {torch.cuda.current_device()}")
    
    # dist.destroy_process_group()

if __name__ == "__main__":
    main()