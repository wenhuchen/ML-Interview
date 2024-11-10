import torch
import torch.multiprocessing as mp
import os
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(local_rank, '#', world_size)
    print(torch.cuda.get_device_name())

if __name__ == "__main__":
    main()