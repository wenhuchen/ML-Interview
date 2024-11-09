import torch
import torch.multiprocessing as mp
import os

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(local_rank, '#', world_size)

if __name__ == "__main__":
    main()
