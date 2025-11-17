import torch
import os
import torch.distributed as dist
import socket
import sys
import datetime

def setup(rank, world_size):
    dist.init_process_group("cuda:nccl,cpu:gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(local_rank, world_size)

    mesh = torch.distributed.init_device_mesh("cuda", mesh_shape=[2, 1, 4], mesh_dim_names=("pp", "fsdp", "ep"))

    tensor = torch.tensor(local_rank % 4, device=torch.device("cuda", local_rank))

    group_ranks = dist.get_process_group_ranks(mesh.get_group("ep"))

    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=mesh.get_group("ep"))

    print(tensor)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
