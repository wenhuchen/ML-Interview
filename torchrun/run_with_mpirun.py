import torch
import os
import socket
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(320, 320, 3, 1) for _ in range(layers)])

def main():
    torch.cuda.reset_peak_memory_stats()

    world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    hostname = socket.gethostname()
    print(f'{hostname}$Process {local_rank}:', local_rank, '#', world_rank, '#', world_size)
    print(f"{hostname}$Process {local_rank}: Default device is set to {torch.cuda.current_device()}")
    net = Net(local_rank + 1)
    net.to('cuda')

    peak_memory = torch.cuda.max_memory_allocated()
    print(f"{hostname}$Process {local_rank}: Peak GPU memory usage: {peak_memory / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    main()