import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_training(rank, world_size):
    setup(rank, world_size)

    # Create model and split it across two GPUs
    model = TwoLayerMLP(784, 128, 10)
    if rank == 0:
        model.layer1.to(rank)
    elif rank == 1:
        model.layer2.to(rank)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(rank)
    params = model.layer1.parameters() if rank == 0 else model.layer2.parameters()
    optimizer = optim.SGD(params, lr=0.1)

    # Simulating a dataset
    input_tensor = torch.randn(64, 784).to(0)  # Input always starts on GPU 0
    target = torch.randint(0, 10, (64,)).to(1)  # Target always on GPU 1

    # Training loop
    for epoch in range(30):
        optimizer.zero_grad()

        # Forward pass
        if rank == 0:
            output = model.layer1(input_tensor)
            dist.send(output, dst=1)
        elif rank == 1:
            output = torch.zeros(64, 128, requires_grad=True).to(rank)
            dist.recv(output, src=0)
            output.retain_grad()
            logits = model.layer2(output)

        # Backward pass
        if rank == 1:
            loss = loss_fn(logits, target)
            loss.backward()
            dist.send(output.grad, dst=0)
        elif rank == 0:
            output_derivative = torch.zeros_like(output).to(rank)
            dist.recv(output_derivative, src=1)
            output.backward(output_derivative)

        optimizer.step()

        if rank == 1:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)