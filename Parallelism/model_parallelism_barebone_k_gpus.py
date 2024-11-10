import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import math

class ShardedLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, with_relu):
        super(ShardedLinear, self).__init__()
        self.world_size = world_size
        self.in_features = in_features
        self.out_features = out_features
        self.with_relu = with_relu

        # Shard the weights and biases
        self.weight = nn.Parameter(torch.empty(out_features // world_size, in_features).cuda())
        self.bias = nn.Parameter(torch.empty(out_features // world_size).cuda())

        # Initialize the parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Perform the linear operation on the shard
        partial_output = torch.matmul(x, self.weight.t()) + self.bias
        if self.with_relu:
            partial_output = torch.relu(partial_output)

        # Gather the partial results from all GPUs
        gathered_output = [torch.zeros_like(partial_output) for _ in range(self.world_size)]
        dist.all_gather(gathered_output, partial_output)

        # partial_output.requires_grad = True
        # partial_output.retain_grad()

        output = torch.concat(gathered_output, dim=1)
        output.requires_grad = True
        output.retain_grad()

        return partial_output, output


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, world_size):
        super(TwoLayerMLP, self).__init__()
        self.world_size = world_size
        self.layer1 = ShardedLinear(input_size, hidden_size, world_size, with_relu=True)
        self.layer2 = ShardedLinear(hidden_size, output_size, world_size, with_relu=False)

    def forward(self, x):
        outputs = []
        outputs.extend(self.layer1(x))
        outputs.extend(self.layer2(outputs[-1]))
        return outputs


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12350'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_training(rank, world_size):
    setup(rank, world_size)

    # Create model
    model = TwoLayerMLP(784, 128, 16, world_size).to('cuda')

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Simulating a dataset (input starts on GPU 0)
    if rank == 0:
        input_tensor = torch.randn(64, 784).to('cuda')
    else:
        input_tensor = torch.zeros(64, 784).to('cuda')
    dist.broadcast(input_tensor, 0)

    if rank == 0:
        target = torch.randint(0, 16, (64,)).to('cuda')
    else:
        target = torch.zeros(64, dtype=torch.long).to('cuda')
    dist.broadcast(target, 0)


    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_tensor)

        # Loss and Backward pass
        loss = loss_fn(outputs[-1], target)
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss.backward()
        # dist.broadcast(outputs[-1].grad, world_size - 1)

        dist.barrier()
        for index in range(len(outputs) - 1, -1, -2):
            if index == len(outputs) - 1:
                o_partial, o_all = outputs[index - 1:index + 1]
                all_grad = torch.chunk(o_all.grad, world_size, 1)
                o_partial.backward(all_grad[rank])
            else:
                o_partial, o_all = outputs[index - 1:index + 1]
                dist.all_reduce(o_all.grad)
                all_grad = torch.chunk(o_all.grad, world_size, 1)
                o_partial.backward(all_grad[rank])
            dist.barrier()

        # Broadcast loss gradients back to all processes
        for param in model.parameters():
            dist.all_reduce(param.grad)

        optimizer.step()

    cleanup()

if __name__ == "__main__":
    world_size = 4  # Adjust this to any number of GPUs available
    mp.spawn(
        run_training, 
        args=(world_size,), 
        nprocs=world_size, 
        join=True)
