import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import torch
from torch.autograd import Function
from typing import Tuple, Any
import sys

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


class CheckpointFunction(Function):
    @staticmethod
    def forward(ctx, run_fn, preserve_rng_state, *args):
        # Optionally stash the RNG state
        if preserve_rng_state:
            ctx.fwd_cpu_rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()

        # Save the function and inputs for backward
        ctx.run_fn = run_fn
        ctx.preserve_rng_state = preserve_rng_state
        ctx.args = args

        ctx.save_for_backward(*args)

        # Execute the forward pass without saving activations
        with torch.no_grad():
            output = run_fn(*args)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Restore the RNG state if needed
        if ctx.preserve_rng_state:
            torch.set_rng_state(ctx.fwd_cpu_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(ctx.fwd_cuda_rng_state)

        # Recompute the forward pass to get the activations
        detached_inputs = detach_variable(ctx.args)
        with torch.enable_grad():
            outputs = ctx.run_fn(*detached_inputs)

        # Compute the gradients using the recomputed activations
        torch.autograd.backward(outputs, grad_outputs)
        grads = tuple(arg.grad for arg in ctx.args if isinstance(arg, torch.Tensor))
        return (None, None) + grads


def checkpoint(run_fn, *args, preserve_rng_state=True):
    return CheckpointFunction.apply(run_fn, preserve_rng_state, *args)


class CheckpointedMLP(nn.Module):
    def __init__(self, checkpoint, input_dim, hidden_dim, output_dim):
        super(CheckpointedMLP, self).__init__()
        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.checkpoint = eval(checkpoint)
        assert isinstance(self.checkpoint, bool)

    def forward(self, x):
        # Define the forward function with checkpointing
        x = self._forward_input(x)
        if self.checkpoint:
            x = checkpoint(self._forward_fc, x)
            x = checkpoint(self._forward_fc, x)
            x = checkpoint(self._forward_fc, x)
            x = checkpoint(self._forward_fc, x)
        else:
            x = self._forward_fc(x)
            x = self._forward_fc(x)
            x = self._forward_fc(x)
            x = self._forward_fc(x)

        return x

    def _forward_input(self, x):
        return self.relu(self.dropout(self.fc1(x)))

    def _forward_fc(self, x):
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Define the model, loss function, and optimizer
    input_dim = 1000
    hidden_dim = 2560
    output_dim = 2560

    # Reset the peak memory statistics before starting measurement
    torch.cuda.reset_peak_memory_stats()

    model = CheckpointedMLP(sys.argv[1], input_dim, hidden_dim, output_dim)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy input and target
    x = torch.randn(64, input_dim).cuda()  # Batch size of 64
    target = torch.randint(0, output_dim, (64,)).cuda()

    # Forward pass with checkpointing
    output = model(x)
    loss = criterion(output, target)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory usage: {peak_memory / (1024 ** 2):.2f} MB with Checkpoint = {sys.argv[1]}")