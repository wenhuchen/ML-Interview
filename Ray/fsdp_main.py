import os
from torch import nn
import torch
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group
import argparse

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--backend', type=str, default='nccl')
args = parser.parse_args()

# Initialize distributed process group
init_process_group(rank=args.rank, world_size=args.world_size, backend=args.backend)

# Set device
device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
print('rank', args.rank, 'device', device)

class Net(nn.Module):

    def __init__(self, num_layers, hidden_size, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.loss_fn = nn.CrossEntropyLoss()

        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.Dropout(0.2))
            self.layers.append(nn.ReLU())

        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.fc_out(x)

    def loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

net = Net(num_layers=5, hidden_size=28*28, num_classes=10)

# Move model to device before wrapping with FSDP
net = net.to(device)
net = FSDP(net, device_id=device if torch.cuda.is_available() else None)
net.train()

# Download and prepare data
data = datasets.MNIST(root='data', download=True, transform=transforms.ToTensor())

train, test = torch.utils.data.random_split(data, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True, num_workers=2)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
net.train()
for epoch in range(1):  # Add epochs for better training
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        y_pred = net.forward(x.view(x.shape[0], -1))
        loss = net.loss(y_pred, y)

        # Reduce loss across all processes
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss = loss / args.world_size  # Average the loss across processes

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            print('rank', args.rank, 'step', batch_idx, 'loss', loss.item())
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')

# Clean up distributed process group
torch.distributed.destroy_process_group()