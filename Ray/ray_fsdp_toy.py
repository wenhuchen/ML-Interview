import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group
import argparse
import ray
import json

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

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

@ray.remote(num_cpus=2, num_gpus=1 if torch.cuda.is_available() else 0)
def train_worker(rank, world_size, config):
    """Ray remote function that runs PyTorch FSDP training."""
    
    # Set rank and world size for this worker
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    print(f'Starting worker {rank}/{world_size}')
    
    # Initialize distributed process group
    init_process_group(rank=rank, world_size=world_size, backend='nccl')
    
    # Set device
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    print(f'Worker {rank} using device: {device}')
    
    # Create model
    net = Net(
        num_layers=config.get("num_layers", 5),
        hidden_size=config.get("hidden_size", 28*28),
        num_classes=config.get("num_classes", 10)
    )
    
    # Move model to device before wrapping with FSDP
    net = net.to(device)
    net = FSDP(net, device_id=device if torch.cuda.is_available() else None)
    net.train()
    
    # Download and prepare data
    data = datasets.MNIST(root='data', download=True, transform=transforms.ToTensor())
    train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=config.get("batch_size", 100), 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=config.get("batch_size", 100), 
        shuffle=True, 
        num_workers=2
    )
    
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get("lr", 0.001))
    
    # Training loop
    net.train()
    for epoch in range(config.get("num_epochs", 1)):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            y_pred = net.forward(x.view(x.shape[0], -1))
            loss = net.loss(y_pred, y)
            
            # Reduce loss across all processes
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / world_size  # Average the loss across processes
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                print(f'Worker {rank}, step {batch_idx}, loss {loss.item()}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Worker {rank}, Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
    
    # Clean up distributed process group
    torch.distributed.destroy_process_group()
    
    return {
        "rank": rank,
        "final_loss": avg_loss,
        "epochs_completed": config.get("num_epochs", 1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2, help='Number of Ray workers')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=28*28, help='Hidden size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    config = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "num_classes": 10
    }
    
    print(f"Starting PyTorch-only FSDP training with {args.num_workers} workers")
    print(f"Configuration: {config}")
    
    # Initialize Ray
    import ray
    try:
        # First, try to shut down any existing Ray instance
        if ray.is_initialized():
            ray.shutdown()
            print("Shut down existing Ray instance")
        
        # Force Ray to start fresh locally, ignoring existing cluster
        ray.init(
            ignore_reinit_error=True,
            local_mode=False,
            _temp_dir="/tmp/ray_new_session",
            _node_ip_address="127.0.0.1"
        )
        print("Ray initialized successfully in local mode")
        
    except Exception as e:
        print(f"Ray initialization failed: {e}")
        print("Trying alternative initialization...")
        
        # Try alternative approach
        try:
            ray.init(
                ignore_reinit_error=True,
                local_mode=True,  # Force local mode
                _temp_dir="/tmp/ray_local"
            )
            print("Ray initialized in local mode")
        except Exception as e2:
            print(f"Alternative Ray initialization also failed: {e2}")
            print("Please check if there are any existing Ray processes running")
            return
    
    # Launch training workers
    futures = []
    for rank in range(args.num_workers):
        future = train_worker.remote(rank, args.num_workers, config)
        futures.append(future)
    
    # Wait for all workers to complete
    print("Waiting for all workers to complete...")
    results = ray.get(futures)
    
    # Print results
    print("\nTraining completed!")
    for result in results:
        print(f"Worker {result['rank']}: Final loss = {result['final_loss']:.4f}")
    
    # Cleanup and shutdown Ray
    try:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shut down successfully")
    except Exception as e:
        print(f"Warning: Error during Ray shutdown: {e}")
    
    print("Training script completed successfully!")

if __name__ == "__main__":
    main()
