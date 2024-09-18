import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_steps):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps

    def forward(self, timestep):
        half_dim = self.dim // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) * 
            (torch.log(torch.tensor(self.max_steps)) / half_dim)
            )
        emb = emb.to(timestep.device)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class UNet(nn.Module):
    def __init__(self, dim: int, max_steps: int):
        super(UNet, self).__init__()

        self.time_embed_dim = 64  # Size of the time embedding
        self.time_embedding = SinusoidalPositionalEmbedding(self.time_embed_dim, max_steps)

        # DownBlock
        self.conv1 = nn.Conv2d(1, dim // 2, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # MidBlock
        self.mid_conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.mid_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        # UpBlock
        self.upconv1 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)  # 7x7 -> 14x14
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)  # Skip connection from conv2
        
        self.upconv2 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)  # 14x14 -> 28x28
        self.conv4 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)  # Skip connection from conv1

        # Final output layer
        self.final_conv = nn.Conv2d(dim // 2, 1, kernel_size=1)
        
        # Activation function
        self.relu = nn.ReLU()

        # Linear layers to process the time embedding and inject it into the network
        self.time_mlp1 = nn.Linear(self.time_embed_dim, dim // 2)
        self.time_mlp2 = nn.Linear(self.time_embed_dim, dim)

    def forward(self, x, t):
        # Downsampling path
        time_emb = self.time_embedding(t)  # (batch_size, time_embed_dim)

        # Process time embedding through the MLPs
        time_emb1 = self.time_mlp1(time_emb).unsqueeze(-1).unsqueeze(-1)  # Reshape for broadcasting
        time_emb2 = self.time_mlp2(time_emb).unsqueeze(-1).unsqueeze(-1)  # Reshape for broadcasting
        
        conv1 = self.relu(self.conv1(x) + time_emb1)  # (1, 28, 28) -> (32, 28, 28)
        pool1 = self.pool1(conv1)         # (32, 28, 28) -> (32, 14, 14)
        
        conv2 = self.relu(self.conv2(pool1) + time_emb2)  # (32, 14, 14) -> (64, 14, 14)
        pool2 = self.pool2(conv2)             # (64, 14, 14) -> (64, 7, 7)
        
        # Mid block
        mid = self.relu(self.mid_conv1(pool2))
        mid = self.relu(self.mid_conv2(mid))

        # Upsampling path with skip connections
        upconv1 = self.upconv1(mid)         # (64, 7, 7) -> (64, 14, 14)
        upconv1 = torch.cat([upconv1, conv2], dim=1)  # Skip connection with conv2

        conv3 = self.relu(self.conv3(upconv1))  # (128, 14, 14) -> (64, 14, 14)
        
        upconv2 = self.upconv2(conv3)           # (64, 14, 14) -> (32, 28, 28)
        upconv2 = torch.cat([upconv2, conv1], dim=1)  # Skip connection with conv1
        
        conv4 = self.relu(self.conv4(upconv2))  # (64, 28, 28) -> (32, 28, 28)
        
        # Final output layer (1 channel output)
        output = self.final_conv(conv4)  # (32, 28, 28) -> (1, 28, 28)
        
        return output