import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 20

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 500),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(500, latent_dim)
        self.fc_logvar = nn.Linear(500, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 500)
        
        self.decoder = nn.Sequential(
            nn.Linear(500, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    # Initialize model and optimizer
    # Set random seed for reproducibility

    # Hyperparameters
    batch_size = 128
    epochs = 30
    lr = 1e-3

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    OPTION = 'eval'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if OPTION == 'train':
        # Training loop
        torch.manual_seed(42)
        model.to(device)

        def train(epoch):
            model.train()
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
            
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

        # Run training
        for epoch in range(1, epochs + 1):
            train(epoch)

        print("Training complete!")

        save_path = 'vae_mnist_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
        }, save_path)
        print(f"Model saved to {save_path}")
    else:
        model.eval()
        def load_model(model, optimizer, load_path):
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            return model, optimizer, epoch

        def save_generated_image(image, filename):
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Image saved as {filename}")

        model, optimizer, epoch = load_model(model, optimizer, 'vae_mnist_model.pth')

        with torch.no_grad():
            x = torch.randn(10, 20).to(device)
            y = model.decode(x).cpu().numpy()
            y = y.reshape((-1, 28))
            plt.figure(figsize=(5, 5))
            plt.imshow(y, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('image', dpi=300, bbox_inches='tight')
            plt.close()