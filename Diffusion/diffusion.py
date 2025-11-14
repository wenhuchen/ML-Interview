import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import wandb
import argparse
from mnist_unet import Unet
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='', type=str)

args = parser.parse_args()

class LinearBetaScheduler:
    def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self.make_beta_schedule()
        self.bar_alpha = self.make_alpha_schedule()

    def make_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)

    def make_alpha_schedule(self):
        alphas_cumprod = torch.cumprod(1 - self.betas, dim=0)
        return alphas_cumprod

    def get_variance(self, t):
        return self.betas[t.cpu().long()].to(t.device)

    def get_alpha(self, t):
        return 1 - self.betas[t.cpu().long()].to(t.device)

    def get_bar_alpha(self, t):
        return self.bar_alpha[t.cpu().long()].to(t.device)


def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):

    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)


def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print('Successfully loaded the model from', load_path)
    return model, optimizer, epoch


def sample(scheduler, t: int, model, x_t):
    t_ = torch.full((x_t.shape[0],), fill_value=t).to(device)
    pred = model(x_t, t=t_)

    alpha_t = scheduler.get_alpha(t_).reshape(x_t.shape[0],1,1,1)
    bar_alpha_t = scheduler.get_bar_alpha(t_).reshape(x_t.shape[0],1,1,1)
    bar_alpha_t_m1 = scheduler.get_bar_alpha(t_ - 1).reshape(x_t.shape[0],1,1,1)
    beta_t = scheduler.get_variance(t_).reshape(x_t.shape[0],1,1,1)
    sqrt_one_minues_bar_alpha_t = torch.sqrt(1 - bar_alpha_t).reshape(x_t.shape[0],1,1,1)

    mean = (1./torch.sqrt(alpha_t))*(x_t - ((1  - alpha_t) / sqrt_one_minues_bar_alpha_t) * pred)

    noise = torch.randn_like(x_t)

    if t > 0:
        std=torch.sqrt(beta_t*(1.-bar_alpha_t_m1)/(1.-bar_alpha_t))
    else:
        std = 0.

    return mean+std*noise


def sample_clip(scheduler, t: int, model, x_t):
    t_ = torch.full((x_t.shape[0],), fill_value=t).to(device)
    pred = model(x_t, t=t_)

    alpha_t = scheduler.get_alpha(t_).reshape(x_t.shape[0],1,1,1)
    bar_alpha_t = scheduler.get_bar_alpha(t_).reshape(x_t.shape[0],1,1,1)
    bar_alpha_t_m1 = scheduler.get_bar_alpha(t_ - 1).reshape(x_t.shape[0],1,1,1)
    beta_t = scheduler.get_variance(t_).reshape(x_t.shape[0],1,1,1)

    x_0_pred=torch.sqrt(1. / bar_alpha_t)*x_t-torch.sqrt(1. / bar_alpha_t - 1.)*pred
    x_0_pred.clamp_(-1., 1.)

    noise = torch.randn_like(x_t)

    if t > 0:
        mean= (beta_t * torch.sqrt(bar_alpha_t_m1) / (1. - bar_alpha_t))*x_0_pred +\
                ((1. - bar_alpha_t_m1) * torch.sqrt(alpha_t) / (1. - bar_alpha_t))*x_t

        std=torch.sqrt(beta_t*(1.-bar_alpha_t_m1)/(1.-bar_alpha_t))
    else:
        mean=(beta_t / (1. - bar_alpha_t))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
        std=0.0

    return mean+std*noise


def save_image(size: int, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.randn(size, 1, 28, 28).to(device)
    # import pdb;pdb.set_trace()

    with torch.no_grad():
        for t in range(max_steps - 1, -1, -1):
            x_t = sample_clip(scheduler, t, model, x_t)

    pred_x_0 = ((x_t + 1) / 2)
    pred_x_0 = pred_x_0.cpu().numpy()
    image = pred_x_0.reshape((-1, 28))
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('image', dpi=300, bbox_inches='tight')
    plt.close()
    print('Saving image to the local folder.')


if __name__ == "__main__":
    # Instantiate the model

    # Hyperparameters
    epochs = 30
    lr = 1e-3
    max_steps = 1000

    model = Unet(timesteps=1000, time_embedding_dim=64, in_channels=1, out_channels=1, dim_mults=[2,4])
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example usage:
    scheduler = LinearBetaScheduler(num_timesteps=max_steps)

    if args.mode == 'train':
        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project="diffusion",
            name="epsilon_pred",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "epochs": epochs,
            },
        )

        batch_size = 128

        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_loader,_=create_mnist_dataloaders(batch_size=batch_size, image_size=28)
        loss_fn=nn.MSELoss(reduction='mean')

        # Define the learning rate scheduler
        lr_scheduler=OneCycleLR(optimizer,lr,total_steps=epochs*len(train_loader),
                                pct_start=0.25,anneal_strategy='cos')

        prev_time = time.time()
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, _) in enumerate(train_loader):
                time_tensor = torch.randint(low=0, high=max_steps, size=(data.shape[0],)).to(device)
                data = data.to(device)
                epsilon = torch.randn_like(data).to(device)

                bar_alpha = scheduler.get_bar_alpha(time_tensor)

                noised_data = data * torch.sqrt(bar_alpha)[:, None, None, None] + epsilon * (1 - torch.sqrt(bar_alpha))[:, None, None, None]
                pred_epsilon = model(noised_data, t=time_tensor)

                # noised_data = noised_data.cpu().numpy().reshape((-1, 28))
                loss = loss_fn(pred_epsilon, epsilon)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                    time_lapse = time.time() - prev_time
                    time_per_sample = time_lapse / (100 * len(data))
                    prev_time = time.time()
                    run.log({'loss': loss.item(), 'time-per-sample': time_per_sample})

            model.eval()
            save_image(2, model)

        # decouple the multi-gpu parallelism
        model_to_save = model.module if hasattr(model, "module") else model

        save_path = 'unet_mnist_model.pth'
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
        }, save_path)
        print(f"Model saved to {save_path}")

        wandb.finish()
    elif args.mode == 'eval':
        batch_size = 1
        model.eval()
        model, optimizer, epoch = load_model(model, optimizer, 'unet_mnist_model.pth')
        save_image(2, model)
    else:
        raise NotImplementedError(args.mode)