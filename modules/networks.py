import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.config import Config


class VAE(nn.Module):
    def __init__(self, cfg: Config):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, cfg.model.latent_dim)
        self.fc22 = nn.Linear(400, cfg.model.latent_dim)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


Networks = {
    'vae': VAE,
}
