import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.d = d
        # Encoder q_phi(z | x)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),   # 28 -> 14
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),                  # 14 -> 7
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, d)
        self.fc_logvar = nn.Linear(64 * 7 * 7, d)
        # Decoder p_theta(x | z): outputs one logit per pixel (Bernoulli)
        self.fc_dec = nn.Linear(d, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),         # 7 -> 14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),                     # 14 -> 28
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps, eps ~ N(0, I), sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 7, 7)
        return self.dec(h)  # logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        z = torch.randn(n, self.d, device=device)
        return torch.sigmoid(self.decode(z))


def kl_divergence(mu, logvar):
    """Closed-form KL(q(z|x) || N(0,I)), summed over latent dims -> shape (B,)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vae_loss(logits, x, mu, logvar):
    """Negative ELBO averaged over the batch. Returns (total, recon, kl)."""
    # BCE summed over the 784 pixels -> shape (B,)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)
    kl = kl_divergence(mu, logvar)
    return (recon + kl).mean(), recon.mean(), kl.mean()