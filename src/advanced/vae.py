import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VariationalAutoencoderV2(nn.Module):
    """
    Variational Autoencoder (VAE) V2 for normal surface modeling and defect synthesis.
    Encodes normal images into a regularized Gaussian latent space.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (256x256 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Latent heads: Mean and Log-Variance
        # Spatial size is 8x8 with base_channels*8 = 256 channels -> flatten to 256*8*8 = 16384
        self.flatten_dim = base_channels * 8 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder mapping
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(-1, 256, 8, 8)
        d5 = self.dec5(h)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        out = self.dec1(d2)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    @staticmethod
    def vae_loss(
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.0005
    ) -> Dict[str, torch.Tensor]:
        """
        Computes VAE loss = Reconstruction Loss + beta * KL Divergence
        """
        recon_loss = F.mse_loss(reconstruction, target, reduction="mean")
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total_loss = recon_loss + beta * kl_div

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_div
        }
