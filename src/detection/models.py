import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, output_padding: int = 0):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=stride, padding=1, 
                output_padding=output_padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class BasicAutoencoderV2(nn.Module):
    """
    Basic Convolutional Autoencoder V2 without skip connections.
    Enforces a strict information bottleneck at the latent representation.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        # Encoder (256x256 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.enc1 = ConvBlock(in_channels, base_channels, stride=2)        # 128x128
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)   # 64x64
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2) # 32x32
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2) # 16x16
        self.enc5 = ConvBlock(base_channels * 8, latent_dim, stride=2)        # 8x8

        # Bottleneck Conv
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder (8x8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.dec5 = DeconvBlock(latent_dim, base_channels * 8, stride=2)      # 16x16
        self.dec4 = DeconvBlock(base_channels * 8, base_channels * 4, stride=2) # 32x32
        self.dec3 = DeconvBlock(base_channels * 4, base_channels * 2, stride=2) # 64x64
        self.dec2 = DeconvBlock(base_channels * 2, base_channels, stride=2)     # 128x128
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Reconstructed image in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        return self.bottleneck(e5)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        d5 = self.dec5(z)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        out = self.dec1(d2)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class SkipConnectionAutoencoderV2(nn.Module):
    """
    Skip Connection Convolutional Autoencoder V2 (U-Net Style).
    Preserves fine spatial boundaries by concatenating encoder feature maps.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        # Encoder stages
        self.enc1 = ConvBlock(in_channels, base_channels, stride=2)        # (32, 128, 128)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)   # (64, 64, 64)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2) # (128, 32, 32)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2) # (256, 16, 16)
        self.enc5 = ConvBlock(base_channels * 8, latent_dim, stride=2)        # (128, 8, 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder stages with skip concatenation
        self.dec5 = DeconvBlock(latent_dim, base_channels * 8, stride=2)  # 16x16
        self.fuse4 = nn.Sequential(
            nn.Conv2d(base_channels * 8 * 2, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dec4 = DeconvBlock(base_channels * 8, base_channels * 4, stride=2)  # 32x32
        self.fuse3 = nn.Sequential(
            nn.Conv2d(base_channels * 4 * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dec3 = DeconvBlock(base_channels * 4, base_channels * 2, stride=2)  # 64x64
        self.fuse2 = nn.Sequential(
            nn.Conv2d(base_channels * 2 * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dec2 = DeconvBlock(base_channels * 2, base_channels, stride=2)  # 128x128
        self.fuse1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        e1 = self.enc1(x)   # (32, 128, 128)
        e2 = self.enc2(e1)  # (64, 64, 64)
        e3 = self.enc3(e2)  # (128, 32, 32)
        e4 = self.enc4(e3)  # (256, 16, 16)
        e5 = self.enc5(e4)  # (128, 8, 8)
        
        b = self.bottleneck(e5)

        # Decoder passes with skip connections
        d5 = self.dec5(b)
        f4 = self.fuse4(torch.cat([d5, e4], dim=1))

        d4 = self.dec4(f4)
        f3 = self.fuse3(torch.cat([d4, e3], dim=1))

        d3 = self.dec3(f3)
        f2 = self.fuse2(torch.cat([d3, e2], dim=1))

        d2 = self.dec2(f2)
        f1 = self.fuse1(torch.cat([d2, e1], dim=1))

        out = self.dec1(f1)
        return out


class GatedSkipAutoencoder(nn.Module):
    """
    Improved Skip Connection Autoencoder with Attention Gates on skip pathways.
    Blocks anomalous high-frequency leakage while preserving normal metal thread geometry.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        self.base = SkipConnectionAutoencoderV2(in_channels, base_channels, latent_dim)
        # 1x1 conv channel gating weights
        self.gate1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.base.enc1(x)
        e2 = self.base.enc2(e1)
        e3 = self.base.enc3(e2)
        e4 = self.base.enc4(e3)
        e5 = self.base.enc5(e4)
        
        b = self.base.bottleneck(e5)

        d5 = self.base.dec5(b)
        f4 = self.base.fuse4(torch.cat([d5, e4], dim=1))

        d4 = self.base.dec4(f4)
        f3 = self.base.fuse3(torch.cat([d4, e3], dim=1))

        d3 = self.base.dec3(f3)
        gated_e2 = e2 * self.gate2(d3)
        f2 = self.base.fuse2(torch.cat([d3, gated_e2], dim=1))

        d2 = self.base.dec2(f2)
        gated_e1 = e1 * self.gate1(d2)
        f1 = self.base.fuse1(torch.cat([d2, gated_e1], dim=1))

        out = self.base.dec1(f1)
        return out


def save_checkpoint(model: nn.Module, save_path: str, epoch: int = 0, optimizer: Any = None, metrics: Dict = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics or {}
    }
    torch.save(state, save_path)
    print(f"Checkpoint successfully saved to: {save_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> Dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return state
