import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def create_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    _1D_window = gaussian_window(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    window: torch.Tensor = None,
    size_average: bool = True,
    full: bool = False
):
    """
    Computes Structural Similarity Index (SSIM) between img1 and img2 in range [0, 1].
    """
    channel = img1.size(1)
    if window is None:
        window = create_window(window_size, channel).to(img1.device)
    else:
        window = window.to(img1.device)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if full:
        return ssim_map.mean(dim=1, keepdim=True)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """
    Structural Dissimilarity Loss: 1 - SSIM(I, I_hat).
    Penalizes structural and textural distortions rather than raw pixel luminance.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", create_window(window_size, channel, sigma))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(img1, img2, window_size=self.window_size, window=self.window, size_average=True)


class CompositeLoss(nn.Module):
    """
    Composite L1 + SSIM Loss:
      L = alpha * L1 + (1 - alpha) * (1 - SSIM)
      
    This prevents blurry reconstructions and false-alarm high errors along sharp edges
    while maintaining strict adherence to fine industrial surface textures.
    """
    def __init__(self, alpha: float = 0.5, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=window_size, channel=channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)
        s_loss = self.ssim_loss(pred, target)
        return self.alpha * l1 + (1.0 - self.alpha) * s_loss


class NutForegroundMasker:
    """
    Extracts the metallic nut foreground region-of-interest (RoI),
    suppressing background lighting fluctuations and inner hole noise.
    """
    @staticmethod
    def extract_mask(image_tensor: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
        """
        image_tensor: (B, C, H, W) normalized [0, 1]
        Returns: (B, 1, H, W) binary mask
        """
        gray = torch.mean(image_tensor, dim=1, keepdim=True)
        mask = (gray > threshold).float()
        return mask
