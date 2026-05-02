import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """residual block for 2D convolutions: Conv→BN→ReLU→Conv→BN + skip"""
    def __init__(self, channels: int = 16, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class DeepMonModel(nn.Module):
    """
    DeepMon架构:
    - Stem: Conv2d(2→16) + BN + ReLU
    - 6个残差块
    - GlobalAvgPool → FC(128) → FC(num_bits)
    """
    def __init__(
        self,
        num_bits: int = 24,
        num_channels: int = 16,
        num_res_blocks: int = 6,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(2, num_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock2D(num_channels) for _ in range(num_res_blocks)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_bits),
        )

    def forward(self, x):
        # x: [B, 2, n_symbols, n_fft]
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)


if __name__ == "__main__":
    model = DeepMonModel()
    x = torch.randn(4, 2, 12, 64)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # [4, 24]
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")