import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    让模型自动学习哪些频率子载波更重要。
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlockWithAttention(nn.Module):
    """残差块 + Channel Attention"""
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
        self.attention = ChannelAttention(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.attention(out)   # attention加在残差之前
        return F.relu(x + out)


class DeepMonAttention(nn.Module):
    """
    DeepMon + Channel Attention版本
    唯一区别：ResBlock换成ResBlockWithAttention
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
            *[ResBlockWithAttention(num_channels) for _ in range(num_res_blocks)]
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
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)


if __name__ == "__main__":
    model = DeepMonAttention()
    x = torch.randn(4, 2, 12, 64)
    out = model(x)
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")