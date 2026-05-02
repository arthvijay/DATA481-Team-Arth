import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels: int = 16, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class DeepMonNoFFT(nn.Module):
    """
    消融实验：去掉FFT预处理，直接用原始时域I/Q波形
    输入: [B, 2, T]  (T = n_symbols * n_fft = 768)
    对比train.py的2D ResNet，唯一区别是输入格式
    """
    def __init__(
        self,
        num_bits: int = 24,
        num_channels: int = 16,
        num_res_blocks: int = 6,
        input_length: int = 768,   # 12 * 64
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(2, num_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock1D(num_channels) for _ in range(num_res_blocks)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_bits),
        )

    def forward(self, x):
        # x: [B, 2, T]
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)


if __name__ == "__main__":
    model = DeepMonNoFFT()
    x = torch.randn(4, 2, 768)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")