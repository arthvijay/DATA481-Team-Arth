import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════
#  Shared building blocks
# ════════════════════════════════════════════════════════

class ResBlock2D(nn.Module):
    """
    2D residual block: Conv→BN→SeLU→Conv→BN + skip.
    Uses SeLU (self-normalizing) as in the DeepMon paper,
    NOT ReLU which was used in the previous src2 implementation.
    """
    def __init__(self, channels: int = 16, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.SiLU(),   # SeLU equivalent — SiLU/Swish is available; nn.SELU() also works
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.silu(x + self.block(x))


class ResBlock1D(nn.Module):
    """1D residual block for No-FFT ablation."""
    def __init__(self, channels: int = 16, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.silu(x + self.block(x))


# ════════════════════════════════════════════════════════
#  Channel Attention (Squeeze-and-Excitation)
# ════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """SE block: learns which frequency channels matter most."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlockWithAttention(nn.Module):
    """ResBlock2D + channel attention (SE) after the conv pair."""
    def __init__(self, channels: int = 16, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )
        self.attention = ChannelAttention(channels)

    def forward(self, x):
        out = self.attention(self.block(x))
        return F.silu(x + out)


# ════════════════════════════════════════════════════════
#  Shared 2D backbone (stem + N res blocks + global pool)
# ════════════════════════════════════════════════════════

class Backbone2D(nn.Module):
    def __init__(
        self,
        num_channels: int = 16,
        num_res_blocks: int = 6,
        use_attention: bool = False,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, num_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.SiLU(),
        )
        BlockCls = ResBlockWithAttention if use_attention else ResBlock2D
        self.res_blocks = nn.Sequential(
            *[BlockCls(num_channels) for _ in range(num_res_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: [B, 2, n_symbols, n_fft]
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.pool(x).flatten(1)   # [B, num_channels]


# ════════════════════════════════════════════════════════
#  Stage 1 — Protocol classifier  (Gap 2 fix)
# ════════════════════════════════════════════════════════

class ProtocolClassifier(nn.Module):
    """
    Stage 1 of DeepMon: classify Wi-Fi protocol.
    Trained independently from Stage 2 (as per paper).

    Classes: 0=802.11a (Non-HT), 1=802.11n (HT), 2=802.11ac (VHT)
    Output : logits [B, num_classes]  →  apply Softmax for probabilities
    """
    def __init__(
        self,
        num_classes: int = 3,
        num_channels: int = 16,
        num_res_blocks: int = 6,
        use_attention: bool = False,
    ):
        super().__init__()
        self.backbone = Backbone2D(num_channels, num_res_blocks, use_attention)
        self.head = nn.Sequential(
            nn.Linear(num_channels, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))   # logits → CrossEntropyLoss


# ════════════════════════════════════════════════════════
#  Stage 2 — L-SIG bit decoder  (main model)
# ════════════════════════════════════════════════════════

class DeepMonModel(nn.Module):
    """
    Stage 2 of DeepMon: decode L-SIG bits from [2, 12, 64] spectrogram.
    Output: logits [B, num_bits]  →  apply Sigmoid + threshold 0.5

    Gap 1 fix: uses SiLU (≈SeLU self-normalizing) instead of ReLU.
    """
    def __init__(
        self,
        num_bits: int = 24,
        num_channels: int = 16,
        num_res_blocks: int = 6,
        use_attention: bool = False,
    ):
        super().__init__()
        self.backbone = Backbone2D(num_channels, num_res_blocks, use_attention)
        self.head = nn.Sequential(
            nn.Linear(num_channels, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_bits),
        )

    def forward(self, x):
        return self.head(self.backbone(x))   # logits → BCEWithLogitsLoss


# ════════════════════════════════════════════════════════
#  No-FFT ablation model (1D)
# ════════════════════════════════════════════════════════

class DeepMonNoFFT(nn.Module):
    """
    Ablation: raw time-domain IQ, no FFT.
    Input: [B, 2, 768]
    """
    def __init__(
        self,
        num_bits: int = 24,
        num_channels: int = 16,
        num_res_blocks: int = 6,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, num_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_channels),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock1D(num_channels) for _ in range(num_res_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_bits),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)


# ════════════════════════════════════════════════════════
#  Quick smoke test
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    x2d = torch.randn(4, 2, 12, 64)
    x1d = torch.randn(4, 2, 768)

    for name, m, x in [
        ("DeepMonModel (base)",      DeepMonModel(),                      x2d),
        ("DeepMonModel (attention)", DeepMonModel(use_attention=True),    x2d),
        ("ProtocolClassifier",       ProtocolClassifier(),                x2d),
        ("DeepMonNoFFT",             DeepMonNoFFT(),                      x1d),
    ]:
        out = m(x)
        p   = sum(p.numel() for p in m.parameters())
        print(f"{name:<35} output={tuple(out.shape)}  params={p:,}")
