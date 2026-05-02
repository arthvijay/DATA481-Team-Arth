from dataclasses import dataclass

@dataclass
class Config:
    # Data path
    root_dir: str = "/home/azureuser/data_unzipped/Data/Zhihui_new/SampleRate_5000000"

    # FFT preprocessing (DeepMon per-OFDM-symbol FFT)
    n_fft: int = 64        # FFT points per symbol
    n_symbols: int = 12    # first 12 OFDM symbols

    # Model
    num_bits: int = 24         # L-SIG bits
    num_channels: int = 16     # ResNet channels
    num_res_blocks: int = 6    # residual blocks

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15
    random_seed: int = 42
    num_workers: int = 4

    # Augmentation
    use_phase_augment: bool = True

    # Output
    output_dir: str = "outputs_deepmon"
    target_protocol: str = "Non-HT"
