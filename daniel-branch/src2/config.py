from dataclasses import dataclass

@dataclass
class Config:
    # 数据路径
    root_dir: str = "/home/azureuser/data_unzipped/Data/Zhihui_new/SampleRate_5000000"
    
    # FFT预处理参数（DeepMon用per-OFDM-symbol FFT）
    n_fft: int = 64          # FFT点数
    n_symbols: int = 12      # 取前12个OFDM symbol
    
    # 模型参数
    num_bits: int = 24       # L-SIG bit数
    num_channels: int = 16   # ResNet通道数
    num_res_blocks: int = 6  # 残差块数量
    
    # 训练参数
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15
    random_seed: int = 42
    num_workers: int = 4
    
    # 数据增强
    use_phase_augment: bool = True   # 随机相位偏移
    
    # 输出路径
    output_dir: str = "outputs_deepmon"
    target_protocol: str = "Non-HT"