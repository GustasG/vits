from pathlib import Path
from typing import Optional, Tuple, List

from pydantic import BaseModel


class TrainingParams(BaseModel):
    log_interval: int
    eval_interval: int
    seed: int
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    eps: float
    batch_size: int
    fp16_run: bool
    lr_decay: float
    segment_size: int
    c_mel: int
    c_kl: float


class DataParams(BaseModel):
    training_files: Path
    validation_files: Path
    text_cleaners: List[str]
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    language: str
    min_text_len: int = 1
    max_text_len: int = 190
    mel_fmax: Optional[float] = None


class ModelParams(BaseModel):
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    upsample_rates: List[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: List[int]
    n_layers_q: int
    use_spectral_norm: bool
    use_sdp: bool
    gin_channels: int


class Params(BaseModel):
    train: TrainingParams
    data: DataParams
    model: ModelParams
