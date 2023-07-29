from functools import cache

import librosa
import torch
import torchaudio


def dynamic_range_compression_torch(x: torch.Tensor, c: int = 1, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * c)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_compression_torch(magnitudes)
    return output


@cache
def get_hann_window(device: torch.device, dtype: torch.dtype, window_length: int) -> torch.Tensor:
    return torch.hann_window(window_length, dtype=dtype, device=device)


@cache
def get_mel_basis(device: torch.device, dtype: torch.dtype, sampling_rate: int, n_fft: int,
                  num_mels: int, fmin: float, fmax: float) -> torch.Tensor:
    mel = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax
    )

    return torch.tensor(mel, device=device, dtype=dtype)


def spectrogram_from_waveform(waveform: torch.Tensor, n_fft: int, hop_size: int,
                              win_size: int, center: bool = False) -> torch.Tensor:
    window = get_hann_window(waveform.device, waveform.dtype, win_size)

    pad = (n_fft - hop_size) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (pad, pad), mode='reflect')
    waveform = waveform.squeeze(1)

    spec = torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        power=None,
        normalized=False,
        center=center,
        pad_mode='reflect',
        onesided=True
    )

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    return spec


def melscale_bank_from_spectrogram(spectrogram: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int,
                                   fmin: float, fmax: float) -> torch.Tensor:
    mel = get_mel_basis(spectrogram.device, spectrogram.dtype, sampling_rate, n_fft, num_mels, fmin, fmax)
    spectrogram = torch.matmul(mel, spectrogram)
    spectrogram = spectral_normalize_torch(spectrogram)

    return spectrogram


def melscale_spectrogram_from_waveform(waveform: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int,
                                       hop_size: int, win_size: int, fmin: float, fmax: float,
                                       center: bool = False) -> torch.Tensor:
    spec = spectrogram_from_waveform(waveform, n_fft, hop_size, win_size, center)
    spec = melscale_bank_from_spectrogram(spec, n_fft, num_mels, sampling_rate, fmin, fmax)

    return spec
