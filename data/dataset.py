import os
import random
from pathlib import Path
from typing import List, Tuple, Iterator

import torch
from torch.utils.data import Dataset

import torchaudio

from model.commons import intersperse
from audio.processing import spectrogram_from_waveform
from text.convert import text_to_sequence
from params import DataParams


class SingleSpeakerDataset(Dataset):
    def __init__(self,
                 *,
                 path: Path,
                 text_cleaners: List[str],
                 sampling_rate: int,
                 filter_length: int,
                 hop_length: int,
                 win_length: int,
                 language: str,
                 min_text_len: int,
                 max_text_len: int):
        super(SingleSpeakerDataset, self).__init__()
        self.file_lines = list(self._load_filepaths_and_text(path))
        self.text_cleaners = text_cleaners
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.language = language
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len

        random.seed(1234)
        random.shuffle(self.file_lines)

        self.file_lines, self.lengths = self._filter(self.file_lines, min_text_len, max_text_len, hop_length)

    @classmethod
    def from_params(cls, path: Path, params: DataParams):
        return cls(
            path=path,
            text_cleaners=params.text_cleaners,
            sampling_rate=params.sampling_rate,
            filter_length=params.filter_length,
            hop_length=params.hop_length,
            win_length=params.win_length,
            language=params.language,
            min_text_len=params.min_text_len,
            max_text_len=params.max_text_len
        )

    @staticmethod
    def _filter(lines: List[Tuple[Path, str]], min_text_length: int, max_text_length: int, hop_length: int) -> \
            Tuple[List[Tuple[Path, str]], List[int]]:
        new_lines, lengths = [], []

        for path, text in lines:
            if min_text_length <= len(text) <= max_text_length:
                new_lines.append((path, text))
                lengths.append(os.path.getsize(path) // (2 * hop_length))

        return new_lines, lengths

    @staticmethod
    def _load_filepaths_and_text(path: Path, sep: str = '|') -> Iterator[Tuple[Path, str]]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                path, text = line.strip().split(sep)
                yield Path(path), text

    def get_text(self, text: str) -> torch.Tensor:
        text_norm = intersperse(text_to_sequence(text, self.text_cleaners, self.language), 0)

        return torch.LongTensor(text_norm)

    def get_audio(self, filename: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, rate = torchaudio.load(filename)
        spec_file = filename.with_suffix('.spec.pt')

        if rate != self.sampling_rate:
            raise ValueError(f'{rate} SR doesn\'t match target {self.sampling_rate} SR')
        
        if spec_file.exists():
            spec = torch.load(spec_file)
        else:
            spec = spectrogram_from_waveform(
                waveform=audio,
                n_fft=self.filter_length,
                hop_size=self.hop_length,
                win_size=self.win_length,
                center=False
            )

            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_file)

        return spec, audio

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, text = self.file_lines[index]

        text = self.get_text(text)
        spec, wav = self.get_audio(path)

        return text, spec, wav

    def __len__(self) -> int:
        return len(self.file_lines)
