import logging
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def latest_checkpoint_path(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    return paths[-1]


def load_checkpoint(path: Path,
                    model: nn.Module,
                    optimizer: Optimizer = None) -> Tuple[float, int]:
    checkpoint_dict = torch.load(path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    saved_state_dict = checkpoint_dict['model']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}

    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except KeyError:
            logger.info('%s is not in the checkpoint', k)
            new_state_dict[k] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    logger.info('Loaded checkpoint "%s" (iteration %d)', path, iteration)

    return learning_rate, iteration


def save_checkpoint(model: nn.Module,
                    optimizer: Optimizer,
                    learning_rate: float,
                    iteration: int,
                    checkpoint_path: Path):
    logger.info('Saving model and optimizer state at iteration %d to %s', iteration, checkpoint_path)

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    payload = {
        'model': state_dict,
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate
    }

    torch.save(payload, checkpoint_path)


def summarize(writer: SummaryWriter,
              global_step: int,
              scalars: Mapping[str, Any] = None,
              histograms: Mapping[str, Any] = None,
              images: Mapping[str, Any] = None,
              audios: Mapping[str, Any] = None,
              audio_sampling_rate: int = 22050):
    if scalars is None:
        scalars = {}
    if histograms is None:
        histograms = {}
    if images is None:
        images = {}
    if audios is None:
        audios = {}

    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)

    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)

    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')

    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)
