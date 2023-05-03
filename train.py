import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler

from audio.processing import melscale_spectrogram_from_waveform, melscale_bank_from_spectrogram
from data.collate import single_speaker_collate
from data.dataset import SingleSpeakerDataset
from data.samplers import DistributedBucketSampler
from model import commons
from model.discriminators import MultiPeriodDiscriminator
from model.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from model.synthesizer import SynthesizerTrn
from utils.checkpoint import load_checkpoint, latest_checkpoint_path
import utils.logging
from utils.plot import plot_spectrogram_to_numpy, plot_alignment_to_numpy
from params import Params


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    if not torch.cuda.is_available():
        print('CPU training is not allowed')
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True, help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    parser.add_argument('-w', '--workers', type=int, default=os.cpu_count(), help='Dataloader worker count')
    args = parser.parse_args()

    params = Params.parse_file(args.config)

    cwd = Path(os.getcwd())
    model_dir = cwd / 'logs' / args.model

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, params, model_dir, args))


def run(rank: int, n_gpus: int, params: Params, model_dir: Path, args):
    global global_step

    if rank == 0:
        logger = utils.logging.get_logger(model_dir, 'train.log')
        writer = SummaryWriter(str(model_dir))
        writer_eval = SummaryWriter(log_dir=str(model_dir / 'eval'))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    torch.manual_seed(params.train.seed)
    torch.cuda.manual_seed(params.train.seed)

    train_dataset = SingleSpeakerDataset.from_params(params.data.training_files, params.data)

    train_sampler = DistributedBucketSampler(
        dataset=train_dataset,
        batch_size=params.train.batch_size,
        boundaries=[32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=True,
        collate_fn=single_speaker_collate,
        batch_sampler=train_sampler,
        num_workers=args.workers
    )

    if rank == 0:
        eval_dataset = SingleSpeakerDataset.from_params(params.data.validation_files, params.data)

        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=params.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=single_speaker_collate,
            num_workers=args.workers
        )

    net_g = SynthesizerTrn.from_params(params).cuda(rank)
    net_d = MultiPeriodDiscriminator(use_spectral_norm=params.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        params=net_g.parameters(),
        lr=params.train.learning_rate,
        betas=params.train.betas,
        eps=params.train.eps
    )

    optim_d = torch.optim.AdamW(
        params=net_d.parameters(),
        lr=params.train.learning_rate,
        betas=params.train.betas,
        eps=params.train.eps
    )

    net_g = DistributedDataParallel(module=net_g, device_ids=[rank])
    net_d = DistributedDataParallel(module=net_d, device_ids=[rank])

    try:
        _, epoch_str = load_checkpoint(latest_checkpoint_path(model_dir, 'G_*.pth'), net_g, optim_g)
        _, epoch_str = load_checkpoint(latest_checkpoint_path(model_dir, 'D_*.pth'), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except IndexError:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optim_g,
        gamma=params.train.lr_decay,
        last_epoch=epoch_str - 2
    )

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optim_d,
        gamma=params.train.lr_decay,
        last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=params.train.fp16_run)

    for epoch in range(epoch_str, params.train.epochs + 1):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            train_and_evaluate(
                rank=rank,
                epoch=epoch,
                params=params,
                model_dir=model_dir,
                nets=[net_g, net_d],
                optims=[optim_g, optim_d],
                scaler=scaler,
                loaders=[train_loader, eval_loader],
                logger=logger,
                writers=(writer, writer_eval)
            )

            logger.info(f'====> Epoch: {epoch}')
        else:
            train_and_evaluate(
                rank=rank,
                epoch=epoch,
                params=params,
                model_dir=model_dir,
                nets=[net_g, net_d],
                optims=[optim_g, optim_d],
                scaler=scaler,
                loaders=[train_loader, None]
            )

        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank: int, epoch: int, params: Params, model_dir: Path,
                       nets, optims, scaler, loaders, logger=None,
                       writers: Tuple[SummaryWriter, SummaryWriter] = None):
    global global_step

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders

    if writers is not None:
        writer, writer_eval = writers

    net_g.train()
    net_d.train()

    for (x, x_lengths, spec, spec_lengths, y, y_lengths) in train_loader:
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        with autocast(enabled=params.train.fp16_run):
            y_hat, l_length, attn, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(x, x_lengths, spec, spec_lengths)

            mel = melscale_bank_from_spectrogram(
                spectrogram=spec,
                n_fft=params.data.filter_length,
                num_mels=params.data.n_mel_channels,
                sampling_rate=params.data.sampling_rate,
                fmin=params.data.mel_fmin,
                fmax=params.data.mel_fmax
            )

            y_mel = commons.slice_segments(
                x=mel,
                ids_str=ids_slice,
                segment_size=params.train.segment_size // params.data.hop_length
            )

            y_hat_mel = melscale_spectrogram_from_waveform(
                waveform=y_hat.squeeze(1),
                n_fft=params.data.filter_length,
                num_mels=params.data.n_mel_channels,
                sampling_rate=params.data.sampling_rate,
                hop_size=params.data.hop_length,
                win_size=params.data.win_length,
                fmin=params.data.mel_fmin,
                fmax=params.data.mel_fmax
            )

            y = commons.slice_segments(
                x=y,
                ids_str=ids_slice * params.data.hop_length,
                segment_size=params.train.segment_size
            )

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=params.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

            with autocast(enabled=False):
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * params.train.c_mel
                loss_dur = torch.sum(l_length.float())
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * params.train.c_kl
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)

        scaler.update()
        global_step += 1

        if rank == 0:
            if global_step % params.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                logger.info(
                    'Epoch: %d. Step: %d -> lr = %f, loss_disc = %f, loss_gen = %f, loss_fm = %f, loss_mel = %f, loss_dur = %f, loss_kl = %f, loss_gen_all = %f',
                    epoch, global_step, lr, loss_disc.item(), loss_gen.item(), loss_fm.item(), loss_mel.item(),
                    loss_dur.item(), loss_kl.item(), loss_gen_all.item()
                )

                scalars = {
                    'loss/g/total': loss_gen_all,
                    'loss/d/total': loss_disc_all,
                    'loss/g/fm': loss_fm,
                    'loss/g/mel': loss_mel,
                    'loss/g/dur': loss_dur,
                    'loss/g/kl': loss_kl,
                    'learning_rate': lr,
                    'grad_norm_d': grad_norm_d,
                    'grad_norm_g': grad_norm_g
                }

                images = {
                    'slice/mel_org': plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    'slice/mel_gen': plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    'all/mel': plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    'all/attn': plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                }

                for i, v in enumerate(losses_gen):
                    scalars[f'loss/g/{i}'] = v

                for i, v in enumerate(losses_disc_r):
                    scalars[f'loss/d_r/{i}'] = v

                for i, v in enumerate(losses_disc_g):
                    scalars[f'loss/d_g/{i}'] = v

                utils.checkpoint.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=images,
                    scalars=scalars,
                    audio_sampling_rate=params.data.sampling_rate
                )

            if global_step % params.train.eval_interval == 0:
                evaluate(params, net_g, eval_loader, writer_eval)

                utils.checkpoint.save_checkpoint(
                    model=net_g,
                    optimizer=optim_g,
                    learning_rate=params.train.learning_rate,
                    iteration=epoch,
                    checkpoint_path=model_dir / f'G_{global_step}.pth'
                )
    
                utils.checkpoint.save_checkpoint(
                    model=net_d,
                    optimizer=optim_d,
                    learning_rate=params.train.learning_rate,
                    iteration=epoch,
                    checkpoint_path=model_dir / f'D_{global_step}.pth'
                )


def evaluate(params: Params, generator, eval_loader, writer_eval: SummaryWriter):
    generator.eval()

    with torch.inference_mode():
        for (x, x_lengths, spec, spec_lengths, y, y_lengths) in eval_loader:
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            break

        y_hat, _, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * params.data.hop_length

        mel = melscale_bank_from_spectrogram(
            spectrogram=spec,
            n_fft=params.data.filter_length,
            num_mels=params.data.n_mel_channels,
            sampling_rate=params.data.sampling_rate,
            fmin=params.data.mel_fmin,
            fmax=params.data.mel_fmax
        )

        y_hat_mel = melscale_spectrogram_from_waveform(
            waveform=y_hat.squeeze(1).float(),
            n_fft=params.data.filter_length,
            num_mels=params.data.n_mel_channels,
            sampling_rate=params.data.sampling_rate,
            hop_size=params.data.hop_length,
            win_size=params.data.win_length,
            fmin=params.data.mel_fmin,
            fmax=params.data.mel_fmax
        )

    images = {
        'gen/mel': plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }

    audios = {
        'gen/audio': y_hat[0, :, :y_hat_lengths[0]]
    }

    if global_step == 0:
        images['gt/mel'] = plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        audios['gt/audio'] = y[0, :, :y_lengths[0]]

    utils.checkpoint.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=images,
        audios=audios,
        audio_sampling_rate=params.data.sampling_rate
    )

    generator.train()


if __name__ == "__main__":
    main()
