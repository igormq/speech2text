import glob
import logging
import os
import re
import shutil
import time

import torch
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tqdm import tqdm

from asr import lr_schedulers, metrics, optimizers
from asr.utils.checks import check_loss

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 serialization_dir,
                 params,
                 model,
                 loss,
                 alphabet,
                 local_rank=0,
                 world_size=1,
                 sync_bn=False,
                 opt_level='O0',
                 keep_batchnorm_fp32=None,
                 loss_scale=1.0):
        self.alphabet = alphabet
        self.clip_grad_norm = params.get('clip_grad_norm', None)
        self.clip_grad_value = params.get('clip_grad_value', None)
        self.warmup_epochs = params.get('warmup_epochs', 0)
        self.label_smoothing = params.get('label_smoothing', 0.0)

        self.local_rank = local_rank
        self.loss = loss.cuda()
        self.model = model
        self.best_monitor = float('inf')
        self.monitor = params.get('monitor', 'loss')
        self.serialization_dir = serialization_dir
        self.distributed = world_size > 1
        self.world_size = world_size
        self.epoch = 0
        self.num_epochs = 0

        self.start_epoch = 0
        self.start_iteration = 0
        self.start_time = 0
        self.iterations_per_epoch = None

        self.time_since_last = time.time()

        self.save_every = params.get('save_every', 60 * 10)  # 10 minutes

        if sync_bn:
            logger.info('Using Apex `sync_bn`')
            self.model = convert_syncbn_model(self.model)

        self.model = model.cuda()

        # Setup optimizer
        parameters = [(n, p) for n, p in self.model.named_parameters()
                      if p.requires_grad]
        self.optimizer = optimizers.from_params(params.pop("optimizer"),
                                                parameters,
                                                world_size=self.world_size)

        self.model, self.optimizer = amp.initialize(
            self.model,
            self.optimizer,
            opt_level=opt_level,
            keep_batchnorm_fp32=keep_batchnorm_fp32,
            loss_scale=loss_scale)

        # Setup lr scheduler
        scheduler_params = params.pop('lr_scheduler', None)
        self.lr_scheduler = None
        if scheduler_params:
            self.lr_scheduler = lr_schedulers.from_params(
                scheduler_params, self.optimizer)

        self.base_lrs = list(
            map(lambda group: group['initial_lr'],
                self.optimizer.param_groups))

        # Setup metrics
        metrics_params = params.pop('metrics', [])
        if 'loss' not in metrics_params:
            metrics_params = ['loss'] + metrics_params

        # Initializing history
        self.metrics = {}
        for phase in ['train', 'val']:
            self.metrics[phase] = metrics.from_params(metrics_params,
                                                      alphabet=alphabet)

        if self.distributed:
            self.model = DistributedDataParallel(self.model,
                                                 delay_allreduce=True)

    def score(self, metrics):
        sign = -1. if self.monitor.startswith('-') else 1.
        name = self.monitor.replace('-', '')
        return sign * metrics[name].avg

    def restore(self):
        self.load_checkpoint()

        self.lr_scheduler.step(max(self.start_epoch - self.warmup_epochs, 0))

        if self.start_epoch or self.start_iteration:
            logger.info(
                f'Restoring from epoch {self.start_epoch + 1} and it. {self.start_iteration}.'
            )

    def preprocess_batch(
            self,
            batch,
    ):
        sample, target, sample_lengths, target_lengths = batch

        return sample.cuda(), target.cuda(), sample_lengths.cuda(
        ), target_lengths.cuda()

    def run(self, train_loader, val_loader=None, num_epochs=1):
        # Trying to restore from a last checkpoint
        self.restore()

        if self.iterations_per_epoch is None:
            self.iterations_per_epoch = len(train_loader)
        else:
            assert len(train_loader) == self.iterations_per_epoch, (
                'len(train_loader) != iterations_per_epoch '
                'found in checkpoint.')

        logger.info(self.model)

        self.start_time = time.time()
        self.time_since_last = time.time(
        )  # reset counter for model checkpoint

        self.num_epochs = num_epochs
        for self.epoch in range(self.start_epoch, self.num_epochs):

            start = time.time()

            train_data_time = self._run_epoch(train_loader,
                                              metrics=self.metrics['train'],
                                              is_train=True)
            train_lap = time.time()

            if val_loader:
                val_data_time = self._run_epoch(val_loader,
                                                metrics=self.metrics['val'],
                                                is_train=False)
                val_lap = time.time()

            tmp = (
                f'Epoch [{self.epoch + 1}/{num_epochs}] - '
                '{phase:>5}: {metrics}. '
                'Data Time: {data_time:.2f}s. Time: {time:.2f}s ({rate:.2f} samples/s)'
            )

            train_time = train_lap - start

            train_string = tmp.format(
                phase='Train',
                metrics=self.metrics['train'].to_str(val=False),
                data_time=train_data_time,
                time=train_time,
                rate=self.metrics['train'].count / train_time)

            logger.info(train_string)

            if val_loader:
                val_time = val_lap - train_lap

                val_string = tmp.format(
                    phase='Val',
                    metrics=self.metrics['val'].to_str(val=False),
                    data_time=val_data_time,
                    time=val_time,
                    rate=self.metrics['val'].count / val_time)

                logger.info(val_string)

            self.save_checkpoint()

            if self.lr_scheduler:
                if not (self.warmup_epochs
                        and self.epoch + 1 < self.warmup_epochs):
                    self.lr_scheduler.step(
                        max(self.epoch + 1 - self.warmup_epochs, 0))

                    logger.info(
                        f"Learning rate set to {self.optimizer.param_groups[0]['lr']:.6f}"
                    )

    def _run_epoch(self, loader, metrics, is_train=False):
        self.model.train(is_train)

        if is_train:
            if loader.sampler and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(self.epoch)

            if loader.batch_sampler and hasattr(loader.batch_sampler,
                                                'set_epoch'):
                loader.batch_sampler.set_epoch(self.epoch)

        if self.start_iteration == 0:
            metrics.reset()

        last_seen = 0
        with tqdm(desc=f'Epoch [{self.epoch + 1}/{self.num_epochs}] - '
                  f'{"Train" if is_train else "Val"}',
                  total=len(loader.dataset),
                  leave=False,
                  miniters=int(0.05 * len(loader.dataset)),
                  maxinterval=3600,
                  unit='samples',
                  disable=(self.local_rank !=
                           0)) as pbar, torch.set_grad_enabled(is_train):

            data_time = 0
            lap = time.time()
            for current_iteration, batch in enumerate(loader):

                # If is the first epoch and start it. is different than curr it., we skip the
                # batches
                if is_train and self.epoch == self.start_epoch and self.start_iteration:
                    if current_iteration < self.start_iteration:
                        continue
                    # Last one, update pbar
                    pbar.set_postfix(metrics.as_dict(), refresh=False)
                    pbar.update(metrics.count - last_seen)
                    last_seen = metrics.count
                    if self.local_rank == 0:
                        pbar.unpause()
                    lap = time.time()
                    self.start_iteration = 0
                    continue

                batch = self.preprocess_batch(batch)
                sample, target, sample_lengths, target_lengths = batch
                data_time += time.time() - lap
                output, output_lengths = self.model(sample, sample_lengths)

                # Ensure float32 loss
                output = output.float()
                loss = self.loss(output, target, output_lengths,
                                 target_lengths)

                if self.label_smoothing:
                    alpha = self.label_smoothing
                    loss = (1 -
                            alpha) * loss + alpha * torch.nn.functional.kl_div(
                                torch.log_softmax(output, dim=-1),
                                torch.empty_like(output).fill_(
                                    1 / output.shape[-1]),
                                reduction='batchmean')

                loss_value = loss.detach()

                if self.distributed:
                    dist.all_reduce(loss_value)
                    loss_value /= self.world_size

                is_valid_loss = check_loss(loss_value)

                if not is_valid_loss:
                    loss_value.zero_()

                metrics.update(
                    (loss_value.repeat(sample.shape[0]), output.detach(),
                     target.detach(), output_lengths.detach(),
                     target_lengths.detach()))

                # Updating progress bar
                pbar.set_postfix(metrics.as_dict(), refresh=False)
                pbar.update((metrics.count - last_seen))
                last_seen = metrics.count

                if not is_train:
                    lap = time.time()
                    continue

                if not is_valid_loss:
                    lap = time.time()
                    if self.local_rank == 0:
                        logger.warning(
                            'Skipping grad update due to invalid loss.')
                    continue

                # LR warmup: for distributed training
                if self.warmup_epochs and self.epoch < self.warmup_epochs:
                    for param_group, lr in zip(self.optimizer.param_groups,
                                               self.base_lrs):

                        param_group['lr'] = (lr - lr / self.world_size) * float(
                            1 + current_iteration +
                            self.epoch * self.iterations_per_epoch) / (
                                self.warmup_epochs * self.iterations_per_epoch
                            ) + lr / self.world_size
                else:
                    param_group = self.optimizer.param_groups[-1]

                pbar.set_postfix_str(
                    (pbar.postfix + f", lr={param_group['lr']:.6f}"),
                    refresh=True)

                # compute gradient
                self.optimizer.zero_grad()

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                # Clipping the norm, avoiding gradient explosion
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), self.clip_grad_norm)
                if self.clip_grad_value:
                    torch.nn.utils.clip_grad_value_(
                        amp.master_params(self.optimizer),
                        self.clip_grad_value)

                # optimizer step
                self.optimizer.step()

                self.save_checkpoint(current_iteration, is_train)

                lap = time.time()

        return data_time

    def save_checkpoint(self, iteration=0, is_train=False):
        if self.local_rank != 0:
            return

        epoch = self.epoch

        # Actually save_checkpoint is called after the iteration ends, so we are really
        # in the next iteration
        current_iteration = iteration + 1 if is_train else self.iterations_per_epoch

        if current_iteration == self.iterations_per_epoch:
            epoch += 1
            current_iteration = 0

        total_iterations = epoch * self.iterations_per_epoch + current_iteration

        if is_train and time.time() - self.time_since_last < self.save_every:
            return

        self.time_since_last = time.time()

        models_dir = os.path.join(self.serialization_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        ckpt_path = os.path.join(models_dir, f'model-{total_iterations}.pth')
        best_ckpt_path = os.path.join(models_dir, f'best-model.pth')

        model = self.model

        klasses = (torch.nn.DataParallel,
                   torch.nn.parallel.DistributedDataParallel,
                   DistributedDataParallel)

        if isinstance(self.model, klasses):
            model = self.model.module

        monitor = self.score(self.metrics['val'])

        chkpt_dict = {
            'model':
            model.state_dict(),
            'epoch':
            epoch,
            'epoch_iterations':
            current_iteration,
            'iterations_per_epoch':
            self.iterations_per_epoch,
            'best_monitor':
            monitor if not is_train and monitor < self.best_monitor else
            self.best_monitor,
            'metrics': {k: m.state_dict()
                        for k, m in self.metrics.items()},
            'optimizer':
            self.optimizer.state_dict(),
        }
        torch.save(chkpt_dict, ckpt_path)

        if not is_train and monitor < self.best_monitor:
            logger.info(f'Best {self.monitor} found ({monitor:.4f} < '
                        f'{self.best_monitor}). Saving...')
            self.best_monitor = monitor

            chkpt_dict['best_monitor'] = monitor
            shutil.copyfile(ckpt_path, best_ckpt_path)

    def load_checkpoint(self):

        if not os.path.exists(os.path.join(self.serialization_dir, 'models')):
            return

        have_checkpoint = (self.serialization_dir is not None and any(
            "model-" in x for x in os.listdir(
                os.path.join(self.serialization_dir, 'models'))))

        if not have_checkpoint:
            return

        serialization_files = glob.glob(
            os.path.join(self.serialization_dir, 'models', '*'))
        model_checkpoints = sorted(
            [f for f in serialization_files if 'model-' in f],
            key=lambda x: int(re.findall(r'-([0-9]+)\.pth', x)[0]))
        total_iterations = int(
            re.search(r"model-([0-9]+)\.pth", model_checkpoints[-1]).group(1))

        ckpt_path = os.path.join(self.serialization_dir, 'models',
                                 f'model-{total_iterations}.pth')

        logger.info(f'Last model checkpoint found in {ckpt_path}. Loading...')

        try:
            ckpt_dict = torch.load(ckpt_path,
                                   map_location=lambda storage, loc: storage)
        except RuntimeError:
            logger.error(
                f'Problem reading file {ckpt_path}. Trying older checkpoints...'
            )
            os.remove(ckpt_path)
            self.load_checkpoint()
            return

        model = self.model

        klasses = (torch.nn.DataParallel,
                   torch.nn.parallel.DistributedDataParallel,
                   DistributedDataParallel)

        if isinstance(self.model, klasses):
            model = model.module

        model.load_state_dict(ckpt_dict['model'])
        self.optimizer.load_state_dict(ckpt_dict['optimizer'])

        self.best_monitor = ckpt_dict['best_monitor']
        for k, state_dict in ckpt_dict['metrics'].items():
            self.metrics[k].load_state_dict(state_dict)

        self.start_epoch = ckpt_dict['epoch']
        self.start_iteration = ckpt_dict['epoch_iterations']
        self.iterations_per_epoch = ckpt_dict['iterations_per_epoch']
