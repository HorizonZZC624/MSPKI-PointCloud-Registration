import os
import os.path as osp
from typing import Dict, Tuple

import torch
import tqdm

from pareconv.engine.base_trainer import BaseTrainer
from pareconv.utils.common import get_log_string
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.torch import to_cuda


class CycleLoader:
    def __init__(self, data_loader, epoch, distributed):
        self.data_loader = data_loader
        self.last_epoch = epoch
        self.distributed = distributed
        self.iterator = self.initialize_iterator()

    def initialize_iterator(self):
        if self.distributed and hasattr(self.data_loader.sampler, 'set_epoch'):
            self.data_loader.sampler.set_epoch(self.last_epoch + 1)
        return iter(self.data_loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.last_epoch += 1
            self.iterator = self.initialize_iterator()
            return next(self.iterator)


class IterBasedTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_iteration,
        snapshot_steps,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_iteration = int(max_iteration)
        self.snapshot_steps = int(snapshot_steps)
        if self.max_iteration < 1 or self.snapshot_steps < 1:
            raise ValueError('max_iteration and snapshot_steps must be positive.')

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_val(self):
        pass

    def after_val(self):
        pass

    def before_train_step(self, iteration, data_dict):
        pass

    def before_val_step(self, iteration, data_dict):
        pass

    def after_train_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_val_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def train_step(self, iteration, data_dict) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def val_step(self, iteration, data_dict) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def after_backward(self, iteration, data_dict, output_dict, result_dict):
        pass

    def check_gradients(self, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            torch.save(data_dict, osp.join(self.snapshot_dir, 'invalid_data.pth'))
            torch.save(
                self.model.state_dict(),
                osp.join(self.snapshot_dir, 'invalid_model.pth'),
            )
            raise FloatingPointError(
                f'Non-finite gradients detected at iteration={iteration}.'
            )

    def inference(self):
        self.set_eval_mode()
        self.before_val()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        if total_iterations == 0:
            raise RuntimeError('The validation loader is empty.')
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict, non_blocking=True)
            self.before_val_step(self.inner_iteration, data_dict)
            timer.add_prepare_time()
            with torch.inference_mode():
                output_dict, result_dict = self.val_step(
                    self.inner_iteration, data_dict
                )
            timer.add_process_time()
            self.after_val_step(
                self.inner_iteration, data_dict, output_dict, result_dict
            )
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            pbar.set_description(
                get_log_string(
                    result_dict=summary_board.summary(),
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    timer=timer,
                )
            )
        self.after_val()
        summary_dict = summary_board.summary()
        self.logger.critical(
            '[Val] '
            + get_log_string(
                summary_dict, iteration=self.iteration, timer=timer
            )
        )
        self.write_event(
            'val', summary_dict, self.iteration // self.snapshot_steps
        )
        self.set_train_mode()

    def run(self):
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError('Train and validation loaders must be registered.')

        if self.args.resume:
            self.load_snapshot(self.resume_snapshot)
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        self.set_train_mode()

        self.summary_board.reset_all()
        self.timer.reset()
        train_loader = CycleLoader(self.train_loader, self.epoch, self.distributed)
        self.before_train()
        self.optimizer.zero_grad(set_to_none=True)

        while self.iteration < self.max_iteration:
            self.iteration += 1
            data_dict = to_cuda(next(train_loader), non_blocking=True)
            self.before_train_step(self.iteration, data_dict)
            self.timer.add_prepare_time()
            output_dict, result_dict = self.train_step(self.iteration, data_dict)
            if 'loss' not in result_dict:
                raise KeyError('train_step must return a loss.')

            (result_dict['loss'] / float(self.grad_acc_steps)).backward()
            self.after_backward(
                self.iteration, data_dict, output_dict, result_dict
            )
            self.check_gradients(
                self.iteration, data_dict, output_dict, result_dict
            )
            is_last = self.iteration == self.max_iteration
            if self.iteration % self.grad_acc_steps == 0 or is_last:
                self.optimizer_step()
                if self.scheduler is not None:
                    self.scheduler.step()

            self.timer.add_process_time()
            self.after_train_step(
                self.iteration, data_dict, output_dict, result_dict
            )
            released = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(released)

            if self.iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                self.logger.info(
                    get_log_string(
                        result_dict=summary_dict,
                        iteration=self.iteration,
                        max_iteration=self.max_iteration,
                        lr=self.get_lr(),
                        timer=self.timer,
                    )
                )
                self.write_event('train', summary_dict, self.iteration)

            if self.iteration % self.snapshot_steps == 0 or is_last:
                self.epoch = train_loader.last_epoch
                self.inference()
                self.save_snapshot(f'iter-{self.iteration}.pth.tar')
                if not self.save_all_snapshots:
                    previous = osp.join(
                        self.snapshot_dir,
                        f'iter-{self.iteration - self.snapshot_steps}.pth.tar',
                    )
                    if osp.exists(previous):
                        os.remove(previous)

        self.after_train()
        self.logger.critical('Training finished.')
        self.finish_tracking()
