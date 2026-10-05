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


class EpochBasedTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_epoch,
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
        self.max_epoch = int(max_epoch)

    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        pass

    def before_val_epoch(self, epoch) -> None:
        pass

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error(
                f'Epoch={epoch}, iteration={iteration}: invalid gradients. '
                'Saving diagnostic tensors.'
            )
            torch.save(data_dict, osp.join(self.snapshot_dir, 'invalid_data.pth'))
            torch.save(self.model.state_dict(), osp.join(self.snapshot_dir, 'invalid_model.pth'))
            raise FloatingPointError('Non-finite gradients detected.')

    def train_epoch(self):
        if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad(set_to_none=True)
        self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)
        total_iterations = len(self.train_loader)
        if total_iterations == 0:
            raise RuntimeError('The training loader is empty.')

        for zero_based_iteration, data_dict in enumerate(self.train_loader):
            self.inner_iteration = zero_based_iteration + 1
            self.iteration += 1
            data_dict = to_cuda(data_dict, non_blocking=True)
            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()

            output_dict, result_dict = self.train_step(
                self.epoch, self.inner_iteration, data_dict
            )
            if 'loss' not in result_dict:
                raise KeyError('train_step must return result_dict containing "loss".')

            group_start = (zero_based_iteration // self.grad_acc_steps) * self.grad_acc_steps
            group_end = min(group_start + self.grad_acc_steps, total_iterations)
            group_size = group_end - group_start
            (result_dict['loss'] / float(group_size)).backward()

            self.after_backward(
                self.epoch,
                self.inner_iteration,
                data_dict,
                output_dict,
                result_dict,
            )
            self.check_gradients(
                self.epoch,
                self.inner_iteration,
                data_dict,
                output_dict,
                result_dict,
            )
            if self.inner_iteration == group_end:
                self.optimizer_step()

            self.timer.add_process_time()
            self.after_train_step(
                self.epoch,
                self.inner_iteration,
                data_dict,
                output_dict,
                result_dict,
            )
            released = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(released)

            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                self.logger.info(
                    get_log_string(
                        result_dict=summary_dict,
                        epoch=self.epoch,
                        max_epoch=self.max_epoch,
                        iteration=self.inner_iteration,
                        max_iteration=total_iterations,
                        lr=self.get_lr(),
                        timer=self.timer,
                    )
                )
                self.write_event('train', summary_dict, self.iteration)

        self.after_train_epoch(self.epoch)
        train_summary = self.summary_board.summary()
        self.logger.critical(
            get_log_string(train_summary, epoch=self.epoch, timer=self.timer)
        )
        if self.scheduler is not None:
            self.scheduler.step()

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        if total_iterations == 0:
            raise RuntimeError('The validation loader is empty.')
        progress = tqdm.tqdm(
            enumerate(self.val_loader), total=total_iterations, ncols=180
        )

        for iteration, data_dict in progress:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict, non_blocking=True)
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()

            with torch.inference_mode():
                output_dict, result_dict = self.val_step(
                    self.epoch, self.inner_iteration, data_dict
                )
            torch.cuda.synchronize()
            timer.add_process_time()
            self.after_val_step(
                self.epoch,
                self.inner_iteration,
                data_dict,
                output_dict,
                result_dict,
            )
            released = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(released)
            progress.set_description(
                get_log_string(
                    result_dict=summary_board.summary(),
                    epoch=self.epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    timer=timer,
                )
            )

        summary_dict = summary_board.summary()
        self.val_summary = summary_dict
        self.after_val_epoch(self.epoch)
        self.logger.critical(
            '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        )
        self.write_event('val', summary_dict, self.epoch)
        self.set_train_mode()

    def run(self):
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError('Train and validation loaders must be registered.')

        if self.args.resume:
            self.load_snapshot(self.resume_snapshot)
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)

        self.set_train_mode()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            self.inference_epoch()


            self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
            if not self.save_all_snapshots and self.epoch > 1:
                previous = osp.join(
                    self.snapshot_dir, f'epoch-{self.epoch - 1}.pth.tar'
                )
                if osp.exists(previous):
                    os.remove(previous)
        self.finish_tracking()
