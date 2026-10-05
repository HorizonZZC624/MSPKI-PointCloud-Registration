from typing import Dict

import torch
from tqdm import tqdm

from pareconv.engine.base_tester import BaseTester
from pareconv.utils.common import get_log_string
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.torch import release_cuda, to_cuda


class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        raise NotImplementedError

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        raise NotImplementedError

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        if self.test_loader is None:
            raise RuntimeError('Test loader has not been registered.')
        if self.model is None:
            raise RuntimeError('Model has not been registered.')

        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        if total_iterations == 0:
            raise RuntimeError('The test loader is empty.')

        with torch.inference_mode():
            pbar = tqdm(enumerate(self.test_loader), total=total_iterations, ncols=180)
            for iteration, data_dict in pbar:
                self.iteration = iteration + 1
                data_dict = to_cuda(data_dict, non_blocking=True)
                self.before_test_step(self.iteration, data_dict)
                torch.cuda.synchronize()
                timer.add_prepare_time()

                output_dict = self.test_step(self.iteration, data_dict)
                torch.cuda.synchronize()
                timer.add_process_time()

                result_dict = self.eval_step(self.iteration, data_dict, output_dict)
                self.after_test_step(
                    self.iteration, data_dict, output_dict, result_dict
                )

                released_result = release_cuda(result_dict)
                summary_board.update_from_result_dict(released_result)
                message = self.summary_string(
                    self.iteration, data_dict, output_dict, released_result
                )
                message += f', {timer.tostring()}'
                self.logger.critical(message)

        self.after_test_epoch()
        summary_dict = summary_board.summary()
        self.logger.critical(
            get_log_string(result_dict=summary_dict, timer=timer)
        )
        return summary_dict
