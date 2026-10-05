

from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data


def _safe_torch_load(path: str | Path, map_location: str | torch.device = 'cpu') -> Any:
    pass







    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)





def all_reduce_tensor(tensor: torch.Tensor, world_size: int | None = None) -> torch.Tensor:
    pass

    if not dist.is_available() or not dist.is_initialized():
        return tensor
    if world_size is None:
        world_size = dist.get_world_size()
    if world_size < 1:
        raise ValueError('world_size must be positive.')
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= float(world_size)
    return reduced_tensor


def all_reduce_tensors(value: Any, world_size: int | None = None) -> Any:
    pass

    if isinstance(value, list):
        return [all_reduce_tensors(item, world_size=world_size) for item in value]
    if isinstance(value, tuple):
        return tuple(all_reduce_tensors(item, world_size=world_size) for item in value)
    if isinstance(value, dict):
        return {
            key: all_reduce_tensors(item, world_size=world_size)
            for key, item in value.items()
        }
    if isinstance(value, torch.Tensor):
        return all_reduce_tensor(value, world_size=world_size)
    return value





def reset_seed_worker_init_fn(worker_id: int) -> None:
    pass

    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset,
    batch_size: int = 1,
    num_workers: int = 1,
    shuffle: bool | None = None,
    collate_fn=None,
    pin_memory: bool = False,
    drop_last: bool = False,
    distributed: bool = False,
):
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    if num_workers < 0:
        raise ValueError('num_workers cannot be negative.')

    sampler = None
    if distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('distributed=True requires an initialized process group.')
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=bool(shuffle))
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )





def initialize(
    seed: int | None = None,
    cudnn_deterministic: bool = True,
    autograd_anomaly_detection: bool = False,
) -> None:
    pass

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = not cudnn_deterministic
    cudnn.deterministic = cudnn_deterministic
    torch.autograd.set_detect_anomaly(autograd_anomaly_detection)


def release_cuda(value: Any) -> Any:
    pass

    if isinstance(value, list):
        return [release_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(release_cuda(item) for item in value)
    if isinstance(value, dict):
        return {key: release_cuda(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.numpy()
    return value


def to_device(value: Any, device: torch.device | str, non_blocking: bool = False) -> Any:
    pass

    if isinstance(value, list):
        return [to_device(item, device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, dict):
        return {
            key: to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    return value


def to_cuda(value: Any, non_blocking: bool = False) -> Any:
    pass

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required but no CUDA device is available.')
    return to_device(value, torch.device('cuda', torch.cuda.current_device()), non_blocking)


def load_weights(model: torch.nn.Module, snapshot: str | Path):
    pass

    checkpoint = _safe_torch_load(snapshot, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError('Checkpoint model state must be a mapping.')
    incompatible = model.load_state_dict(state_dict, strict=False)
    return set(incompatible.missing_keys), set(incompatible.unexpected_keys)





class CosineAnnealingFunction(Callable):
    def __init__(self, max_epoch: int, eta_min: float = 0.0):
        if max_epoch < 1:
            raise ValueError('max_epoch must be positive.')
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def __call__(self, last_epoch: int) -> float:
        next_epoch = last_epoch + 1
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (
            1.0 + math.cos(math.pi * next_epoch / self.max_epoch)
        )


class WarmUpCosineAnnealingFunction(Callable):
    def __init__(
        self,
        total_steps: int,
        warmup_steps: int,
        eta_init: float = 0.1,
        eta_min: float = 0.1,
    ):
        if total_steps < 1:
            raise ValueError('total_steps must be positive.')
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError('warmup_steps must satisfy 0 <= warmup_steps < total_steps.')
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.normal_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step: int) -> float:
        next_step = last_step + 1
        if self.warmup_steps > 0 and next_step < self.warmup_steps:
            return self.eta_init + (
                (1.0 - self.eta_init) / self.warmup_steps * next_step
            )
        if next_step > self.total_steps:
            return self.eta_min
        next_step -= self.warmup_steps
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (
            1.0 + math.cos(math.pi * next_step / self.normal_steps)
        )


def build_warmup_cosine_lr_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    eta_init: float = 0.1,
    eta_min: float = 0.1,
    grad_acc_steps: int = 1,
):
    if grad_acc_steps < 1:
        raise ValueError('grad_acc_steps must be positive.')
    total_steps = max(1, total_steps // grad_acc_steps)
    warmup_steps = warmup_steps // grad_acc_steps
    cosine_func = WarmUpCosineAnnealingFunction(
        total_steps,
        warmup_steps,
        eta_init=eta_init,
        eta_min=eta_min,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_func)
