

from __future__ import annotations

import torch


def safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pass






    if not logits.is_floating_point():
        raise TypeError('safe_softmax expects a floating-point tensor.')
    valid = torch.isfinite(logits)
    finite_min = torch.finfo(logits.dtype).min
    safe_logits = torch.where(valid, logits, torch.full_like(logits, finite_min))
    max_logits = safe_logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(safe_logits - max_logits) * valid.to(logits.dtype)
    normalizer = exp_logits.sum(dim=dim, keepdim=True)
    return torch.where(
        normalizer > 0,
        exp_logits / normalizer.clamp_min(torch.finfo(logits.dtype).tiny),
        torch.zeros_like(exp_logits),
    )
