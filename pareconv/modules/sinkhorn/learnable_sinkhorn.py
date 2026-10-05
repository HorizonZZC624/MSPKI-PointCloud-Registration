import torch
import torch.nn as nn


class LearnableLogOptimalTransport(nn.Module):
    def __init__(self, num_iterations, inf=1e12):
        pass
        super().__init__()
        if num_iterations < 1:
            raise ValueError('num_iterations must be positive.')
        self.num_iterations = int(num_iterations)
        self.register_parameter('alpha', nn.Parameter(torch.tensor(1.0)))
        self.inf = float(inf)

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks=None, col_masks=None):
        if scores.ndim != 3:
            raise ValueError('scores must have shape (B, M, N).')
        scores = torch.nan_to_num(scores, nan=0.0, posinf=self.inf, neginf=-self.inf)
        batch_size, num_row, num_col = scores.shape
        device = scores.device

        if row_masks is None:
            row_masks = torch.ones(
                (batch_size, num_row), dtype=torch.bool, device=device
            )
        else:
            row_masks = row_masks.to(device=device, dtype=torch.bool)
        if col_masks is None:
            col_masks = torch.ones(
                (batch_size, num_col), dtype=torch.bool, device=device
            )
        else:
            col_masks = col_masks.to(device=device, dtype=torch.bool)

        padded_row_masks = torch.zeros(
            (batch_size, num_row + 1), dtype=torch.bool, device=device
        )
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(
            (batch_size, num_col + 1), dtype=torch.bool, device=device
        )
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = padded_row_masks.unsqueeze(2) | padded_col_masks.unsqueeze(1)

        alpha = self.alpha.to(dtype=scores.dtype)
        padded_col = alpha.expand(batch_size, num_row, 1)
        padded_row = alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat(
            [torch.cat([scores, padded_col], dim=-1), padded_row], dim=1
        )
        padded_scores = padded_scores.masked_fill(padded_score_masks, -self.inf)

        num_valid_row = row_masks.to(scores.dtype).sum(1)
        num_valid_col = col_masks.to(scores.dtype).sum(1)
        total_valid = (num_valid_row + num_valid_col).clamp_min(1.0)
        norm = -torch.log(total_valid)

        log_mu = torch.empty(
            (batch_size, num_row + 1), dtype=scores.dtype, device=device
        )
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col.clamp_min(1.0)) + norm
        log_mu = log_mu.masked_fill(padded_row_masks, -self.inf)

        log_nu = torch.empty(
            (batch_size, num_col + 1), dtype=scores.dtype, device=device
        )
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row.clamp_min(1.0)) + norm
        log_nu = log_nu.masked_fill(padded_col_masks, -self.inf)

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm[:, None, None]
        return torch.nan_to_num(outputs, nan=-self.inf, posinf=self.inf, neginf=-self.inf)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_iterations={self.num_iterations})'
