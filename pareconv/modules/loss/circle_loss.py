import torch
import torch.nn.functional as F
import torch.nn as nn


def _two_sided_masked_mean(loss_row, row_masks, loss_col, col_masks):
    pass




    row_values = loss_row[row_masks]
    col_values = loss_col[col_masks]
    if row_values.numel() > 0 and col_values.numel() > 0:
        return (row_values.mean() + col_values.mean()) / 2
    if row_values.numel() > 0:
        return row_values.mean()
    if col_values.numel() > 0:
        return col_values.mean()
    return (loss_row.sum() + loss_col.sum()) * 0.0


def circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
):

    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()


    pos_weights = feat_dists - 1e5 * (~pos_masks).float()
    pos_weights = pos_weights - pos_optimal
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights).detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()
    neg_weights = neg_optimal - neg_weights
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights).detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = _two_sided_masked_mean(loss_row, row_masks, loss_col, col_masks)

    return loss


def weighted_circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
    pos_scales=None,
    neg_scales=None,
):

    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()


    pos_weights = feat_dists - 1e5 * (~pos_masks).float()
    pos_weights = pos_weights - pos_optimal
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()
    neg_weights = neg_optimal - neg_weights
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = _two_sided_masked_mean(loss_row, row_masks, loss_col, col_masks)

    return loss


class CircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(CircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists):
        return circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
        )


class WeightedCircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists, pos_scales=None, neg_scales=None):
        return weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )
