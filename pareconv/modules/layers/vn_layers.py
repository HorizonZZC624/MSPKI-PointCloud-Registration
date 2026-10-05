import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        pass
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        out_channels = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d * d).sum(2, keepdim=True).clamp_min(EPS)
        reflected = x - (dotprod / d_norm_sq) * d
        return self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * reflected
        )


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        negative_slope=0.2,
    ):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        direction_channels = 1 if share_nonlinearity else out_channels
        self.map_to_dir = nn.Linear(in_channels, direction_channels, bias=False)

    def forward(self, x):
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


        p = F.normalize(p, p=2, dim=2, eps=EPS)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)

        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).to(p.dtype)
        d_norm_sq = (d * d).sum(2, keepdim=True).clamp_min(EPS)
        reflected = p - (dotprod / d_norm_sq) * d
        return self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * reflected
        )


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        use_batchnorm='norm',
        negative_slope=0.2,
    ):
        super().__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(
            out_channels,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

    def forward(self, x):
        x = self.linear(x)
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        return self.leaky_relu(x)


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
        else:
            raise ValueError(f'Unsupported VNBatchNorm dim={dim}.')

    def forward(self, x):
        if self.num_features == 1:
            return x
        norm = torch.linalg.norm(x, dim=2).clamp_min(EPS)
        norm_bn = self.bn(norm)
        return x / norm.unsqueeze(2) * norm_bn.unsqueeze(2)


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        if x.ndim < 4:
            raise ValueError('VNMaxPool expects a sample dimension at the end.')
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        indices = dotprod.argmax(dim=-1, keepdim=True)
        gather_index = indices.expand(*x.shape[:-1], 1)
        return torch.gather(x, dim=-1, index=gather_index).squeeze(-1)


class VNStdFeature(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=4,
        normalize_frame=False,
        share_nonlinearity=False,
        negative_slope=0.2,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn_lin = nn.Linear(
            in_channels // 4,
            2 if normalize_frame else 3,
            bias=False,
        )

    def forward(self, x):
        z0 = self.vn1(x)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            u1 = v1 / torch.linalg.norm(v1, dim=1, keepdim=True).clamp_min(EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
            u2 = v2 / torch.linalg.norm(v2, dim=1, keepdim=True).clamp_min(EPS)
            u3 = torch.cross(u1, u2, dim=1)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        else:
            raise NotImplementedError(f'Unsupported VNStdFeature dim={self.dim}.')
        return x_std, z0
