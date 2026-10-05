import math

import torch
import torch.nn as nn

from pareconv.modules.ops import index_select
from pareconv.modules.kpconv.kernel_points import load_kernels


class KPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        bias=False,
        dimension=3,
        inf=1e6,
        eps=1e-9,
    ):
        pass
















        super(KPConv, self).__init__()


        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension

        self.inf = inf
        self.eps = eps


        self.weights = nn.Parameter(torch.zeros(self.kernel_size, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()


        kernel_points = self.initialize_kernel_points()
        self.register_buffer('kernel_points', kernel_points)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def initialize_kernel_points(self):
        pass
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed='center')
        return torch.from_numpy(kernel_points).float()

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        pass










        s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)
        neighbors = index_select(s_points, neighbor_indices, dim=0)
        neighbors = neighbors - q_points.unsqueeze(1)


        neighbors = neighbors.unsqueeze(2)
        differences = neighbors - self.kernel_points
        sq_distances = torch.sum(differences ** 2, dim=3)
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)
        neighbor_weights = torch.transpose(neighbor_weights, 1, 2)


        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)


        weighted_feats = weighted_feats.permute(1, 0, 2)
        kernel_outputs = torch.matmul(weighted_feats, self.weights)
        output_feats = torch.sum(kernel_outputs, dim=0, keepdim=False)


        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_feats = output_feats / neighbor_num.unsqueeze(1)


        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'kernel_size: {}'.format(self.kernel_size)
        format_string += ', in_channels: {}'.format(self.in_channels)
        format_string += ', out_channels: {}'.format(self.out_channels)
        format_string += ', radius: {:g}'.format(self.radius)
        format_string += ', sigma: {:g}'.format(self.sigma)
        format_string += ', bias: {}'.format(self.bias is not None)
        format_string += ')'
        return format_string
