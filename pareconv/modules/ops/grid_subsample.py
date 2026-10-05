

from pareconv.modules.ops.gpu_preprocess import voxel_grid_subsample_stacked


def grid_subsample(points, lengths, voxel_size):
    return voxel_grid_subsample_stacked(points, lengths, voxel_size)
