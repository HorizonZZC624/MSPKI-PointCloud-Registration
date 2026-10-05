from typing import Optional

import torch
import torch.nn.functional as F


def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    pass























    if points.shape[-1] != 3:
        raise ValueError(f'points must end with coordinate dimension 3, got {tuple(points.shape)}')
    if normals is not None and points.shape != normals.shape:
        raise ValueError('points and normals must have identical shapes.')
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]
        translation = transform[:, None, :3, 3]
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points


def apply_rotation(points: torch.Tensor, rotation: torch.Tensor, normals: Optional[torch.Tensor] = None):
    pass




















    if points.shape[-1] != 3:
        raise ValueError(f'points must end with coordinate dimension 3, got {tuple(points.shape)}')
    if normals is not None and points.shape != normals.shape:
        raise ValueError('points and normals must have identical shapes.')
    if rotation.ndim == 2:
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2))
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif rotation.ndim == 3 and points.ndim == 3:
        points = torch.matmul(points, rotation.transpose(-1, -2))
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and rotation{}.'.format(tuple(points.shape), tuple(rotation.shape))
        )
    if normals is not None:
        return points, normals
    else:
        return points


def get_rotation_translation_from_transform(transform):
    pass








    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation


def get_transform_from_rotation_translation(rotation, translation):
    pass








    input_shape = rotation.shape
    rotation = rotation.reshape(-1, 3, 3)
    translation = translation.reshape(-1, 3)
    transform = torch.eye(4).to(rotation).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    output_shape = input_shape[:-2] + (4, 4)
    transform = transform.reshape(*output_shape)
    return transform


def inverse_transform(transform):
    pass







    rotation, translation = get_rotation_translation_from_transform(transform)
    inv_rotation = rotation.transpose(-1, -2)
    inv_translation = -torch.matmul(inv_rotation, translation.unsqueeze(-1)).squeeze(-1)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)
    return inv_transform


def skew_symmetric_matrix(inputs):
    pass











    input_shape = inputs.shape
    output_shape = input_shape[:-1] + (3, 3)
    skews = torch.zeros(output_shape, device=inputs.device, dtype=inputs.dtype)
    skews[..., 0, 1] = -inputs[..., 2]
    skews[..., 0, 2] = inputs[..., 1]
    skews[..., 1, 0] = inputs[..., 2]
    skews[..., 1, 2] = -inputs[..., 0]
    skews[..., 2, 0] = -inputs[..., 1]
    skews[..., 2, 1] = inputs[..., 0]
    return skews


def rodrigues_rotation_matrix(axes, angles):
    pass











    input_shape = axes.shape
    axes = axes.reshape(-1, 3)
    angles = angles.reshape(-1)
    axes = F.normalize(axes, p=2, dim=1)
    skews = skew_symmetric_matrix(axes)
    sin_values = torch.sin(angles).view(-1, 1, 1)
    cos_values = torch.cos(angles).view(-1, 1, 1)
    eyes = torch.eye(3, device=skews.device, dtype=skews.dtype).unsqueeze(0).expand_as(skews)
    rotations = eyes + sin_values * skews + (1.0 - cos_values) * torch.matmul(skews, skews)
    output_shape = input_shape[:-1] + (3, 3)
    rotations = rotations.reshape(*output_shape)
    return rotations


def rodrigues_alignment_matrix(src_vectors, tgt_vectors):
    pass




    if src_vectors.shape != tgt_vectors.shape or src_vectors.shape[-1] != 3:
        raise ValueError('src_vectors and tgt_vectors must have identical shape (..., 3).')
    input_shape = src_vectors.shape
    src = F.normalize(src_vectors.reshape(-1, 3), dim=-1, p=2, eps=1e-8)
    tgt = F.normalize(tgt_vectors.reshape(-1, 3), dim=-1, p=2, eps=1e-8)
    batch = src.shape[0]
    eye = torch.eye(3, device=src.device, dtype=src.dtype).unsqueeze(0).repeat(batch, 1, 1)

    cross = torch.cross(src, tgt, dim=-1)
    sin = torch.linalg.norm(cross, dim=-1)
    cos = (src * tgt).sum(dim=-1).clamp(-1.0, 1.0)
    rotations = eye.clone()

    general = sin > 1e-7
    if general.any():
        axis = cross[general] / sin[general, None]
        skew = skew_symmetric_matrix(axis)
        sin_g = sin[general, None, None]
        cos_g = cos[general, None, None]
        eye_g = eye[general]
        rotations[general] = (
            eye_g + sin_g * skew + (1.0 - cos_g) * (skew @ skew)
        )

    opposite = (~general) & (cos < 0.0)
    if opposite.any():
        src_o = src[opposite]

        basis_indices = src_o.abs().argmin(dim=-1)
        basis = torch.zeros_like(src_o)
        basis.scatter_(1, basis_indices[:, None], 1.0)
        axis = F.normalize(torch.cross(src_o, basis, dim=-1), dim=-1, eps=1e-8)

        rotations[opposite] = 2.0 * axis[:, :, None] * axis[:, None, :] - eye[opposite]

    return rotations.reshape(*input_shape[:-1], 3, 3)
