from typing import Tuple, List, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation





def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    pass
    q_points = np.asarray(q_points)
    s_points = np.asarray(s_points)
    if q_points.ndim != 2 or s_points.ndim != 2:
        raise ValueError('q_points and s_points must be two-dimensional arrays.')
    if q_points.shape[1] != s_points.shape[1]:
        raise ValueError('Query/support point dimensions must match.')
    if s_points.shape[0] == 0:
        raise ValueError('Support points must not be empty.')
    if q_points.shape[0] == 0:
        distances = np.empty((0,), dtype=np.float64)
        indices = np.empty((0,), dtype=np.int64)
        return (distances, indices) if return_index else distances
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1)
    if return_index:
        return distances, indices
    else:
        return distances


def regularize_normals(points, normals, positive=True):
    pass




    dot_products = -(points * normals).sum(axis=1, keepdims=True)
    direction = dot_products > 0
    if positive:
        normals = normals * direction - normals * (1 - direction)
    else:
        normals = normals * (1 - direction) - normals * direction
    return normals





def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def compose_transforms(transforms: List[np.ndarray]) -> np.ndarray:
    pass



    if not transforms:
        raise ValueError('transforms must contain at least one matrix.')
    final_transform = np.asarray(transforms[0]).copy()
    for transform in transforms[1:]:
        final_transform = np.matmul(transform, final_transform)
    return final_transform


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pass








    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass








    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    pass







    rotation, translation = get_rotation_translation_from_transform(transform)
    inv_rotation = rotation.T
    inv_translation = -np.matmul(inv_rotation, translation)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)
    return inv_transform


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:

    euler = np.random.rand(3) * np.pi * 2 * rotation_factor
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def random_sample_rotation_v2() -> np.ndarray:
    axis = np.random.rand(3) - 0.5
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    theta = np.pi * np.random.rand()
    euler = axis * theta
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation.astype(np.float32)


def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform





def random_sample_keypoints(
    points: np.ndarray,
    feats: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.random.choice(num_points, num_keypoints, replace=False)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_scores(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.argsort(-scores)[:num_keypoints]
        points = points[indices]
        feats = feats[indices]
    return points, feats


def random_sample_keypoints_with_scores(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.arange(num_points)
        scores = np.nan_to_num(np.asarray(scores), nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.clip(scores, 0.0, None)
        total = scores.sum()
        probs = None if total <= 0 else scores / total
        indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_nms(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if len(indices) == num_keypoints:
                    break
                if i + 1 < num_points:
                    current_masks = np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
                    masks[i + 1 :] = masks[i + 1 :] & ~current_masks
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats


def random_sample_keypoints_with_nms(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if i + 1 < num_points:
                    current_masks = np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
                    masks[i + 1 :] = masks[i + 1 :] & ~current_masks
        indices = np.array(indices)
        if len(indices) > num_keypoints:
            sorted_scores = scores[sorted_indices]
            scores = sorted_scores[indices]
            probs = scores / np.sum(scores)
            indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats

def uniform_2_sphere(num: int = None):
    pass











    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)





def convert_depth_mat_to_points(
    depth_mat: np.ndarray, intrinsics: np.ndarray, scaling_factor: float = 1000.0, distance_limit: float = 6.0
):
    pass









    focal_x = intrinsics[0, 0]
    focal_y = intrinsics[1, 1]
    center_x = intrinsics[0, 2]
    center_y = intrinsics[1, 2]
    height, width = depth_mat.shape
    coords = np.arange(height * width)
    u = coords % width
    v = coords // width
    depth = depth_mat.flatten()
    z = depth / scaling_factor
    z[z > distance_limit] = 0.0
    x = (u - center_x) * z / focal_x
    y = (v - center_y) * z / focal_y
    points = np.stack([x, y, z], axis=1)
    points = points[depth > 0]
    return points
