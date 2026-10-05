import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pareconv.utils.pointcloud import (
    apply_transform,
    get_nearest_neighbor,
    get_rotation_translation_from_transform,
)





def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    pass










    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    pass










    return np.linalg.norm(gt_translation - est_translation)


def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
    pass









    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte


def compute_rotation_mse_and_mae(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    pass
    gt_euler_angles = Rotation.from_matrix(gt_rotation).as_euler('xyz', degrees=True)
    est_euler_angles = Rotation.from_matrix(est_rotation).as_euler('xyz', degrees=True)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae


def compute_translation_mse_and_mae(gt_translation: np.ndarray, est_translation: np.ndarray):
    pass
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae


def compute_transform_mse_and_mae(gt_transform: np.ndarray, est_transform: np.ndarray):
    pass
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def compute_registration_rmse(src_points: np.ndarray, gt_transform: np.ndarray, est_transform: np.ndarray):
    pass











    gt_points = apply_transform(src_points, gt_transform)
    est_points = apply_transform(src_points, est_transform)
    squared_errors = np.sum((gt_points - est_points) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_errors))) if squared_errors.size else 0.0


def compute_modified_chamfer_distance(
    raw_points: np.ndarray,
    ref_points: np.ndarray,
    src_points: np.ndarray,
    gt_transform: np.ndarray,
    est_transform: np.ndarray,
):
    pass

    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = get_nearest_neighbor(aligned_src_points, raw_points).mean()

    composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = get_nearest_neighbor(ref_points, aligned_raw_points).mean()

    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance


def compute_correspondence_residual(ref_corr_points, src_corr_points, transform):
    pass
    if len(ref_corr_points) == 0:
        return 0.0
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    return float(np.mean(residuals))


def compute_inlier_ratio(ref_corr_points, src_corr_points, transform, positive_radius=0.1):
    pass
    if len(ref_corr_points) == 0:
        return 0.0
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    return float(np.mean(residuals < positive_radius))



def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    pass
    if len(ref_points) == 0 or len(src_points) == 0:
        return 0.0
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    return float(np.mean(nn_distances < positive_radius))
def compute_overlap_mask(ref_points, src_points, transform=None, positive_radius=0.1):
    pass
    if len(ref_points) == 0 or len(src_points) == 0:
        return (
            np.zeros(len(ref_points), dtype=bool),
            np.zeros(len(src_points), dtype=bool),
        )
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    ref_nn_distances = get_nearest_neighbor(ref_points, src_points)
    ref_overlap_mask = ref_nn_distances < positive_radius
    src_nn_distances = get_nearest_neighbor(src_points, ref_points)
    src_overlap_mask = src_nn_distances < positive_radius
    return ref_overlap_mask, src_overlap_mask




def get_correspondences(ref_points, src_points, transform, matching_radius):
    pass



    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    pairs = [(i, j) for i, indices in enumerate(indices_list) for j in indices]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)





def extract_corr_indices_from_feats(
    ref_feats: np.ndarray,
    src_feats: np.ndarray,
    mutual: bool = False,
    bilateral: bool = False,
):
    pass











    ref_feats = np.asarray(ref_feats)
    src_feats = np.asarray(src_feats)
    if ref_feats.ndim != 2 or src_feats.ndim != 2:
        raise ValueError('Feature arrays must have shape (N, C) and (M, C).')
    if ref_feats.shape[1] != src_feats.shape[1]:
        raise ValueError('Reference/source feature dimensions must match.')
    if ref_feats.shape[0] == 0 or src_feats.shape[0] == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy()
    ref_nn_indices = get_nearest_neighbor(ref_feats, src_feats, return_index=True)[1]
    if mutual or bilateral:
        src_nn_indices = get_nearest_neighbor(src_feats, ref_feats, return_index=True)[1]
        ref_indices = np.arange(ref_feats.shape[0])
        if mutual:
            ref_masks = np.equal(src_nn_indices[ref_nn_indices], ref_indices)
            ref_corr_indices = ref_indices[ref_masks]
            src_corr_indices = ref_nn_indices[ref_corr_indices]
        else:
            src_indices = np.arange(src_feats.shape[0])
            ref_corr_indices = np.concatenate([ref_indices, src_nn_indices], axis=0)
            src_corr_indices = np.concatenate([ref_nn_indices, src_indices], axis=0)
    else:
        ref_corr_indices = np.arange(ref_feats.shape[0])
        src_corr_indices = ref_nn_indices
    return ref_corr_indices, src_corr_indices


def extract_correspondences_from_feats(
    ref_points: np.ndarray,
    src_points: np.ndarray,
    ref_feats: np.ndarray,
    src_feats: np.ndarray,
    mutual: bool = False,
    return_feat_dist: bool = False,
):
    pass
    ref_corr_indices, src_corr_indices = extract_corr_indices_from_feats(ref_feats, src_feats, mutual=mutual)

    ref_corr_points = ref_points[ref_corr_indices]
    src_corr_points = src_points[src_corr_indices]
    outputs = [ref_corr_points, src_corr_points]
    if return_feat_dist:
        ref_corr_feats = ref_feats[ref_corr_indices]
        src_corr_feats = src_feats[src_corr_indices]
        feat_dists = np.linalg.norm(ref_corr_feats - src_corr_feats, axis=1)
        outputs.append(feat_dists)
    return outputs





def evaluate_correspondences(ref_points, src_points, transform, positive_radius=0.1):
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=positive_radius)
    inlier_ratio = compute_inlier_ratio(ref_points, src_points, transform, positive_radius=positive_radius)
    residual = compute_correspondence_residual(ref_points, src_points, transform)

    return {
        'overlap': overlap,
        'inlier_ratio': inlier_ratio,
        'residual': residual,
        'num_corr': ref_points.shape[0],
    }


def evaluate_sparse_correspondences(ref_points, src_points, ref_corr_indices, src_corr_indices, gt_corr_indices):
    ref_corr_indices = np.asarray(ref_corr_indices, dtype=np.int64).reshape(-1)
    src_corr_indices = np.asarray(src_corr_indices, dtype=np.int64).reshape(-1)
    gt_corr_indices = np.asarray(gt_corr_indices, dtype=np.int64).reshape(-1, 2)
    if ref_corr_indices.shape != src_corr_indices.shape:
        raise ValueError('Predicted reference/source correspondence indices must match.')
    if np.any(ref_corr_indices < 0) or np.any(ref_corr_indices >= len(ref_points)):
        raise IndexError('Reference correspondence index out of range.')
    if np.any(src_corr_indices < 0) or np.any(src_corr_indices >= len(src_points)):
        raise IndexError('Source correspondence index out of range.')
    ref_gt_corr_indices = gt_corr_indices[:, 0]
    src_gt_corr_indices = gt_corr_indices[:, 1]

    gt_corr_mat = np.zeros((ref_points.shape[0], src_points.shape[0]))
    gt_corr_mat[ref_gt_corr_indices, src_gt_corr_indices] = 1.0
    num_gt_correspondences = gt_corr_mat.sum()

    pred_corr_mat = np.zeros_like(gt_corr_mat)
    pred_corr_mat[ref_corr_indices, src_corr_indices] = 1.0
    num_pred_correspondences = pred_corr_mat.sum()

    pos_corr_mat = gt_corr_mat * pred_corr_mat
    num_pos_correspondences = pos_corr_mat.sum()

    precision = num_pos_correspondences / (num_pred_correspondences + 1e-12)
    recall = num_pos_correspondences / (num_gt_correspondences + 1e-12)

    pos_corr_mat = pos_corr_mat > 0
    gt_corr_mat = gt_corr_mat > 0
    ref_hit_ratio = np.any(pos_corr_mat, axis=1).sum() / (np.any(gt_corr_mat, axis=1).sum() + 1e-12)
    src_hit_ratio = np.any(pos_corr_mat, axis=0).sum() / (np.any(gt_corr_mat, axis=0).sum() + 1e-12)
    hit_ratio = 0.5 * (ref_hit_ratio + src_hit_ratio)

    return {
        'precision': precision,
        'recall': recall,
        'hit_ratio': hit_ratio,
    }
