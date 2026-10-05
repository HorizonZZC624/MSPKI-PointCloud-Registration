from pathlib import Path

import matplotlib.colors as mpl_colors
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def _require_open3d():
    if o3d is None:
        raise ImportError(
            'open3d is required for visualization and RANSAC evaluation. '
            'Install it with `pip install open3d`.'
        )


def get_color(color_name):
    custom = {
        'custom_yellow': np.asarray([255.0, 204.0, 102.0]) / 255.0,
        'custom_blue': np.asarray([102.0, 153.0, 255.0]) / 255.0,
    }
    if color_name in custom:
        return custom[color_name]
    if color_name not in mpl_colors.CSS4_COLORS:
        raise ValueError(f'Unknown color: {color_name}')
    return np.asarray(mpl_colors.to_rgb(mpl_colors.CSS4_COLORS[color_name]))


def make_scaling_along_axis(points, axis=2, alpha=0):
    points = np.asarray(points)
    if isinstance(axis, int):
        new_axis = np.zeros(3, dtype=np.float64)
        new_axis[axis] = 1.0
        axis = new_axis
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError('axis must be non-zero.')
    axis = axis / norm
    projections = points @ axis
    upper, lower = np.max(projections), np.min(projections)
    span = max(float(upper - lower), 1e-12)
    return 1.0 - ((projections - lower) / span * (1.0 - alpha) + alpha)


def make_open3d_colors(points, base_color, scaling_axis=2, scaling_alpha=0):
    points = np.asarray(points)
    base_color = np.asarray(base_color)
    point_colors = np.ones_like(points) * base_color
    scales = make_scaling_along_axis(
        points, axis=scaling_axis, alpha=scaling_alpha
    )
    return point_colors * scales.reshape(-1, 1)


def make_open3d_point_cloud(points, colors=None, normals=None):
    _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))
    return pcd


def estimate_normals(points):
    pcd = make_open3d_point_cloud(points)
    pcd.estimate_normals()
    return np.asarray(pcd.normals)


def voxel_downsample(points, voxel_size, normals=None):
    pcd = make_open3d_point_cloud(points, normals=normals)
    pcd = pcd.voxel_down_sample(voxel_size)
    output_points = np.asarray(pcd.points)
    if normals is not None:
        return output_points, np.asarray(pcd.normals)
    return output_points


def make_open3d_registration_feature(data):
    _require_open3d()
    features = o3d.pipelines.registration.Feature()
    features.data = np.asarray(data).T
    return features


def make_open3d_axis(axis_vector=None, origin=None, scale=1.0):
    _require_open3d()
    origin = np.zeros((1, 3)) if origin is None else np.asarray(origin).reshape(1, 3)
    if axis_vector is None:
        axis_vector = np.array([[1.0, 0.0, 0.0]])
    axis_vector = np.asarray(axis_vector, dtype=np.float64).reshape(1, 3) * scale
    points = np.concatenate([origin, origin + axis_vector], axis=0)
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    axes.paint_uniform_color(get_color('red'))
    return axes


def make_open3d_axes(axis_vectors=None, origin=None, scale=1.0):
    _require_open3d()
    origin = np.zeros((1, 3)) if origin is None else np.asarray(origin).reshape(1, 3)
    if axis_vectors is None:
        axis_vectors = np.eye(3)
    axis_vectors = np.asarray(axis_vectors, dtype=np.float64) * scale
    points = np.concatenate([origin, origin + axis_vectors], axis=0)
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(
        np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    )
    axes.colors = o3d.utility.Vector3dVector(np.eye(3))
    return axes


def make_open3d_corr_lines(ref_corr_points, src_corr_points, label):
    _require_open3d()
    num_correspondences = ref_corr_points.shape[0]
    corr_points = np.concatenate([ref_corr_points, src_corr_points], axis=0)
    corr_indices = np.asarray(
        [(index, index + num_correspondences) for index in range(num_correspondences)],
        dtype=np.int32,
    )
    corr_lines = o3d.geometry.LineSet()
    corr_lines.points = o3d.utility.Vector3dVector(corr_points)
    corr_lines.lines = o3d.utility.Vector2iVector(corr_indices)
    if label == 'pos':
        corr_lines.paint_uniform_color(np.asarray([0.0, 1.0, 0.0]))
    elif label == 'neg':
        corr_lines.paint_uniform_color(np.asarray([1.0, 0.0, 0.0]))
    else:
        raise ValueError(f'Unsupported correspondence label: {label}')
    return corr_lines


def draw_geometries(*geometries):
    _require_open3d()
    o3d.visualization.draw_geometries(list(geometries))


def draw_and_save(geometries, save_path='output.png', width=1024, height=768):
    pass
    _require_open3d()
    geometries = list(geometries)
    if not geometries:
        raise ValueError('At least one geometry is required.')

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'

    min_bound = np.full(3, np.inf)
    max_bound = np.full(3, -np.inf)
    for index, geometry in enumerate(geometries):
        renderer.scene.add_geometry(f'geometry_{index}', geometry, material)
        bounds = geometry.get_axis_aligned_bounding_box()
        min_bound = np.minimum(min_bound, bounds.get_min_bound())
        max_bound = np.maximum(max_bound, bounds.get_max_bound())

    combined_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    renderer.setup_camera(60.0, combined_bounds, combined_bounds.get_center())
    image = renderer.render_to_image()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(output_path), image):
        raise IOError(f'Failed to save visualization to {output_path}.')
    renderer.scene.clear_geometry()


def _ransac_criteria(num_iterations):

    try:
        return o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(num_iterations), 0.999
        )
    except TypeError:
        return o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(num_iterations), int(num_iterations)
        )


def registration_with_ransac_from_feats(
    src_points,
    ref_points,
    src_feats,
    ref_feats,
    distance_threshold=0.1,
    ransac_n=3,
    num_iterations=50000,
    val_iterations=1000,
):
    del val_iterations
    _require_open3d()
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)
    src_features = make_open3d_registration_feature(src_feats)
    ref_features = make_open3d_registration_feature(ref_feats)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd,
        ref_pcd,
        src_features,
        ref_features,
        True,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=_ransac_criteria(num_iterations),
    )
    return result.transformation


def registration_with_ransac_from_correspondences(
    src_points,
    ref_points,
    correspondences=None,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=10000,
):
    _require_open3d()
    src_points = np.asarray(src_points)
    ref_points = np.asarray(ref_points)
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)
    if correspondences is None:
        indices = np.arange(src_points.shape[0], dtype=np.int32)
        correspondences = np.stack([indices, indices], axis=1)
    correspondence_vector = o3d.utility.Vector2iVector(
        np.asarray(correspondences, dtype=np.int32)
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        ref_pcd,
        correspondence_vector,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=_ransac_criteria(num_iterations),
    )
    return result.transformation
