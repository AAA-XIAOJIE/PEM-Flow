from __future__ import annotations

import cv2
import numpy as np

from pemflow.pipeline.geometry import solve_camera_pose


def warp_points(points, matrix):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    if np.any(np.abs(q[:, 2]) < 1e-12):
        raise ValueError("Homography maps a point to infinity")
    return q[:, :2] / q[:, 2:]


def candidate_matrices(matrices, label_frames, *, reference_index=None):
    frames = sorted(set(map(int, label_frames)))
    if not frames or frames[0] < 1:
        raise ValueError("Positive, one-based label frames are required")
    missing = [f - 1 for f in frames if f - 1 not in matrices]
    if missing:
        raise KeyError(f"Missing production matrix keys: {missing[:10]}")
    bridge = np.eye(3)
    if reference_index is not None:
        if reference_index not in matrices:
            raise KeyError(f"Missing calibration reference key {reference_index}")
        bridge = np.linalg.inv(matrices[reference_index])
    result = {}
    for frame in frames:
        h = bridge @ np.asarray(matrices[frame - 1], dtype=float)
        if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
            raise ValueError(f"Invalid transform for label frame {frame}")
        result[frame] = h / h[2, 2]
    return result


def pixel_to_plane(points, profile, z=0.0):
    camera = profile["camera_parameters"]
    k = np.asarray(camera["matrix"], dtype=float)
    d = np.asarray(camera["distortion"], dtype=float)
    r, c = solve_camera_pose(k, d, profile["world_points"], profile["image_points"])
    uv = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, k, d).reshape(-1, 2)
    rays = np.column_stack([rays, np.ones(len(rays))]) @ r
    if np.any(np.abs(rays[:, 2]) < 1e-09):
        raise ValueError("Ray parallel to projection plane")
    xyz = c + ((float(z) - c[2]) / rays[:, 2])[:, None] * rays
    return xyz[:, :2]


def residual_summary(current_points, reference_points, matrix, profile=None):
    q = warp_points(current_points, matrix)
    reference = np.asarray(reference_points, dtype=float)
    delta = q - reference
    distance = np.linalg.norm(delta, axis=1)
    if len(distance) == 0:
        raise ValueError("Cannot summarise an empty correspondence set")
    result = dict(
        points=len(distance),
        median_px=float(np.median(distance)),
        p95_px=float(np.quantile(distance, 0.95)),
        max_px=float(distance.max()),
        signed_dx_median_px=float(np.median(delta[:, 0])),
        signed_dy_median_px=float(np.median(delta[:, 1])),
    )
    if profile is not None:
        metres = np.linalg.norm(
            pixel_to_plane(q, profile) - pixel_to_plane(reference, profile), axis=1
        )
        result.update(
            road_plane_equivalent_median_m=float(np.median(metres)),
            road_plane_equivalent_p95_m=float(np.quantile(metres, 0.95)),
        )
    return result
