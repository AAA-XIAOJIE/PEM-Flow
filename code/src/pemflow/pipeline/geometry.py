from __future__ import annotations

import cv2
import numpy as np


def solve_camera_pose(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    world_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ok, rotation_vector, translation = cv2.solvePnP(
        np.asarray(world_points_3d, dtype=np.float32),
        np.asarray(image_points_2d, dtype=np.float32),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP did not converge")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_position = (-rotation.T @ translation).reshape(3)
    return (rotation, camera_position)
