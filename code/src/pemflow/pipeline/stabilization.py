from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def _project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
        return homogeneous[:, :2] / homogeneous[:, 2:]


def _distance_summary(distances: np.ndarray) -> dict:
    values = np.asarray(distances, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "median_px": float(np.median(finite)) if len(finite) else None,
        "p95_px": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "max_px": float(finite.max()) if len(finite) else None,
    }


def _grid_coverage(points: np.ndarray, width: int, height: int) -> dict:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    valid = points[
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    ]
    cells = np.floor(valid / [width, height] * 4).astype(int)
    occupied = len(np.unique(cells, axis=0)) if len(cells) else 0
    return {"rows": 4, "columns": 4, "occupied_cells": occupied, "fraction": occupied / 16}


def _frame_diagnostics(
    *,
    frame_index,
    reference_frame_index,
    reference_keypoints,
    current_keypoints,
    current_xy,
    reference_xy,
    inlier_mask,
    matrix,
    previous_matrix,
    width,
    height,
    status,
    reason,
    reused_from_frame_index,
    elapsed_seconds,
):
    current_xy = np.asarray(current_xy, dtype=float).reshape(-1, 2)
    reference_xy = np.asarray(reference_xy, dtype=float).reshape(-1, 2)
    inliers = (
        np.asarray(inlier_mask).ravel().astype(bool)
        if inlier_mask is not None
        else np.zeros(len(current_xy), dtype=bool)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        residuals = np.linalg.norm(_project_points(current_xy, matrix) - reference_xy, axis=1)
    finite = bool(np.isfinite(matrix).all())
    determinant = condition = None
    if finite:
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            determinant = float(np.linalg.det(matrix))
            try:
                condition = float(np.linalg.cond(matrix))
            except np.linalg.LinAlgError:
                pass
        determinant = determinant if np.isfinite(determinant) else None
        condition = condition if condition is not None and np.isfinite(condition) else None
    corner_jump = None
    if previous_matrix is not None:
        corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        with np.errstate(invalid="ignore", over="ignore"):
            corner_jump = _distance_summary(
                np.linalg.norm(
                    _project_points(corners, matrix) - _project_points(corners, previous_matrix),
                    axis=1,
                )
            )
    return {
        "frame_index": int(frame_index),
        "reference_frame_index": int(reference_frame_index),
        "index_convention": "zero_based_source_video",
        "image_width_px": int(width),
        "image_height_px": int(height),
        "reference_keypoints": int(reference_keypoints),
        "current_keypoints": int(current_keypoints),
        "good_matches": int(len(current_xy)),
        "ransac_inlier_count": int(inliers.sum()) if inlier_mask is not None else None,
        "ransac_inlier_ratio": float(inliers.mean())
        if inlier_mask is not None and len(inliers)
        else None,
        "all_match_residual": _distance_summary(residuals),
        "inlier_residual": _distance_summary(residuals[inliers]),
        "inlier_grid_coverage": {
            "current": _grid_coverage(current_xy[inliers], width, height),
            "reference": _grid_coverage(reference_xy[inliers], width, height),
        },
        "matrix_finite": finite,
        "matrix_determinant": determinant,
        "matrix_condition_number": condition,
        "corner_jump_from_previous_px": corner_jump,
        "status": status,
        "fallback_reason": reason,
        "reused_from_frame_index": reused_from_frame_index,
        "elapsed_seconds": float(elapsed_seconds),
    }


def generate_stabilization_matrices(
    video_path: str | Path,
    reference_frame_index: int,
    *,
    nfeatures: int = 2000,
    ratio_threshold: float = 0.7,
    minimum_matches: int = 21,
    ransac_threshold_px: float = 3.0,
    show_progress: bool = False,
    random_seed: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
    diagnostics_callback: Callable[[dict], None] | None = None,
) -> dict[int, np.ndarray]:
    cv2.setRNGSeed(int(random_seed))
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 <= reference_frame_index < frame_count:
            raise ValueError(
                f"Reference frame {reference_frame_index} outside 0..{frame_count - 1}"
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_index)
        ok, reference = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read reference frame {reference_frame_index}")
        detector = cv2.SIFT_create(nfeatures=nfeatures)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        reference_points, reference_descriptors = detector.detectAndCompute(reference_gray, None)
        if reference_descriptors is None:
            raise RuntimeError("No SIFT descriptors in reference frame")
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        result: dict[int, np.ndarray] = {}
        last_valid = np.eye(3, dtype=float)
        last_valid_frame = None
        previous_matrix = None
        frame_indices = range(frame_count)
        if show_progress:
            from tqdm import tqdm

            frame_indices = tqdm(frame_indices, total=frame_count, unit="frame", desc="Stabilizing")
        for frame_index in frame_indices:
            started = perf_counter() if diagnostics_callback is not None else None
            ok, frame = capture.read()
            if not ok:
                break
            current_xy = reference_xy = np.empty((0, 1, 2), dtype=np.float32)
            inlier_mask = None
            reason = None
            reused_from = last_valid_frame
            if frame_index == reference_frame_index:
                matrix = np.eye(3, dtype=float)
                last_valid = matrix
                last_valid_frame = frame_index
                point_count = len(reference_points)
                status = "reference"
                reused_from = None
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                points, descriptors = detector.detectAndCompute(gray, None)
                point_count = len(points) if points is not None else 0
                matrix = last_valid
                status = (
                    "reused_previous" if last_valid_frame is not None else "reused_initial_identity"
                )
                if descriptors is None:
                    reason = "missing_descriptors"
                elif len(points) <= 10:
                    reason = "insufficient_keypoints"
                else:
                    pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
                    good = [
                        first
                        for pair in pairs
                        if len(pair) == 2
                        for first, second in [pair]
                        if first.distance < ratio_threshold * second.distance
                    ]
                    if len(good) >= minimum_matches or diagnostics_callback is not None:
                        reference_xy = np.float32(
                            [reference_points[item.queryIdx].pt for item in good]
                        ).reshape(-1, 1, 2)
                        current_xy = np.float32(
                            [points[item.trainIdx].pt for item in good]
                        ).reshape(-1, 1, 2)
                    if len(good) >= minimum_matches:
                        candidate, inlier_mask = cv2.findHomography(
                            current_xy, reference_xy, cv2.RANSAC, ransac_threshold_px
                        )
                        if candidate is not None:
                            matrix = candidate
                            last_valid = matrix
                            last_valid_frame = frame_index
                            status = "estimated"
                            reused_from = None
                        else:
                            reason = "homography_estimation_failed"
                            inlier_mask = None
                    else:
                        reason = "insufficient_good_matches"
            result[frame_index] = matrix
            if diagnostics_callback is not None:
                diagnostics_callback(
                    _frame_diagnostics(
                        frame_index=frame_index,
                        reference_frame_index=reference_frame_index,
                        reference_keypoints=len(reference_points),
                        current_keypoints=point_count,
                        current_xy=current_xy,
                        reference_xy=reference_xy,
                        inlier_mask=inlier_mask,
                        matrix=matrix,
                        previous_matrix=previous_matrix,
                        width=frame.shape[1],
                        height=frame.shape[0],
                        status=status,
                        reason=reason,
                        reused_from_frame_index=reused_from,
                        elapsed_seconds=perf_counter() - started,
                    )
                )
            previous_matrix = matrix
            if progress_callback:
                progress_callback(frame_index + 1, frame_count)
        if len(result) != frame_count:
            raise RuntimeError(
                f"Decoded {len(result)} of {frame_count} declared video frames; refusing a partial matrix file"
            )
        return result
    finally:
        capture.release()


def save_stabilization_matrices(matrices: dict[int, np.ndarray], output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, matrices, allow_pickle=True)
