from __future__ import annotations

import cv2
import numpy as np

from pemflow.pipeline.alignment import candidate_matrices


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _matrices(values, count):
    values = np.asarray(values, dtype=float)
    if values.shape != (count, 3, 3):
        raise ValueError("Base matrices must cover every expected video frame as (N, 3, 3)")
    if not np.isfinite(values).all() or np.any(np.abs(values[:, 2, 2]) < 1e-12):
        raise ValueError("Non-finite or unnormalizable base matrix")
    if np.any(np.abs(np.linalg.det(values)) < 1e-12):
        raise ValueError("Singular base matrix")
    return values


def _read_image(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = capture.read()
    if not ok or image is None:
        raise ValueError(f"Cannot decode zero-based video frame {index}")
    if image.shape[:2] != (2160, 3840):
        raise ValueError(f"Unexpected native frame dimensions {image.shape}")
    return image


def _masked_features(image, frame_observations, sift):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (1920, 1080))
    mask = np.full(gray.shape, 255, np.uint8)
    for row in frame_observations:
        x, y, w, h = np.asarray(row[3:7]) * [1920, 1080, 1920, 1080]
        a = (max(0, int(x - w * 0.7 - 4)), max(0, int(y - h * 0.7 - 4)))
        b = (min(1919, int(x + w * 0.7 + 4)), min(1079, int(y + h * 0.7 + 4)))
        cv2.rectangle(mask, a, b, 0, -1)
    return sift.detectAndCompute(gray, mask)


def _correspondences(reference, current, *, diagnostic_fit=True):
    kr, dr = reference
    kc, dc = current
    if dr is None or dc is None:
        raise ValueError("No SIFT descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(dc, dr, k=2)
    reverse = matcher.knnMatch(dr, dc, k=2)
    rev = {
        a.queryIdx: a.trainIdx
        for pair in reverse
        if len(pair) == 2
        for a, b in [pair]
        if a.distance < 0.7 * b.distance
    }
    good = [
        a
        for pair in forward
        if len(pair) == 2
        for a, b in [pair]
        if a.distance < 0.7 * b.distance and rev.get(a.trainIdx) == a.queryIdx
    ]
    if len(good) < 30:
        raise ValueError(f"Only {len(good)} mutual feature matches")
    current_xy = np.array([kc[m.queryIdx].pt for m in good]) * 2
    reference_xy = np.array([kr[m.trainIdx].pt for m in good]) * 2
    if not diagnostic_fit:
        return (current_xy, reference_xy, None)
    cv2.setRNGSeed(20260927)
    fitted, inlier = cv2.findHomography(current_xy, reference_xy, cv2.RANSAC, 3.0)
    if fitted is None or inlier is None or inlier.sum() < 25:
        raise ValueError("Insufficient geometrically consistent correspondences")
    return (current_xy, reference_xy, inlier.ravel().astype(bool))


def rebase_video_matrices(matrices: np.ndarray, calibration_reference_index: int) -> np.ndarray:
    values = np.asarray(matrices, dtype=float)
    if values.ndim != 3 or len(values) == 0:
        raise ValueError("Require a nonempty (N, 3, 3) matrix array")
    values = _matrices(values, len(values))
    index = _integer(calibration_reference_index, "calibration_reference_index")
    if index >= len(values):
        raise ValueError("Calibration reference lies outside matrix coverage")
    aligned = candidate_matrices(
        dict(enumerate(values)), range(1, len(values) + 1), reference_index=index
    )
    return np.stack([aligned[frame] for frame in range(1, len(values) + 1)])
