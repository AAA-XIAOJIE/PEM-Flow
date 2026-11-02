from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

MIXED_RUNS = {"G05", "G06", "G07", "G08", "G15", "G16"}


def _angle_difference(first: float, second: float) -> float:
    return float((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def estimate_headings(scene: pd.DataFrame) -> dict[tuple[int, int], float]:
    selected = scene[scene["class"].isin([3, 9])].sort_values(["track_id", "frame_id"])
    headings: dict[tuple[int, int], float] = {}
    for track_id, group in selected.groupby("track_id"):
        current = group.reset_index(drop=True)
        n = len(current)
        if n == 0:
            continue
        if n == 1:
            headings[int(track_id), int(current.loc[0, "frame_id"])] = 0.0
            continue
        x = current["real_x"].to_numpy(float)
        y = current["real_y"].to_numpy(float)
        speed = current["velocity"].to_numpy(float)
        cls = int(current["class"].iloc[0])
        minimum_displacement = 0.3 if cls == 3 else 0.18
        raw = np.full(n, np.nan)
        half_window = 30
        for index in range(n):
            left = max(0, index - half_window)
            right = min(n - 1, index + half_window)
            dx = x[right] - x[left]
            dy = y[right] - y[left]
            if np.hypot(dx, dy) >= minimum_displacement:
                raw[index] = np.arctan2(dy, dx)
        valid = np.isfinite(raw)
        if not valid.any():
            dx = x[-1] - x[0]
            dy = y[-1] - y[0]
            raw[:] = np.arctan2(dy, dx) if np.hypot(dx, dy) >= 0.05 else 0.0
            valid = np.isfinite(raw)
        if not valid.all():
            indices = np.arange(n)
            raw = np.arctan2(
                np.interp(indices, indices[valid], np.sin(raw[valid])),
                np.interp(indices, indices[valid], np.cos(raw[valid])),
            )
        sg_window = min(61, n if n % 2 else n - 1)
        if sg_window >= 5:
            smooth_sin = savgol_filter(np.sin(raw), sg_window, 2)
            smooth_cos = savgol_filter(np.cos(raw), sg_window, 2)
        else:
            smooth_sin, smooth_cos = (np.sin(raw), np.cos(raw))
        target = np.arctan2(smooth_sin, smooth_cos)
        final = np.zeros(n)
        final[0] = target[0]
        ema_sin, ema_cos = (np.sin(target[0]), np.cos(target[0]))
        for index in range(1, n):
            if speed[index] < 0.08 and (not valid[index]):
                final[index] = final[index - 1]
                ema_sin = np.sin(final[index])
                ema_cos = np.cos(final[index])
                continue
            alpha = 0.08 if speed[index] < 0.3 else 0.15
            ema_sin = alpha * np.sin(target[index]) + (1 - alpha) * ema_sin
            ema_cos = alpha * np.cos(target[index]) + (1 - alpha) * ema_cos
            requested = np.arctan2(ema_sin, ema_cos)
            max_step = np.deg2rad(2.0 if speed[index] < 0.3 else 6.0)
            step = np.clip(_angle_difference(requested, final[index - 1]), -max_step, max_step)
            final[index] = final[index - 1] + step
        for frame_id, heading in zip(current["frame_id"], final, strict=False):
            headings[int(track_id), int(frame_id)] = float(heading)
    return headings


def _rectangle_distance(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    heading: float,
    half_length: float,
    half_width: float,
) -> np.ndarray:
    cosine, sine = (np.cos(heading), np.sin(heading))
    local_x = (x - cx) * cosine + (y - cy) * sine
    local_y = -(x - cx) * sine + (y - cy) * cosine
    dx = np.maximum(np.abs(local_x) - half_length, 0.0)
    dy = np.maximum(np.abs(local_y) - half_width, 0.0)
    return np.hypot(dx, dy)


def _circle_distance(
    x: np.ndarray, y: np.ndarray, cx: float, cy: float, radius: float
) -> np.ndarray:
    return np.maximum(np.hypot(x - cx, y - cy) - radius, 0.0)


def _neighbor_ratios(
    labels: np.ndarray, classes: dict[int, int], requested: set[int]
) -> dict[int, float]:
    pairs: set[tuple[int, int]] = set()
    for first, second in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        valid = (first >= 0) & (second >= 0) & (first != second)
        for left, right in zip(first[valid], second[valid], strict=False):
            pairs.add(tuple(sorted((int(left), int(right)))))
    neighbours: dict[int, set[int]] = defaultdict(set)
    for left, right in pairs:
        neighbours[left].add(right)
        neighbours[right].add(left)
    output: dict[int, float] = {}
    for entity in requested:
        adjacent = neighbours.get(entity, set())
        if not adjacent:
            output[entity] = float("nan")
        else:
            output[entity] = sum((classes[item] == 9 for item in adjacent)) / len(adjacent)
    return output
