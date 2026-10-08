from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pemflow.io import COLUMNS
from pemflow.pipeline.geometry import solve_camera_pose

LABEL_COLUMNS = [
    "frame",
    "source_track_id",
    "raw_class",
    "cx_norm",
    "cy_norm",
    "width_norm",
    "height_norm",
    "confidence",
]
MOTION_WINDOW = 31
MOTION_ORDER = 2
MIN_MOTION_WINDOW = 25


def label_frame(values, count):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABEL_COLUMNS):
        raise ValueError("Expected frame,id,class,cx,cy,width,height,confidence")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite label values")
    if np.any(values[:, :3] != np.floor(values[:, :3])):
        raise ValueError("Frame, identity and class must be integers")
    if np.any(values[:, 0] < 1) or np.any(values[:, 0] > count):
        raise ValueError("Labels use one-based local video frames")
    if np.any(values[:, 1] <= 0) or np.any(values[:, 2] < 0):
        raise ValueError("Invalid identity or negative detector class")
    if np.any(values[:, 3:] < 0) or np.any(values[:, 3:] > 1):
        raise ValueError("Invalid normalized label values")
    if np.any(values[:, 5:7] <= 0):
        raise ValueError("Detection boxes must have positive area")
    data = pd.DataFrame(values, columns=LABEL_COLUMNS)
    for column in LABEL_COLUMNS[:3]:
        data[column] = data[column].astype(np.int64)
    if data.duplicated(["frame", "source_track_id"]).any():
        raise ValueError("Duplicate source identity within one video frame")
    data["observation_id"] = np.arange(1, len(data) + 1, dtype=np.int64)
    return data


def majority(values):
    count = values.value_counts()
    maximum = count.max()
    tied = set(count[count.eq(maximum)].index)
    return next((v for v in values if v in tied))


def project(data, profile, matrices, *, chunk_size=32768):
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Expected a dense zero-based matrix sequence")
    if not np.isfinite(matrices).all() or np.any(np.abs(np.linalg.det(matrices)) < 1e-12):
        raise ValueError("Non-finite or singular stabilization transform")
    reference = int(profile["reference_frame_index"])
    if not np.allclose(matrices[reference], np.eye(3), rtol=0, atol=1e-09):
        raise ValueError("Matrices are not expressed in the calibration reference image")
    camera = profile["camera_parameters"]
    k = np.asarray(camera["matrix"], dtype=float)
    d = np.asarray(camera["distortion"], dtype=float)
    rotation, centre = solve_camera_pose(k, d, profile["world_points"], profile["image_points"])
    sizes = [profile["image_width_px"], profile["image_height_px"]]
    heights = {int(k): float(v) for k, v in profile["class_reference_height_m"].items()}
    xy = np.empty((len(data), 2), dtype=float)
    for start in range(0, len(data), chunk_size):
        part = data.iloc[start : start + chunk_size]
        indices = part.frame.to_numpy(dtype=int) - 1
        if np.any(indices < 0) or np.any(indices >= len(matrices)):
            raise ValueError("Matrix coverage is incomplete")
        pixels = part[["cx_norm", "cy_norm"]].to_numpy() * sizes
        q = np.einsum("nij,nj->ni", matrices[indices], np.c_[pixels, np.ones(len(part))])
        if np.any(np.abs(q[:, 2]) < 1e-12):
            raise ValueError("Homography projects a box centre to infinity")
        uv = q[:, :2] / q[:, 2:]
        rays = cv2.undistortPoints(uv.reshape(-1, 1, 2), k, d).reshape(-1, 2)
        rays = np.c_[rays, np.ones(len(part))] @ rotation
        if np.any(np.abs(rays[:, 2]) < 1e-09):
            raise ValueError("Projection ray parallel to the reference plane")
        z = part["class"].map(heights).to_numpy(dtype=float)
        scale = (z - centre[2]) / rays[:, 2]
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Intersection is missing or behind the camera")
        xy[start : start + len(part)] = (centre + scale[:, None] * rays)[:, :2]
    return xy


def continuous_motion(
    data,
    xy,
    *,
    fps=30.0,
    window=MOTION_WINDOW,
    order=MOTION_ORDER,
    minimum_window=MIN_MOTION_WINDOW,
):
    if (
        fps <= 0
        or window <= order
        or window % 2 != 1
        or (order < 1)
        or (minimum_window <= order)
        or (minimum_window % 2 != 1)
        or (minimum_window > window)
    ):
        raise ValueError("Invalid Savitzky-Golay settings")
    out = data.copy()
    out[["raw_x", "raw_y"]] = np.asarray(xy, dtype=np.float64)
    out = out.sort_values(["id", "frame", "observation_id"], kind="stable").reset_index(drop=True)
    cut = (
        out.id.ne(out.id.shift())
        | out.frame.diff().ne(1)
        | out.source_track_id.ne(out.source_track_id.shift())
    )
    out["motion_segment"] = cut.cumsum().astype(np.int64)
    smooth = out[["raw_x", "raw_y"]].to_numpy().copy()
    raw = smooth.copy()
    speed = np.full(len(out), np.nan)
    velocity_components = np.full((len(out), 2), np.nan)
    acceleration_components = np.full((len(out), 2), np.nan)
    windows_used = np.zeros(len(out), dtype=np.int64)
    segment_sizes = out.groupby("motion_segment").size()
    out["motion_segment_frames"] = out.motion_segment.map(segment_sizes).astype(np.int64)
    for idx in out.groupby("motion_segment", sort=False).indices.values():
        window_used = min(window, len(idx))
        if window_used % 2 == 0:
            window_used -= 1
        if window_used >= minimum_window:
            windows_used[idx] = window_used
            smooth[idx] = savgol_filter(raw[idx], window_used, order, axis=0, mode="interp")
            velocity = savgol_filter(
                raw[idx], window_used, order, deriv=1, delta=1.0 / fps, axis=0, mode="interp"
            )
            speed[idx] = np.linalg.norm(velocity, axis=1)
            velocity_components[idx] = velocity
            if order >= 2:
                acceleration_components[idx] = savgol_filter(
                    raw[idx], window_used, order, deriv=2, delta=1.0 / fps, axis=0, mode="interp"
                )
    out[["real_x", "real_y"]] = smooth
    out["velocity"] = speed
    out[["vx", "vy"]] = velocity_components
    out[["ax", "ay"]] = acceleration_components
    out["acceleration_norm"] = np.linalg.norm(acceleration_components, axis=1)
    out["tangential_acceleration"] = np.divide(
        np.sum(velocity_components * acceleration_components, axis=1),
        speed,
        out=np.full(len(out), np.nan),
        where=speed > 1e-12,
    )
    out["time_s"] = (out.frame - 1) / fps
    out["motion_window"] = windows_used
    out["smoothed"] = windows_used > 0
    out["adaptive_motion_window"] = out.smoothed & out.motion_window.lt(window)
    out["motion_status"] = np.select(
        [windows_used == window, windows_used > 0],
        ["same_polynomial_fit", "adaptive_polynomial_fit"],
        default="insufficient_window",
    )
    delta = out.groupby("motion_segment")[["real_x", "real_y"]].diff()
    out["dist"] = np.hypot(delta.real_x, delta.real_y)
    out.loc[cut, "velocity"] = np.nan
    out.loc[cut, ["vx", "vy", "ax", "ay", "acceleration_norm", "tangential_acceleration"]] = np.nan
    out.loc[cut & out.smoothed, "motion_status"] = "segment_start"
    return out


def clean_csv(frame):
    out = frame.copy()
    out["input_row"] = np.arange(1, len(out) + 1)
    reason = pd.Series("", index=out.index)
    for name in ["frame", "id"]:
        v = pd.to_numeric(out[name], errors="coerce")
        bad = ~np.isfinite(v) | v.le(0) | v.ne(np.floor(v))
        reason.loc[bad & reason.eq("")] = "invalid_" + name
    reason.loc[~out["class"].isin([0, 3, 9]) & reason.eq("")] = "invalid_class"
    positions = out[["real_x", "real_y"]].to_numpy(dtype=float)
    reason.loc[~np.isfinite(positions).all(axis=1) & reason.eq("")] = "nonfinite_position"
    for name in ["velocity", "dist"]:
        v = out[name].to_numpy(dtype=float)
        bad = np.isinf(v) | np.isfinite(v) & (v < 0)
        if np.any(bad & reason.eq("").to_numpy()):
            raise ValueError(f"Invalid derived {name}; recompute motion, do not delete positions")
    valid = out.loc[reason.eq("")]
    exact = valid.duplicated(COLUMNS, keep="first")
    reason.loc[valid.index[exact]] = "identical_seven_column_duplicate"
    removed = out.loc[reason.ne("")].copy()
    removed["reason"] = reason.loc[removed.index]
    cleaned = out.loc[reason.eq("")].copy()
    if cleaned.duplicated(["frame", "id"]).any():
        raise ValueError("Conflicting frame/ID records require an explicit identity decision")
    return (cleaned.drop(columns="input_row"), removed)


def describe(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, median=None, p95=None, p99=None, maximum=None)
    return dict(
        n=len(x),
        median=float(np.median(x)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        maximum=float(x.max()),
    )


def quality_summary(frame, fps=30.0):
    x = frame.sort_values(["id", "frame"], kind="stable").copy()
    continuous = x.id.eq(x.id.shift()) & x.frame.diff().eq(1)
    raw_delta = x[["raw_x", "raw_y"]].diff()
    raw_step = np.hypot(raw_delta.raw_x, raw_delta.raw_y).where(continuous)
    acceleration = x.tangential_acceleration
    same_segment = continuous & x.motion_segment.eq(x.motion_segment.shift())
    acceleration = acceleration.where(same_segment)
    flags = x[["observation_id", "frame", "id", "source_track_id", "class"]].copy()
    flags["raw_step_m"] = raw_step
    flags["velocity_m_s"] = x.velocity
    flags["acceleration_m_s2"] = acceleration
    flags["motion_window"] = x.motion_window
    flags["motion_status"] = x.motion_status
    flags["short_motion_segment"] = ~x.smoothed
    flags["adaptive_motion_window"] = x.adaptive_motion_window
    flags["raw_step_over_1m"] = raw_step.gt(1.0)
    flags["speed_over_12m_s"] = x.velocity.gt(12.0)
    flags["abs_acceleration_over_10m_s2"] = acceleration.abs().gt(10.0)
    reasons = [
        "short_motion_segment",
        "adaptive_motion_window",
        "raw_step_over_1m",
        "speed_over_12m_s",
        "abs_acceleration_over_10m_s2",
    ]
    chosen = flags[reasons].any(axis=1)
    sizes = x.groupby("id").size()
    segments = x.drop_duplicates("motion_segment")
    result = dict(
        rows=len(x),
        frames=int(x.frame.nunique()),
        identities=int(x.id.nunique()),
        duplicate_frame_id_rows=int(x.duplicated(["frame", "id"], keep=False).sum()),
        finite_positions=bool(np.isfinite(x[["real_x", "real_y"]]).all().all()),
        missing_velocity_rows=int(x.velocity.isna().sum()),
        adaptive_motion_segments=int(segments.adaptive_motion_window.sum()),
        insufficient_motion_segments=int((~segments.smoothed).sum()),
        motion_window_rows={str(int(k)): int(v) for k, v in x.motion_window.value_counts().items()},
        tracks_shorter_than31_observations=int(sizes.lt(31).sum()),
        gaps=int((x.id.eq(x.id.shift()) & x.frame.diff().gt(1)).sum()),
        raw_step_m=describe(raw_step),
        speed_m_s=describe(x.velocity),
        abs_acceleration_m_s2=describe(acceleration.abs()),
        acceleration_fraction_abs_le2=float(acceleration.dropna().abs().le(2).mean())
        if acceleration.notna().any()
        else None,
        raw_step_over_1m=int(raw_step.gt(1).sum()),
        speed_over_12m_s=int(x.velocity.gt(12).sum()),
        acceleration_over_10m_s2=int(acceleration.abs().gt(10).sum()),
        class_rows={str(int(k)): int(v) for k, v in x["class"].value_counts().items()},
        curation_id_rows=int(x.identity_edit.sum()),
        explicit_class_edit_rows=int(x.class_edit.sum()),
        majority_class_changed_rows=int(x.class_unified.sum()),
    )
    return (result, flags.loc[chosen])


def process(values, profile, matrices):
    data = label_frame(values, profile["frames"])
    data = prepare_tracks(data)
    events = []
    xy = project(data, profile, matrices)
    full = continuous_motion(data, xy, fps=profile["fps"])
    clean, removed = clean_csv(full)
    stats, flags = quality_summary(clean, profile["fps"])
    return (full, clean, removed, events, stats, flags)


def prepare_tracks(data):
    """Set one majority class per automatic source track before projection."""
    out = data.copy()
    out["id"] = out.source_track_id
    out["corrected_class"] = out.raw_class
    out["class_edit"] = False
    out["identity_edit"] = False
    ordered = out.sort_values(["id", "frame", "observation_id"], kind="stable")
    classes = ordered.groupby("id", sort=False).corrected_class.agg(majority)
    out["class"] = out.id.map(classes).astype(np.int64)
    if not out["class"].isin([0, 3, 9]).all():
        raise ValueError("Unsupported final track class")
    out["class_unified"] = out["class"].ne(out.corrected_class)
    return out
