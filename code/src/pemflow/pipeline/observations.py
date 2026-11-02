from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

RECOVERY_METHOD = "selective_common_box_image_recovery"
RECOVERY_VERSION = 1


@dataclass(frozen=True)
class RecoveryConfig:
    max_gap_frames: int = 15
    anchor_margin: int = 2
    neighbor_diagonals: float = 1.5
    min_features: int = 6
    max_features: int = 80
    feature_quality: float = 0.01
    feature_spacing_px: float = 3.0
    box_inset_fraction: float = 0.15
    lk_window_px: int = 15
    lk_levels: int = 3
    max_forward_backward_px: float = 1.0
    max_lk_error: float = 25.0
    min_common_frames: int = 2
    min_common_area_ratio: float = 1.5
    max_direction_disagreement_height: float = 0.1
    max_endpoint_error_height: float = 0.1
    min_identity_margin_height: float = 0.3
    max_pair_iou: float = 0.7
    min_center_separation_height: float = 0.3
    patch_size_fraction: float = 0.3
    min_patch_ncc: float = 0.65
    min_patch_peak_margin: float = 0.04
    patch_search_radius_height: float = 0.35
    max_reacquisition_center_error_height: float = 0.1


def box(row, shape):
    size = np.array([shape[1], shape[0]], float)
    center, wh = (row[3:5] * size, row[5:7] * size)
    return np.r_[center - wh / 2, center + wh / 2]


def center(rect):
    return (rect[:2] + rect[2:]) / 2


def overlap(a, b):
    inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    return float(inter / max(1e-09, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter))


def _consecutive_count(values):
    longest = run = 0
    previous = None
    for value in sorted(set(values)):
        run = run + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, run)
        previous = value
    return longest


def _clean_anchors(frames, partners, a, b, shape, config):
    for frame in (a, a + 1, b - 1, b):
        rows = frames.get(frame, np.empty((0, 8)))
        pair = [rows[rows[:, 1] == identity] for identity in partners]
        if any((len(rows) != 1 for rows in pair)):
            return False
        rects = [box(rows[0], shape) for rows in pair]
        centers = np.array([center(rect) for rect in rects])
        heights = [rect[3] - rect[1] for rect in rects]
        if min(heights) <= 0 or overlap(*rects) > config.max_pair_iou:
            return False
        if np.linalg.norm(centers[0] - centers[1]) < config.min_center_separation_height * min(
            heights
        ):
            return False
        for i, rect in enumerate(rects):
            other = centers[1 - i]
            if np.all((other >= rect[:2]) & (other <= rect[2:])):
                return False
    for identity in partners:
        for first, second in ((a, a + 1), (b - 1, b)):
            areas = [
                np.prod(
                    box(frames[f][frames[f][:, 1] == identity][0], shape)[2:]
                    - box(frames[f][frames[f][:, 1] == identity][0], shape)[:2]
                )
                for f in (first, second)
            ]
            if min(areas) <= 0 or max(areas) / min(areas) >= config.min_common_area_ratio:
                return False
    return True


def find_close_gap_events(labels, shape, config, frame_window=None):
    ordered = labels[np.argsort(labels[:, 0], kind="stable")]
    distinct, starts = np.unique(ordered[:, 0], return_index=True)
    frames = {int(f): part for f, part in zip(distinct, np.split(ordered, starts[1:]))}
    events = []
    for identity in np.unique(labels[:, 1]):
        rows = labels[labels[:, 1] == identity]
        rows = rows[np.argsort(rows[:, 0])]
        for left, right in zip(rows[:-1], rows[1:]):
            count = int(right[0] - left[0] - 1)
            if not 1 <= count <= config.max_gap_frames:
                continue
            if frame_window and (not (frame_window[0] <= left[0] and right[0] <= frame_window[1])):
                continue
            a, b = (int(left[0]) - config.anchor_margin, int(right[0]) + config.anchor_margin)
            if a not in frames or b not in frames:
                continue
            ra = frames[a][frames[a][:, 1] == identity]
            rb = frames[b][frames[b][:, 1] == identity]
            if len(ra) != 1 or len(rb) != 1:
                continue
            seed = box(ra[0], shape)
            partners = [int(identity)]
            anchors = [(ra[0], rb[0])]
            for row in frames[a]:
                if row[1] == identity or row[2] != left[2]:
                    continue
                other = frames[b][(frames[b][:, 1] == row[1]) & (frames[b][:, 2] == row[2])]
                if len(other) != 1:
                    continue
                if np.linalg.norm(
                    center(box(row, shape)) - center(seed)
                ) <= config.neighbor_diagonals * np.linalg.norm(seed[2:] - seed[:2]):
                    partners.append(int(row[1]))
                    anchors.append((row, other[0]))
            if len(partners) != 2:
                continue

            def common_observation(f, *, strong=False):
                t = (f - a) / (b - a)
                priors = [box(x, shape) * (1 - t) + box(y, shape) * t for x, y in anchors]
                for row in frames.get(f, np.empty((0, 8))):
                    if int(row[1]) not in partners:
                        continue
                    rect = box(row, shape)
                    cs = np.array([center(p) for p in priors])
                    if np.all((cs >= rect[:2]) & (cs <= rect[2:]), axis=1).sum() == 2:
                        index = partners.index(int(row[1]))
                        endpoint_areas = [
                            np.prod(box(x, shape)[2:] - box(x, shape)[:2]) for x in anchors[index]
                        ]
                        area_ratio = np.prod(rect[2:] - rect[:2]) / max(
                            1e-09, np.median(endpoint_areas)
                        )
                        if not strong or area_ratio >= config.min_common_area_ratio:
                            return True
                return False

            core_frames = [f for f in range(a + 1, b) if common_observation(f, strong=True)]
            touched = _consecutive_count(core_frames) >= config.min_common_frames and any(
                (common_observation(f) for f in range(int(left[0]) + 1, int(right[0])))
            )
            if touched:
                first, last = (int(left[0]) + 1, int(right[0]) - 1)
                while first > a + 1 and common_observation(first - 1):
                    first -= 1
                while last < b - 1 and common_observation(last + 1):
                    last += 1
                new_a, new_b = (first - 1 - config.anchor_margin, last + 1 + config.anchor_margin)
                if new_a not in frames or new_b not in frames:
                    continue
                new_anchors = []
                for partner in partners:
                    x, y = (
                        frames[new_a][frames[new_a][:, 1] == partner],
                        frames[new_b][frames[new_b][:, 1] == partner],
                    )
                    if len(x) != 1 or len(y) != 1:
                        break
                    new_anchors.append((x[0], y[0]))
                if len(new_anchors) != 2:
                    continue
                if not _clean_anchors(frames, partners, new_a, new_b, shape, config):
                    continue
                sizes = {
                    partner: np.median(
                        [np.prod(box(row, shape)[2:] - box(row, shape)[:2]) for row in pair]
                    )
                    for partner, pair in zip(partners, new_anchors)
                }
                carrier_keys, final_core_frames = ([], [])
                for f in range(first, last + 1):
                    t = (f - new_a) / (new_b - new_a)
                    cs = np.array(
                        [
                            center(box(x, shape) * (1 - t) + box(y, shape) * t)
                            for x, y in new_anchors
                        ]
                    )
                    for row in frames.get(f, np.empty((0, 8))):
                        identity_at_frame = int(row[1])
                        if identity_at_frame not in partners:
                            continue
                        rect = box(row, shape)
                        if not np.all((cs >= rect[:2]) & (cs <= rect[2:]), axis=1).all():
                            continue
                        if (
                            np.prod(rect[2:] - rect[:2]) / max(1e-09, sizes[identity_at_frame])
                            >= config.min_common_area_ratio
                        ):
                            carrier_keys.append((f, identity_at_frame))
                            final_core_frames.append(f)
                if _consecutive_count(final_core_frames) < config.min_common_frames:
                    continue
                events.append(
                    {
                        "missing_source_id": int(identity),
                        "partner_ids": partners,
                        "missing_start": first,
                        "missing_end": last,
                        "original_missing_start": int(left[0]) + 1,
                        "original_missing_end": int(right[0]) - 1,
                        "anchor_start": new_a,
                        "anchor_end": new_b,
                        "common_core_frames": sorted(set(final_core_frames)),
                        "common_carrier_keys": carrier_keys,
                        "anchor_control_frames": [new_a + 1, new_b - 1],
                        "anchors": np.array(new_anchors),
                    }
                )
    return events


def track_direction(images, frames, rect, all_rects, index, config):
    start = frames[0]
    gray = cv2.cvtColor(images[start], cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    size = rect[2:] - rect[:2]
    low = np.ceil(rect[:2] + size * config.box_inset_fraction).astype(int)
    high = np.floor(rect[2:] - size * config.box_inset_fraction).astype(int)
    mask[max(0, low[1]) : high[1], max(0, low[0]) : high[0]] = 255
    yy, xx = np.indices(gray.shape)
    cs = np.array([center(r) for r in all_rects])
    distance = np.stack([(xx - c[0]) ** 2 + (yy - c[1]) ** 2 for c in cs])
    mask[np.argmin(distance, axis=0) != index] = 0
    points = cv2.goodFeaturesToTrack(
        gray,
        config.max_features,
        config.feature_quality,
        config.feature_spacing_px,
        mask=mask,
        blockSize=3,
    )
    if points is None or len(points) < config.min_features:
        return (
            {},
            {
                "status": "INSUFFICIENT_INITIAL_FEATURES",
                "initial_features": 0 if points is None else len(points),
            },
        )
    origin = points.copy()
    initial = len(points)
    result = {
        start: {
            "box": rect.copy(),
            "features": initial,
            "fb_p95_px": 0.0,
            "points": points[:, 0].copy(),
        }
    }
    kwargs = dict(
        winSize=(config.lk_window_px, config.lk_window_px),
        maxLevel=config.lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    for frame in frames[1:]:
        target = cv2.cvtColor(images[frame], cv2.COLOR_BGR2GRAY)
        following, good, error = cv2.calcOpticalFlowPyrLK(gray, target, points, None, **kwargs)
        if following is None:
            return (result, {"status": "FLOW_FAILED", "frame": frame, "initial_features": initial})
        back, reverse_good, _ = cv2.calcOpticalFlowPyrLK(target, gray, following, None, **kwargs)
        fb = np.linalg.norm(back[:, 0] - points[:, 0], axis=1)
        keep = good[:, 0].astype(bool) & reverse_good[:, 0].astype(bool)
        keep &= (fb <= config.max_forward_backward_px) & (error[:, 0] <= config.max_lk_error)
        keep &= np.all(
            (following[:, 0] >= [0, 0]) & (following[:, 0] < [gray.shape[1], gray.shape[0]]), axis=1
        )
        displacement = following[:, 0] - origin[:, 0]
        if keep.sum() >= config.min_features:
            median = np.median(displacement[keep], axis=0)
            residual = np.linalg.norm(displacement - median, axis=1)
            med = np.median(residual[keep])
            mad = np.median(abs(residual[keep] - med))
            keep &= residual <= min(4.0, max(1.5, med + 3 * 1.4826 * mad))
        if keep.sum() < config.min_features:
            return (
                result,
                {
                    "status": "INSUFFICIENT_SURVIVING_FEATURES",
                    "frame": frame,
                    "initial_features": initial,
                    "remaining": int(keep.sum()),
                },
            )
        shift = np.median(displacement[keep], axis=0)
        result[frame] = {
            "box": rect + np.r_[shift, shift],
            "features": int(keep.sum()),
            "fb_p95_px": float(np.percentile(fb[keep], 95)),
            "points": following[keep, 0].copy(),
        }
        points, origin, gray = (following[keep].copy(), origin[keep].copy(), target)
    return (
        result,
        {"status": "COMPLETE", "initial_features": initial, "final_features": len(points)},
    )


def patch_peak(image, template, search_rect):
    side = len(template)
    half = side // 2
    lo = np.maximum([half, half], np.ceil(search_rect[:2]).astype(int))
    hi = np.minimum(
        [image.shape[1] - half - 1, image.shape[0] - half - 1],
        np.floor(search_rect[2:]).astype(int),
    )
    if np.any(hi < lo):
        return None
    cut = image[lo[1] - half : hi[1] + half + 1, lo[0] - half : hi[0] + half + 1]
    score = cv2.matchTemplate(cut, template, cv2.TM_CCOEFF_NORMED)
    _, value, _, peak = cv2.minMaxLoc(score)
    position = np.array(peak, float) + lo
    remaining = score.copy()
    yy, xx = np.indices(score.shape)
    remaining[(xx - peak[0]) ** 2 + (yy - peak[1]) ** 2 <= max(3.0, side * 0.4) ** 2] = -1
    alternative = float(remaining.max()) if np.any(remaining > -1) else -1.0
    return (position, float(value), float(value - alternative))


def track_patch(images, frames, rect, point, template, config):
    result = {}
    initial = point.copy()
    height = rect[3] - rect[1]
    minimum_score = minimum_margin = 1.0
    for frame in frames:
        radius = max(8.0, height * config.patch_search_radius_height)
        found = patch_peak(images[frame], template, np.r_[point - radius, point + radius])
        if found is None:
            return (result, {"status": "PATCH_SEARCH_EMPTY", "frame": frame})
        position, score, margin = found
        if score < config.min_patch_ncc or margin < config.min_patch_peak_margin:
            return (
                result,
                {"status": "PATCH_AMBIGUOUS", "frame": frame, "ncc": score, "peak_margin": margin},
            )
        minimum_score = min(minimum_score, score)
        minimum_margin = min(minimum_margin, margin)
        shift = position - initial
        result[frame] = {
            "box": rect + np.r_[shift, shift],
            "features": 0,
            "fb_p95_px": None,
            "points": position.reshape(1, 2),
            "method": "independent_color_patch",
            "ncc": score,
            "peak_margin": margin,
            "support_pixels": int(template.shape[0] * template.shape[1]),
        }
        point = position
    return (
        result,
        {
            "status": "COMPLETE",
            "measurement_method": "color_patch_correlation",
            "minimum_ncc": minimum_score,
            "minimum_peak_margin": minimum_margin,
            "patch_size": template.shape[0],
            "initial_features": 0,
        },
    )


def track_patch_pair(images, a, b, left, right, config):
    height = min(left[3] - left[1], right[3] - right[1])
    side = (
        max(
            11,
            int(
                round(
                    min(*left[2:] - left[:2], *right[2:] - right[:2]) * config.patch_size_fraction
                )
            ),
        )
        | 1
    )
    half = side // 2
    low = np.ceil(left[:2] + half).astype(int)
    high = np.floor(left[2:] - half).astype(int)
    choices = []
    for y in range(low[1], high[1] + 1, max(3, side // 4)):
        for x in range(low[0], high[0] + 1, max(3, side // 4)):
            point = np.array([x, y], float)
            template = images[a][y - half : y + half + 1, x - half : x + half + 1].copy()
            if template.shape[:2] != (side, side) or float(template.std()) < 8:
                continue
            peak = patch_peak(images[b], template, np.r_[right[:2] + half, right[2:] - half])
            if peak is None:
                continue
            later, score, margin = peak
            displacement_error = np.linalg.norm(later - point - (center(right) - center(left)))
            if (
                score >= config.min_patch_ncc
                and margin >= config.min_patch_peak_margin
                and (displacement_error <= config.max_endpoint_error_height * height)
            ):
                choices.append((score + 0.2 * margin, point, later, template, score, margin))
    if not choices:
        return (
            {},
            {},
            {"status": "NO_STABLE_ENDPOINT_PATCH"},
            {"status": "NO_STABLE_ENDPOINT_PATCH"},
        )
    choices.sort(key=lambda q: q[0], reverse=True)
    _, point, later, template, score, margin = choices[0]
    px, py = np.round(later).astype(int)
    reverse_template = images[b][py - half : py + half + 1, px - half : px + half + 1].copy()
    fwd, fqa = track_patch(images, list(range(a, b + 1)), left, point, template, config)
    rev, rqa = track_patch(
        images, list(range(b, a - 1, -1)), right, later, reverse_template, config
    )
    for q in [fqa, rqa]:
        q["selected_left_point"] = point.tolist()
        q["selected_right_point"] = later.tolist()
        q["endpoint_patch_ncc"] = score
        q["endpoint_patch_margin"] = margin
        q["patch_candidates"] = len(choices)
    return (fwd, rev, fqa, rqa)


def recover_event(images, event, shape, roi, config, observations):
    a, b = (event["anchor_start"], event["anchor_end"])
    offsets = np.array([roi[0], roi[1], roi[0], roi[1]])
    left = np.array([box(pair[0], shape) - offsets for pair in event["anchors"]])
    right = np.array([box(pair[1], shape) - offsets for pair in event["anchors"]])
    directions, audit = ([], [])
    for i, identity in enumerate(event["partner_ids"]):
        fwd, fqa = track_direction(images, list(range(a, b + 1)), left[i], left, i, config)
        rev, rqa = track_direction(images, list(range(b, a - 1, -1)), right[i], right, i, config)
        sparse_failure = None
        if fqa["status"] != "COMPLETE" or rqa["status"] != "COMPLETE":
            sparse_failure = {"forward": fqa, "reverse": rqa}
            fwd, rev, fqa, rqa = track_patch_pair(images, a, b, left[i], right[i], config)
        height = min(left[i][3] - left[i][1], right[i][3] - right[i][1])
        endpoint = None
        if b in fwd and a in rev:
            endpoint = [
                float(np.linalg.norm(center(fwd[b]["box"]) - center(right[i]))),
                float(np.linalg.norm(center(rev[a]["box"]) - center(left[i]))),
            ]
        complete = fqa["status"] == rqa["status"] == "COMPLETE"
        endpoint_ok = complete and max(endpoint) <= config.max_endpoint_error_height * height
        own_margin = None
        if complete:
            j = 1 - i
            own_margin = float(
                min(
                    np.linalg.norm(center(fwd[b]["box"]) - center(right[j])) - endpoint[0],
                    np.linalg.norm(center(rev[a]["box"]) - center(left[j])) - endpoint[1],
                )
            )
        audit.append(
            {
                "source_id": identity,
                "forward": fqa,
                "reverse": rqa,
                "endpoint_error_px": endpoint,
                "endpoint_gate_px": config.max_endpoint_error_height * height,
                "identity_margin_px": own_margin,
                "identity_margin_gate_px": config.min_identity_margin_height * height,
                "sparse_flow_failure": sparse_failure,
                "endpoint_passed": bool(
                    endpoint_ok and own_margin >= config.min_identity_margin_height * height
                ),
            }
        )
        directions.append((fwd, rev, height))
    rows, per_frame = ([], [])
    for frame in range(a, b + 1):
        t = (frame - a) / (b - a)
        measures = []
        for i, (fwd, rev, height) in enumerate(directions):
            if frame not in fwd or frame not in rev:
                continue
            difference = float(
                np.linalg.norm(center(fwd[frame]["box"]) - center(rev[frame]["box"]))
            )
            rect = (1 - t) * fwd[frame]["box"] + t * rev[frame]["box"] + offsets
            measures.append(
                {
                    "source_id": event["partner_ids"][i],
                    "box": rect,
                    "direction_disagreement_px": difference,
                    "direction_gate_px": config.max_direction_disagreement_height * height,
                    "forward_features": fwd[frame]["features"],
                    "reverse_features": rev[frame]["features"],
                    "forward_fb_p95_px": fwd[frame]["fb_p95_px"],
                    "reverse_fb_p95_px": rev[frame]["fb_p95_px"],
                    "forward_method": fwd[frame].get("method", "sparse_lk"),
                    "reverse_method": rev[frame].get("method", "sparse_lk"),
                    "forward_patch_ncc": fwd[frame].get("ncc"),
                    "reverse_patch_ncc": rev[frame].get("ncc"),
                    "forward_points": fwd[frame]["points"] + [roi[0], roi[1]],
                    "reverse_points": rev[frame]["points"] + [roi[0], roi[1]],
                }
            )
        accepted = len(measures) == 2 and all((q["endpoint_passed"] for q in audit))
        shared_fraction = None
        if len(measures) == 2:
            for name in ["forward_points", "reverse_points"]:
                p, q = (measures[0][name], measures[1][name])
                near = np.linalg.norm(p[:, None] - q[None], axis=2) < 2.0
                current = max(float(near.any(axis=1).mean()), float(near.any(axis=0).mean()))
                shared_fraction = (
                    current if shared_fraction is None else max(shared_fraction, current)
                )
            pair_iou = overlap(measures[0]["box"], measures[1]["box"])
            separation = np.linalg.norm(center(measures[0]["box"]) - center(measures[1]["box"]))
            minheight = min((d[2] for d in directions))
            accepted &= all(
                (m["direction_disagreement_px"] <= m["direction_gate_px"] for m in measures)
            )
            accepted &= (
                pair_iou <= config.max_pair_iou
                and separation >= config.min_center_separation_height * minheight
            )
            accepted &= shared_fraction <= 0.1
        else:
            pair_iou = separation = None
        record = {
            "frame": frame,
            "is_gap": event["missing_start"] <= frame <= event["missing_end"],
            "accepted": bool(accepted),
            "pair_iou": pair_iou,
            "center_separation_px": None if separation is None else float(separation),
            "shared_feature_fraction": shared_fraction,
            "measurements": measures,
        }
        per_frame.append(record)
    all_gap = [f for f in per_frame if f["is_gap"]]
    passed = len(all_gap) > 0 and all((f["accepted"] for f in all_gap))
    final_end = event["missing_end"]
    settling_checks, actions, control_checks = ([], [], [])
    endpoint_areas = [
        np.median([np.prod(rect[2:] - rect[:2]) for rect in pair]) for pair in zip(left, right)
    ]
    image_core_keys = set()
    for record in all_gap:
        if not record["accepted"]:
            continue
        centers = np.array([center(m["box"]) for m in record["measurements"]])
        for i, measure in enumerate(record["measurements"]):
            old = observations[
                (observations[:, 0] == record["frame"])
                & (observations[:, 1] == measure["source_id"])
            ]
            if len(old) != 1:
                continue
            rect = box(old[0], shape)
            contains_both = np.all((centers >= rect[:2]) & (centers <= rect[2:]), axis=1).all()
            area_ratio = np.prod(rect[2:] - rect[:2]) / max(1e-09, endpoint_areas[i])
            if contains_both and area_ratio >= config.min_common_area_ratio:
                image_core_keys.add((record["frame"], measure["source_id"]))
    core_passed = _consecutive_count([f for f, _ in image_core_keys]) >= config.min_common_frames
    carrier_ids = {identity for _, identity in image_core_keys}
    passed &= core_passed
    for record in per_frame:
        if record["frame"] not in event.get("anchor_control_frames", [a + 1, b - 1]):
            continue
        errors = []
        for i, measure in enumerate(record["measurements"]):
            old = observations[
                (observations[:, 0] == record["frame"])
                & (observations[:, 1] == measure["source_id"])
            ]
            delta = (
                None
                if len(old) != 1
                else float(np.linalg.norm(center(measure["box"]) - center(box(old[0], shape))))
            )
            gate = config.max_reacquisition_center_error_height * directions[i][2]
            errors.append(
                {"source_id": measure["source_id"], "center_delta_px": delta, "gate_px": gate}
            )
        good = (
            record["accepted"]
            and len(errors) == 2
            and all(
                (
                    e["center_delta_px"] is not None and e["center_delta_px"] <= e["gate_px"]
                    for e in errors
                )
            )
        )
        control_checks.append({"frame": record["frame"], "passed": bool(good), "errors": errors})
        passed &= good
    passed &= len(control_checks) == 2
    if passed:
        for record in per_frame:
            frame = record["frame"]
            in_gap = event["missing_start"] <= frame <= event["missing_end"]
            in_tail = event["missing_end"] < frame < b
            if not (in_gap or in_tail) or not record["accepted"]:
                continue
            centers = np.array([center(m["box"]) for m in record["measurements"]])
            errors, tail_changed = ([], False)
            for i, measure in enumerate(record["measurements"]):
                identity = measure["source_id"]
                old = observations[(observations[:, 0] == frame) & (observations[:, 1] == identity)]
                rect = measure["box"]
                delta = (
                    None
                    if len(old) != 1
                    else float(np.linalg.norm(center(rect) - center(box(old[0], shape))))
                )
                gate = config.max_reacquisition_center_error_height * directions[i][2]
                action, reason = (
                    "preserve_existing",
                    "existing_observation_without_common_box_evidence",
                )
                if in_gap and len(old) == 0:
                    action, reason = (
                        "add_missing",
                        "missing_observation_in_confirmed_common_box_event",
                    )
                elif len(old) == 1:
                    old_rect = box(old[0], shape)
                    contains_both = np.all(
                        (centers >= old_rect[:2]) & (centers <= old_rect[2:]), axis=1
                    ).all()
                    area_ratio = np.prod(old_rect[2:] - old_rect[:2]) / max(
                        1e-09, endpoint_areas[i]
                    )
                    own_distance = np.linalg.norm(center(old_rect) - centers[i])
                    other_distance = np.linalg.norm(center(old_rect) - centers[1 - i])
                    independent = (
                        area_ratio < config.min_common_area_ratio
                        and delta <= gate
                        and (own_distance < other_distance)
                    )
                    if independent:
                        reason = "independent_observation_agrees_with_image_measurement"
                    if (
                        in_gap
                        and identity in carrier_ids
                        and ((frame, identity) in image_core_keys)
                    ):
                        action, reason = (
                            "replace_common",
                            "inflated_common_box_contains_both_image_centers",
                        )
                    elif in_gap and identity in carrier_ids and contains_both and (not independent):
                        action, reason = (
                            "replace_common",
                            "adjacent_common_contact_of_confirmed_carrier",
                        )
                    elif in_tail and identity in carrier_ids and (delta > gate):
                        action, reason = (
                            "replace_residual",
                            "common_carrier_reacquisition_disagrees_with_image_measurement",
                        )
                elif in_tail:
                    action, reason = ("preserve_missing", "outside_confirmed_common_box_gap")
                replacement = None
                if action in {"add_missing", "replace_common", "replace_residual"}:
                    size = np.array([shape[1], shape[0]])
                    replacement = np.r_[
                        frame,
                        identity,
                        event["anchors"][i, 0, 2],
                        center(rect) / size,
                        (rect[2:] - rect[:2]) / size,
                        0.0,
                    ]
                    rows.append(replacement)
                    if in_tail:
                        tail_changed = True
                        final_end = max(final_end, frame)
                actions.append(
                    {
                        "frame": frame,
                        "source_id": identity,
                        "action": action,
                        "reason": reason,
                        "original_observation": None if len(old) == 0 else old[0].copy(),
                        "replacement_observation": replacement,
                        "center_delta_px": delta,
                        "gate_px": gate,
                    }
                )
                errors.append(
                    {
                        "source_id": identity,
                        "center_delta_px": delta,
                        "gate_px": gate,
                        "common_box_carrier": identity in carrier_ids,
                        "action": action,
                    }
                )
            if in_tail:
                settling_checks.append(
                    {
                        "frame": frame,
                        "errors": errors,
                        "replace_residual_common_box_state": tail_changed,
                    }
                )
    return (
        np.array(rows) if passed else np.empty((0, 8)),
        {
            "status": "IMAGE_MEASUREMENTS_ACCEPTED" if passed else "UNRESOLVED",
            "source_ids": event["partner_ids"],
            "missing_source_id": event["missing_source_id"],
            "missing_frames": [event["original_missing_start"], event["original_missing_end"]],
            "repair_frames": [event["missing_start"], final_end],
            "anchor_frames": [a, b],
            "common_core_frames": sorted({f for f, _ in image_core_keys}),
            "common_carrier_ids": sorted(carrier_ids),
            "core_image_evidence_passed": bool(core_passed),
            "anchor_control_checks": control_checks,
            "actions": actions,
            "reacquisition_checks": settling_checks,
            "roi_xyxy": roi.tolist(),
            "directions": audit,
            "frames": per_frame,
            "accepted_gap_frames": sum((f["accepted"] for f in all_gap)),
            "gap_frames": len(all_gap),
        },
    )


def recover_close_gaps(video_path, labels, *, config=None, frame_window=None):
    config = config or RecoveryConfig()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        raise ValueError("Cannot open source video")
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    events = find_close_gap_events(labels, shape, config, frame_window)
    output = labels.copy()
    audits, replacements = ([], [])
    for event in events:
        rects = np.array([box(row, shape) for row in event["anchors"].reshape(-1, 8)])
        padding = np.max(np.linalg.norm(rects[:, 2:] - rects[:, :2], axis=1))
        low = np.maximum(0, np.floor(rects[:, :2].min(axis=0) - padding)).astype(int)
        high = np.minimum([shape[1], shape[0]], np.ceil(rects[:, 2:].max(axis=0) + padding)).astype(
            int
        )
        roi = np.r_[low, high]
        cap.set(cv2.CAP_PROP_POS_FRAMES, event["anchor_start"] - 1)
        images = {}
        for frame in range(event["anchor_start"], event["anchor_end"] + 1):
            ok, image = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame:
                raise ValueError(f"Cannot decode exact source frame {frame}")
            images[frame] = image[low[1] : high[1], low[0] : high[0]].copy()
        proposals, audit = recover_event(images, event, shape, roi, config, output)
        audits.append(audit)
        if len(proposals):
            keys = {(int(row[0]), int(row[1])) for row in proposals}
            if len(keys) != len(proposals):
                raise AssertionError("Recovery proposed duplicate frame/source keys")
            scope = np.array([(int(row[0]), int(row[1])) in keys for row in output], dtype=bool)
            replacements.append(
                {
                    "event": len(audits) - 1,
                    "original_observations": output[scope].copy(),
                    "new_observations": proposals.copy(),
                    "actions": [
                        action
                        for action in audit["actions"]
                        if action["replacement_observation"] is not None
                    ],
                    "origin": "bidirectional_image_optical_flow",
                    "detector_confidence": None,
                    "confidence_field": "0 sentinel; not current-frame detector confidence",
                }
            )
            output = np.vstack([output[~scope], proposals])
    cap.release()
    output = output[np.lexsort((output[:, 1], output[:, 0]))]
    if len(np.unique(output[:, :2], axis=0)) != len(output):
        raise AssertionError("Recovery produced duplicate frame/source keys")
    return (
        output,
        {
            "config": asdict(config),
            "events": audits,
            "replacements": replacements,
            "original_rows": len(labels),
            "output_rows": len(output),
            "candidate_or_gt_used": False,
            "manual_decisions_used": False,
            "measurement_kind": "image optical flow, independently checked in both directions",
            "detector_score_policy": "0 sentinel for recovered measurements; never use corrected labels for detector AP",
        },
    )
