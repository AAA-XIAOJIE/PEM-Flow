from __future__ import annotations

import math

import numpy as np

CLASSES = ("pedestrian", "e-bike", "motor-vehicle")
RAW_CLASSES = {0: "pedestrian", 9: "e-bike", 3: "motor-vehicle"}


def overlaps(boxes, truths):
    a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    b = np.asarray(truths, dtype=float).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def ranked_ap(tp, fp, gt_count):
    tp, fp = (np.asarray(tp, dtype=int), np.asarray(fp, dtype=int))
    if tp.shape != fp.shape or np.any(tp + fp != 1) or np.any(tp < 0) or np.any(fp < 0):
        raise ValueError("Each scored prediction must be exactly one TP or FP")
    if gt_count < 0 or int(tp.sum()) > gt_count:
        raise ValueError("Invalid GT count or repeated matching of ground truth")
    levels = np.linspace(0, 1, 101)
    envelope = np.zeros(101, dtype=float)
    if gt_count == 0:
        return (None, None, levels, envelope)
    recall = tp.cumsum() / gt_count
    precision = tp.cumsum() / np.maximum(tp.cumsum() + fp.cumsum(), 1)
    if len(tp):
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(recall, levels, side="left")
        valid = positions < len(precision)
        envelope[valid] = precision[positions[valid]]
    all_point = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return (float(envelope.mean()), all_point, levels, envelope)


def evaluate(frames, *, threshold=0.5, ignore_iou=0.5, operating_confidence=0.3):
    if not all((math.isfinite(v) and 0 < v <= 1 for v in (threshold, ignore_iou))):
        raise ValueError("IoU thresholds must lie in (0,1]")
    if not 0 <= operating_confidence <= 1:
        raise ValueError("Invalid operating confidence")
    keys = [(f["task"], f["image"]) for f in frames]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated task/image in evaluation")
    events = []
    gt_counts = dict.fromkeys(CLASSES, 0)
    for frame in sorted(frames, key=lambda f: (f["task"], f["image"])):
        ignored = [g["xyxy"] for g in frame["gt"] if g["status"] == "ignore"]
        for cls in CLASSES:
            truths = [g for g in frame["gt"] if g["status"] == "valid" and g["label"] == cls]
            gt_counts[cls] += len(truths)
            predictions = [
                (index, p)
                for index, p in enumerate(frame["detections"])
                if RAW_CLASSES.get(p["raw_class"]) == cls
            ]
            predictions.sort(key=lambda item: (-item[1]["confidence"], item[0]))
            for _, prediction in predictions:
                conf = prediction["confidence"]
                if not math.isfinite(conf) or not 0 <= conf <= 1:
                    raise ValueError("Nonfinite or out-of-range prediction confidence")
            valid_iou = overlaps([p["xyxy"] for _, p in predictions], [g["xyxy"] for g in truths])
            ignored_iou = overlaps([p["xyxy"] for _, p in predictions], ignored)
            used = set()
            for pi, (original_index, prediction) in enumerate(predictions):
                candidates = [
                    gi
                    for gi in range(len(truths))
                    if gi not in used and valid_iou[pi, gi] >= threshold
                ]
                chosen = (
                    max(candidates, key=lambda gi: (valid_iou[pi, gi], -gi)) if candidates else None
                )
                if chosen is not None:
                    used.add(chosen)
                    status = "tp"
                elif ignored and float(ignored_iou[pi].max()) >= ignore_iou:
                    status = "ignore"
                else:
                    status = "fp"
                events.append(
                    {
                        "task": frame["task"],
                        "image": frame["image"],
                        "group": frame["group"],
                        "stratum": frame["stratum"],
                        "class": cls,
                        "prediction_index": original_index,
                        "confidence": prediction["confidence"],
                        "iou_threshold": threshold,
                        "status": status,
                        "matched_gt_index": truths[chosen]["index"] if chosen is not None else None,
                    }
                )
    rows, curves = ([], [])
    for cls in CLASSES:
        items = [e for e in events if e["class"] == cls and e["status"] != "ignore"]
        items.sort(key=lambda e: (-e["confidence"], e["task"], e["image"], e["prediction_index"]))
        tp_flags = [int(e["status"] == "tp") for e in items]
        fp_flags = [int(e["status"] == "fp") for e in items]
        ap101, ap_all, levels, envelope = ranked_ap(tp_flags, fp_flags, gt_counts[cls])
        selected = [e for e in items if e["confidence"] >= operating_confidence]
        tp = sum((e["status"] == "tp" for e in selected))
        fp = len(selected) - tp
        fn = gt_counts[cls] - tp
        rows.append(
            {
                "class": cls,
                "iou_threshold": threshold,
                "gt": gt_counts[cls],
                "tp_at_0_3": tp,
                "fp_at_0_3": fp,
                "fn_at_0_3": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / gt_counts[cls] if gt_counts[cls] else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "ap101": ap101,
                "ap_all_points": ap_all,
                "retained_predictions": len(items),
                "ignored_predictions": sum(
                    (e["class"] == cls and e["status"] == "ignore" for e in events)
                ),
                "maximum_recall": sum(tp_flags) / gt_counts[cls] if gt_counts[cls] else None,
            }
        )
        curves.extend(
            (
                {
                    "class": cls,
                    "iou_threshold": threshold,
                    "recall": float(r),
                    "interpolated_precision": float(p),
                }
                for r, p in zip(levels, envelope, strict=True)
            )
        )
    return (rows, curves, events)
