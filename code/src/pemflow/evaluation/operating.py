from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("pedestrian", "e-bike", "motor-vehicle")
TASKS = ("G01_tierA_detection", "G07_tierA_detection", "G15_tierA_detection")
REVIEWED_STATUSES = {"valid", "ignore", "ambiguous"}


def _safe_float(value: str | None) -> float:
    try:
        result = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")
    return result


def _attributes(box: ET.Element) -> dict[str, str]:
    return {node.get("name", ""): node.text or "" for node in box.findall("attribute")}


def _load_boxes(
    path: Path, *, reviewed: bool
) -> tuple[dict[str, list[dict[str, object]]], dict[str, tuple[int, int]]]:
    result: dict[str, list[dict[str, object]]] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    root = ET.parse(path).getroot()
    for image in root.findall("image"):
        name = image.get("name")
        if not name or name in result:
            raise ValueError(f"Missing or duplicate image name in {path}: {name!r}")
        try:
            width = int(image.get("width", "0"))
            height = int(image.get("height", "0"))
        except ValueError:
            width, height = (0, 0)
        dimensions[name] = (width, height)
        items: list[dict[str, object]] = []
        for box in image.findall("box"):
            attributes = _attributes(box)
            item: dict[str, object] = {
                "label": box.get("label"),
                "xtl": _safe_float(box.get("xtl")),
                "ytl": _safe_float(box.get("ytl")),
                "xbr": _safe_float(box.get("xbr")),
                "ybr": _safe_float(box.get("ybr")),
                "status": attributes.get("review_status", "unreviewed")
                if reviewed
                else "prediction",
            }
            items.append(item)
        result[name] = items
    return (result, dimensions)


def _validate_pair(
    task: str,
    predictions: dict[str, list[dict[str, object]]],
    ground_truth: dict[str, list[dict[str, object]]],
    prediction_dimensions: dict[str, tuple[int, int]],
    ground_truth_dimensions: dict[str, tuple[int, int]],
) -> None:
    errors: list[str] = []
    if len(predictions) != 90 or len(ground_truth) != 90:
        errors.append(
            f"expected 90 images; prediction={len(predictions)}, reviewed={len(ground_truth)}"
        )
    if set(predictions) != set(ground_truth):
        missing_review = sorted(set(predictions) - set(ground_truth))
        missing_prediction = sorted(set(ground_truth) - set(predictions))
        errors.append(
            f"image-name mismatch; missing reviewed={missing_review[:5]}, missing prediction={missing_prediction[:5]}"
        )
    for name in sorted(set(predictions) & set(ground_truth)):
        prediction_size = prediction_dimensions.get(name, (0, 0))
        reviewed_size = ground_truth_dimensions.get(name, (0, 0))
        if prediction_size != reviewed_size or min(*reviewed_size) <= 0:
            errors.append(
                f"{name}: invalid or inconsistent image dimensions; prediction={prediction_size}, reviewed={reviewed_size}"
            )
    for name, boxes in ground_truth.items():
        width, height = ground_truth_dimensions.get(name, (0, 0))
        for index, box in enumerate(boxes):
            label = str(box["label"])
            status = str(box["status"])
            coordinates = [float(box[key]) for key in ("xtl", "ytl", "xbr", "ybr")]
            if label not in CLASSES:
                errors.append(f"{name} box {index}: invalid label {label!r}")
            if status not in REVIEWED_STATUSES:
                errors.append(f"{name} box {index}: incomplete review status {status!r}")
            if not all((math.isfinite(value) for value in coordinates)):
                errors.append(f"{name} box {index}: non-finite coordinates")
            elif not (
                0 <= coordinates[0] < coordinates[2] <= width
                and 0 <= coordinates[1] < coordinates[3] <= height
            ):
                errors.append(f"{name} box {index}: invalid or out-of-bounds coordinates")
            if len(errors) >= 20:
                break
        if len(errors) >= 20:
            break
    for name, boxes in predictions.items():
        width, height = prediction_dimensions.get(name, (0, 0))
        for index, box in enumerate(boxes):
            if str(box["label"]) not in CLASSES:
                errors.append(f"{name} prediction {index}: invalid label {box['label']!r}")
            coordinates = [float(box[key]) for key in ("xtl", "ytl", "xbr", "ybr")]
            if not all((math.isfinite(value) for value in coordinates)):
                errors.append(f"{name} prediction {index}: non-finite coordinates")
            elif not (
                0 <= coordinates[0] < coordinates[2] <= width
                and 0 <= coordinates[1] < coordinates[3] <= height
            ):
                errors.append(f"{name} prediction {index}: invalid or out-of-bounds coordinates")
            if len(errors) >= 20:
                break
        if len(errors) >= 20:
            break
    if errors:
        raise ValueError(f"Tier A task {task} is incomplete:\n- " + "\n- ".join(errors))


def _iou(first: dict[str, object], second: dict[str, object]) -> float:
    x1 = max(float(first["xtl"]), float(second["xtl"]))
    y1 = max(float(first["ytl"]), float(second["ytl"]))
    x2 = min(float(first["xbr"]), float(second["xbr"]))
    y2 = min(float(first["ybr"]), float(second["ybr"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first["xbr"]) - float(first["xtl"])) * max(
        0.0, float(first["ybr"]) - float(first["ytl"])
    )
    second_area = max(0.0, float(second["xbr"]) - float(second["xtl"])) * max(
        0.0, float(second["ybr"]) - float(second["ytl"])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _greedy_match(
    predictions: list[dict[str, object]],
    ground_truth: list[dict[str, object]],
    threshold: float,
    *,
    class_aware: bool,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(ground_truth):
            if class_aware and prediction["label"] != truth["label"]:
                continue
            score = _iou(prediction, truth)
            if score >= threshold:
                candidates.append((score, prediction_index, truth_index))
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, prediction_index, truth_index in candidates:
        if prediction_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_index)
        matches.append((prediction_index, truth_index, score))
    return (matches, used_predictions, used_truth)


def _stratum(name: str) -> str:
    for value in ("no_vehicle", "with_vehicle", "high_occupancy"):
        if value in name:
            return value
    raise ValueError(f"Cannot infer Tier A stratum from image name: {name}")


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else float("nan")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = (math.floor(position), math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def evaluate(
    prediction_dir: Path,
    reviewed_dir: Path,
    tasks: list[str],
    threshold: float,
    *,
    sample_metadata: dict | None = None,
) -> tuple[list[dict[str, object]], Counter[tuple[str, str]], list[dict[str, object]]]:
    aggregates: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(
        lambda: Counter(tp=0, fp=0, fn=0, ignored_pred=0)
    )
    confusion: Counter[tuple[str, str]] = Counter()
    localization: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    frame_rows: list[dict[str, object]] = []
    for task in tasks:
        prediction_path = prediction_dir / f"{task}.xml"
        reviewed_path = reviewed_dir / f"{task}_reviewed.xml"
        predictions, prediction_dimensions = _load_boxes(prediction_path, reviewed=False)
        ground_truth, ground_truth_dimensions = _load_boxes(reviewed_path, reviewed=True)
        _validate_pair(
            task, predictions, ground_truth, prediction_dimensions, ground_truth_dimensions
        )
        for name, prediction_items in predictions.items():
            truth_items = ground_truth[name]
            valid = [item for item in truth_items if item["status"] == "valid"]
            excluded = [item for item in truth_items if item["status"] in {"ignore", "ambiguous"}]
            group = (
                sample_metadata[task, name]["evaluation_cohort"]
                if sample_metadata is not None
                else _stratum(name)
            )
            row: dict[str, object] = {
                "task": task,
                "stratum": group,
                "image": name,
                "predictions": len(prediction_items),
                "valid_gt": len(valid),
            }
            for class_name in CLASSES:
                class_predictions = [
                    item for item in prediction_items if item["label"] == class_name
                ]
                class_truth = [item for item in valid if item["label"] == class_name]
                matches, used_predictions, used_truth = _greedy_match(
                    class_predictions, class_truth, threshold, class_aware=True
                )
                ignored = {
                    index
                    for index, prediction in enumerate(class_predictions)
                    if index not in used_predictions
                    and any((_iou(prediction, item) >= threshold for item in excluded))
                }
                true_positive = len(matches)
                false_positive = len(class_predictions) - len(used_predictions) - len(ignored)
                false_negative = len(class_truth) - len(used_truth)
                keys = (
                    (task, group, class_name),
                    (task, "ALL", class_name),
                    ("ALL", group, class_name),
                    ("ALL", "ALL", class_name),
                )
                for key in keys:
                    aggregates[key].update(
                        tp=true_positive,
                        fp=false_positive,
                        fn=false_negative,
                        ignored_pred=len(ignored),
                    )
                scores = [score for _, _, score in matches]
                localization[task, group, class_name].extend(scores)
                localization["ALL", "ALL", class_name].extend(scores)
                row[f"{class_name}_tp"] = true_positive
                row[f"{class_name}_fp"] = false_positive
                row[f"{class_name}_fn"] = false_negative
            matches, _, _ = _greedy_match(prediction_items, valid, threshold, class_aware=False)
            for prediction_index, truth_index, _ in matches:
                confusion[
                    str(valid[truth_index]["label"]),
                    str(prediction_items[prediction_index]["label"]),
                ] += 1
            frame_rows.append(row)
    metrics: list[dict[str, object]] = []
    for (task, group, class_name), counts in sorted(aggregates.items()):
        true_positive = counts["tp"]
        false_positive = counts["fp"]
        false_negative = counts["fn"]
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = (
            _safe_divide(2 * precision * recall, precision + recall)
            if math.isfinite(precision) and math.isfinite(recall)
            else float("nan")
        )
        scores = localization.get((task, group, class_name), [])
        metrics.append(
            {
                "task": task,
                "stratum": group,
                "class": class_name,
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "ignored_predictions": counts["ignored_pred"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_iou_median": statistics.median(scores) if scores else float("nan"),
                "matched_iou_p05": _percentile(scores, 0.05),
                "matched_iou_count": len(scores),
            }
        )
    return (metrics, confusion, frame_rows)
