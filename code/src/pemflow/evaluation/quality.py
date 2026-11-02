"""Image-space detection/tracking and projected-centre evaluation against reviewed GT."""

from __future__ import annotations

import collections
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.geometry import Polygon

from pemflow.evaluation import operating
from pemflow.evaluation.tracking import evaluate_tracking
from pemflow.io import intermediate_paths, read_json, table_paths, write_json
from pemflow.pipeline.motion import describe, project

CLASSES = {0: "pedestrian", 9: "e-bike", 3: "motor-vehicle"}


def attrs(element):
    return {item.get("name"): item.text for item in element.findall("attribute")}


def write_rows(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def manual_boxes(dataset):
    """Read the same explicit boxes that generated the curated CSV."""
    path = Path(dataset) / "applied_rules.json"
    result = {}
    if path.is_file():
        for rule in read_json(path)["rules"]:
            if rule["kind"] in ("position", "insert") and rule["space"] in (
                "bbox_pixels",
                "bbox_normalized",
            ):
                for item in rule["values"]:
                    result[rule["segment"], item["frame"], item["source_id"]] = (
                        rule["space"],
                        item["value"],
                    )
    return result


def evaluate_dataset(input_root, automatic_root, validation, config, output):
    input_root, automatic_root, validation, output = map(
        Path, (input_root, automatic_root, validation, output)
    )
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    dataset = read_json(automatic_root / "dataset.json")
    with (validation / "sampling_plan.csv").open(encoding="utf-8-sig") as stream:
        samples = list(csv.DictReader(stream))
    with (validation / "sample_metadata.csv").open(encoding="utf-8-sig") as stream:
        metadata = {(row["task"], row["image"]): row for row in csv.DictReader(stream)}
    truth, roots, images = {}, {}, {}
    for task in sorted({row["task"] for row in samples}):
        source = ET.parse(validation / "reviewed_work" / (task + "_reviewed.xml")).getroot()
        roots[task] = ET.Element("annotations")
        for image in source.findall("image"):
            key = task, image.get("name")
            truth[key] = image
            images[key] = ET.SubElement(roots[task], "image", image.attrib)
    roi = Polygon(read_json(config)["roi_polygon_m"])
    boxes = manual_boxes(input_root)
    frames_b, differences, mapping = [], [], []
    for item in dataset["segments"]:
        key = item["key"]
        chosen = [row for row in samples if row["video_file"] == item["source_video"]["filename"]]
        if not chosen:
            continue
        paths = table_paths(input_root, key)
        scene, lineage = pd.read_csv(paths["trajectory"]), pd.read_csv(paths["lineage"])
        if not np.array_equal(scene[["frame", "id", "class"]], lineage[["frame", "id", "class"]]):
            raise ValueError("CSV and lineage keys differ: " + key)
        scene[["source_track_id", "observation_id", "raw_x", "raw_y"]] = lineage[
            ["source_track_id", "observation_id", "raw_x", "raw_y"]
        ]
        inputs = intermediate_paths(automatic_root, key)
        with np.load(inputs["observations"], allow_pickle=False) as archive:
            labels = archive["observations"].copy()
        with np.load(inputs["matrices"], allow_pickle=False) as archive:
            matrices = archive["matrices"].copy()
        by_frame = {
            int(frame): group
            for frame, group in scene.groupby("frame")
            if int(frame) in {int(row["local_frame"]) for row in chosen}
        }
        for sample in chosen:
            frame, task, image_name = (
                int(sample["local_frame"]),
                sample["task"],
                sample["output_image"],
            )
            reference = truth[task, image_name]
            width, height = int(reference.get("width")), int(reference.get("height"))
            rows = by_frame.get(frame, scene.iloc[:0])
            inside = rows.loc[contains_xy(roi, rows.real_x.to_numpy(), rows.real_y.to_numpy())]
            counts = inside["class"].value_counts()
            info = metadata[task, image_name]
            info.update(
                n_in_roi=str(len(inside)),
                n_ped=str(int(counts.get(0, 0))),
                n_ebike=str(int(counts.get(9, 0))),
                n_vehicle=str(int(counts.get(3, 0))),
                phase="with_vehicle" if counts.get(3, 0) else "no_vehicle",
            )
            if sample["tier"] == "A":
                info["evaluation_cohort"] = info["phase"]
            preds = []
            for row in rows.itertuples():
                manual = boxes.get((key, frame, int(row.source_track_id)))
                if manual:
                    space, bbox = manual
                    bbox = np.asarray(bbox, float)
                    if space == "bbox_pixels":
                        bbox /= [
                            item["profile"]["image_width_px"],
                            item["profile"]["image_height_px"],
                        ] * 2
                    cx, cy, w, h = bbox
                else:
                    if row.observation_id < 1:
                        raise ValueError("Manual CSV insertion is missing its submitted bbox")
                    raw = labels[int(row.observation_id) - 1]
                    if raw[0] != frame or raw[1] != row.source_track_id:
                        raise ValueError("Prediction observation points to a different source row")
                    cx, cy, w, h = raw[3:7]
                bb = dict(
                    xtl=max(0.0, (cx - w / 2) * width),
                    ytl=max(0.0, (cy - h / 2) * height),
                    xbr=min(float(width), (cx + w / 2) * width),
                    ybr=min(float(height), (cy + h / 2) * height),
                )
                cls = int(rows.loc[row.Index, "class"])
                prediction = dict(
                    label=CLASSES[cls],
                    identity=int(row.id),
                    **bb,
                    real_x=row.real_x,
                    real_y=row.real_y,
                    raw_x=row.raw_x,
                    raw_y=row.raw_y,
                )
                preds.append(prediction)
                box = ET.SubElement(
                    images[task, image_name],
                    "box",
                    label=CLASSES[cls],
                    occluded="0",
                    z_order="0",
                    **{k: str(v) for k, v in bb.items()},
                )
                for name, value in [
                    ("pred_track_id", int(row.id)),
                    ("source_track_id", int(row.source_track_id)),
                ]:
                    ET.SubElement(box, "attribute", name=name).text = str(value)
                mapping.append(
                    dict(
                        task=task,
                        image=image_name,
                        segment=key,
                        frame=frame,
                        observation_id=int(row.observation_id),
                        id=int(row.id),
                        source_track_id=int(row.source_track_id),
                        class_=cls,
                    )
                )
            valid, ignored = [], []
            for box in reference.findall("box"):
                attributes = attrs(box)
                bb = dict(
                    label=box.get("label"),
                    identity=int(attributes.get("gt_id") or 0),
                    **{k: float(box.get(k)) for k in ["xtl", "ytl", "xbr", "ybr"]},
                )
                (valid if attributes["review_status"] == "valid" else ignored).append(bb)
            if sample["tier"] == "B":
                frames_b.append(
                    SimpleNamespace(
                        task=task,
                        image_name=image_name,
                        frame_number=int(sample["global_frame"]),
                        valid_gt=tuple(SimpleNamespace(**box) for box in valid),
                        ignored_gt=tuple(SimpleNamespace(**box) for box in ignored),
                        tracking_predictions=tuple(SimpleNamespace(**box) for box in preds),
                    )
                )
            matches, _, _ = operating._greedy_match(preds, valid, 0.5, class_aware=True)
            if matches:
                reverse = {value: code for code, value in CLASSES.items()}
                centres = pd.DataFrame(
                    [
                        dict(
                            frame=frame,
                            cx_norm=(valid[j]["xtl"] + valid[j]["xbr"]) / 2 / width,
                            cy_norm=(valid[j]["ytl"] + valid[j]["ybr"]) / 2 / height,
                            **{"class": reverse[valid[j]["label"]]},
                        )
                        for _, j, _ in matches
                    ]
                )
                projected = project(centres, item["profile"], matrices)
                for (i, _, iou), xy in zip(matches, projected, strict=True):
                    prediction = preds[i]
                    differences.append(
                        dict(
                            tier=sample["tier"],
                            task=task,
                            image=image_name,
                            segment=key,
                            frame=frame,
                            class_=prediction["label"],
                            iou=iou,
                            raw_distance_m=float(
                                np.linalg.norm(
                                    np.array([prediction["raw_x"], prediction["raw_y"]]) - xy
                                )
                            ),
                            distance_m=float(
                                np.linalg.norm(
                                    np.array([prediction["real_x"], prediction["real_y"]]) - xy
                                )
                            ),
                        )
                    )
        print(f"{key} evaluated: {len(chosen)} reviewed frames", flush=True)
    for task, root in roots.items():
        ET.ElementTree(root).write(
            predictions / (task + ".xml"), encoding="utf-8", xml_declaration=True
        )
    write_rows(output / "sample_metadata.csv", list(metadata.values()))
    metrics_a, confusion, frame_a = operating.evaluate(
        predictions,
        validation / "reviewed_work",
        list(operating.TASKS),
        0.5,
        sample_metadata=metadata,
    )
    pooled = {
        name: collections.Counter(tp=0, fp=0, fn=0, ignored_predictions=0)
        for name in [*CLASSES.values(), "ALL"]
    }
    for row in metrics_a:
        if row["task"] == "ALL" and row["stratum"] == "ALL":
            for field in ("tp", "fp", "fn", "ignored_predictions"):
                pooled[row["class"]][field] += row[field]
                pooled["ALL"][field] += row[field]
    for value in pooled.values():
        tp, fp, fn = value["tp"], value["fp"], value["fn"]
        value.update(
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        )
    metrics_b, events = [], []
    for task in [*sorted({frame.task for frame in frames_b}), "ALL"]:
        selected = (
            frames_b if task == "ALL" else [frame for frame in frames_b if frame.task == task]
        )
        for cls in [None, *CLASSES.values()]:
            result = evaluate_tracking(
                selected,
                threshold=0.5,
                class_name=cls,
                events=events if cls is None and task != "ALL" else None,
            )
            tp, fp, fn = result["matches"], result["false_positives"], result["false_negatives"]
            result.update(
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            )
            metrics_b.append(dict(task=task, **result))
    geometry_summary = [
        dict(
            tier=tier,
            class_=cls,
            raw=describe(
                [
                    row["raw_distance_m"]
                    for row in differences
                    if row["tier"] == tier and (cls == "ALL" or row["class_"] == cls)
                ]
            ),
            processed=describe(
                [
                    row["distance_m"]
                    for row in differences
                    if row["tier"] == tier and (cls == "ALL" or row["class_"] == cls)
                ]
            ),
        )
        for tier in ["A", "B"]
        for cls in [*CLASSES.values(), "ALL"]
    ]
    for name, rows in [
        ("tier_a_metrics", metrics_a),
        ("tier_a_frames", frame_a),
        ("tier_b_metrics", metrics_b),
        ("tier_b_events", events),
        ("position_differences", differences),
        ("observation_mapping", mapping),
    ]:
        write_rows(output / (name + ".csv"), rows)
    summary = dict(
        status="COMPLETE",
        scope="Full sampled roadway; reviewed ignore regions are retained.",
        tier_a=dict(
            frames=len(frame_a), pooled=pooled, class_agnostic_matches=sum(confusion.values())
        ),
        tier_b=[row for row in metrics_b if row["task"] == "ALL"],
        position_difference=geometry_summary,
        position_reference="Same-camera projected reviewed box centres; not independent surveyed accuracy.",
    )
    write_json(output / "summary.json", summary)
    return summary
