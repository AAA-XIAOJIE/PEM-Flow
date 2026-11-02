"""Low-threshold detector AP on the fixed native-resolution Tier A frames."""

from __future__ import annotations

import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from pemflow.evaluation.detection import CLASSES, evaluate
from pemflow.evaluation.quality import attrs
from pemflow.io import read_json, runtime_metadata, write_json


def ground_truth_frames(validation, sample_metadata=None):
    validation = Path(validation)
    with (validation / "sampling_plan.csv").open(encoding="utf-8-sig") as stream:
        samples = [row for row in csv.DictReader(stream) if row["tier"] == "A"]
    truth = {}
    for task in {row["task"] for row in samples}:
        root = ET.parse(validation / "reviewed_work" / (task + "_reviewed.xml")).getroot()
        for image in root.findall("image"):
            width, height = int(image.get("width")), int(image.get("height"))
            boxes = []
            for index, box in enumerate(image.findall("box")):
                attributes = attrs(box)
                boxes.append(
                    dict(
                        index=index,
                        label=box.get("label"),
                        status=attributes["review_status"],
                        xyxy=[
                            float(box.get(name)) / divisor
                            for name, divisor in zip(
                                ["xtl", "ytl", "xbr", "ybr"],
                                [width, height, width, height],
                                strict=True,
                            )
                        ],
                    )
                )
            truth[task, image.get("name")] = boxes
    metadata = {}
    if sample_metadata:
        with Path(sample_metadata).open(encoding="utf-8-sig") as stream:
            metadata = {(row["task"], row["image"]): row for row in csv.DictReader(stream)}
    return [
        dict(
            row,
            image=row["output_image"],
            gt=truth[row["task"], row["output_image"]],
            stratum=metadata[row["task"], row["output_image"]]["evaluation_cohort"]
            if metadata
            else row["stratum"],
        )
        for row in samples
    ]


def infer_frames(frames, video_root, model, config, device, output):
    import cv2

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from ultralytics import YOLO

    detector = YOLO(str(model))
    cfg = read_json(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capture, current = None, None
    try:
        for index, frame in enumerate(frames, 1):
            if current != frame["video_file"]:
                if capture is not None:
                    capture.release()
                current = frame["video_file"]
                capture = cv2.VideoCapture(str(Path(video_root) / current))
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame["local_frame"]) - 1)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode {current} frame {frame['local_frame']}")
            prediction = detector.predict(
                image,
                conf=cfg["evaluation_detection_confidence_threshold"],
                iou=cfg["nms_iou_threshold"],
                imgsz=cfg["image_size"],
                agnostic_nms=cfg["agnostic_nms"],
                max_det=300,
                half=False,
                rect=True,
                batch=1,
                augment=False,
                classes=None,
                device=device,
                save=False,
                verbose=False,
            )[0]
            frame["detections"] = [
                dict(
                    raw_class=int(cls),
                    confidence=float(conf),
                    xyxy=[float(value) for value in xyxy],
                )
                for cls, conf, xyxy in zip(
                    prediction.boxes.cls.cpu().tolist(),
                    prediction.boxes.conf.cpu().tolist(),
                    prediction.boxes.xyxyn.cpu().tolist(),
                    strict=True,
                )
            ]
            write_json(
                output / (Path(frame["image"]).stem + ".json"),
                dict(task=frame["task"], image=frame["image"], detections=frame["detections"]),
            )
            if index % 30 == 0 or index == len(frames):
                print(f"Detector AP inference {index}/{len(frames)}", flush=True)
    finally:
        if capture is not None:
            capture.release()
    return frames


def score_frames(frames, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, all_curves, events = [], [], []
    cohorts = [("ALL", frames)] + [
        (group, [frame for frame in frames if frame["group"] == group])
        for group in ["G01", "G07", "G15"]
    ]
    cohorts += [
        (name, [frame for frame in frames if frame["stratum"] == name])
        for name in sorted({frame["stratum"] for frame in frames})
    ]
    for cohort, selected in cohorts:
        for threshold in [round(0.5 + 0.05 * index, 2) for index in range(10)]:
            rows, curves, frame_events = evaluate(
                selected, threshold=threshold, ignore_iou=0.5, operating_confidence=0.3
            )
            all_rows.extend(dict(cohort=cohort, **row) for row in rows)
            if cohort == "ALL":
                all_curves.extend(curves)
                if threshold == 0.5:
                    events = frame_events

    def mean(values):
        available = [value for value in values if value is not None]
        return sum(available) / len(available) if available else None

    pooled = [row for row in all_rows if row["cohort"] == "ALL"]
    summary = dict(
        status="COMPLETE",
        frame_count=len(frames),
        map50=mean(row["ap101"] for row in pooled if row["iou_threshold"] == 0.5),
        map50_95=mean(row["ap101"] for row in pooled),
        per_class=[
            dict(
                class_=cls,
                ap50=mean(
                    row["ap101"]
                    for row in pooled
                    if row["class"] == cls and row["iou_threshold"] == 0.5
                ),
                ap50_95=mean(row["ap101"] for row in pooled if row["class"] == cls),
            )
            for cls in CLASSES
        ],
        gt_counts=dict(
            valid=sum(box["status"] == "valid" for frame in frames for box in frame["gt"]),
            ignore=sum(box["status"] != "valid" for frame in frames for box in frame["gt"]),
        ),
        runtime=runtime_metadata(),
    )
    pd.DataFrame(all_rows).to_csv(output / "ap_by_cohort_class_iou.csv", index=False)
    pd.DataFrame(all_curves).to_csv(output / "precision_recall_curves.csv", index=False)
    pd.DataFrame(events).to_csv(output / "prediction_events_iou50.csv", index=False)
    write_json(output / "summary.json", summary)
    return summary
