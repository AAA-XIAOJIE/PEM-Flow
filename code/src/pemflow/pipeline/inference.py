"""Sequential full-video YOLO11m/ByteTrack inference."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from pemflow.io import read_json, runtime_metadata, sha256_file, write_json
from pemflow.pipeline.serialization import LABEL_COLUMNS, _ChunkWriter, serialize_result


def infer_video_segment(
    dataset_path,
    key,
    video_root,
    model,
    config_root,
    output,
    *,
    device="0",
    max_frames=None,
    progress_every=150,
):
    import cv2
    import torch

    started = time.monotonic()
    dataset = read_json(dataset_path)
    item = next(item for item in dataset["segments"] if item["key"] == key)
    video = Path(video_root) / item["source_video"]["filename"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    # Model import/initialization precedes seeding, as in the verified producer.
    from ultralytics import YOLO

    detector = YOLO(str(model))
    torch.manual_seed(0)
    cv2.setRNGSeed(0)
    limit = int(item["source_video"]["frames"])
    if max_frames is not None:
        limit = min(limit, max_frames)
    parameters = dict(
        dataset["inference"]["parameters"],
        source=str(video),
        device=device,
        tracker=str(Path(config_root) / "bytetrack_pemflow.yaml"),
        project=str(output),
        name="runtime",
        exist_ok=True,
    )
    writer = _ChunkWriter(output / "_chunks", 65536)
    stream = detector.track(**parameters)
    empty_frames, frame = [], 0
    try:
        with (output / "frame_counts.csv").open("w", encoding="utf-8", newline="") as handle:
            counts = csv.writer(handle, lineterminator="\n")
            counts.writerow(["frame", "observations"])
            for frame, result in enumerate(stream, 1):
                if int(detector.predictor.dataset.frame) != frame:
                    raise ValueError("Decoded source frame numbering is inconsistent")
                if tuple(result.orig_shape) != (
                    item["profile"]["image_height_px"],
                    item["profile"]["image_width_px"],
                ):
                    raise ValueError("Video dimensions differ from camera calibration")
                values = serialize_result(result, frame)
                writer.append(values)
                counts.writerow([frame, len(values)])
                if not len(values):
                    empty_frames.append(frame)
                if frame % progress_every == 0 or frame == limit:
                    report = dict(
                        key=key,
                        stage="inference",
                        frame=frame,
                        total=limit,
                        observations=writer.rows,
                        elapsed_seconds=time.monotonic() - started,
                    )
                    write_json(output / "progress.json", report)
                    print(
                        f"{key} inference {frame}/{limit} frames, {writer.rows} observations",
                        flush=True,
                    )
                if max_frames is not None and frame == limit:
                    break
        if frame != limit:
            raise ValueError(f"Incomplete video decoding: {frame}/{limit}")
        digest = writer.finalize(output / "labels.npz")
        write_json(output / "empty_frames.json", dict(frames=empty_frames))
        report = dict(
            key=key,
            frames=frame,
            rows=writer.rows,
            label_columns=LABEL_COLUMNS,
            observations_sha256=digest,
            model_sha256=sha256_file(model),
            parameters=parameters,
            runtime=runtime_metadata(),
            elapsed_seconds=time.monotonic() - started,
            complete=max_frames is None,
        )
        write_json(output / ("done.json" if max_frames is None else "sample_done.json"), report)
        return report
    finally:
        stream.close()
