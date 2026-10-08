"""Video observations and camera calibration to trajectory and motion tables."""

from __future__ import annotations

import collections
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from pemflow.io import COLUMNS, read_json, runtime_metadata, write_json
from pemflow.pipeline.matrix_features import rebase_video_matrices
from pemflow.pipeline.matrix_quality import AutomaticRepairConfig, apply_automatic_matrix_repairs
from pemflow.pipeline.motion import process
from pemflow.pipeline.observations import RecoveryConfig, recover_close_gaps
from pemflow.pipeline.stabilization import generate_stabilization_matrices

MOTION_COLUMNS = [
    "frame",
    "id",
    "class",
    "time_s",
    "velocity",
    "vx",
    "vy",
    "ax",
    "ay",
    "acceleration_norm",
    "tangential_acceleration",
    "motion_segment",
    "motion_segment_frames",
    "motion_window",
    "motion_status",
    "adaptive_motion_window",
]
PROVENANCE_COLUMNS = [
    "observation_origin",
    "recovery_action",
    "recovery_reason",
    "recovery_event_index",
    "raw_inference_key_present",
    "raw_inference_frame",
    "raw_inference_source_track_id",
    "raw_inference_observation_id",
    "raw_inference_confidence",
]
LINEAGE_COLUMNS = [
    "observation_id",
    "frame",
    "source_track_id",
    "id",
    "raw_class",
    "class",
    "class_edit",
    "identity_edit",
    "class_unified",
    "motion_segment",
    "motion_segment_frames",
    "smoothed",
    "raw_x",
    "raw_y",
    "motion_window",
    "motion_status",
    "adaptive_motion_window",
] + PROVENANCE_COLUMNS


def save_matrices(path, matrices):
    np.savez_compressed(path, frame_index=np.arange(len(matrices)), matrices=matrices)


def produce(dataset_path, key, video_root, labels_path, output, *, threads=8, resume=False):
    dataset = read_json(dataset_path)
    item = next(item for item in dataset["segments"] if item["key"] == key)
    count = item["profile"]["frames"]
    video = Path(video_root) / item["source_video"]["filename"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(labels_path, allow_pickle=False) as archive:
        raw_observations = archive["observations"].copy()
    started = time.monotonic()

    def progress(stage, frame, total):
        if frame % 150 == 0 or frame == total:
            write_json(
                output / "progress.json", dict(key=key, stage=stage, frame=frame, total=total)
            )
            print(f"{key} {stage} {frame}/{total}", flush=True)

    base_path = output / "base_matrices.npz"
    diagnostic_path = output / "stabilization_frames.jsonl"
    if resume and (output / "base_done.json").exists():
        with np.load(base_path, allow_pickle=False) as archive:
            base = archive["matrices"].copy()
        diagnostics = [
            json.loads(line) for line in diagnostic_path.read_text(encoding="utf-8").splitlines()
        ]
    else:
        cv2.setNumThreads(threads)
        diagnostics = []
        with diagnostic_path.open("w", encoding="utf-8", newline="\n") as stream:

            def record(value):
                diagnostics.append(value)
                stream.write(json.dumps(value, allow_nan=False) + "\n")

            matrices = generate_stabilization_matrices(
                video,
                item["stabilization_reference_index"],
                **dataset["stabilization"],
                diagnostics_callback=record,
                progress_callback=lambda frame, total: progress("stabilization", frame, total),
            )
        if set(matrices) != set(range(count)):
            raise ValueError("Stabilization coverage is incomplete")
        base = np.stack([matrices[index] for index in range(count)])
        save_matrices(base_path, base)
        write_json(output / "base_done.json", dict(frames=count))
    cv2.setNumThreads(1)
    matrix_path = output / "matrices.npz"
    if resume and (output / "matrix_done.json").exists():
        with np.load(matrix_path, allow_pickle=False) as archive:
            matrices = archive["matrices"].copy()
    else:
        repaired, audit, quality = apply_automatic_matrix_repairs(
            video,
            raw_observations,
            base,
            item["stabilization_reference_index"],
            diagnostics,
            AutomaticRepairConfig(**dataset["automatic_matrix_repairs"]["config"]),
            progress_callback=lambda frame, total: progress("matrix audit", frame, total),
        )
        matrices = rebase_video_matrices(repaired, item["profile"]["reference_frame_index"])
        save_matrices(matrix_path, matrices)
        write_json(output / "matrix_repair_audit.json", audit)
        write_json(output / "matrix_quality_summary.json", quality)
        write_json(output / "matrix_done.json", dict(frames=count))
    observations_path = output / "observations.npz"
    if resume and (output / "observations_done.json").exists():
        with np.load(observations_path, allow_pickle=False) as archive:
            observations = archive["observations"].copy()
        audit = read_json(output / "observation_recovery.json")
    else:
        print(f"{key} image recovery", flush=True)
        observations, audit = recover_close_gaps(
            video,
            raw_observations,
            config=RecoveryConfig(**dataset["automatic_observations"]["config"]),
        )
        np.savez_compressed(observations_path, observations=observations)
        write_json(output / "observation_recovery.json", audit)
        write_json(output / "observations_done.json", dict(rows=len(observations)))
    from pemflow.pipeline.provenance import observation_provenance
    from pemflow.pipeline.review import write_review

    cv2.setNumThreads(1)
    _, clean, _, _, statistics, flags = process(observations, item["profile"], matrices)
    provenance = observation_provenance(raw_observations, observations, audit)
    clean = clean.merge(
        provenance, on=["frame", "source_track_id"], how="left", validate="one_to_one"
    )
    destination = output / "trajectory"
    destination.mkdir(exist_ok=True)
    options = dict(index=False, float_format="%.10f", na_rep="", lineterminator="\n")
    clean[COLUMNS].to_csv(destination / "trajectory.csv", **options)
    clean[MOTION_COLUMNS].to_csv(destination / "motion.csv", **options)
    clean[LINEAGE_COLUMNS].to_csv(
        destination / "lineage.csv.gz",
        index=False,
        float_format="%.10f",
        compression={"method": "gzip", "mtime": 0},
    )
    flags.to_csv(destination / "flags.csv", index=False, float_format="%.10f")
    if (destination / "review").exists():
        shutil.rmtree(destination / "review")
    write_review(
        observations,
        clean,
        flags,
        item["profile"],
        key,
        destination,
        labels_path=observations_path,
        video=video,
        diagnostics_path=diagnostic_path,
        matrix_audit_path=output / "matrix_repair_audit.json",
        matrix_quality_path=output / "matrix_quality_summary.json",
    )
    result = dict(
        key=key,
        rows=len(clean),
        frames=count,
        statistics=statistics,
        base_status_counts=dict(collections.Counter(value["status"] for value in diagnostics)),
        runtime=runtime_metadata(),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "done.json", result)
    print(f"{key} CSV complete: {len(clean)} rows", flush=True)
    return result
