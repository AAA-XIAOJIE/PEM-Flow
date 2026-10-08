from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pemflow.io import sha256_file
from pemflow.pipeline.motion import label_frame

EVENT_COLUMNS = [
    "event_id",
    "segment",
    "reason",
    "source_id_a",
    "source_id_b",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "support_frames",
    "details_json",
]
QUALITY_REASONS = [
    "short_motion_segment",
    "adaptive_motion_window",
    "raw_step_over_1m",
    "speed_over_12m_s",
    "abs_acceleration_over_10m_s2",
]


def review_events(observations, clean, flags, profile, key, diagnostics_path=None):
    raw = label_frame(observations, profile["frames"])
    fps = float(profile["fps"])
    events = []

    def add(reason, a, b, start, end, support, details):
        start, end = (int(start), int(end))
        token = f"{key}:{reason}:{a}:{b}:{start}:{end}"
        events.append(
            dict(
                event_id=hashlib.sha256(token.encode()).hexdigest()[:20],
                segment=key,
                reason=reason,
                source_id_a=a,
                source_id_b=b,
                start_frame=start,
                end_frame=end,
                start_time_s=(start - 1) / fps,
                end_time_s=(end - 1) / fps,
                support_frames=int(support),
                details_json=json.dumps(
                    details, ensure_ascii=False, sort_keys=True, allow_nan=False
                ),
            )
        )

    def intervals(reason, a, b, frames, details):
        frames = np.unique(frames).astype(int)
        for block in np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1):
            if len(block):
                add(reason, a, b, block[0], block[-1], len(block), details)

    for source, part in raw.groupby("source_track_id", sort=True):
        part = part.sort_values("frame", kind="stable")
        frames = part.frame.to_numpy(dtype=int)
        for index in np.flatnonzero(np.diff(frames) > 1):
            start, end = (frames[index] + 1, frames[index + 1] - 1)
            add(
                "source_track_gap",
                int(source),
                None,
                start,
                end,
                end - start + 1,
                dict(
                    before_frame=int(frames[index]),
                    after_frame=int(frames[index + 1]),
                    missing_frames=int(end - start + 1),
                    filled=False,
                ),
            )
        votes = part.raw_class.value_counts().sort_index()
        if len(votes) > 1:
            final = clean.loc[clean.source_track_id.eq(source), "class"].unique()
            add(
                "source_class_disagreement",
                int(source),
                None,
                frames[0],
                frames[-1],
                len(part),
                dict(
                    raw_class_votes={str(int(c)): int(n) for c, n in votes.items()},
                    automatic_majority_classes=[int(c) for c in final],
                    largest_vote_fraction=float(votes.max() / len(part)),
                ),
            )
    pairs = {}
    for frame, part in raw.groupby("frame", sort=True):
        if len(part) < 2:
            continue
        centres = part[["cx_norm", "cy_norm"]].to_numpy()
        size = part[["width_norm", "height_norm"]].to_numpy()
        low, high = (centres - size / 2, centres + size / 2)
        extent = np.maximum(
            0, np.minimum(high[:, None], high[None, :]) - np.maximum(low[:, None], low[None, :])
        )
        intersection = extent[..., 0] * extent[..., 1]
        area = size[:, 0] * size[:, 1]
        union = area[:, None] + area[None, :] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        rows, cols = np.where(np.triu(iou >= 0.8, k=1))
        ids = part.source_track_id.to_numpy(dtype=int)
        for left, right in zip(rows, cols):
            pair = tuple(sorted((int(ids[left]), int(ids[right]))))
            pairs.setdefault(pair, []).append((int(frame), float(iou[left, right])))
    for (a, b), evidence in sorted(pairs.items()):
        intervals(
            "overlapping_source_boxes",
            a,
            b,
            [x[0] for x in evidence],
            dict(
                report_iou_threshold=0.8,
                maximum_iou=max((x[1] for x in evidence)),
                boxes_changed=False,
            ),
        )
    for reason in QUALITY_REASONS:
        if reason not in flags:
            continue
        selected = flags.loc[flags[reason].fillna(False).astype(bool)]
        for source, part in selected.groupby("source_track_id", sort=True):
            details = dict(rows_preserved=True)
            if "motion_window" in part:
                details["motion_windows_used"] = sorted(
                    part.motion_window.unique().astype(int).tolist()
                )
            for field in ["raw_step_m", "velocity_m_s", "acceleration_m_s2"]:
                values = part[field].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    details["maximum_abs_" + field] = float(np.abs(values).max())
            intervals(reason, int(source), None, part.frame.to_numpy(), details)
    if diagnostics_path is not None:
        fallback = {}
        with Path(diagnostics_path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("fallback_reason"):
                    reason = str(record["fallback_reason"])
                    fallback.setdefault(reason, []).append(int(record["frame_index"]) + 1)
        for reason, frames in sorted(fallback.items()):
            intervals(
                "stabilization_fallback",
                None,
                None,
                frames,
                dict(
                    fallback_reason=reason,
                    matrices_changed=False,
                    evidence="existing estimation diagnostics; not independent accuracy",
                ),
            )
    table = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return table.sort_values(["start_frame", "reason", "event_id"], kind="stable").reset_index(
        drop=True
    )


def write_review(
    observations,
    clean,
    flags,
    profile,
    key,
    trajectory_dir,
    *,
    labels_path,
    video,
    diagnostics_path,
    matrix_audit_path=None,
    matrix_quality_path=None,
):
    root = Path(trajectory_dir)
    protected = [
        root / name for name in ["trajectory.csv", "motion.csv", "lineage.csv.gz", "flags.csv"]
    ]
    hashes = {p.name: sha256_file(p) for p in protected}
    if clean.identity_edit.any() or clean.class_edit.any():
        raise ValueError("Automatic production must not apply manual label edits")
    if not clean.id.eq(clean.source_track_id).all():
        raise ValueError("Automatic production must preserve source identities")
    events = review_events(observations, clean, flags, profile, key, diagnostics_path)
    unresolved = []
    if matrix_audit_path is not None:
        with Path(matrix_audit_path).open(encoding="utf-8-sig") as handle:
            audit = json.load(handle)
        if isinstance(audit, dict):
            audit = audit["frames"]
        for record in audit:
            if str(record.get("status", "")).upper().startswith("UNRESOLVED"):
                index = record.get("video_index", record.get("frame_index"))
                frame = int(index) + 1
                details = dict(
                    status=record["status"],
                    original_matrix_retained=True,
                    screen_reasons=record.get("screen_reasons", []),
                    trigger_reasons=record.get("trigger_reasons", []),
                    validation_gates=record.get("validation_gates", {}),
                    audit_gates=record.get("audit_gates", {}),
                    evidence="independent background image audit; unresolved is not certified quality",
                )
                unresolved.append(
                    dict(
                        event_id=len(events) + len(unresolved),
                        segment=key,
                        reason="stabilization_quality_unresolved",
                        source_id_a=None,
                        source_id_b=None,
                        start_frame=frame,
                        end_frame=frame,
                        start_time_s=(frame - 1) / profile["fps"],
                        end_time_s=(frame - 1) / profile["fps"],
                        support_frames=1,
                        details_json=json.dumps(details, ensure_ascii=False, allow_nan=False),
                    )
                )
        if unresolved:
            events = pd.DataFrame(
                events.to_dict(orient="records") + unresolved, columns=EVENT_COLUMNS
            )
            events = events.sort_values(
                ["start_frame", "reason", "event_id"], kind="stable"
            ).reset_index(drop=True)
    destination = root / "review"
    destination.mkdir(exist_ok=False)
    events.to_csv(
        destination / "events.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
        mode="x",
    )
    summary = dict(
        schema_version=1,
        status="REPORT_ONLY",
        key=key,
        events=len(events),
        reason_counts=dict(Counter(events.reason)),
        csv_rows=len(clean),
        changes_applied=0,
        production_requires_human_input=False,
        historical_curation_read=False,
        gt_used=False,
        source_video=str(Path(video).resolve()),
        labels_sha256=sha256_file(labels_path),
        generated_products_sha256=hashes,
        stabilization_diagnostics_sha256=sha256_file(diagnostics_path),
        stabilization_quality_unresolved_frames=len(unresolved),
        review_policy=dict(
            overlap_iou=0.8,
            gaps="all internal source gaps",
            class_disagreement="more than one raw class per source",
            motion_flags="existing quality_summary thresholds",
            meaning="possible anomalies; not confirmed errors",
        ),
    )
    if matrix_audit_path is not None:
        summary["matrix_repair_audit_sha256"] = sha256_file(matrix_audit_path)
    if matrix_quality_path is not None:
        summary["matrix_quality_summary_sha256"] = sha256_file(matrix_quality_path)
        with Path(matrix_quality_path).open(encoding="utf-8-sig") as handle:
            summary["stabilization_quality"] = json.load(handle)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
