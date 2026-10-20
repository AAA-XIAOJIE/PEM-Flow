"""Reproduce the marked-line height comparison from the measurement inputs.

The example calls the installed PEM-Flow projection and motion functions,
then averages four single-pass RMSEs from two analysed round trips.
All input paths are relative to --input.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pemflow.pipeline.motion import process
from pemflow.io import PACKAGE_ROOT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pass_samples(clean, trials, reference, field_xy_sign, source_rows):
    a = np.asarray(reference["measured_A_xyz_m"][:2], dtype=float)
    b = np.asarray(reference["constructed_B_xyz_m"][:2], dtype=float)
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    samples = []
    for t in trials:
        selected = clean[
            clean.source_track_id.isin(t["source_track_ids"])
            & clean.frame.between(t["start_frame"], t["end_frame"])
        ].sort_values("frame")
        if selected.frame.duplicated().any():
            raise ValueError(f"Multiple target observations in one frame: {t['trial_id']}")
        for mode, columns in (("raw", ["raw_x", "raw_y"]), ("smooth", ["real_x", "real_y"])):
            xy = selected[columns].to_numpy(float) * field_xy_sign
            if not np.isfinite(xy).all():
                raise ValueError(f"Non-finite position: {t['trial_id']}")
            signed = (xy - a) @ normal
            samples.append(pd.DataFrame({
                "trial_id": t["trial_id"], "roundtrip": t["roundtrip"],
                "direction": t["direction"], "class_id": t["class_id"],
                "mode": mode, "frame": selected.frame.to_numpy(),
                "source_track_id": selected.source_track_id.to_numpy(),
                "observation_id": selected.observation_id.to_numpy(),
                "source_observation_row": source_rows[selected.observation_id.to_numpy(int) - 1],
                "observed_class": selected["class"].to_numpy(),
                "raw_class": selected.raw_class.to_numpy(),
                "x_m": xy[:, 0], "y_m": xy[:, 1],
                "signed_error_m": signed, "absolute_error_m": np.abs(signed),
                "motion_window": selected.motion_window.to_numpy(),
                "smoothed": selected.smoothed.to_numpy(),
            }))
    return pd.concat(samples, ignore_index=True)


def reduce_passes(samples, trials):
    lookup = {t["trial_id"]: t for t in trials}
    rows = []
    for (tid, cls, mode), group in samples.groupby(["trial_id", "class_id", "mode"]):
        t = lookup[tid]
        e = group.signed_error_m.to_numpy(float)
        expected = t["end_frame"] - t["start_frame"] + 1
        if len(e) == 0:
            raise ValueError(f"No target observations: {tid}")
        rows.append({
            "trial_id": tid, "class_id": cls, "mode": mode,
            "roundtrip": t["roundtrip"], "direction": t["direction"],
            "n": len(e), "expected_frames": expected, "missing_frames": expected-len(e),
            "coverage": len(e)/expected, "smoothed_n": int(group.smoothed.sum()),
            "class_mismatch_n": int(group.observed_class.ne(cls).sum()),
            "rmse_m": float(np.sqrt(np.mean(e**2))),
            "mae_m": float(np.mean(np.abs(e))),
            "signed_bias_m": float(e.mean()),
            "p95_absolute_error_m": float(np.quantile(np.abs(e), .95)),
            "max_absolute_error_m": float(np.max(np.abs(e))),
        })
    return pd.DataFrame(rows)


def summarize(table, heights):
    rows = []
    for (cls, mode), group in table.groupby(["class_id", "mode"]):
        if len(group) != 4 or set(group["roundtrip"]) != {1, 2}:
            raise ValueError("The objective requires four passes from two analysed round trips")
        rows.append({
            "class_id": cls, "mode": mode, "height_above_road_m": heights[cls],
            "mean_four_pass_rmse_m": float(group.rmse_m.mean()),
            "n": int(group.n.sum()), "expected_frames": int(group.expected_frames.sum()),
            "n_passes": 4, "n_roundtrips": 2,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--data", dest="data", type=Path,
                        default=PACKAGE_ROOT / "data/height_selection",
                        help="Extracted height_selection folder (default: package data/height_selection)")
    parser.add_argument("--output", type=Path, required=True, help="New result folder")
    args = parser.parse_args()
    inputs = args.data / "inputs"
    config = read_json(inputs / "measurement_config.json")
    reference = read_json(inputs / "reference_line.json")
    trials = pd.read_csv(inputs / "trials.csv").to_dict("records")
    for t in trials:
        t["source_track_ids"] = [int(v) for v in str(t["source_track_ids"]).split(";")]
    if {t["trial_id"] for t in trials} != {"P1","P2","P3","P4","E1","E2","E3","E4"}:
        raise ValueError("Expected the supplied eight fixed passes")
    with np.load(inputs / "observations.npz", allow_pickle=False) as z:
        observations = z["observations"]
        source_rows = z["source_observation_row"]
    with np.load(inputs / "matrices.npz", allow_pickle=False) as z:
        matrices = z["matrices"]
    road_z = float(config["road_z_m"])
    field_xy_sign = np.asarray(config["processing_to_field_xy_sign"], dtype=float)
    classes = sorted({int(t["class_id"]) for t in trials})
    # Keep the distributed result tables as a stable reference.
    args.output.mkdir(parents=True, exist_ok=False)

    def calculate(heights):
        profile = copy.deepcopy(config["profile"])
        # Formal class_reference_height_m stores absolute world Z. The grid
        # variable h is height above this experiment's road at Z=-0.11 m.
        for cls, h in heights.items():
            profile["class_reference_height_m"][str(cls)] = road_z + h
        _, clean, removed, _, _, _ = process(observations, profile, matrices)
        if len(removed):
            raise ValueError("Unexpected observation removal during reconstruction")
        samples = pass_samples(clean, trials, reference, field_xy_sign, source_rows)
        passes = reduce_passes(samples, trials)
        return samples, passes, summarize(passes, heights)

    # Directly call the formal full-track reconstruction at every grid height.
    # Extract the fixed pass windows after complete-track smoothing.
    summaries, curves = [], []
    for i in range(201):
        h = i / 100
        _, passes, summary = calculate({cls: h for cls in classes})
        passes["height_above_road_m"] = h
        passes["projection_world_z_m"] = road_z + h
        curves.append(passes)
        summaries.append(summary)
        if i % 20 == 0:
            print(f"Height grid: {i+1}/201 ({h:.2f} m above road)", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    primary = summary[summary["mode"].eq("smooth")]
    selected = {
        cls: float(primary[primary.class_id.eq(cls)].sort_values(
            ["mean_four_pass_rmse_m", "height_above_road_m"]).iloc[0].height_above_road_m)
        for cls in classes
    }
    samples, passes, selected_summary = calculate(selected)
    samples["height_above_road_m"] = samples.class_id.map(selected)
    samples["projection_world_z_m"] = road_z + samples.height_above_road_m
    passes["height_above_road_m"] = passes.class_id.map(selected)
    rounds = passes.groupby(["class_id", "mode", "roundtrip"], as_index=False).agg(
        mean_two_pass_rmse_m=("rmse_m", "mean"), n=("n", "sum"))
    print("Selected heights above road (m):", selected, flush=True)
    for name, table in [
        ("height_summary.csv", summary),
        ("height_trials.csv", pd.concat(curves, ignore_index=True)),
        ("selected_trials.csv", passes), ("selected_summary.csv", selected_summary),
        ("selected_roundtrips.csv", rounds), ("selected_samples.csv.gz", samples),
    ]:
        table.to_csv(args.output / name, index=False,
                     compression={"method": "gzip", "mtime": 0} if name.endswith(".gz") else "infer")
    result = {
        "height_above_road_m": {str(k): v for k,v in selected.items()},
        "projection_world_z_m": {str(k): road_z+v for k,v in selected.items()},
        "road_z_m": road_z,
        "objective": "arithmetic mean of four single-pass RMSEs, formal SG output",
        "grid_min_m": 0, "grid_max_m": 2, "grid_step_m": .01,
        "complete_target_observations": len(observations), "matrix_frames": len(matrices),
        "results": selected_summary.to_dict("records"),
    }
    (args.output / "result_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(selected_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
