"""Rebuild automatic CSVs from archived image observations and stabilization.

Run from code/ after installing PEM-Flow:
    python examples/reproject_cached.py --output ../outputs/reprojected
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pemflow.io import CONFIG_ROOT, PACKAGE_ROOT, COLUMNS, read_json, table_paths, write_json
from pemflow.pipeline.motion import process
from pemflow.pipeline.production import LINEAGE_COLUMNS, MOTION_COLUMNS, PROVENANCE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PACKAGE_ROOT / "data/automatic")
    parser.add_argument("--dataset", type=Path, default=CONFIG_ROOT / "dataset.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", nargs="+")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new output directory.")
    dataset = read_json(args.dataset)
    items = [x for x in dataset["segments"] if not args.segments or x["key"] in args.segments]
    if not items:
        parser.error("No selected segment exists in the dataset configuration.")
    cv2.setNumThreads(1)
    write_json(args.output / "dataset.json", {**dataset, "segments": items})
    summaries = []
    for item in items:
        key = item["key"]
        inputs = args.input / "intermediates" / key
        with np.load(inputs / "observations.npz", allow_pickle=False) as z:
            observations = z["observations"].copy()
        with np.load(inputs / "matrices.npz", allow_pickle=False) as z:
            matrices = z["matrices"].copy()
        _, clean, _, _, statistics, _ = process(observations, item["profile"], matrices)
        source = table_paths(args.input, key)
        keys = ["observation_id", "frame", "source_track_id"]
        provenance = pd.read_csv(
            source["lineage"], usecols=keys + PROVENANCE_COLUMNS,
            dtype={c: str for c in PROVENANCE_COLUMNS}, keep_default_na=False,
        )
        # Provenance is keyed to the archived image observation, not world coordinates.
        clean = clean.merge(provenance, on=keys, how="left", validate="one_to_one")
        if clean[PROVENANCE_COLUMNS].isna().any().any():
            raise ValueError(f"{key}: archived observation provenance is incomplete.")
        outputs = table_paths(args.output, key)
        for name, columns in (("trajectory", COLUMNS), ("motion", MOTION_COLUMNS), ("lineage", LINEAGE_COLUMNS)):
            outputs[name].parent.mkdir(parents=True, exist_ok=True)
            options = dict(index=False, float_format="%.10f", na_rep="", lineterminator="\n")
            if name == "lineage":
                options.update(compression={"method": "gzip", "mtime": 0}, lineterminator="\r\n")
            clean[columns].to_csv(outputs[name], **options)
        target = args.output / "intermediates" / key
        target.mkdir(parents=True, exist_ok=True)
        for name in ("observations.npz", "matrices.npz"):
            shutil.copy2(inputs / name, target / name)
        summaries.append(dict(key=key, rows=len(clean), statistics=statistics))
        print(f"{key}: {len(clean)} rows", flush=True)
    write_json(args.output / "run.json", dict(
        status="COMPLETE", source="Archived observations and stabilization matrices",
        rows=sum(x["rows"] for x in summaries), segments=summaries,
    ))


if __name__ == "__main__":
    main()
