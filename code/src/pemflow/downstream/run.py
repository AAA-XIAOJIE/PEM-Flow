"""ROI density, flow and motion metrics for a trajectory dataset."""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path

import pandas as pd

from pemflow.downstream.metrics import aggregate, segment
from pemflow.io import read_json, runtime_metadata, table_paths, write_json


def compute_metrics(input_root, config, output, *, workers=2, resume=False):
    input_root, output = Path(input_root), Path(output)
    dataset = read_json(input_root / "dataset.json")
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for item in dataset["segments"]:
        sources = table_paths(input_root, item["key"])
        # Count records without loading a second copy of the scene table.
        with sources["trajectory"].open(encoding="utf-8-sig") as stream:
            expected_rows = sum(1 for _ in stream) - 1
        jobs.append(
            dict(
                item=item,
                sources={k: str(v) for k, v in sources.items()},
                config=str(config),
                output=str(output),
                expected_rows=expected_rows,
                resume=resume,
            )
        )
    started = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(segment, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            write_json(
                output / "progress.json",
                dict(status="RUNNING", completed=len(results), total=len(jobs)),
            )
    order = {item["key"]: n for n, item in enumerate(dataset["segments"])}
    results.sort(key=lambda item: order[item["key"]])
    summaries = aggregate(results, output)
    report = dict(
        status="COMPLETE",
        source_rows=sum(r["source_rows"] for r in results),
        source_frames=sum(r["frames"] for r in results),
        files=len(results),
        runs=len({r["run"] for r in results}),
        roi_rows=sum(r["rows"] for r in results),
        roi_area_m2=results[0]["roi_area_m2"],
        segments=results,
        overall=summaries["TOTAL"],
        runs_summary={k: v for k, v in summaries.items() if k != "TOTAL"},
        runtime=runtime_metadata(),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "trajectory_metrics.json", report)
    definitions = dict(
        roi_selection="Strict interior of the configured polygon.",
        density="1 / max(full domain cell area, class area floor).",
        frame_density="sum(ROI cell area * individual density) / ROI area.",
        frame_flow="sum(ROI cell area * density * stored speed) / ROI area; missing if any selected speed is unavailable.",
        shape="Finite-domain shape-distance assignment at the configured grid spacing.",
        motion="Stored position, speed and both acceleration components from the same quadratic SG fit.",
        headings="Split at actual frame gaps, stored motion boundaries, source IDs and class changes.",
        missing="Unknown derivatives remain NaN. Empty ROI frames have zero density and flow.",
    )
    write_json(output / "definitions.json", definitions)
    write_json(
        output / "roi/summary.json",
        dict(
            status="COMPLETE",
            roi_area_m2=report["roi_area_m2"],
            source_rows=report["source_rows"],
            frames=report["source_frames"],
            roi_rows=report["roi_rows"],
            segments=results,
            overall=summaries["TOTAL"]["roi"]["frame_statistics"],
            definitions=definitions,
        ),
    )
    pd.DataFrame(
        [
            {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool)) or v is None}
            for r in results
        ]
    ).to_csv(output / "segment_statistics.csv", index=False)
    pd.DataFrame(
        [dict(run=k, **v["roi"]["frame_statistics"]) for k, v in summaries.items() if k != "TOTAL"]
    ).to_csv(output / "run_statistics.csv", index=False)
    write_json(
        output / "progress.json",
        dict(status="COMPLETE", completed=len(results), total=len(results)),
    )
    print(
        f"Metrics complete: {report['source_rows']} source rows, {report['roi_rows']} ROI rows",
        flush=True,
    )
    return report
