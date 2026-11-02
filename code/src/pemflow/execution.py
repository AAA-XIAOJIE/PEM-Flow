"""Concurrent segment scheduling; each segment retains serial frame processing."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import threading
from pathlib import Path

from pemflow.io import CODE_ROOT, read_json, table_paths, write_json

ENVIRONMENT = dict(
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="2",
    PYTHONUTF8="1",
    PYTHONUNBUFFERED="1",
)


def run_worker(python, stage, args, key, destination, *, resume=False):
    log_path = destination.parent / (destination.name + ".log")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if stage == "inference" and destination.exists() and not (destination / "done.json").exists():
        # ByteTrack state is serial: an interrupted inference restarts this video.
        shutil.rmtree(destination)
    command = [
        str(python),
        "-m",
        "pemflow.execution",
        stage,
        "--dataset",
        str(args.dataset),
        "--key",
        key,
        "--video-root",
        str(args.video_root),
        "--output",
        str(destination),
    ]
    if stage == "inference":
        command += [
            "--model",
            str(args.model),
            "--config-root",
            str(args.config_root),
            "--device",
            str(args.device),
        ]
    else:
        command += [
            "--labels",
            str(destination / "inference" / "labels.npz"),
            "--threads",
            str(args.threads),
        ]
        if resume:
            command.append("--resume")
    environment = dict(os.environ, **ENVIRONMENT, PYTHONPATH=str(CODE_ROOT / "src"))
    with log_path.open("a" if resume else "w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line.rstrip(), flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{key} {stage} failed ({returncode}); see {log_path}")


def run(args):
    args.dataset = Path(args.dataset).resolve()
    args.config_root = args.dataset.parent
    args.output = Path(args.output).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(args.dataset)
    items = [
        item for item in protocol["segments"] if not args.segments or item["key"] in args.segments
    ]
    if not items:
        raise ValueError("No selected segment exists in the dataset configuration")
    for item in items:
        if not (Path(args.video_root) / item["source_video"]["filename"]).is_file():
            raise FileNotFoundError(Path(args.video_root) / item["source_video"]["filename"])
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    progress = dict(
        status="RUNNING", completed=[], total=len(items), workers=args.workers, segments={}
    )
    lock = threading.Lock()
    write_json(args.output / "dataset.json", {**protocol, "segments": items})

    def update(key, stage):
        with lock:
            progress["segments"][key] = stage
            if stage == "complete" and key not in progress["completed"]:
                progress["completed"].append(key)
            write_json(args.output / "progress.json", progress)
            print(
                f"Completed {len(progress['completed'])}/{len(items)} videos | {key}: {stage}",
                flush=True,
            )

    def segment(item):
        key = item["key"]
        work = args.output / "intermediates" / key
        inference = work / "inference"
        if not (args.resume and (inference / "done.json").exists()):
            update(key, "inference")
            run_worker(args.inference_python, "inference", args, key, inference, resume=args.resume)
        if not (args.resume and (work / "done.json").exists()):
            update(key, "stabilization and trajectory")
            run_worker(args.cpu_python, "production", args, key, work, resume=args.resume)
        paths = table_paths(args.output, key)
        for name, target in paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                work / "trajectory" / ("lineage.csv.gz" if name == "lineage" else name + ".csv"),
                target,
            )
        update(key, "complete")
        return read_json(work / "done.json")

    try:
        # Long segments start first. Scheduling changes no per-video parameters.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(segment, item)
                for item in sorted(items, key=lambda item: -item["source_video"]["frames"])
            ]
            results = [future.result() for future in futures]
        progress["status"] = "COMPLETE"
        write_json(args.output / "progress.json", progress)
        receipt = dict(
            status="COMPLETE",
            rows=sum(value["rows"] for value in results),
            frames=sum(value["frames"] for value in results),
            segments=results,
        )
        write_json(args.output / "run.json", receipt)
        return receipt
    except Exception as error:
        progress.update(status="FAILED", error=str(error))
        write_json(args.output / "progress.json", progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inference", "production"])
    for name in ["dataset", "video-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "inference":
        from pemflow.pipeline.inference import infer_video_segment

        infer_video_segment(
            args.dataset,
            args.key,
            args.video_root,
            args.model,
            args.config_root,
            args.output,
            device=args.device,
        )
    else:
        from pemflow.pipeline.production import produce

        produce(
            args.dataset,
            args.key,
            args.video_root,
            args.labels,
            args.output,
            threads=args.threads,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
