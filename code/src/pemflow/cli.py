"""PEM-Flow command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pemflow.io import CODE_ROOT, CONFIG_ROOT, PACKAGE_ROOT, write_json


def main():
    parser = argparse.ArgumentParser(
        prog="pemflow",
        description="Video trajectories, explicit CSV revisions and dataset metrics.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Compute automatic trajectories from source videos.")
    run.add_argument("--dataset", type=Path, default=CONFIG_ROOT / "demo_dataset.json")
    run.add_argument("--video-root", type=Path, default=PACKAGE_ROOT / "assets/demos")
    run.add_argument("--model", type=Path, default=PACKAGE_ROOT / "assets/model/train12/best.pt")
    run.add_argument("--output", type=Path, default=PACKAGE_ROOT / "outputs/demo")
    run.add_argument("--segments", nargs="+", help="Dataset keys, for example G3-1 G3-7-2.")
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--device", default="0")
    run.add_argument(
        "--threads", type=int, default=8, help="OpenCV threads for base stabilization."
    )
    run.add_argument("--inference-python", type=Path, default=Path(sys.executable))
    run.add_argument("--cpu-python", type=Path, default=Path(sys.executable))
    run.add_argument("--resume", action="store_true")
    revise = commands.add_parser("revise", help="Apply the recorded human CSV decisions.")
    revise.add_argument("--input", type=Path, default=PACKAGE_ROOT / "data/automatic")
    revise.add_argument("--rules", type=Path, default=CODE_ROOT / "corrections/final_rules.json")
    revise.add_argument("--output", type=Path, default=PACKAGE_ROOT / "outputs/revised")
    metrics = commands.add_parser("metrics", help="Compute ROI density/flow and motion statistics.")
    metrics.add_argument("--input", type=Path, default=PACKAGE_ROOT / "data/revised")
    metrics.add_argument("--config", type=Path, default=CONFIG_ROOT / "pipeline.json")
    metrics.add_argument("--output", type=Path, default=PACKAGE_ROOT / "outputs/downstream")
    metrics.add_argument("--workers", type=int, default=2)
    metrics.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate CSV detection/tracking against reviewed GT."
    )
    evaluate.add_argument("--input", type=Path, default=PACKAGE_ROOT / "data/revised")
    evaluate.add_argument("--automatic", type=Path, default=PACKAGE_ROOT / "data/automatic")
    evaluate.add_argument("--validation", type=Path, default=CODE_ROOT / "validation")
    evaluate.add_argument("--config", type=Path, default=CONFIG_ROOT / "pipeline.json")
    evaluate.add_argument("--output", type=Path, default=PACKAGE_ROOT / "outputs/evaluation")
    evaluate.add_argument(
        "--ap",
        action="store_true",
        help="Also calculate detector AP from the native-resolution video frames.",
    )
    evaluate.add_argument("--video-root", type=Path, default=PACKAGE_ROOT / "assets/videos")
    evaluate.add_argument(
        "--model", type=Path, default=PACKAGE_ROOT / "assets/model/train12/best.pt"
    )
    evaluate.add_argument("--inference-config", type=Path, default=CONFIG_ROOT / "inference.json")
    evaluate.add_argument("--device", default="0")
    args = parser.parse_args()
    if hasattr(args, "workers") and args.workers < 1:
        parser.error("--workers must be positive")
    if args.command == "run":
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            parser.error("Choose a new --output directory or use --resume.")
        from pemflow.execution import run

        run(args)
    elif args.command == "revise":
        from pemflow.revision.apply import apply_revisions

        apply_revisions(args.input, args.rules, args.output)
    elif args.command == "metrics":
        from pemflow.downstream.run import compute_metrics

        compute_metrics(
            args.input, args.config, args.output, workers=args.workers, resume=args.resume
        )
    else:
        from pemflow.evaluation.quality import evaluate_dataset

        summary = evaluate_dataset(
            args.input, args.automatic, args.validation, args.config, args.output / "quality"
        )
        if args.ap:
            from pemflow.evaluation.ap import ground_truth_frames, infer_frames, score_frames

            frames = ground_truth_frames(
                args.validation, args.output / "quality/sample_metadata.csv"
            )
            frames = infer_frames(
                frames,
                args.video_root,
                args.model,
                args.inference_config,
                args.device,
                args.output / "ap/predictions",
            )
            summary["detector_ap"] = score_frames(frames, args.output / "ap")
        write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
