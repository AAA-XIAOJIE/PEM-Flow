"""Recalculate detector AP from the published low-confidence predictions and GT."""
import argparse
from pathlib import Path

from pemflow.io import CODE_ROOT, PACKAGE_ROOT, read_json
from pemflow.evaluation.ap import ground_truth_frames, score_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=CODE_ROOT / "validation")
    parser.add_argument("--predictions", type=Path, default=PACKAGE_ROOT / "data/downstream/detection/predictions")
    parser.add_argument("--metadata", type=Path, default=PACKAGE_ROOT / "data/downstream/quality/sample_metadata.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new output directory.")
    frames = ground_truth_frames(args.validation, args.metadata)
    for frame in frames:
        prediction = read_json(args.predictions / (Path(frame["image"]).stem + ".json"))
        if prediction["task"] != frame["task"] or prediction["image"] != frame["image"]:
            raise ValueError("Prediction and reference image do not match.")
        frame["detections"] = prediction["detections"]
    summary = score_frames(frames, args.output)
    print(f"Frames: {summary['frame_count']}; mAP50: {summary['map50']:.9f}; mAP50-95: {summary['map50_95']:.9f}")


if __name__ == "__main__":
    main()
