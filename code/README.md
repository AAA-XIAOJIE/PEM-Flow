# PEM-Flow code

The processing sequence is:

**Source video → YOLO11m + ByteTrack → background stabilization and image audit
→ close-gap image recovery → world-coordinate projection → continuous-segment
motion → automatic CSV.**

Human review produces explicit rules applied by `revise`. The resulting CSV
edition is used by `metrics` and `evaluate`.

## Installation

Use Python 3.11 or later; the release was tested with Python 3.12. From this
`code/` directory, install the tested inference profile and the editable package:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements/gpu.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

The CUDA wheel above matches the validated machine. Choose a PyTorch wheel
appropriate for your own device, or use `--device cpu`. Installation profiles
record tested dependency versions. Execution records installed versions in the
output metadata and proceeds with the available environment.

For a separate CPU environment, install `requirements/cpu.txt` followed by
`pip install -e . --no-deps`. The full video command accepts separate
`--inference-python` and `--cpu-python` interpreters. A single inference
environment also contains all packages needed by the other commands.

## Inputs

Put the 19 original files listed in `../assets/video_manifest.csv` into
`../assets/videos/`. The train12 weight is already in
`../assets/model/train12/best.pt`. Camera intrinsics, surveyed control points,
reference image indices and numerical parameters are in `configs/dataset.json`.
ROI and shape parameters are in `configs/pipeline.json`.

`G3-7-1` reads `G3-7.mp4` and writes `G3-7.csv`; `G3-7-2` reads
`h_G3-7-1.mp4` and writes `G3-7_hmi.csv`. They are independent video segments
with local frame numbers and identities; both aggregate into experiment G07.

## Commands

```powershell
.venv\Scripts\python -m pemflow run --workers 2 --device 0 --output ../outputs/automatic
.venv\Scripts\python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
.venv\Scripts\python -m pemflow metrics --input ../outputs/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../outputs/revised --automatic ../outputs/automatic --output ../outputs/evaluation
```

These commands use the environment created above and are run from `code/`.
In an activated environment, `pemflow` and `python -m pemflow` provide the same
commands. Default input paths are resolved from the installed editable package,
so they also work from other working directories. Explicit relative arguments
such as `--output` refer to your working directory.

Two workers can run on the same GPU. Each video keeps sequential frame decoding,
batch size 1 and its own tracker state. The console reports the video, current
stage, processed frames and completed video count. `progress.json` and individual
stage logs are written to the output folder.

To resume a stopped job, repeat the `run` command with `--resume`. Completed
stages are retained. Interrupted tracking restarts that video because tracker
state depends on preceding frames; completed stabilization and recovery stages
can be resumed separately. Use a new output directory for a new scientific
configuration or dataset.

To process selected source segments:

```powershell
.venv\Scripts\python -m pemflow run --segments G3-1 G3-7-2 --workers 2 --output ../outputs/selected
```

`revise` uses `corrections/final_rules.json`. It selects immutable source track
IDs and one-based inclusive frame ranges. Deletions omit those rows. Explicit
ID mappings are applied simultaneously. Submitted boxes are projected using
the production matrices; affected complete identities are recalculated with the
same motion functions. Other CSV field tokens are copied unchanged. Each revised
row and its reason are recorded in an audit. Rules were supplied by human review
of this automatic edition; they should be reviewed again when using different
source observations.

`metrics` reads the trajectory, motion and lineage tables. It outputs
`trajectory_metrics.json`, summary CSVs, and ROI individual/frame tables.
Trajectory summaries report speed, absolute SG tangential acceleration, SG
acceleration magnitude and within-segment position increments.
`evaluate` compares the fixed 270 detection frames and 600 tracking frames
against reviewed GT and writes precision/recall/F1, MOTA, IDF1, ID switches,
fragmentations and projected box-centre differences.

Detector AP requires low-threshold predictions on native-resolution source
images. Add `--ap` in the inference environment:

```powershell
.venv\Scripts\python -m pemflow evaluate --ap --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation
```

The AP entry decodes the fixed 270 sampled frames from the source videos,
predicts at confidence 0.001, and scores IoU 0.50–0.95 using the 101-point
precision envelope. CSV tracking predictions and image-recovered boxes are
separate from this detector calculation.

## Outputs and scientific definitions

An automatic dataset contains `dataset.json`, `trajectory/`, `motion/`,
`lineage/`, `intermediates/`, `run.json` and progress/log files.
Trajectory CSV columns are:

```text
frame,id,class,real_x,real_y,velocity,dist
```

`frame` starts at 1 within each video, `real_x/real_y/dist` are metres and
`velocity` is m/s. Class codes are 0 pedestrian, 9 e-bike and 3 motor-vehicle
probe. The `motion/` table stores both velocity and acceleration components,
tangential acceleration, actual window and segment status. Empty fields denote
unavailable derivatives.

Position, velocity and acceleration use the same quadratic Savitzky–Golay fit.
The maximum window is 31 frames. A continuous 25–30-frame segment uses its
largest legal odd window; shorter segments retain measured positions and missing
derivatives. Automatic source tracks use one majority class; actual missing
frames, source identity changes and final ID changes split motion segments.
Matrix index `f-1` projects label frame `f`, after the
stabilization matrices are expressed in the calibration reference image.

See [methods.md](docs/methods.md) for ROI and diagnostic definitions and
[data dictionary](docs/data_dictionary.md) for file fields.

## Source layout

`src/pemflow/pipeline` implements video processing; `revision` applies explicit
human rules; `downstream` computes trajectory/ROI metrics; `evaluation` scores
reviewed GT. `cli.py` defines the four commands, `execution.py` schedules segment
workers, and `io.py` defines dataset paths and file helpers.

Original code uses the BSD 3-Clause license. Ultralytics, its derived tracker
configuration and the detector weight retain AGPL component notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
