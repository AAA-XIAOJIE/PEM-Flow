# PEM-Flow code

The processing sequence is:

**Video → YOLO11m and ByteTrack → background stabilization and image audit
→ image recovery → world-coordinate projection → motion calculation → CSV.**

Recorded human decisions produce the reviewed dataset. Downstream commands
calculate motion, density, flow and detection/tracking quality.

## Installation

Use Python 3.11 or later. The tested environment uses Python 3.12. Run from `code/`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements/gpu.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

Choose the PyTorch wheel appropriate for your device. CPU inference uses
`--device cpu`; `requirements/cpu.txt` provides a CPU dependency profile.
The program records installed versions in its output metadata.

Merge the Zenodo data archive's `PEM-Flow/` directory into this repository.
The train12 weight is in `../assets/model/train12/best.pt`.

## Demonstration videos

The three clips in `../assets/demos/` contain 24, 48 and 72 participants with
50% e-bikes. Process them with:

```powershell
.venv\Scripts\python -m pemflow run --dataset configs/demo_dataset.json --video-root ../assets/demos --workers 2 --device 0 --output ../outputs/demo
```

The matching output tables are in `../data/demo/`. Each clip has its own
one-based frame numbers and tracker IDs. `assets/demos/clip_manifest.json`
records source frame ranges and encoding. Camera calibration is retained;
G15's calibration reference is clip frame 151 (zero-based index 150).

## Recalculate the dataset

The archived observations and stabilization matrices in `../data/automatic/`
are inputs to the production projection and motion functions:

```powershell
.venv\Scripts\python examples/reproject_cached.py --output ../outputs/reprojected
.venv\Scripts\python -m pemflow revise --input ../outputs/reprojected --output ../outputs/revised
.venv\Scripts\python -m pemflow metrics --input ../outputs/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../outputs/revised --automatic ../outputs/reprojected --output ../outputs/evaluation
```

This calculates all 19 trajectory segments, applies the recorded human
decisions, and evaluates the resulting dataset. The reference results are
`../data/automatic/`, `../data/revised/` and `../data/downstream/`.

Recalculate detector AP from the 270 archived low-confidence prediction records
and reviewed annotations:

```powershell
.venv\Scripts\python examples/score_archived_ap.py --output ../outputs/ap
```

Recalculate the reference-line height comparison:

```powershell
.venv\Scripts\python examples/reproduce_height_selection.py --output ../outputs/height_selection
```

Class heights are 0.72 m for pedestrians, 0.20 m for e-bikes and 0.60 m for
motor vehicles. The first two minimize the mean of four single-pass RMSEs from
two round trips, using motion-processed positions: 4.02225 cm and 4.16203 cm.
The height experiment uses its own camera calibration and road elevation,
stored in `../data/height_selection/`.

## Process source videos

Put the 19 files listed in `../assets/video_manifest.csv` in `../assets/videos/`.
The manifest specifies filenames, sizes and SHA-256 hashes.

```powershell
.venv\Scripts\python -m pemflow run --workers 2 --device 0 --output ../outputs/automatic
.venv\Scripts\python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
.venv\Scripts\python -m pemflow metrics --input ../outputs/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../outputs/revised --automatic ../outputs/automatic --output ../outputs/evaluation
```

`configs/dataset.json` contains camera parameters, control points, reference
indices and processing parameters. `configs/pipeline.json` specifies ROI and
shape parameters.

`G3-7-1` reads `G3-7.mp4` and produces `G3-7.csv`; `G3-7-2` reads
`h_G3-7-1.mp4` and produces `G3-7_hmi.csv`. Each segment has local frame and
identity keys; experiment statistics aggregate them as G07.

Two workers can share a GPU. Each video is decoded sequentially with batch
size 1 and its own tracker state. Progress appears in the console and
`progress.json`. Add `--resume` to continue an interrupted run.
Use `--segments G3-1 G3-7-2` to select source segments.
Separate environments can be specified with `--inference-python` and
`--cpu-python`.

The commands above run from `code/`. Default inputs resolve from the package;
explicit relative arguments resolve from the working directory.

## Human revision and evaluation

`corrections/final_rules.json` records 77 human decisions against the source
IDs and inclusive frame ranges of `data/automatic/`. Deletions filter rows;
ID mappings are applied simultaneously; submitted boxes are projected with
the production matrices. Affected complete trajectories use the same motion
calculation. Row-level records preserve each change and its reason.

`metrics` calculates motion summaries and ROI individual/frame density and
flow. `evaluate` uses 270 detection frames and 600 tracking frames to calculate
precision, recall, F1, MOTA, IDF1, ID switches, fragmentations and projected
box-centre differences.

To rerun detection on the source images for AP:

```powershell
.venv\Scripts\python -m pemflow evaluate --ap --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation
```

AP uses confidence 0.001 predictions, IoU 0.50–0.95 and the 101-point precision
envelope. The 270 source images are decoded from the source videos.

## Outputs and source layout

Automatic output contains `trajectory/`, `motion/`, `lineage/`,
`intermediates/`, `dataset.json`, `run.json` and progress logs.

```text
frame,id,class,real_x,real_y,velocity,dist
```

Coordinates and distance use metres; speed uses m/s. Class codes are
0 pedestrian, 9 e-bike and 3 motor vehicle. Position, velocity and acceleration
use a quadratic Savitzky–Golay fit with a maximum 31-frame window.
Continuous segments of 25–30 frames use the largest permitted odd window;
shorter segments retain observed positions and missing derivatives.

`src/pemflow/pipeline/` implements video processing; `revision/` applies human
rules; `downstream/` calculates trajectory and ROI metrics; `evaluation/`
scores reviewed annotations. `cli.py` provides commands, `execution.py`
schedules workers, and `io.py` handles dataset paths.

See [methods](docs/methods.md), [field definitions](docs/data_dictionary.md)
and [licences](THIRD_PARTY_NOTICES.md).
