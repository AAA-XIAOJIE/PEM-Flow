# Using PEM-Flow

Video → detection and tracking → background stabilization → image recovery →
world-coordinate projection → motion calculation → CSV.
Human revision rules produce the reviewed dataset; metric and evaluation
commands operate on the supplied datasets.

## Installation

Use Python 3.11 or later; the tested environment uses Python 3.12.
The following commands run from `code/`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements/gpu.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

Choose the PyTorch wheel for your device. CPU use is supported by
`requirements/cpu.txt` and `--device cpu`. Dependency versions are recorded
in run metadata. Git LFS retrieves `assets/model/train12/best.pt` and the demo
videos; run `git lfs pull` if an existing checkout contains pointer files.

## Run the demonstrations

```powershell
.venv\Scripts\python -m pemflow run --workers 2 --device 0 --output ../outputs/demo
```

The defaults read `configs/demo_dataset.json` and `../assets/demos/`.
The three clips each contain 300 frames at 3840 × 2160 and 30 fps. Their
trajectory, motion and lineage results are supplied in `../data/demo/`.
Clip frames and IDs are local; `../assets/demos/clip_manifest.json` maps their
frames to the original recordings.

Two workers share the GPU while preserving each video's sequential tracker.
The terminal and `progress.json` report progress. Add `--resume` to continue
an interrupted run; `--segments G3-6` selects one demo.

## Work with the dataset

Merge the Zenodo archive's `PEM-Flow/` directory into the repository.
Automatic and human-reviewed trajectory CSVs are in `data/automatic/trajectory/`
and `data/revised/trajectory/`. Their corresponding motion tables, lineage and
plots are in the adjacent directories.

Calculate metrics and evaluate the human-reviewed data:

```powershell
.venv\Scripts\python -m pemflow metrics --input ../data/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation
.venv\Scripts\python examples/score_archived_ap.py --output ../outputs/ap
```

The AP example uses the supplied confidence-ranked predictions and annotations.
To calculate trajectories from the supplied image observations and stabilization
matrices, then apply the recorded human decisions:

```powershell
.venv\Scripts\python examples/reproject_cached.py --output ../outputs/automatic
.venv\Scripts\python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
```

The rules in `corrections/final_rules.json` contain 77 human decisions.
Their source IDs and inclusive frame ranges refer to the automatic dataset.
Deletion filters the selected rows; simultaneous ID mappings and submitted
boxes use the recorded decisions. Affected complete tracks use the same
projection and motion calculation. `data/revision_records/` records each change.

## Configuration and outputs

`configs/demo_dataset.json` describes the demos; `configs/dataset.json` describes
the 19 dataset segments. Each entry contains the input filename, frame count,
camera parameters, 3D/2D control-point pairs and calibration reference frame.
`configs/pipeline.json` specifies the ROI and target shapes. Class reference
heights are 0.72 m for pedestrians, 0.20 m for e-bikes and 0.60 m for motor vehicles.

For another recording, supply a dataset JSON with that recording's camera and
control points, dimensions, frame count, file size and SHA-256:

```powershell
.venv\Scripts\python -m pemflow run --dataset configs/my_dataset.json --video-root C:/my_videos --output ../outputs/my_run
```

Place the dataset JSON beside `inference.json` and `bytetrack_pemflow.yaml`.
Use a calibration reference index inside the video. The command uses the chosen
input specification and writes `trajectory/`, `motion/`, `lineage/` and processing
metadata. Explicit relative paths resolve from the working directory.

The trajectory columns are:

```text
frame,id,class,real_x,real_y,velocity,dist
```

Positions and distance are in metres; speed is in m/s. Classes are 0 pedestrian,
9 e-bike and 3 motor vehicle. Position, velocity and acceleration use a quadratic
Savitzky–Golay fit with at most 31 frames. Continuous segments of 25–30 frames
use the largest valid odd window; shorter segments retain positions and missing
derivatives. The two G07 segments use `G3-7.csv` and `G3-7_hmi.csv`.

`src/pemflow/pipeline/` implements video processing; `revision/` applies human
rules; `downstream/` calculates motion and ROI metrics; `evaluation/` scores
annotations. `cli.py` provides commands, `execution.py` schedules workers, and
`io.py` resolves paths. See [methods](docs/methods.md),
[field definitions](docs/data_dictionary.md) and [licences](THIRD_PARTY_NOTICES.md).
