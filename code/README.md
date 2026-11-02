# Using PEM-Flow

Video → detection and tracking → background stabilization → image recovery →
world-coordinate projection → smoothing and motion → automatic CSVs.
The `revise` command applies human revision rules to produce reviewed CSVs.

## Installation

Use Python 3.12. Run the following commands from `PEM-Flow/code/`.

Create a virtual environment:

```shell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in a Linux shell:

```bash
source .venv/bin/activate
```

Choose one installation below. For an NVIDIA GPU with CUDA 12.8 driver support:

```shell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/gpu.txt
python -m pip install -e . --no-deps
```

For CPU processing:

```shell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/cpu.txt
python -m pip install -e . --no-deps
```

All commands below use this environment. On Windows, `.venv\Scripts\python.exe`
can replace `python` to run commands without activating the environment.
Use `git lfs pull` from the repository root to download the model and demo videos.

## Run the demonstrations

```shell
python -m pemflow run --workers 2 --device 0 --output ../outputs/demo
```

For CPU processing, replace `--device 0` with `--device cpu`.
The defaults use `configs/demo_dataset.json` and `../assets/demos/`.
The three clips each have 300 frames; their matching results are in `../data/demo/`.
Clip frames and IDs are local. Source-frame mappings are in
`../assets/demos/clip_manifest.json`.

Use `--segments G3-6` to process one demo. Add `--resume` with the same
configuration and output directory to continue an interrupted run. Use a new
output directory for a separate run. Two workers process two videos concurrently;
tracking within each video remains sequential. Progress appears in the terminal
and the output directory's `progress.json`.

## Use the dataset

The [GitHub repository](https://github.com/AAA-XIAOJIE/PEM-Flow) supplies
the processing code, model, configurations, annotation XML, revision rules and
demonstrations. The complete experiment results, annotation images, projection
inputs and downstream results are supplied in `PEMFlow_data.zip` on Zenodo.

Extract that archive, then copy its `PEM-Flow/code/` and `PEM-Flow/data/` into
the cloned repository root, merging matching directories. Annotation images
will be at `code/validation/images/`; complete results will be under `data/`.
The following commands run from `code/`.

| Task | Command |
| --- | --- |
| Apply human revisions | `python -m pemflow revise --input ../data/automatic --output ../outputs/revised` |
| Calculate motion, density and flow | `python -m pemflow metrics --input ../data/revised --output ../outputs/downstream --workers 2` |
| Evaluate detection and tracking | `python -m pemflow evaluate --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation` |
| Calculate detector AP | `python examples/score_ap.py --output ../outputs/ap` |

`revise` uses `corrections/final_rules.json`. Its source IDs and inclusive frame
ranges refer to the automatic dataset. The AP command uses the supplied
confidence-ranked predictions and annotations.

To calculate automatic CSVs from the supplied image observations and stabilization
matrices, then apply the human revision rules:

```shell
python examples/project_observations.py --input ../data/automatic --output ../outputs/automatic
python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
```

## Process source videos

Select the input folder with `--video-root`; filenames must match the dataset
configuration. To process the recordings described by `configs/dataset.json`:

```shell
python -m pemflow run --dataset configs/dataset.json --video-root ../videos --output ../outputs/automatic --workers 2 --device 0
```

For another recording, create `configs/my_dataset.json` using the
[configuration guide](docs/configuration.md), then pass it to `--dataset`.
Use that recording's camera parameters, control points and reference frame.

## Outputs

Paths below are relative to the chosen `--output` directory.

| Directory | Contents |
| --- | --- |
| `trajectory/` | Positions, speed and position increments |
| `motion/` | Velocity and acceleration components and motion segments |
| `lineage/` | Source observations, identities and row provenance |
| `intermediates/` | Image observations and stabilization matrices from video processing |

The automatic and reviewed datasets use the same CSV structure. Frames start at
1 and IDs are local to each video. The two G7 segments use `G3-7.csv` and
`G3-7_hmi.csv`. Empty numeric fields mean unavailable values.

See [field definitions](docs/data_dictionary.md), [configuration](docs/configuration.md)
and [licences](THIRD_PARTY_NOTICES.md). Use `python -m pemflow --help`, or add
`--help` after a command, for its available options.
