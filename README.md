# PEM-Flow

Trajectories of pedestrian–e-bike mixed flow with motor-vehicle probes.
The repository contains the processing code, trained model, three demonstration
videos and their outputs. The dataset contains **18 experiments in 19 video
segments**: **7,143,147 automatic records** and **6,887,738 human-reviewed records**.

## Quick start

Install [Git LFS](https://git-lfs.com/), then clone the repository:

```shell
git lfs install
git clone https://github.com/AAA-XIAOJIE/PEM-Flow.git
cd PEM-Flow/code
python -m pip install -r requirements/gpu.txt
python -m pip install -e . --no-deps
python -m pemflow run --workers 2 --device 0
```

This processes the three demonstration clips and writes `outputs/demo/`.
For a CPU, install `requirements/cpu.txt` and use `--device cpu`.
See [installation and commands](code/README.md) or [中文使用说明](code/README_zh.md).

## Files

| Location | Contents |
| --- | --- |
| `code/src/pemflow/` | Video processing, human revision, metrics and evaluation |
| `code/configs/` | Camera, detection, tracking and motion settings |
| `code/corrections/` | Human revision rules for the automatic dataset |
| `code/validation/` | Detection and tracking annotations |
| `assets/model/train12/best.pt` | Detector weight |
| `assets/demos/` | Three 10-second source-video clips, copied without re-encoding |
| `data/demo/` | Matching demonstration trajectory, motion and lineage tables |

The demonstrations represent 24, 48 and 72 participants, each with 50% e-bikes.
They retain the original 3840 × 2160 pixels and 30 fps. Each clip has local frame
numbers and tracker identities; input details are in `assets/demos/clip_manifest.json`.

## Dataset

Extract the Zenodo archive `PEMFlow_data.zip` and merge its `PEM-Flow/` directory
into the repository:

| Location | Contents |
| --- | --- |
| `data/automatic/` | Automatic CSVs, trajectory plots and processing observations |
| `data/revised/` | Human-reviewed CSVs and trajectory plots |
| `data/downstream/` | Motion, density, flow and quality metrics |
| `data/revision_records/` | Human decisions and row-level change records |
| `data/height_selection/` | Reference-line experiment data |
| `code/validation/images/` | Images corresponding to the annotations |

Both CSV datasets are supplied as formal results. Files with the same segment
name correspond to the same recording. [Field definitions](code/docs/data_dictionary.md)
describe units and identity keys; [processing methods](code/docs/methods.md)
describe the calculations.

## Licences

Data: [CC BY 4.0](DATA_LICENSE.txt). Software and model terms:
[third-party notices](code/THIRD_PARTY_NOTICES.md).
