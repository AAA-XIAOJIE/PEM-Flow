# PEM-Flow

Pedestrian–e-bike mixed-flow trajectories with motor-vehicle probes.
This repository provides the processing code, detector model, three demonstration
videos and their matching CSV results.

## Start here

Install [Git LFS](https://git-lfs.com/), then download the repository:

```shell
git lfs install
git clone https://github.com/AAA-XIAOJIE/PEM-Flow.git
cd PEM-Flow
git lfs pull
cd code
```

Follow the [installation steps](code/README.md#installation)
([中文说明](code/README_zh.md#安装)), then process the demonstrations:

```shell
python -m pemflow run --workers 2 --device 0 --output ../outputs/demo
```

Use `--device cpu` with the CPU installation. The command reads
`assets/demos/` and writes trajectory, motion and lineage tables to
`outputs/demo/`. Supplied demonstration results are in `data/demo/`.

## Repository layout

| Location | Contents |
| --- | --- |
| `code/README.md` · `code/README_zh.md` | Installation and commands |
| `code/src/pemflow/` | Video processing, human revision and dataset metrics |
| `code/configs/` | Video, camera, detection, tracking, motion and ROI settings |
| `code/corrections/final_rules.json` | Human revision rules |
| `code/validation/` | Detection and tracking annotations and frame mappings |
| `code/examples/` | Projection from image observations and AP calculation |
| `code/docs/` | Configuration guide and CSV field definitions |
| `assets/model/train12/best.pt` | Detector model |
| `assets/demos/` | Three 10-second source-video clips and source-frame mappings |
| `data/demo/` | Matching demonstration results |

## Use the dataset

The code, model, configurations, annotation XML, revision rules and three
demonstrations with matching results are supplied in this GitHub repository.
The complete experiment results and companion data are supplied as
`PEMFlow_data.zip` on Zenodo.

Extract the data archive, then copy its `PEM-Flow/code/` and `PEM-Flow/data/`
directories into this repository, merging directories with the same names.
This adds the annotation images and complete experiment data to the code and
demonstrations already present here. Follow the commands in `code/README.md`
or `code/README_zh.md` from the combined repository.

| Location | Contents |
| --- | --- |
| `data/automatic/` | Automatic trajectories, motion tables and processing inputs |
| `data/revised/` | Human-reviewed trajectories and motion tables |
| `data/downstream/` | Motion, density, flow and detection/tracking metrics |
| `data/revision_records/` | Human decisions and row-level changes |
| `code/validation/images/` | Annotation images |

See the [usage guide](code/README.md#use-the-dataset) for commands and the
[data dictionary](code/docs/data_dictionary.md) for fields, units and identity keys.

## Licences and citation

Data: [CC BY 4.0](DATA_LICENSE.txt). Software and model:
[licences](code/THIRD_PARTY_NOTICES.md). Citation: [CITATION.cff](CITATION.cff).
