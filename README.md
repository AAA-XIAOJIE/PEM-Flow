# PEM-Flow

PEM-Flow converts videos of pedestrian–e-bike interactions into trajectories,
applies recorded human revisions, and calculates motion, density, flow and
detection/tracking quality metrics.

## Package

| Location | Contents |
| --- | --- |
| GitHub | Processing code, configurations, train12 model, human revision rules and annotation records |
| Zenodo | Demonstration videos, trajectory datasets, downstream results and calculation inputs |

Merge the data archive's `PEM-Flow/` directory into the cloned repository.

| Directory | Contents |
| --- | --- |
| `code/` | Python package, configurations, examples and reviewed annotations |
| `assets/model/train12/` | Detector weight |
| `assets/demos/` | Three 10-second videos: 24, 48 and 72 participants, each with 50% e-bikes |
| `assets/videos/` | Input directory for the 19 source videos listed in `assets/video_manifest.csv` |
| `data/demo/` | The demonstration clips' trajectory, motion and lineage tables |
| `data/automatic/` | Automatic trajectories, matching figures and projection inputs |
| `data/revised/` | Trajectories and figures after recorded human review |
| `data/downstream/` | Motion, density, flow and quality metrics |
| `data/revision_records/` | Human decisions, submitted boxes and row-level revision records |
| `data/height_selection/` | Reference-line experiment inputs and height-comparison results |

The automatic dataset contains **7,143,147** trajectory records and the
human-reviewed dataset **6,887,738**, covering **19 video segments** from
**18 experiments** and **132,663 frames**. The reviewed ROI contains
**2,819,547** records.

Install [Git LFS](https://git-lfs.com/), run `git lfs install`, and clone the
repository. Use `git lfs pull` to retrieve the model in an existing clone.
Follow [English instructions](code/README.md) or
[中文使用说明](code/README_zh.md) to process the demos, recalculate the dataset
from archived observations, or process source videos.

## Projection heights

The class reference heights are **0.72 m for pedestrians**, **0.20 m for
e-bikes**, and **0.60 m for motor vehicles**. The pedestrian and e-bike heights
minimize the mean of four single-pass RMSEs from two round trips, using
motion-processed positions. The respective mean lateral deviations from the
reference line are **4.02225 cm** and **4.16203 cm**.

`code/examples/reproduce_height_selection.py` calculates these values from
`data/height_selection/`. See [methods](code/docs/methods.md) and the
[data dictionary](code/docs/data_dictionary.md) for definitions.

## Licences

Data use [CC BY 4.0](DATA_LICENSE.txt). Software and model terms are listed in
[THIRD_PARTY_NOTICES.md](code/THIRD_PARTY_NOTICES.md).
