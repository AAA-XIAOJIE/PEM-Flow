# PEM-Flow

PEM-Flow processes 19 source videos from 18 experiments into trajectories,
applies the recorded human CSV revisions, and calculates motion, density, flow
and detection/tracking quality metrics.

| Directory | Contents |
| --- | --- |
| `assets/videos/` | Place the source videos here, using `assets/video_manifest.csv`. |
| `assets/model/train12/best.pt` | The train12 detector weight used by the processing chain. |
| `code/` | Python package, scientific configurations, final human rules and reviewed GT. |
| `data/automatic/` | Automatic CSVs, motion tables, row lineage, trajectory figures and projection inputs. |
| `data/revised/` | Human-revised CSVs, motion tables, row lineage and trajectory figures. |
| `data/downstream/` | Metrics calculated from the revised data. |
| `data/revision_records/` | Final human decisions, submitted boxes and per-row revision audit. |
| `manifest.json` | Package inventory and file hashes. |

The automatic edition contains **7,143,147** scene records; the revised edition
contains **6,887,738**. Both cover the same **132,663** source video frames.
The revised ROI contains **2,820,289** records. CSVs and their matching figures
are stored together in each edition. Figures use the same world-coordinate
extent and English labels.

Start with [the code instructions](code/README.md) or
[中文使用说明](code/README_zh.md). Scientific definitions are described in
[code/docs/methods.md](code/docs/methods.md).

Install [Git LFS](https://git-lfs.com/) before cloning this repository, then run
`git lfs install` and clone normally. In an existing clone, run `git lfs pull`
to download the model, reviewed images and dataset files. The source videos are
placed separately in `assets/videos/` using the video manifest.

## Source videos

The 19 source MP4 files are distributed through the Zenodo data archive. The
Zenodo deposit is being prepared; its public download link will be added here
when the record is published. The videos total 46.20 GB.

Download the original MP4 files and place them directly in `assets/videos/`,
keeping the names listed in `assets/video_manifest.csv`. Each file's byte count
and SHA-256 are included in that manifest. Keep the original files without
re-encoding. After placement, follow the installation and commands in
[code/README.md](code/README.md) or [code/README_zh.md](code/README_zh.md).

## Licences

PEM-Flow data use CC BY 4.0; see [DATA_LICENSE.txt](DATA_LICENSE.txt).
Software and model terms are described in
[code/THIRD_PARTY_NOTICES.md](code/THIRD_PARTY_NOTICES.md).
