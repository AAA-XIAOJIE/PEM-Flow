# Demonstration results

These tables correspond to the three source-video clips in `assets/demos/`.
Run `python -m pemflow run` from `code/`; results are written to `outputs/demo/`.

| Clip | Trajectory CSV | Motion CSV | Lineage |
| --- | --- | --- | --- |
| low_G6_24p_50pct_10s.mp4 | trajectory/G3-6.csv | motion/G3-6.csv | lineage/G3-6.csv.gz |
| medium_G7_48p_50pct_10s.mp4 | trajectory/G3-7.csv | motion/G3-7.csv | lineage/G3-7.csv.gz |
| high_G15_72p_50pct_10s.mp4 | trajectory/G3-15.csv | motion/G3-15.csv | lineage/G3-15.csv.gz |

Frame numbers start at 1 and tracker IDs are local to each clip.
`dataset.json` records the input configuration; `manifest.json` lists checksums.
The complete dataset's automatic and human-reviewed trajectories are in
`data/automatic/` and `data/revised/` in the Zenodo data archive.
