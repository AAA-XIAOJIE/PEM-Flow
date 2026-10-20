Three 10-second clips: 24, 48 and 72 participants; nominal 50% e-bikes.
Place the downloaded clips in this directory.
Run from code/:
python -m pemflow run --dataset configs/demo_dataset.json --video-root ../assets/demos --workers 2 --device 0 --output ../outputs/demo
The corresponding trajectory, motion and lineage tables are in data/demo/.
Each clip uses local frame numbers and tracker IDs.
See clip_manifest.json for source frame ranges and encoding.
