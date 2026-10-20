Place the 19 source videos directly in this directory.
Use the filenames, byte counts and SHA-256 hashes in assets/video_manifest.csv.
G7 has two independently indexed video segments.
The detector weight is assets/model/train12/best.pt.
Run from code/:
python -m pemflow run --workers 2 --device 0 --output ../outputs/automatic
Installation and workflow: code/README.md or code/README_zh.md.
