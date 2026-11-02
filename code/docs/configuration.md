# Configuration guide

## Files

| File | Purpose |
| --- | --- |
| `configs/demo_dataset.json` | Three demonstration clips |
| `configs/dataset.json` | The 19 dataset recording segments |
| `configs/inference.json` | Detector image size, confidence and NMS settings |
| `configs/bytetrack_pemflow.yaml` | Tracker settings |
| `configs/pipeline.json` | ROI polygon, object shapes and density/flow settings |

## Describe an input video

Copy a dataset JSON into the same configuration directory. Retain its top-level
processing settings and edit the `segments` list for the desired inputs.
The detector and tracker configuration files are read from that same directory.

| Segment field | Value to supply |
| --- | --- |
| `key` | Unique segment key, used by `--segments` |
| `run`, `segment` | Experiment and recording labels |
| `source_video.filename` | Filename relative to `--video-root` |
| `source_video.frames`, `bytes`, `sha256` | Video frame count, file size and SHA-256 |
| `profile.frames`, `fps` | Frame count and frame rate |
| `profile.image_width_px`, `image_height_px` | Native image dimensions |
| `profile.camera_parameters.matrix` | 3 × 3 intrinsic matrix in pixels |
| `profile.camera_parameters.distortion` | OpenCV distortion coefficients |
| `profile.world_points` | Surveyed XYZ points in metres |
| `profile.image_points` | Corresponding pixel points in the same order |
| `profile.reference_frame_index` | Zero-based calibration image index |
| `stabilization_reference_index` | Zero-based stabilization reference index |
| `profile.class_reference_height_m` | Projection height in metres for each class |

Calibration pixel points must use the video's image coordinates and the
calibration reference frame. The reference indices must lie within the video.
Use consistent world axes and metre units for all measured points.

```shell
python -m pemflow run --dataset configs/my_dataset.json --video-root ../videos --output ../outputs/my_run
```

Explicit relative command-line paths resolve from the current working directory.

## Processing settings

The dataset JSON contains `stabilization`, `automatic_matrix_repairs`,
`automatic_observations` and `inference` settings. These control background
alignment and image-based recovery before world-coordinate projection.
Each source track uses its majority class for projection.

| Class | Code | Projection height |
| --- | --- | --- |
| Pedestrian | 0 | 0.72 m |
| E-bike | 9 | 0.20 m |
| Motor vehicle | 3 | 0.60 m |

Position, velocity and acceleration use a quadratic Savitzky–Golay fit within
continuous segments. The window is 31 frames; segments of 25–30 frames use the
largest valid odd window. Shorter segments retain positions and missing derivatives.

## ROI metrics

Pass `--config configs/pipeline.json` to `metrics` to choose the ROI and shape
settings. The ROI vertices use the trajectory coordinate system. Object centres
strictly inside the polygon contribute to ROI statistics. The program uses the
polygon's calculated area for normalization.

The shape settings use a pedestrian disc, an oriented e-bike rectangle and an
oriented vehicle rectangle. `voronoi` controls their dimensions, grid spacing,
domain margins and minimum cell areas. Individual density is the reciprocal of
effective cell area; flow is density times stored speed. ROI-clipped cell areas
supply frame-level weights. See the [data dictionary](data_dictionary.md) for outputs.
