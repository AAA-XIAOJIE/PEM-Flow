# Processing definitions

## Geometry and observation order

YOLO11m train12 and ByteTrack run on each video as one serial stream at
confidence 0.3, NMS IoU 0.5, image size 1280, rectangular inference, FP32 and
batch size 1. Original inference observations contain frame, source track ID,
raw class, normalized centre/size and confidence. Serialization reproduces the
producer's six-significant-digit text convention.

Background stabilization uses SIFT2000, the configured reference frame, ratio
0.7, at least 21 matches and a 3-pixel RANSAC threshold. Diagnostics are generated
for every source frame. Selected abnormal/low-support frames receive masked
background SIFT correspondence checks. Correspondence groups are split into
50% fitting, 25% validation and 25% independent audit subsets. A replacement
matrix requires the configured coverage, residual and improvement criteria;
otherwise the measured base matrix is retained and its unresolved status is
reported. `matrix_quality.py` defines the checks.

The current matrices are rebased to the calibration reference image before
projection. The label frame `f` uses matrix index `f-1`. PnP uses the configured
intrinsics, distortion and surveyed 3D control points. Reference heights are
0.9 m for pedestrian centres, 0.2 m for e-bike centres and 0.6 m for the vehicle
probe. The trajectory's majority class is established before height selection.

Close-gap image recovery operates after matrix estimation, using the current
video and raw observations. It uses optical flow and image patch checks around
short gaps near another source track. It accepts only the current configured
geometric and image-evidence criteria. Observations retain their source identity
and record detector versus image-recovery provenance. Detector AP uses independent
low-threshold image predictions.

## Motion

Automatic source tracks use one majority class. Continuous segments split at
frame gaps, source track transitions or final ID transitions. Coordinates are
fitted with a quadratic Savitzky–Golay polynomial at 30 Hz. The same fit gives positions, velocity
components and acceleration components.

| Continuous segment length | Window |
| --- | --- |
| At least 31 frames | 31 |
| 25–30 frames | Largest odd length not exceeding the segment |
| Fewer than 25 frames | Measured positions; unavailable derivatives |

Speed is `hypot(vx, vy)`, acceleration magnitude is `hypot(ax, ay)`, and
tangential acceleration is `(vx*ax + vy*ay) / speed` when speed exceeds 1e-12.
`dist` is the Euclidean position increment within a motion segment. Segment
starts have missing increments and derivatives. Each row records actual window
and boundary/status information.

## ROI and finite-domain cells

The ROI polygon is specified by four surveyed world-coordinate vertices in
`configs/pipeline.json`; its actual area is 111.82971704 m². Selection uses the
strict polygon interior of the stored trajectory centres.

Shape-distance Voronoi cells use a 0.1 m grid over a finite domain expanded by
3 m longitudinally and 0 m laterally. Pedestrians use a 0.3 m disc; e-bikes use
1.6 × 0.65 m rectangles; the vehicle uses a 5 × 2 m rectangle. Headings split
on actual gaps, class/source changes and stored motion boundaries. Ties use the
stable ascending identity order.

Individual effective area is `max(domain_cell_area, class_floor)` with floors
0.3, 0.5 and 2 m² respectively. Individual density is its reciprocal. Individual
flow is density times stored speed. The clipped ROI cell area supplies weights
for frame density and flow; normalization uses the actual ROI polygon area.
Empty ROI frames yield zero. A frame with an unavailable selected speed retains
missing flow.

The 2 m/s² acceleration proportion is a reasonableness diagnostic. Diagnostic
flags for 12 m/s speed, 10 m/s² acceleration and 1 m increments identify possible
anomalies and produce reports. These values do not select or delete people.

## Human revisions and evaluation

The final rules were written from human review of the automatic trajectories
and video frames. Source IDs remain the selection keys even when final IDs are
mapped. User boxes use normalized or native-pixel centre coordinates and sizes.
Manual insertions have negative observation IDs and explicit manual provenance;
detector confidences remain unavailable. Affected full identities are projected
and fitted again with the production functions; unchanged tokens are retained.

Quality evaluation uses the unchanged full-roadway 870 reviewed frames:
270 for detection operating metrics/AP and 600 for tracking. Reviewed staff and
other ignore regions retain their GT status. CSV-based P/R use class-aware
IoU 0.5 matching; tracking uses one-to-one assignments and task-local GT IDs.
AP predicts at confidence 0.001 and integrates the confidence-ranked 101-point
precision envelope at IoU 0.50:0.05:0.95. Projected box-centre differences are
same-camera consistency measurements, rather than surveyed absolute accuracy.
