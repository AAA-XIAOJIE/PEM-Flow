# Data dictionary

| File | Main fields / meaning |
| --- | --- |
| `trajectory/*.csv` | `frame` one-based local video frame; `id` final output identity; `class` 0 pedestrian, 9 e-bike, 3 vehicle probe; `real_x`, `real_y` world position in m; `velocity` m/s; `dist` within-segment position increment in m. |
| `motion/*.csv` | Same frame/ID/class keys and speed; `time_s=(frame-1)/30`; `vx,vy` m/s; `ax,ay` m/s²; `acceleration_norm`; signed `tangential_acceleration`; `motion_segment`, segment length, actual window, motion status and adaptive-window flag. |
| `lineage/*.csv.gz` | Original observation/source identity and raw class, output identity/class, raw projected position, motion boundaries and observation/recovery provenance. Curated rows also carry manual source information. |
| `intermediates/*/observations.npz` | Eight-column automatic trajectory observations; `frame,source_track_id,raw_class,cx_norm,cy_norm,width_norm,height_norm,confidence`. |
| `intermediates/*/matrices.npz` | `frame_index` zero-based and `matrices` N × 3 × 3 in the calibration reference image. |
| `roi/individual/*.csv` | Original source row, frame, class, position, stored speed, ROI-clipped cell area, full domain cell area, effective area, local density, flow Q and neighbor composition. |
| `roi/frame/*.csv` | Source-frame counts, class counts, count density, shape-cell density, flow, missing speed count and vehicle phase. |
| `downstream/trajectory_metrics.csv` | Scene/ROI statistics by video segment, experiment and overall cohort. Four metric prefixes: `speed` (m/s), `sg_tangential` (m/s²), `acceleration_norm` (m/s²), and `position_step` (m). Each has finite sample counts, distribution statistics and reasonableness-threshold summaries. |
| `revision_records/audit/*` | Per-row before/after field tokens, deleted/inserted rows, source observation and direct/causal human rule IDs. |

CSV float output uses ten decimal places. Empty numeric fields mean unavailable
values and must remain missing when calculating later statistics. Identity
numbers are local to each source video; they must not be joined across files
merely because the numbers coincide.

`tangential_acceleration` in the motion table is signed: positive means speeding
up and negative means slowing down. The `sg_tangential_*` distribution and
threshold summaries use its absolute value. `acceleration_norm` is the
nonnegative magnitude of the two SG acceleration components and also reflects
changes in motion direction. These summaries read the stored SG derivatives.

`plots/` contains figures paired with that edition's CSVs. `dataset.json` carries
the source segment profiles and scientific settings. Human review rules and
submitted coordinates are stored separately from automatic generation inputs.
