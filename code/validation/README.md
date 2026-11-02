# Detection and tracking annotations

| File or directory | Contents |
| --- | --- |
| `reviewed_work/*_reviewed.xml` | Image dimensions, boxes, classes, review status and tracking identities |
| `sampling_plan.csv` | Mapping from each annotation image to its video and frame |
| `sample_metadata.csv` | Sample groups and image metadata |
| `images/` | Annotation images, supplied in `PEMFlow_data.zip` |

Tier A contains 270 detection frames. Tier B contains 600 frames for tracking.
The XML files use the classes `pedestrian`, `e-bike` and `motor-vehicle`.
`review_status=valid` identifies evaluated objects; `ignore` and `ambiguous`
regions retain their specified treatment. Tracking identities are local to each task.

From `code/`, evaluate the dataset with:

```shell
python -m pemflow evaluate --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation
```

Calculate AP from the supplied detector predictions with:

```shell
python examples/score_ap.py --output ../outputs/ap
```
