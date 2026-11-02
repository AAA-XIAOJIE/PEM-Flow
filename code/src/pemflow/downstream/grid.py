from __future__ import annotations

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from pemflow.downstream.shapes import (
    _circle_distance,
    _neighbor_ratios,
    _rectangle_distance,
    estimate_headings,
)


def segment_headings(frame, lineage):
    scene = frame.rename(columns={"id": "original_id", "frame": "frame_id"}).copy()
    scene["track_id"] = lineage.motion_segment.to_numpy()
    scene["t"] = (scene.frame_id - 1) / 30
    directions = estimate_headings(scene)
    lookup = dict(zip(scene.track_id, scene.original_id, strict=True))
    return {(int(lookup[s]), f): value for (s, f), value in directions.items()}


class Grid:
    def __init__(self, cfg):
        self.s = cfg["voronoi"]
        self.resolution = float(self.s["grid_resolution_m"])
        self.unit = self.resolution**2
        self.polygon = Polygon(cfg["roi_polygon_m"])
        if not self.polygon.is_valid or self.polygon.area <= 0 or self.resolution <= 0:
            raise ValueError("Invalid ROI/grid")
        self.area = float(self.polygon.area)
        low = np.asarray(cfg["roi_polygon_m"]).min(axis=0)
        high = np.asarray(cfg["roi_polygon_m"]).max(axis=0)
        margins = np.array([self.s["longitudinal_margin_m"], self.s["lateral_margin_m"]])
        self.low, self.high = (low - margins, high + margins)
        axes = [
            np.arange(a, b + self.resolution * 0.5, self.resolution)
            for a, b in zip(self.low, self.high, strict=True)
        ]
        x, y = np.meshgrid(*axes)
        self.shape = x.shape
        self.x, self.y = (x.ravel(), y.ravel())
        self.roi_mask = contains_xy(self.polygon, self.x, self.y)
        self.minimum = {int(k): float(v) for k, v in self.s["minimum_cell_area_m2"].items()}

    def cells(self, frame, headings, mixed):
        if frame.empty:
            return []
        frame = frame.sort_values("id", kind="stable")
        xy = frame[["real_x", "real_y"]].to_numpy(float)
        nearby = np.flatnonzero(((xy >= self.low) & (xy <= self.high)).all(axis=1))
        inside = contains_xy(self.polygon, xy[:, 0], xy[:, 1])
        if not len(nearby):
            return []
        classes = frame["class"].to_numpy(int)
        ids = frame.id.to_numpy(int)
        fr = int(frame.frame.iloc[0])
        minimum_distance = np.full(len(self.x), np.inf)
        labels = np.full(len(self.x), -1, dtype=int)
        for ordinal, ix in enumerate(nearby):
            c, (cx, cy) = (classes[ix], xy[ix])
            if c == 0:
                distance = _circle_distance(self.x, self.y, cx, cy, self.s["pedestrian_radius_m"])
            else:
                prefix = "motor_vehicle" if c == 3 else "e_bike"
                angle = headings.get((int(ids[ix]), fr), 0.0)
                distance = _rectangle_distance(
                    self.x,
                    self.y,
                    cx,
                    cy,
                    angle,
                    self.s[prefix + "_length_m"] / 2,
                    self.s[prefix + "_width_m"] / 2,
                )
            closer = distance < minimum_distance
            minimum_distance[closer] = distance[closer]
            labels[closer] = ordinal
        area_domain = np.bincount(labels, minlength=len(nearby)) * self.unit
        area_roi = np.bincount(labels[self.roi_mask], minlength=len(nearby)) * self.unit
        requested = {i for i, ix in enumerate(nearby) if inside[ix]}
        ratios = {}
        if mixed:
            clipped = labels.reshape(self.shape).copy()
            clipped[~self.roi_mask.reshape(self.shape)] = -1
            ratios = _neighbor_ratios(
                clipped, {i: int(classes[ix]) for i, ix in enumerate(nearby)}, requested
            )
        result = []
        for i in sorted(requested):
            row = frame.iloc[nearby[i]]
            effective = max(float(area_domain[i]), self.minimum[int(row["class"])])
            density = 1 / effective
            result.append(
                dict(
                    source_row=int(row.source_row),
                    frame=fr,
                    id=int(row.id),
                    class_=int(row["class"]),
                    real_x=float(row.real_x),
                    real_y=float(row.real_y),
                    velocity=float(row.velocity),
                    cell_area=float(area_roi[i]),
                    domain_cell_area=float(area_domain[i]),
                    effective_cell_area=effective,
                    local_density=density,
                    Q=density * float(row.velocity),
                    p_ebike_neighbor=ratios.get(i, np.nan),
                    zero_domain_cell=bool(area_domain[i] == 0),
                    zero_roi_cell=bool(area_roi[i] == 0),
                )
            )
        return result
