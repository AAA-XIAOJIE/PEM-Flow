from __future__ import annotations
import csv
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from shapely import contains_xy
from pemflow.downstream import grid as grid_module
from pemflow.downstream import shapes as shapes_module
from pemflow.downstream.grid import Grid, segment_headings
from pemflow.downstream.shapes import MIXED_RUNS

CLASS = {0: "pedestrian", 9: "e_bike", 3: "motor_vehicle_probe"}
FIELDS = ["speed", "sg_tangential", "acceleration_norm", "position_step"]
ICOLS = [
    "source_file",
    "source_row",
    "frame",
    "time_s",
    "id",
    "class_",
    "real_x",
    "real_y",
    "velocity",
    "cell_area",
    "domain_cell_area",
    "effective_cell_area",
    "local_density",
    "Q",
    "p_ebike_neighbor",
    "zero_domain_cell",
    "zero_roi_cell",
]
FCOLS = [
    "frame",
    "time_s",
    "n_in_roi",
    "n_ped",
    "n_ebike",
    "n_vehicle",
    "density_count",
    "density_total",
    "density_ped",
    "density_ebike",
    "density_vehicle",
    "flow_total",
    "n_velocity_missing",
    "vehicle_phase",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def finite_max(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else None


def describe(values, field):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if field == "sg_tangential":
        x = np.abs(x)
    result = {"n": int(x.size)}
    result.update({k: None for k in ["minimum", "mean", "median", "p95", "p99", "maximum"]})
    if x.size:
        result.update(
            minimum=float(x.min()),
            mean=float(x.mean()),
            median=float(np.median(x)),
            p95=float(np.quantile(x, 0.95)),
            p99=float(np.quantile(x, 0.99)),
            maximum=float(x.max()),
        )
    thresholds = {
        "speed": [12],
        "sg_tangential": [2, 10],
        "acceleration_norm": [2, 10],
        "position_step": [1],
    }[field]
    result["thresholds"] = {
        str(v): {
            "above": int((x > v).sum()),
            "fraction_at_or_below": float((x <= v).mean()) if x.size else None,
        }
        for v in thresholds
    }
    return result


def aligned_tables(trajectory, motion, lineage, fps=30):
    t = pd.read_csv(trajectory, dtype={"frame": "int32", "id": "int32", "class": "int8"})
    m = pd.read_csv(
        motion,
        dtype={
            "frame": "int32",
            "id": "int32",
            "class": "int8",
            "motion_segment": "int32",
            "motion_segment_frames": "int32",
            "motion_window": "int16",
            "motion_status": "category",
            "adaptive_motion_window": "bool",
        },
    )
    lineage_frame = pd.read_csv(
        lineage,
        usecols=["frame", "id", "source_track_id", "motion_segment"],
        dtype={"frame": "int32", "id": "int32", "motion_segment": "int32"},
    )
    if len(t) != len(m) or len(t) != len(lineage_frame):
        raise ValueError("Trajectory/motion/lineage lengths differ")
    for other in (m, lineage_frame):
        if not np.array_equal(t[["frame", "id"]], other[["frame", "id"]]):
            raise ValueError("Trajectory/motion/lineage original row keys differ")
    if not np.array_equal(t["class"], m["class"]) or not np.allclose(
        t.velocity, m.velocity, rtol=0, atol=0, equal_nan=True
    ):
        raise ValueError("Stored class/speed differs between trajectory and motion")
    if not np.array_equal(m.motion_segment, lineage_frame.motion_segment):
        raise ValueError("Stored motion/lineage boundaries differ")
    if t.duplicated(["frame", "id"]).any():
        raise ValueError("Duplicate frame/final-ID keys in revised CSV")
    t["source_row"] = np.arange(1, len(t) + 1, dtype=np.int64)
    order = np.lexsort((t.source_row.to_numpy(), t.frame.to_numpy(), t.id.to_numpy()))
    sorted_source = bool(np.array_equal(order, np.arange(len(t))))
    t = t.iloc[order].reset_index(drop=True)
    m = m.iloc[order].reset_index(drop=True)
    lineage_frame = lineage_frame.iloc[order].reset_index(drop=True)
    ids, frames, classes = (t.id.to_numpy(), t.frame.to_numpy(), t["class"].to_numpy())
    source = lineage_frame.source_track_id.to_numpy()
    stored = m.motion_segment.to_numpy()
    same_source = (source[1:] == source[:-1]) | pd.isna(source[1:]) & pd.isna(source[:-1])
    continuous = np.r_[
        False,
        (ids[1:] == ids[:-1])
        & (frames[1:] == frames[:-1] + 1)
        & (classes[1:] == classes[:-1])
        & same_source
        & (stored[1:] == stored[:-1]),
    ]
    heading_lineage = pd.DataFrame({"motion_segment": np.cumsum(~continuous)})
    return (t, m, lineage_frame, heading_lineage, continuous, sorted_source)


def metric_rows(key, run, t, m, continuous, inside):
    speed = t.velocity.to_numpy()
    tangent = m.tangential_acceleration.to_numpy()
    norm = m.acceleration_norm.to_numpy()
    xy = t[["real_x", "real_y"]].to_numpy()
    step = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    values = {
        "speed": speed,
        "sg_tangential": tangent,
        "acceleration_norm": norm,
        "position_step": np.where(continuous, step, np.nan),
    }
    rows, cache = ([], {})
    for scope, selection in [("scene", np.ones(len(t), bool)), ("roi", inside)]:
        for code in [0, 9, 3, "ALL"]:
            mask = selection if code == "ALL" else selection & (t["class"].to_numpy() == code)
            row = dict(
                scope=scope,
                segment=key,
                run=run,
                class_code=code,
                class_name="all_classes" if code == "ALL" else CLASS[code],
                rows=int(mask.sum()),
                identities=int(t.loc[mask, "id"].nunique()),
                missing_speed=int((mask & ~np.isfinite(speed)).sum()),
                missing_sg_tangential=int((mask & ~np.isfinite(tangent)).sum()),
                missing_acceleration_norm=int((mask & ~np.isfinite(norm)).sum()),
            )
            for field, array in values.items():
                selected = array[mask & np.isfinite(array)]
                cache[f"{scope}__{code}__{field}"] = selected
                row[field] = describe(selected, field)
            rows.append(row)
    component = (
        np.isfinite(speed)
        & np.isfinite(m.vx)
        & np.isfinite(m.vy)
        & np.isfinite(m.ax)
        & np.isfinite(m.ay)
    )
    checks = dict(
        nonfinite_positions=int((~np.isfinite(xy).all(axis=1)).sum()),
        duplicate_frame_id_rows=0,
        speed_hypot_max_residual=finite_max(np.abs(np.hypot(m.vx, m.vy) - speed)),
        acceleration_norm_max_residual=finite_max(np.abs(np.hypot(m.ax, m.ay) - norm)),
        tangential_weighted_max_residual=finite_max(
            np.abs(
                tangent[component] * speed[component]
                - (
                    m.vx.to_numpy()[component] * m.ax.to_numpy()[component]
                    + m.vy.to_numpy()[component] * m.ay.to_numpy()[component]
                )
            )
        ),
        time_max_residual_s=finite_max(np.abs(m.time_s.to_numpy() - (t.frame.to_numpy() - 1) / 30)),
        negative_speed_rows=int((np.isfinite(speed) & (speed < 0)).sum()),
        negative_acceleration_norm_rows=int((np.isfinite(norm) & (norm < 0)).sum()),
    )
    return (rows, cache, checks)


def segment(job):
    key, run, output = (job["item"]["key"], job["item"]["run"], Path(job["output"]))
    source = {k: Path(v) for k, v in job["sources"].items()}
    cfg = read(job["config"])
    source_hash = {k: sha(v) for k, v in source.items()}
    binding = dict(
        sources=source_hash,
        configuration_sha256=sha(job["config"]),
        implementation_sha256=sha(__file__),
        quality_roi_sha256=sha(Path(grid_module.__file__)),
        voronoi_sha256=sha(Path(shapes_module.__file__)),
    )
    done = output / "segments" / (key + ".json")
    if job.get("resume") and done.exists():
        result = read(done)
        if result["binding"] != binding:
            return segment({**job, "resume": False})
        for path, digest in result["products_sha256"].items():
            if sha(output / path) != digest:
                raise ValueError("Downstream product changed: " + path)
        print("RESUME", key, flush=True)
        return result
    started = time.monotonic()
    t, m, lineage_frame, heading_lineage, continuous, was_sorted = aligned_tables(**source)
    print("LOADED", key, len(t), "source_sorted", was_sorted, flush=True)
    total_frames = int(job["item"]["profile"]["frames"])
    if len(t) != job["expected_rows"] or t.frame.min() < 1 or t.frame.max() > total_frames:
        raise ValueError(key + " revised row count/frame coverage differs")
    grid = Grid(cfg)
    inside = contains_xy(grid.polygon, t.real_x.to_numpy(), t.real_y.to_numpy())
    rows, cache, checks = metric_rows(key, run, t, m, continuous, inside)
    if (
        checks["nonfinite_positions"]
        or checks["negative_speed_rows"]
        or checks["negative_acceleration_norm_rows"]
    ):
        raise ValueError(key + " invalid stored position/derivative values")
    headings = segment_headings(t, heading_lineage)
    print(
        "HEADINGS",
        key,
        len(headings),
        "elapsed_s",
        round(time.monotonic() - started, 1),
        flush=True,
    )
    frames = t.sort_values(["frame", "id", "source_row"], kind="stable").reset_index(drop=True)
    indexes = frames.frame.to_numpy()
    empty = frames.iloc[:0]
    individual = output / "roi/individual" / (key + ".csv")
    frame_path = output / "roi/frame" / (key + ".csv")
    individual.parent.mkdir(parents=True, exist_ok=True)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    count_density = np.zeros(total_frames)
    density = np.zeros(total_frames)
    flows = np.zeros(total_frames)
    n_roi = np.zeros(total_frames, np.int32)
    missing = np.zeros(total_frames, np.int32)
    clip = {c: [] for c in CLASS}
    zero_domain = zero_roi = records = neighbor_finite = 0
    with (
        individual.open("w", encoding="utf-8", newline="") as fi,
        frame_path.open("w", encoding="utf-8", newline="") as ff,
    ):
        iw, fw = (csv.DictWriter(fi, ICOLS), csv.DictWriter(ff, FCOLS))
        iw.writeheader()
        fw.writeheader()
        for fr in range(1, total_frames + 1):
            left, right = np.searchsorted(indexes, [fr, fr + 1])
            frame = frames.iloc[left:right] if left != right else empty
            cells = grid.cells(frame, headings, run in MIXED_RUNS)
            counts = {c: sum((r["class_"] == c for r in cells)) for c in CLASS}
            den = {
                c: sum((r["cell_area"] * r["local_density"] for r in cells if r["class_"] == c))
                / grid.area
                for c in CLASS
            }
            miss = sum((not np.isfinite(r["velocity"]) for r in cells))
            flow = (
                sum((r["cell_area"] * r["Q"] for r in cells)) / grid.area if miss == 0 else np.nan
            )
            item = dict(
                frame=fr,
                time_s=(fr - 1) / 30,
                n_in_roi=len(cells),
                n_ped=counts[0],
                n_ebike=counts[9],
                n_vehicle=counts[3],
                density_count=len(cells) / grid.area,
                density_total=sum(den.values()),
                density_ped=den[0],
                density_ebike=den[9],
                density_vehicle=den[3],
                flow_total=flow,
                n_velocity_missing=miss,
                vehicle_phase="with_vehicle" if counts[3] else "no_vehicle",
            )
            fw.writerow(
                {
                    k: "" if isinstance(v, float) and (not np.isfinite(v)) else v
                    for k, v in item.items()
                }
            )
            for cell in cells:
                cell.update(source_file=source["trajectory"].name, time_s=(fr - 1) / 30)
                iw.writerow(
                    {
                        k: "" if isinstance(v, float) and (not np.isfinite(v)) else v
                        for k, v in cell.items()
                    }
                )
                clip[cell["class_"]].append(cell["cell_area"] / cell["effective_cell_area"])
            records += len(cells)
            zero_domain += sum((c["zero_domain_cell"] for c in cells))
            zero_roi += sum((c["zero_roi_cell"] for c in cells))
            neighbor_finite += sum((np.isfinite(c["p_ebike_neighbor"]) for c in cells))
            count_density[fr - 1] = item["density_count"]
            density[fr - 1] = item["density_total"]
            flows[fr - 1] = flow
            n_roi[fr - 1] = len(cells)
            missing[fr - 1] = miss
            if fr % 500 == 0:
                dump(
                    output / "progress" / (key + ".json"),
                    dict(
                        key=key,
                        frame=fr,
                        total=total_frames,
                        rows=records,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
                print(
                    "ROI",
                    key,
                    fr,
                    "/",
                    total_frames,
                    "elapsed_s",
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
    if records != int(inside.sum()) or int(missing.sum()) != int(
        (inside & ~np.isfinite(t.velocity)).sum()
    ):
        raise ValueError(key + " ROI coverage/missing-speed mismatch")
    cache.update(
        frame_count_density=count_density,
        frame_density=density,
        frame_flow=flows,
        frame_n_roi=n_roi,
        frame_missing_velocity=missing,
    )
    cache_path = output / "cache" / (key + ".npz")
    cache_path.parent.mkdir(exist_ok=True)
    np.savez(cache_path, **cache)
    ratios = density[count_density > 0] / count_density[count_density > 0]
    result = dict(
        key=key,
        run=run,
        source_rows=len(t),
        frames=total_frames,
        rows=records,
        source_originally_sorted=was_sorted,
        continuous_heading_segments=int(heading_lineage.motion_segment.max()),
        stored_motion_segments=int(m.motion_segment.nunique()),
        gaps=int(
            (
                (t.id.to_numpy()[1:] == t.id.to_numpy()[:-1]) & (np.diff(t.frame.to_numpy()) > 1)
            ).sum()
        ),
        speed_missing_rows=int(t.velocity.isna().sum()),
        sg_tangential_missing_rows=int(m.tangential_acceleration.isna().sum()),
        acceleration_norm_missing_rows=int(m.acceleration_norm.isna().sum()),
        motion_window_rows={str(k): int(v) for k, v in m.motion_window.value_counts().items()},
        roi_area_m2=grid.area,
        empty_roi_frames=int((n_roi == 0).sum()),
        missing_velocity_rows=int(missing.sum()),
        missing_frame_flow=int((~np.isfinite(flows)).sum()),
        zero_roi_cell_rows=int(zero_roi),
        zero_domain_cell_rows=int(zero_domain),
        neighbor_finite_rows=int(neighbor_finite),
        maximum_frame_density=finite_max(density),
        maximum_frame_flow=finite_max(flows),
        count_density_correlation=float(np.corrcoef(count_density, density)[0, 1]),
        ratio_mean=float(ratios.mean()) if ratios.size else None,
        ratio_sd=float(ratios.std(ddof=1)) if ratios.size > 1 else None,
        clip_full_by_class={
            str(c): dict(n=len(v), mean=float(np.mean(v)) if v else None) for c, v in clip.items()
        },
        metric_rows=rows,
        checks=checks,
        binding=binding,
        sources={k: str(v) for k, v in source.items()},
        elapsed_seconds=time.monotonic() - started,
    )
    for k, path in source.items():
        if sha(path) != source_hash[k]:
            raise ValueError("Source changed while calculating: " + str(path))
    result.update(individual_sha256=sha(individual), frame_sha256=sha(frame_path))
    result["products_sha256"] = {
        str(p.relative_to(output)).replace("\\", "/"): sha(p)
        for p in [individual, frame_path, cache_path]
    }
    dump(done, result)
    print(
        "DONE",
        key,
        "ROI_rows",
        records,
        "elapsed_s",
        round(result["elapsed_seconds"], 1),
        flush=True,
    )
    return result


def aggregate(results, output):
    tables, summaries = ([], {})
    groups = [("TOTAL", results)] + [
        (run, [s for s in results if s["run"] == run])
        for run in sorted({s["run"] for s in results})
    ]
    for label, segments in groups:
        archives = [
            np.load(output / "cache" / (s["key"] + ".npz"), allow_pickle=False) for s in segments
        ]
        summary = {}
        try:
            for scope in ["scene", "roi"]:
                summary[scope] = {}
                for code in [0, 9, 3, "ALL"]:
                    original = [
                        r
                        for s in segments
                        for r in s["metric_rows"]
                        if r["scope"] == scope and r["class_code"] == code
                    ]
                    row = dict(
                        scope=scope,
                        segment=label,
                        run=label,
                        class_code=code,
                        class_name="all_classes" if code == "ALL" else CLASS[code],
                    )
                    for field in [
                        "rows",
                        "identities",
                        "missing_speed",
                        "missing_sg_tangential",
                        "missing_acceleration_norm",
                    ]:
                        row[field] = sum((r[field] for r in original))
                    for field in FIELDS:
                        row[field] = describe(
                            np.concatenate([a[f"{scope}__{code}__{field}"] for a in archives]),
                            field,
                        )
                    summary[scope][str(code)] = row
                    tables.append(row)
            counts = np.concatenate([a["frame_count_density"] for a in archives])
            densities = np.concatenate([a["frame_density"] for a in archives])
            flow = np.concatenate([a["frame_flow"] for a in archives])
            ratio = densities[counts > 0] / counts[counts > 0]
            summary["roi"]["frame_statistics"] = dict(
                frames=len(counts),
                mean_count_density=float(counts.mean()),
                mean_density=float(densities.mean()),
                mean_finite_flow=float(flow[np.isfinite(flow)].mean()),
                maximum_frame_density=finite_max(densities),
                maximum_frame_flow=finite_max(flow),
                count_density_correlation=float(np.corrcoef(counts, densities)[0, 1]),
                nonempty_frame_ratio_mean=float(ratio.mean()),
                nonempty_frame_ratio_sd=float(ratio.std(ddof=1)),
                empty_roi_frames=int((counts == 0).sum()),
                missing_frame_flow=int((~np.isfinite(flow)).sum()),
            )
            summaries[label] = summary
        finally:
            for archive in archives:
                archive.close()
    flat = []
    for row in [r for s in results for r in s["metric_rows"]] + tables:
        item = {k: v for k, v in row.items() if k not in FIELDS}
        for field in FIELDS:
            item.update({field + "_" + k: v for k, v in row[field].items() if k != "thresholds"})
            for threshold, values in row[field]["thresholds"].items():
                item.update(
                    {field + "_threshold_" + threshold + "_" + k: v for k, v in values.items()}
                )
        flat.append(item)
    pd.DataFrame(flat).to_csv(output / "trajectory_metrics.csv", index=False)
    return summaries
