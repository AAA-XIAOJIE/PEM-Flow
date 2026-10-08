from __future__ import annotations
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from pemflow.pipeline import motion as quality_pipeline
from pemflow.pipeline.motion import continuous_motion, label_frame, project

TABLES = ("trajectory.csv", "motion.csv", "lineage.csv.gz")
INDEX_COLUMNS = [
    "observation_id",
    "frame",
    "source_track_id",
    "id",
    "raw_class",
    "class",
    "motion_segment",
    "class_edit",
    "identity_edit",
    "class_unified",
]
MOTION_COLUMNS = [
    "real_x",
    "real_y",
    "velocity",
    "dist",
    "time_s",
    "vx",
    "vy",
    "ax",
    "ay",
    "acceleration_norm",
    "tangential_acceleration",
    "motion_segment",
    "motion_segment_frames",
    "motion_window",
    "motion_status",
    "smoothed",
    "adaptive_motion_window",
    "raw_x",
    "raw_y",
]
MANUAL_LINEAGE_COLUMNS = [
    "observation_origin",
    "original_observation_id",
    "original_source_key",
    "manual_source_key",
    "manual_role",
]
SPACES = {
    "bbox_center_pixels": 2,
    "bbox_center_normalized": 2,
    "bbox_pixels": 4,
    "bbox_normalized": 4,
    "raw_xy_m": 2,
}


def _open_csv(path, mode="rt"):
    if Path(path).suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8-sig" if "r" in mode else "utf-8", newline="")
    return Path(path).open(
        mode.replace("t", ""), encoding="utf-8-sig" if "r" in mode else "utf-8", newline=""
    )


def _token(value):
    if isinstance(value, bool | np.bool_):
        return "True" if value else "False"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        if not math.isfinite(float(value)):
            if math.isnan(float(value)):
                return ""
            raise ValueError("Infinite revised value")
        return f"{float(value):.10f}"
    return str(value)


def _json_value(value):
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _normalized_box(value, space, profile):
    value = np.asarray(value, dtype=float)
    if space.endswith("pixels"):
        value = value / np.asarray(
            [profile["image_width_px"], profile["image_height_px"]] * (len(value) // 2)
        )
    if np.any(value < 0) or np.any(value > 1) or (len(value) == 4 and np.any(value[2:] <= 0)):
        raise ValueError("Explicit bbox coordinates outside normalized bounds")
    return value


def revise_segment(index, observations, matrices, profile, rules):
    before = index.copy().reset_index(drop=True)
    if before.observation_id.duplicated().any() or before.duplicated(["frame", "id"]).any():
        raise ValueError("Automatic lineage contains duplicate observation or frame/ID keys")
    for column in (
        "observation_id",
        "frame",
        "source_track_id",
        "id",
        "raw_class",
        "class",
        "motion_segment",
    ):
        values = before[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Noninteger automatic lineage identifier")
        before[column] = before[column].astype(np.int64)
    box_data = label_frame(observations, profile["frames"])
    positions = before.observation_id.to_numpy() - 1
    if np.any(positions < 0) or np.any(positions >= len(box_data)):
        raise ValueError("Lineage observation ID is outside automatic observations")
    chosen = box_data.iloc[positions].reset_index(drop=True)
    for name in ("frame", "source_track_id", "raw_class", "observation_id"):
        if not np.array_equal(chosen[name], before[name]):
            raise ValueError("Automatic observations and lineage are not the same source rows")
    data = chosen.copy()
    for name in ("id", "class", "class_edit", "identity_edit", "class_unified"):
        data[name] = before[name].to_numpy()
    deleted = np.zeros(len(data), dtype=bool)
    direct = defaultdict(list)
    direct_fields = defaultdict(set)
    used = defaultdict(set)
    rule_tracks = {}
    raw_overrides = {}
    roles = {}
    insertion_rules = []
    for rule in rules:
        if rule["kind"] == "insert":
            insertion_rules.append(rule)
            continue
        scope = before.source_track_id.isin(rule["source_ids"]) & before.frame.between(
            rule["start_frame"], rule["end_frame"]
        )
        selected = before.index[scope].to_numpy()
        if len(selected) != rule["expected_rows"]:
            raise ValueError(
                f"Rule {rule['rule_id']} expected {rule['expected_rows']} rows, found {len(selected)}"
            )
        if set(before.loc[scope, "source_track_id"]) != set(rule["source_ids"]):
            raise ValueError("Rule does not match every requested source ID")
        field = {
            "class": "class",
            "identity_map": "id",
            "delete": "deleted",
            "position": "position",
            "role": "role",
        }[rule["kind"]]
        for row in selected:
            if field in used[row] or "deleted" in used[row] or (field == "deleted" and used[row]):
                raise ValueError("Overlapping contradictory revision scopes")
            used[row].add(field)
            observation = int(before.at[row, "observation_id"])
            direct[observation].append(rule["rule_id"])
            direct_fields[observation].add(field)
        if rule["kind"] == "class":
            if not before.loc[scope, "class"].eq(rule["expected_class"]).all():
                raise ValueError("Class precondition does not match automatic CSV")
            data.loc[scope, "class"] = rule["class"]
            data.loc[scope, "class_edit"] = True
            data.loc[scope, "class_unified"] = False
        elif rule["kind"] == "identity_map":
            for source in rule["source_ids"]:
                subset = scope & before.source_track_id.eq(source)
                if not before.loc[subset, "id"].eq(rule["expected_ids"][str(source)]).all():
                    raise ValueError("Identity precondition does not match automatic CSV")
                data.loc[subset, "id"] = rule["mapping"][str(source)]
                data.loc[subset, "identity_edit"] = True
        elif rule["kind"] == "delete":
            deleted[selected] = True
        elif rule["kind"] == "role":
            for row in selected:
                roles[int(before.at[row, "observation_id"])] = rule["role"]
        else:
            supplied = {(v["frame"], v["source_id"]): v["value"] for v in rule["values"]}
            actual = set(
                zip(before.loc[scope, "frame"], before.loc[scope, "source_track_id"], strict=True)
            )
            if set(supplied) != actual:
                raise ValueError("Position values must cover the selected source rows exactly")
            for row in selected:
                value = np.asarray(
                    supplied[before.at[row, "frame"], before.at[row, "source_track_id"]],
                    dtype=float,
                )
                space = rule["space"]
                if space == "raw_xy_m":
                    raw_overrides[int(before.at[row, "observation_id"])] = value
                    continue
                value = _normalized_box(value, space, profile)
                names = ["cx_norm", "cy_norm", "width_norm", "height_norm"][: len(value)]
                data.loc[row, names] = value
        if rule["kind"] != "role":
            rule_tracks[rule["rule_id"]] = set(before.loc[scope, "id"]) | set(data.loc[scope, "id"])
    original_before = before.copy()
    inserted = {}
    source_keys = set(zip(box_data.frame, box_data.source_track_id, strict=True))
    for rule in insertion_rules:
        tracks = set()
        for supplied in rule["values"]:
            key = (supplied["frame"], supplied["source_id"])
            if key in source_keys:
                raise ValueError("Insert source key already exists in automatic observations")
            source_keys.add(key)
            observation = -1 - len(inserted)
            box = _normalized_box(supplied["value"], rule["space"], profile)
            item = dict(
                frame=supplied["frame"],
                source_track_id=supplied["source_id"],
                observation_id=observation,
                id=supplied["id"],
                raw_class=np.nan,
                confidence=np.nan,
                **{"class": supplied["class"]},
                cx_norm=box[0],
                cy_norm=box[1],
                width_norm=box[2],
                height_norm=box[3],
                class_edit=False,
                identity_edit=False,
                class_unified=False,
            )
            data = pd.concat([data, pd.DataFrame([item])], ignore_index=True)
            before = pd.concat(
                [before, pd.DataFrame([{**item, "motion_segment": -1}])[before.columns]],
                ignore_index=True,
            )
            deleted = np.append(deleted, False)
            direct[observation] = [rule["rule_id"]]
            direct_fields[observation] = {"inserted"}
            inserted[observation] = {
                "manual_source_key": f"{rule['rule_id']}:{key[0]}:{key[1]}",
                "record": item,
            }
            tracks.add(supplied["id"])
        rule_tracks[rule["rule_id"]] = tracks
    kept = data.loc[~deleted]
    if kept.duplicated(["frame", "id"]).any():
        raise ValueError("Identity revision creates duplicate frame/ID keys")
    touched_ids = set().union(*rule_tracks.values()) if rule_tracks else set()
    affected = before.id.isin(touched_ids) | data.id.isin(touched_ids)
    retained = affected & ~deleted
    recalc = data.loc[retained].copy()
    patches = {}
    if len(recalc):
        xy = project(recalc, profile, matrices)
        for i, observation in enumerate(recalc.observation_id):
            if int(observation) in raw_overrides:
                xy[i] = raw_overrides[int(observation)]
        computed = continuous_motion(recalc, xy, fps=profile["fps"])
        original_groups = original_before.groupby("motion_segment").observation_id.agg(
            lambda x: frozenset(x)
        )
        group_ids = {members: int(group) for group, members in original_groups.items()}
        next_id = int(original_before.motion_segment.max()) + 1 if len(original_before) else 1
        for _, block in computed.groupby("motion_segment", sort=False):
            members = frozenset(block.observation_id)
            number = group_ids.get(members)
            if number is None:
                number = next_id
                next_id += 1
            computed.loc[block.index, "motion_segment"] = number
        for _, row in computed.iterrows():
            observation = int(row.observation_id)
            patches[observation] = {
                name: row[name]
                for name in MOTION_COLUMNS
                + ["id", "class", "class_edit", "identity_edit", "class_unified"]
            }
    scopes = {}
    for observation, role in roles.items():
        patches.setdefault(observation, {})["manual_role"] = role
    for row in before.index[affected | deleted | before.observation_id.isin(roles)]:
        observation = int(before.at[row, "observation_id"])
        is_inserted = observation in inserted
        causal = [
            rid
            for rid, tracks in rule_tracks.items()
            if before.at[row, "id"] in tracks or data.at[row, "id"] in tracks
        ]
        causal = list(dict.fromkeys(causal + direct[observation]))
        scopes[observation] = dict(
            observation_id=observation,
            frame=int(before.at[row, "frame"]),
            source_track_id=int(before.at[row, "source_track_id"]),
            original_id=None if is_inserted else int(before.at[row, "id"]),
            original_class=None if is_inserted else int(before.at[row, "class"]),
            inserted=is_inserted,
            observation_origin="manual_user_bbox" if is_inserted else "automatic",
            original_observation_id=None if is_inserted else observation,
            original_source_key=None
            if is_inserted
            else f"{int(before.at[row, 'frame'])}:{int(before.at[row, 'source_track_id'])}",
            manual_source_key=inserted[observation]["manual_source_key"] if is_inserted else None,
            manual_role=roles.get(observation),
            revised_id=int(data.at[row, "id"]),
            revised_class=int(data.at[row, "class"]),
            direct_rule_ids=direct[observation],
            causal_rule_ids=causal,
            direct_fields=sorted(direct_fields[observation]),
            deleted=bool(deleted[row]),
            recomputed=bool(retained[row]),
            automatic_box=None
            if is_inserted
            else {
                name: _json_value(chosen.at[row, name]) for name in quality_pipeline.LABEL_COLUMNS
            },
            revised_box={
                name: _json_value(data.at[row, name]) for name in quality_pipeline.LABEL_COLUMNS
            },
            raw_xy_override_m=raw_overrides.get(observation, np.array([])).tolist() or None,
            before=None if is_inserted else {},
            after={} if not deleted[row] else None,
        )
    return (patches, scopes)


def _write_revised_table(source, destination, table, index, patches, scopes):
    count = 0
    manual_lineage = table == "lineage.csv.gz" and any(
        (entry["inserted"] or entry["manual_role"] for entry in scopes.values())
    )
    with _open_csv(source) as src, _open_csv(destination, "xt") as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames)
        if manual_lineage:
            if set(MANUAL_LINEAGE_COLUMNS[1:]) & set(fieldnames):
                raise ValueError("Input lineage is already a manual revision")
            fieldnames += [name for name in MANUAL_LINEAGE_COLUMNS if name not in fieldnames]
        writer = csv.DictWriter(dst, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        identifiers = index[["observation_id", "frame", "id", "class"]].itertuples(
            index=False, name=None
        )
        for row, identity in zip(reader, identifiers, strict=True):
            observation, frame, track, category = identity
            if any(
                (
                    float(row[name]) != value
                    for name, value in (("frame", frame), ("id", track), ("class", category))
                )
            ):
                raise ValueError("CSV/motion/lineage row order or identity binding differs")
            if table == "lineage.csv.gz" and int(row["observation_id"]) != observation:
                raise ValueError("Lineage observation row order differs")
            before_tokens = dict(row)
            if manual_lineage:
                row.setdefault("observation_origin", "automatic")
                row.update(
                    original_observation_id=str(observation),
                    original_source_key=f"{frame}:{row['source_track_id']}",
                    manual_source_key="",
                    manual_role="",
                )
            if observation in scopes:
                entry = scopes[observation]
                if table == "lineage.csv.gz":
                    entry["observation_origin"] = before_tokens.get(
                        "observation_origin", "automatic"
                    )
                entry["before"][table] = before_tokens
                if entry["deleted"]:
                    continue
                for name, value in patches.get(observation, {}).items():
                    if name in row:
                        row[name] = _token(value)
                entry["after"][table] = dict(row)
            writer.writerow(row)
            count += 1
        for entry in sorted(
            (item for item in scopes.values() if item["inserted"]),
            key=lambda item: (item["frame"], item["revised_id"]),
        ):
            observation = entry["observation_id"]
            values = {
                **entry["revised_box"],
                **patches[observation],
                "observation_id": observation,
                "frame": entry["frame"],
                "source_track_id": entry["source_track_id"],
                "observation_origin": "manual_user_bbox",
                "original_observation_id": "",
                "original_source_key": "",
                "manual_source_key": entry["manual_source_key"],
                "manual_role": entry["manual_role"] or "",
            }
            row = {
                name: _token(values[name]) if values.get(name) is not None else ""
                for name in fieldnames
            }
            entry["after"][table] = dict(row)
            writer.writerow(row)
            count += 1
    return count


def _write_audit(output, key, scopes, rules, batch_id):
    metadata = {
        rule["rule_id"]: {
            **{
                name: rule[name]
                for name in (
                    "rule_id",
                    "kind",
                    "actor",
                    "time",
                    "reason",
                    "evidence",
                    "user_approved",
                    "approval_reference",
                )
            },
            **({"category": rule["category"]} if rule["kind"] == "delete" else {}),
        }
        for rule in rules
    }
    changed = direct = deleted = 0
    deletion_categories = defaultdict(int)
    with (output / "row_changes.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        with (output / "affected_rows.csv").open("x", encoding="utf-8", newline="") as affected:
            writer = csv.DictWriter(
                affected,
                fieldnames=[
                    "segment",
                    "observation_id",
                    "frame",
                    "source_track_id",
                    "original_id",
                    "original_class",
                    "type",
                    "recomputed",
                    "values_changed",
                    "rule_ids",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            for entry in scopes.values():
                fields = []
                if entry["inserted"]:
                    fields = ["inserted"]
                elif not entry["deleted"]:
                    for table, before in entry["before"].items():
                        fields += [
                            table + ":" + name
                            for name in before.keys() | entry["after"][table].keys()
                            if before.get(name) != entry["after"][table].get(name)
                        ]
                else:
                    fields = ["deleted"]
                is_changed = (
                    bool(fields)
                    or entry["automatic_box"] != entry["revised_box"]
                    or entry["raw_xy_override_m"] is not None
                )
                direct += bool(entry["direct_rule_ids"])
                deleted += entry["deleted"]
                if entry["deleted"]:
                    deletion_categories[metadata[entry["direct_rule_ids"][0]]["category"]] += 1
                changed += is_changed
                kind = (
                    "direct_manual_deletion"
                    if entry["deleted"]
                    else "direct_manual_insertion"
                    if entry["inserted"]
                    else "direct_manual_role"
                    if entry["manual_role"] and (not entry["recomputed"])
                    else "direct_manual_and_derived"
                    if entry["direct_rule_ids"]
                    else "derived_recomputation"
                )
                item = dict(
                    segment=key,
                    batch_id=batch_id,
                    type=kind,
                    **entry,
                    changed_fields=fields,
                    rule_metadata=[metadata[rid] for rid in entry["causal_rule_ids"]],
                )
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                writer.writerow(
                    {
                        "segment": key,
                        **{
                            k: entry[k]
                            for k in (
                                "observation_id",
                                "frame",
                                "source_track_id",
                                "original_id",
                                "original_class",
                            )
                        },
                        "type": kind,
                        "recomputed": entry["recomputed"],
                        "values_changed": is_changed,
                        "rule_ids": json.dumps(entry["causal_rule_ids"], ensure_ascii=False),
                    }
                )
    return dict(
        affected_rows=len(scopes),
        direct_rows=direct,
        derived_only_rows=len(scopes) - direct,
        deleted_rows=deleted,
        changed_rows=changed,
        recomputed_rows=sum((entry["recomputed"] for entry in scopes.values())),
        inserted_rows=sum((entry["inserted"] for entry in scopes.values())),
        role_marked_rows=sum((bool(entry["manual_role"]) for entry in scopes.values())),
        deleted_rows_by_category=dict(deletion_categories),
    )
