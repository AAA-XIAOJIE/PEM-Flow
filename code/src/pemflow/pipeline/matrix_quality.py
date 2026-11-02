from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pemflow.pipeline import matrix_features as video_matrix_repairs

AUTO_REPAIR_METHOD = "automatic_held_out_background_stabilization"
AUTO_REPAIR_VERSION = 1


@dataclass(frozen=True)
class AutomaticStabilizationConfig:
    seed: int = 20261003
    nfeatures: int = 4000
    correspondence_group_radius_px: float = 4.0
    train_fraction: float = 0.5
    validation_fraction: float = 0.25
    grid_columns: int = 4
    grid_rows: int = 4
    screen_reference_cells_lt: int = 8
    screen_corner_jump_gt_px: float = 10.0
    screen_inlier_p95_gt_px: float = 3.0
    trigger_global_median_gt_px: float = 1.0
    trigger_global_p95_gt_px: float = 3.0
    trigger_region_median_gt_px: float = 1.0
    trigger_region_fraction_within3_lt: float = 0.8
    region_min_points: int = 8
    coverage_cell_min_points: int = 3
    coverage_min_cells: int = 6
    coverage_min_span_fraction: float = 0.5
    validation_min_points: int = 40
    train_inliers_min_points: int = 30
    fit_ransac_threshold_px: float = 3.0
    fit_max_iterations: int = 2000
    fit_confidence: float = 0.995
    candidate_median_max_px: float = 0.5
    candidate_p95_max_px: float = 3.0
    candidate_fraction_within3_min: float = 0.9
    candidate_region_median_max_px: float = 1.0
    candidate_region_fraction_within3_min: float = 0.8
    improvement_factor_max: float = 0.5

    def __post_init__(self):
        if self.nfeatures < 30 or self.seed < 0 or self.correspondence_group_radius_px <= 0:
            raise ValueError("Invalid extraction or deterministic split configuration")
        if (
            not 0 < self.train_fraction < 1
            or not 0 < self.validation_fraction < 1 - self.train_fraction
        ):
            raise ValueError("Train/validation/audit splits must all be nonempty fractions")
        if (
            self.grid_columns < 1
            or self.grid_rows < 1
            or self.coverage_min_cells > self.grid_columns * self.grid_rows
        ):
            raise ValueError("Invalid spatial coverage configuration")
        if (
            min(
                self.region_min_points,
                self.coverage_cell_min_points,
                self.validation_min_points,
                self.train_inliers_min_points,
                self.fit_max_iterations,
            )
            < 1
        ):
            raise ValueError("Support and iteration counts must be positive")
        if not 0 < self.fit_confidence < 1 or not 0 < self.improvement_factor_max <= 1:
            raise ValueError("Invalid fit confidence or improvement factor")


AutomaticRepairConfig = AutomaticStabilizationConfig


def policy_metadata(config=AutomaticStabilizationConfig()):
    sources = [Path(__file__), Path(video_matrix_repairs.__file__)]
    return dict(
        method=AUTO_REPAIR_METHOD,
        version=AUTO_REPAIR_VERSION,
        config=asdict(config),
        stage="post_base_stabilization_pre_reference_rebase",
        matrix_basis="native_current_image_to_original_stabilization_reference",
        extraction="half-resolution masked SIFT4000, mutual BF-L2 ratio 0.7",
        population="raw mutual correspondences; no all-point inlier filtering",
        splitting="physical near-duplicate groups; train/validation/audit 50/25/25",
        seed_salt="SHA256(str(one_based_frame)) first four bytes little-endian",
        independent_audit_can_veto=True,
        refit_after_audit=False,
        full_road_accuracy_certified=False,
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    )


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def select_audit_frames(
    diagnostics,
    config=AutomaticStabilizationConfig(),
    *,
    expected_frame_count=None,
    reference_index=None,
):
    if isinstance(diagnostics, (str, Path)):
        with Path(diagnostics).open(encoding="utf-8-sig") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        records = list(diagnostics)
    indices = [int(record["frame_index"]) for record in records]
    count = len(records) if expected_frame_count is None else int(expected_frame_count)
    if indices != list(range(count)):
        raise ValueError("Stabilization diagnostics must cover every source frame once in order")
    selected = []
    for record in records:
        index, status = (int(record["frame_index"]), record.get("status"))
        if index == reference_index or status == "reference":
            continue
        reasons = []
        determinant = record.get("matrix_determinant")
        invalid_determinant = determinant is not None and (
            not np.isfinite(determinant) or abs(determinant) <= 1e-12
        )
        if (
            not record.get("matrix_finite", False)
            or invalid_determinant
            or status not in {"estimated", "reference"}
        ):
            reasons.append("fallback_or_invalid")
        cells = (record.get("inlier_grid_coverage") or {}).get("reference") or {}
        if (cells.get("occupied_cells") or 0) < config.screen_reference_cells_lt:
            reasons.append("poor_coverage")
        jump = (record.get("corner_jump_from_previous_px") or {}).get("max_px")
        if jump is not None and (not np.isfinite(jump) or jump > config.screen_corner_jump_gt_px):
            reasons.append("corner_jump")
        tail = (record.get("inlier_residual") or {}).get("p95_px")
        if tail is not None and (not np.isfinite(tail) or tail > config.screen_inlier_p95_gt_px):
            reasons.append("inlier_tail")
        if reasons:
            selected.append(
                dict(
                    frame_index=index,
                    frame=index + 1,
                    screen_reasons=reasons,
                    base_diagnostics=_json_value(record),
                )
            )
    return selected


def group_correspondences(current, reference, config=AutomaticStabilizationConfig()):
    current, reference = (np.asarray(current, float), np.asarray(reference, float))
    if current.shape != reference.shape or current.ndim != 2 or current.shape[1] != 2:
        raise ValueError("Correspondences must be equal Nx2 native-pixel arrays")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Correspondences contain non-finite coordinates")
    parent = np.arange(len(current))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    for first, second in sorted(
        cKDTree(current).query_pairs(config.correspondence_group_radius_px)
    ):
        if (
            np.linalg.norm(reference[first] - reference[second])
            <= config.correspondence_group_radius_px
        ):
            a, b = (root(first), root(second))
            if a != b:
                parent[max(a, b)] = min(a, b)
    grouped = {}
    for index in range(len(parent)):
        grouped.setdefault(root(index), []).append(index)
    members = list(grouped.values())
    return (
        np.asarray([current[group].mean(axis=0) for group in members]).reshape(-1, 2),
        np.asarray([reference[group].mean(axis=0) for group in members]).reshape(-1, 2),
        members,
    )


def split_group_indices(group_count, frame, config=AutomaticStabilizationConfig()):
    if int(frame) != frame or frame < 1 or int(group_count) != group_count or (group_count < 0):
        raise ValueError("Group count and one-based frame must be valid integers")
    salt = int.from_bytes(hashlib.sha256(str(int(frame)).encode()).digest()[:4], "little")
    seed = (config.seed + salt) % (2**31 - 1)
    order = np.random.default_rng(seed).permutation(int(group_count))
    train_count, validation_count = (
        int(config.train_fraction * len(order)),
        int(config.validation_fraction * len(order)),
    )
    return (
        dict(
            train=order[:train_count],
            validation=order[train_count : train_count + validation_count],
            audit=order[train_count + validation_count :],
        ),
        seed,
    )


def _cells(points, width, height, config):
    cells = np.floor(points / [width, height] * [config.grid_columns, config.grid_rows]).astype(int)
    return np.clip(cells[:, 1], 0, config.grid_rows - 1) * config.grid_columns + np.clip(
        cells[:, 0], 0, config.grid_columns - 1
    )


def _valid_matrix(matrix):
    return bool(
        matrix.shape == (3, 3)
        and np.isfinite(matrix).all()
        and (abs(matrix[2, 2]) > 1e-12)
        and (abs(np.linalg.det(matrix)) > 1e-12)
    )


def _quality(current, reference, matrix, width, height, config):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        projected = np.c_[current, np.ones(len(current))] @ matrix.T
        error = np.linalg.norm(projected[:, :2] / projected[:, 2:] - reference, axis=1)
    cell = _cells(reference, width, height, config)

    def stats(values):
        valid = bool(len(values) and np.isfinite(values).all())
        return dict(
            median_px=float(np.median(values)) if valid else None,
            p95_px=float(np.quantile(values, 0.95)) if valid else None,
            fraction_within3px=float(np.mean(values <= config.fit_ransac_threshold_px))
            if len(values)
            else None,
        )

    regions = [
        dict(
            cell=index,
            count=int((cell == index).sum()),
            quality_supported=int((cell == index).sum()) >= config.region_min_points,
            **stats(error[cell == index]),
        )
        for index in range(config.grid_columns * config.grid_rows)
        if np.any(cell == index)
    ]
    span = np.ptp(reference, axis=0) / [width, height] if len(reference) else np.zeros(2)
    populated = sum((region["count"] >= config.coverage_cell_min_points for region in regions))
    coverage = dict(
        cells_with_at_least3=populated,
        quality_supported_cells=sum((region["quality_supported"] for region in regions)),
        reference_span_fraction=span.tolist(),
        sufficient=bool(
            len(error) >= config.validation_min_points
            and populated >= config.coverage_min_cells
            and np.all(span >= config.coverage_min_span_fraction)
        ),
    )
    return dict(count=len(error), **stats(error), coverage=coverage, regions=regions)


def _region_bad(region, config):
    return bool(
        region["quality_supported"]
        and (
            region["median_px"] is None
            or region["median_px"] > config.trigger_region_median_gt_px
            or region["fraction_within3px"] < config.trigger_region_fraction_within3_lt
        )
    )


def _global_triggers(stats, config):
    reasons = []
    if stats["median_px"] is not None and stats["median_px"] > config.trigger_global_median_gt_px:
        reasons.append("global_median")
    if stats["p95_px"] is not None and stats["p95_px"] > config.trigger_global_p95_gt_px:
        reasons.append("global_tail")
    return reasons


def _acceptance(stats, config):
    return dict(
        coverage=stats["coverage"]["sufficient"],
        median=stats["median_px"] is not None
        and stats["median_px"] <= config.candidate_median_max_px,
        tail=stats["p95_px"] is not None and stats["p95_px"] <= config.candidate_p95_max_px,
        fraction_within3=stats["fraction_within3px"] is not None
        and stats["fraction_within3px"] >= config.candidate_fraction_within3_min,
        local_regions=all(
            (
                region["median_px"] is not None
                and region["median_px"] <= config.candidate_region_median_max_px
                and (region["fraction_within3px"] >= config.candidate_region_fraction_within3_min)
                for region in stats["regions"]
                if region["quality_supported"]
            )
        ),
    )


def evaluate_correspondences(
    current,
    reference,
    base,
    frame,
    config=AutomaticStabilizationConfig(),
    *,
    width=3840,
    height=2160,
):
    base = np.asarray(base, float)
    if base.shape != (3, 3) or width <= 0 or height <= 0:
        raise ValueError("Expected a 3x3 matrix and positive native image dimensions")
    record = dict(
        frame=int(frame),
        frame_index=int(frame) - 1,
        input_base_matrix=_json_value(base),
        output_matrix=_json_value(base),
        candidate_matrix=None,
        candidate_attempts=0,
        matrix_changed=False,
        quality_passed=False,
        status="UNRESOLVED",
        reason=None,
    )
    repaired = base.copy()
    try:
        current, reference = (np.asarray(current, float), np.asarray(reference, float))
        cu, re, members = group_correspondences(current, reference, config)
        parts, seed = split_group_indices(len(cu), frame, config)
        record.update(
            raw_correspondences=len(current),
            independent_groups=len(cu),
            seed=seed,
            split_count={name: len(indices) for name, indices in parts.items()},
            group_members=members,
            split_indices={name: indices.tolist() for name, indices in parts.items()},
        )
        before = _quality(
            cu[parts["validation"]], re[parts["validation"]], base, width, height, config
        )
        before_audit = _quality(cu[parts["audit"]], re[parts["audit"]], base, width, height, config)
        validation_regions = {region["cell"]: region for region in before["regions"]}
        audit_regions = {region["cell"]: region for region in before_audit["regions"]}
        confirmed_local = sorted(
            (
                cell
                for cell in validation_regions.keys() & audit_regions.keys()
                if _region_bad(validation_regions[cell], config)
                and _region_bad(audit_regions[cell], config)
            )
        )
        reasons = _global_triggers(before, config)
        if confirmed_local:
            reasons.append("local_region")
        if not _valid_matrix(base):
            reasons.append("invalid_base_matrix")
        record.update(
            base_validation=before,
            base_audit=before_audit,
            trigger_reasons=reasons,
            confirmed_local_cells=confirmed_local,
        )
        if not before["coverage"]["sufficient"] or not before_audit["coverage"]["sufficient"]:
            record.update(
                status="UNRESOLVED_INSUFFICIENT_SUPPORT",
                reason="validation_or_audit_coverage_insufficient",
            )
            return (repaired, _json_value(record))
        if not reasons:
            audit_disagreement = bool(
                _global_triggers(before_audit, config)
                or any(
                    (
                        _region_bad(region, config)
                        for region in before["regions"] + before_audit["regions"]
                    )
                )
            )
            record.update(
                status="UNRESOLVED_BASE_AUDIT_DISAGREEMENT"
                if audit_disagreement
                else "RETAINED_GOOD_BASE",
                quality_passed=not audit_disagreement,
                reason="held_out_splits_have_unconfirmed_base_error"
                if audit_disagreement
                else "base_has_no_confirmed_repair_trigger",
            )
            return (repaired, _json_value(record))
        train = parts["train"]
        if len(train) < config.train_inliers_min_points:
            record.update(status="UNRESOLVED_TRAINING_SUPPORT", reason="too_few_training_groups")
            return (repaired, _json_value(record))
        record["candidate_attempts"] = 1
        cv2.setRNGSeed(seed)
        candidate, mask = cv2.findHomography(
            cu[train],
            re[train],
            cv2.RANSAC,
            config.fit_ransac_threshold_px,
            maxIters=config.fit_max_iterations,
            confidence=config.fit_confidence,
        )
        if candidate is None or mask is None or (not _valid_matrix(candidate)):
            record.update(status="UNRESOLVED_FIT_FAILED", reason="homography_fit_failed_or_invalid")
            return (repaired, _json_value(record))
        candidate = candidate / candidate[2, 2]
        inliers = mask.ravel().astype(bool)
        inlier_cells = len(np.unique(_cells(re[train][inliers], width, height, config)))
        structural = bool(
            inliers.sum() >= config.train_inliers_min_points
            and inlier_cells >= config.coverage_min_cells
        )
        after = _quality(
            cu[parts["validation"]], re[parts["validation"]], candidate, width, height, config
        )
        after_audit = _quality(
            cu[parts["audit"]], re[parts["audit"]], candidate, width, height, config
        )
        improvement = {}
        if "global_median" in reasons:
            improvement["median_halved"] = (
                after["median_px"] is not None
                and after["median_px"] <= config.improvement_factor_max * before["median_px"]
            )
        if "global_tail" in reasons:
            improvement["tail_halved"] = (
                after["p95_px"] is not None
                and after["p95_px"] <= config.improvement_factor_max * before["p95_px"]
            )
        if confirmed_local:
            after_regions = {region["cell"]: region for region in after["regions"]}
            improvement["local_improvement"] = all(
                (
                    after_regions[cell]["median_px"] is not None
                    and (
                        after_regions[cell]["median_px"]
                        <= config.improvement_factor_max * validation_regions[cell]["median_px"]
                        or after_regions[cell]["fraction_within3px"]
                        >= config.candidate_fraction_within3_min
                    )
                    for cell in confirmed_local
                    if validation_regions[cell]["median_px"] is not None
                )
            )
        if "invalid_base_matrix" in reasons:
            improvement["invalid_matrix_replaced"] = _valid_matrix(candidate)
        validation_gates, audit_gates = (
            _acceptance(after, config),
            _acceptance(after_audit, config),
        )
        validation_gates["training_structure"] = structural
        passed = bool(
            all(validation_gates.values())
            and all(audit_gates.values())
            and all(improvement.values())
        )
        record.update(
            candidate_matrix=candidate.tolist(),
            training_inliers=int(inliers.sum()),
            training_inlier_cells=inlier_cells,
            training_structure_passed=structural,
            candidate_validation=after,
            candidate_audit=after_audit,
            validation_gates=validation_gates,
            audit_gates=audit_gates,
            improvement_gates=improvement,
            validation_accepted=bool(all(validation_gates.values()) and all(improvement.values())),
            independent_audit_passed=bool(all(audit_gates.values())),
        )
        if passed:
            repaired = candidate
            record.update(
                status="AUTOMATIC_REPAIR_ACCEPTED",
                reason="one_training_fit_passed_validation_and_independent_audit",
                quality_passed=True,
                matrix_changed=not np.array_equal(base, repaired),
                output_matrix=repaired.tolist(),
            )
        else:
            status = (
                "UNRESOLVED_AUDIT_REJECTED"
                if record["validation_accepted"]
                else "UNRESOLVED_VALIDATION_REJECTED"
            )
            record.update(
                status=status, reason="candidate_failed_structure_quality_or_improvement_gates"
            )
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
        record.update(
            status="UNRESOLVED_EVALUATION_ERROR", reason=str(error), error_type=type(error).__name__
        )
    return (repaired, _json_value(record))


def apply_automatic_matrix_repairs(
    video,
    observations,
    base,
    reference_index,
    diagnostics,
    config=AutomaticStabilizationConfig(),
    progress_callback=None,
):
    started = perf_counter()
    base = np.asarray(base, float)
    observations = np.asarray(observations, float)
    if base.ndim != 3 or base.shape[1:] != (3, 3) or (not 0 <= reference_index < len(base)):
        raise ValueError("Matrices/reference must cover the source video sequence")
    if not _valid_matrix(base[reference_index]) or not np.allclose(
        base[reference_index], np.eye(3), rtol=0, atol=1e-08
    ):
        raise ValueError(
            "Automatic stabilization requires the original identity SIFT-reference basis"
        )
    if (
        observations.ndim != 2
        or observations.shape[1] != 8
        or (not np.isfinite(observations).all())
    ):
        raise ValueError("Expected finite normalized Nx8 detector/tracker observations")
    if len(observations) and (
        np.any(observations[:, 0] != np.floor(observations[:, 0]))
        or observations[:, 0].min() < 1
        or observations[:, 0].max() > len(base)
    ):
        raise ValueError("Observation frames do not address this source video")
    specs = select_audit_frames(
        diagnostics, config, expected_frame_count=len(base), reference_index=reference_index
    )
    repaired, records = (base.copy(), [])
    frames = observations[:, 0]
    if np.any(np.diff(frames) < 0):
        observations = observations[np.argsort(frames, kind="stable")]
        frames = observations[:, 0]

    def rows(index):
        return observations[
            np.searchsorted(frames, index + 1, "left") : np.searchsorted(frames, index + 1, "right")
        ]

    capture = None
    reference_features = None
    reference_error = None
    if specs:
        try:
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise ValueError(f"Cannot open source video: {video}")
            reference_image = video_matrix_repairs._read_image(capture, reference_index)
            reference_features = video_matrix_repairs._masked_features(
                reference_image, rows(reference_index), cv2.SIFT_create(nfeatures=config.nfeatures)
            )
            if reference_features[1] is None:
                raise ValueError("No masked SIFT descriptors in reference frame")
        except (ValueError, RuntimeError, cv2.error) as error:
            reference_error = dict(reason=str(error), error_type=type(error).__name__)
    try:
        for processed, spec in enumerate(specs, 1):
            index = spec["frame_index"]
            frame_started = perf_counter()
            if reference_error:
                record = dict(
                    status="UNRESOLVED_REFERENCE_EXTRACTION",
                    **reference_error,
                    quality_passed=False,
                    matrix_changed=False,
                    candidate_attempts=0,
                    input_base_matrix=_json_value(base[index]),
                    output_matrix=_json_value(base[index]),
                    candidate_matrix=None,
                )
            else:
                try:
                    image = video_matrix_repairs._read_image(capture, index)
                    current_features = video_matrix_repairs._masked_features(
                        image, rows(index), cv2.SIFT_create(nfeatures=config.nfeatures)
                    )
                    current, reference, _ = video_matrix_repairs._correspondences(
                        reference_features, current_features, diagnostic_fit=False
                    )
                    repaired[index], record = evaluate_correspondences(
                        current,
                        reference,
                        base[index],
                        spec["frame"],
                        config,
                        width=image.shape[1],
                        height=image.shape[0],
                    )
                    record["feature_counts"] = dict(
                        reference=len(reference_features[0]), current=len(current_features[0])
                    )
                except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
                    record = dict(
                        status="UNRESOLVED_EXTRACTION_ERROR",
                        reason=str(error),
                        error_type=type(error).__name__,
                        quality_passed=False,
                        matrix_changed=False,
                        candidate_attempts=0,
                        input_base_matrix=_json_value(base[index]),
                        output_matrix=_json_value(base[index]),
                        candidate_matrix=None,
                    )
            record.update(spec, elapsed_seconds=perf_counter() - frame_started)
            records.append(_json_value(record))
            if progress_callback:
                progress_callback(processed, len(specs))
    finally:
        if capture is not None:
            capture.release()
    counts = Counter((record["status"] for record in records))
    unresolved = [
        record["frame"] for record in records if record["status"].startswith("UNRESOLVED")
    ]
    changed = [record["frame"] for record in records if record["matrix_changed"]]
    summary = dict(
        method=AUTO_REPAIR_METHOD,
        version=AUTO_REPAIR_VERSION,
        complete=True,
        total_source_frames=len(base),
        screened_frames=len(specs),
        processed_frames=len(records),
        repaired_frames=counts.get("AUTOMATIC_REPAIR_ACCEPTED", 0),
        retained_good_frames=counts.get("RETAINED_GOOD_BASE", 0),
        unresolved_frames=len(unresolved),
        changed_frames=changed,
        unresolved_frame_numbers=unresolved,
        status_counts=dict(counts),
        screened_quality_passed=not unresolved,
        full_road_accuracy_certified=False,
        reference_frame_index=int(reference_index),
        reference_identity_preserved=bool(
            np.array_equal(base[reference_index], repaired[reference_index])
        ),
        elapsed_seconds=perf_counter() - started,
    )
    audit = dict(
        policy=policy_metadata(config),
        screen_is_quality_certification=False,
        summary=summary,
        frames=records,
        input_base_matrices_sha256=hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest(),
        output_matrices_sha256=hashlib.sha256(np.ascontiguousarray(repaired).tobytes()).hexdigest(),
        limitations=[
            "Background alignment is not independently surveyed road/world-coordinate accuracy.",
            "Unscreened frames are retained, not held-out quality certified.",
            "Current detector masks can miss moving objects; parallax/false matches can remain.",
        ],
    )
    return (repaired, _json_value(audit), _json_value(summary))
