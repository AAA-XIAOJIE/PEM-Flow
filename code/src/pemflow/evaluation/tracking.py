import math
from collections import Counter, defaultdict


def maximum_assignment(matrix):
    if not matrix or not matrix[0]:
        return []
    n, m = (len(matrix), len(matrix[0]))
    if any((len(row) != m for row in matrix)):
        raise ValueError("Ragged assignment matrix")
    if n > m:
        return [
            (j, i) for i, j in maximum_assignment([list(row) for row in zip(*matrix, strict=False)])
        ]
    u, v = ([0.0] * (n + 1), [0.0] * (m + 1))
    matched, previous = ([0] * (m + 1), [0] * (m + 1))
    for row in range(1, n + 1):
        matched[0] = row
        column = 0
        minimum, visited = ([math.inf] * (m + 1), [False] * (m + 1))
        while True:
            visited[column] = True
            current = matched[column]
            delta, next_column = (math.inf, 0)
            for j in range(1, m + 1):
                if visited[j]:
                    continue
                reduced = -matrix[current - 1][j - 1] - u[current] - v[j]
                if reduced < minimum[j]:
                    minimum[j], previous[j] = (reduced, column)
                if minimum[j] < delta:
                    delta, next_column = (minimum[j], j)
            for j in range(m + 1):
                if visited[j]:
                    u[matched[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while column:
            parent = previous[column]
            matched[column] = matched[parent]
            column = parent
    return [(matched[j] - 1, j - 1) for j in range(1, m + 1) if matched[j]]


def box_iou(a, b):
    intersection = max(0.0, min(a.xbr, b.xbr) - max(a.xtl, b.xtl)) * max(
        0.0, min(a.ybr, b.ybr) - max(a.ytl, b.ytl)
    )
    union = (a.xbr - a.xtl) * (a.ybr - a.ytl) + (b.xbr - b.xtl) * (b.ybr - b.ytl) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_tracking(frames, *, threshold=0.5, class_name=None, events=None):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    totals = Counter(gt=0, pred=0, tp=0, fp=0, fn=0, ignored=0, idsw=0, frag=0)
    pair_counts, gt_classes = (Counter(), {})
    last_match, previous_frame, visible_was_matched = ({}, {}, {})
    ever_matched, seen_frames = (set(), set())
    matched_iou_sum = 0.0
    for frame in sorted(frames, key=lambda f: (f.task, f.frame_number)):
        frame_key = (frame.task, frame.frame_number)
        if frame_key in seen_frames:
            raise ValueError(f"Duplicate evaluation frame: {frame_key}")
        seen_frames.add(frame_key)
        for kind, items in (("GT", frame.valid_gt), ("prediction", frame.tracking_predictions)):
            seen = set()
            for b in items:
                if (
                    isinstance(b.identity, bool)
                    or not isinstance(b.identity, int)
                    or b.identity <= 0
                ):
                    raise ValueError(f"{frame_key} {kind}: missing/non-integer positive identity")
                if b.identity in seen:
                    raise ValueError(f"{frame_key}: duplicate {kind} ID {b.identity} in same frame")
                seen.add(b.identity)
                if (
                    not all((math.isfinite(x) for x in (b.xtl, b.ytl, b.xbr, b.ybr)))
                    or b.xbr <= b.xtl
                    or b.ybr <= b.ytl
                ):
                    raise ValueError("Invalid evaluation box")
                if kind == "GT":
                    key = (frame.task, b.identity)
                    if key in gt_classes and gt_classes[key] != b.label:
                        raise ValueError(f"GT identity changes class: {key}")
                    gt_classes[key] = b.label
        gt = sorted(
            (b for b in frame.valid_gt if class_name is None or b.label == class_name),
            key=lambda b: b.identity,
        )
        pred = sorted(
            (b for b in frame.tracking_predictions if class_name is None or b.label == class_name),
            key=lambda b: b.identity,
        )
        similarity = [[box_iou(g, p) if g.label == p.label else 0.0 for p in pred] for g in gt]
        prev = previous_frame.get(frame.task, {})
        score = [
            [
                (1000 if prev.get(g.identity) == p.identity else 0) + similarity[i][j]
                if similarity[i][j] >= threshold
                else 0.0
                for j, p in enumerate(pred)
            ]
            for i, g in enumerate(gt)
        ]
        matches = [(i, j) for i, j in maximum_assignment(score) if score[i][j] > 0]
        used_p, used_g = ({j for _, j in matches}, {i for i, _ in matches})
        excluded = {
            j
            for j, p in enumerate(pred)
            if j not in used_p
            and any((box_iou(p, ignored) >= threshold for ignored in frame.ignored_gt))
        }
        totals.update(
            gt=len(gt),
            pred=len(pred) - len(excluded),
            tp=len(matches),
            fp=len(pred) - len(used_p) - len(excluded),
            fn=len(gt) - len(used_g),
            ignored=len(excluded),
        )

        def record_event(kind, g=None, p=None, previous_id=None, *, event_frame=frame):
            if events is not None:
                events.append(
                    dict(
                        task=event_frame.task,
                        sample_index=event_frame.frame_number,
                        image=getattr(event_frame, "image_name", ""),
                        event=kind,
                        gt_id=g.identity if g is not None else None,
                        pred_id=p.identity if p is not None else None,
                        previous_pred_id=previous_id,
                        class_=g.label if g is not None else p.label,
                    )
                )

        for i, g in enumerate(gt):
            if i not in used_g:
                record_event("FN", g=g)
        for j, p in enumerate(pred):
            if j not in used_p and j not in excluded:
                record_event("FP", p=p)
        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                if j not in excluded and similarity[i][j] >= threshold:
                    pair_counts[frame.task, g.identity, p.identity] += 1
        by_gt = {i: j for i, j in matches}
        current = {}
        for i, g in enumerate(gt):
            key = (frame.task, g.identity)
            found = i in by_gt
            if found:
                p = pred[by_gt[i]]
                if key in last_match and last_match[key] != p.identity:
                    totals["idsw"] += 1
                    record_event("IDSW", g=g, p=p, previous_id=last_match[key])
                if key in ever_matched and (not visible_was_matched.get(key, False)):
                    totals["frag"] += 1
                    record_event("FRAG", g=g, p=p)
                last_match[key] = current[g.identity] = p.identity
                ever_matched.add(key)
                matched_iou_sum += similarity[i][by_gt[i]]
            visible_was_matched[key] = found
        previous_frame[frame.task] = current
    by_task = defaultdict(dict)
    for (task, g, p), count in pair_counts.items():
        by_task[task][g, p] = count
    idtp = 0
    for pairs in by_task.values():
        gt_ids, pred_ids = (sorted({g for g, _ in pairs}), sorted({p for _, p in pairs}))
        matrix = [[pairs.get((g, p), 0) for p in pred_ids] for g in gt_ids]
        idtp += sum((matrix[i][j] for i, j in maximum_assignment(matrix)))
    return dict(
        class_=class_name or "ALL",
        gt_detections=totals["gt"],
        evaluated_predictions=totals["pred"],
        matches=totals["tp"],
        false_positives=totals["fp"],
        false_negatives=totals["fn"],
        ignored_predictions=totals["ignored"],
        id_switches=totals["idsw"],
        fragmentations=totals["frag"],
        mota=1 - (totals["fn"] + totals["fp"] + totals["idsw"]) / totals["gt"]
        if totals["gt"]
        else math.nan,
        motp_iou=matched_iou_sum / totals["tp"] if totals["tp"] else math.nan,
        idtp=idtp,
        idfp=totals["pred"] - idtp,
        idfn=totals["gt"] - idtp,
        idf1=2 * idtp / (totals["gt"] + totals["pred"])
        if totals["gt"] + totals["pred"]
        else math.nan,
    )
