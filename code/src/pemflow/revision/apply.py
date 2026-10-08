"""Apply the explicit human-reviewed rules to an automatic dataset."""

from __future__ import annotations

import collections
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pemflow.io import (
    intermediate_paths,
    read_json,
    runtime_metadata,
    sha256_file,
    table_paths,
    write_json,
)
from pemflow.revision.rules import INDEX_COLUMNS, _write_audit, _write_revised_table, revise_segment


def apply_revisions(input_root, rules_path, output):
    input_root, output = Path(input_root), Path(output)
    rules_path = Path(rules_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new revision output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = read_json(input_root / "dataset.json")
    document = read_json(rules_path)
    grouped = collections.defaultdict(list)
    for rule in document["rules"]:
        grouped[rule["segment"]].append(rule)
    keys = {item["key"] for item in dataset["segments"]}
    if set(grouped) - keys:
        raise ValueError("Revision rules refer to unknown source segments")
    write_json(output / "dataset.json", dataset)
    write_json(output / "applied_rules.json", document)
    cv2.setNumThreads(1)
    start = time.monotonic()
    results = []
    for item in dataset["segments"]:
        key = item["key"]
        source = table_paths(input_root, key)
        target = table_paths(output, key)
        for path in target.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        rules = grouped[key]
        if not rules:
            for name, path in source.items():
                shutil.copyfile(path, target[name])
            results.append(dict(key=key, status="UNCHANGED", rule_ids=[]))
            continue
        index = pd.read_csv(source["lineage"], usecols=INDEX_COLUMNS)
        inputs = intermediate_paths(input_root, key)
        with np.load(inputs["observations"], allow_pickle=False) as archive:
            observations = archive["observations"].copy()
        with np.load(inputs["matrices"], allow_pickle=False) as archive:
            matrices = archive["matrices"].copy()
        patches, scopes = revise_segment(index, observations, matrices, item["profile"], rules)
        counts = [
            _write_revised_table(
                source[name],
                target[name],
                "lineage.csv.gz" if name == "lineage" else name + ".csv",
                index,
                patches,
                scopes,
            )
            for name in source
        ]
        if len(set(counts)) != 1:
            raise ValueError(f"{key} trajectory/motion/lineage lengths differ")
        audit = output / "audit" / key
        audit.mkdir(parents=True)
        statistics = _write_audit(audit, key, scopes, rules, document["batch_id"])
        results.append(
            dict(
                key=key,
                automatic_rows=len(index),
                revised_rows=counts[0],
                rule_ids=[rule["rule_id"] for rule in rules],
                **statistics,
            )
        )
        print(f"{key} revised: {len(index)} -> {counts[0]} rows", flush=True)
        write_json(
            output / "progress.json", dict(completed=len(results), total=len(dataset["segments"]))
        )
    manifest = dict(
        status="COMPLETE",
        batch_id=document["batch_id"],
        rules_sha256=sha256_file(rules_path),
        input_dataset=str(input_root.resolve()),
        segments=results,
        runtime=runtime_metadata(),
        elapsed_seconds=time.monotonic() - start,
        method="Explicit human source IDs, frame ranges and boxes; affected identities use the production projection and motion functions; other field tokens are retained.",
    )
    write_json(output / "revision.json", manifest)
    return manifest
