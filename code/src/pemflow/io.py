"""Dataset paths and small file helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
import uuid
from pathlib import Path

COLUMNS = ["frame", "id", "class", "real_x", "real_y", "velocity", "dist"]
CODE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = CODE_ROOT.parent
CONFIG_ROOT = CODE_ROOT / "configs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_write_text(path, text):
    """Replace a small status file, including when Windows briefly locks it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    try:
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    def convert(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert) + "\n"
    try:
        atomic_write_text(path, text)
    except PermissionError:
        if Path(path).name != "progress.json":
            raise
        print("Progress file is temporarily locked; computation continues.", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def runtime_metadata():
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "shapely",
        "ultralytics",
        "torch",
        "torchvision",
        "opencv-python",
        "opencv-python-headless",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages)


def csv_filename(key):
    return {"G3-7-1": "G3-7.csv", "G3-7-2": "G3-7_hmi.csv"}.get(key, key + ".csv")


def table_paths(dataset, key):
    dataset = Path(dataset)
    filename = csv_filename(key)
    return dict(
        trajectory=dataset / "trajectory" / filename,
        motion=dataset / "motion" / filename,
        lineage=dataset / "lineage" / (filename + ".gz"),
    )


def intermediate_paths(dataset, key):
    root = Path(dataset) / "intermediates" / key
    return dict(observations=root / "observations.npz", matrices=root / "matrices.npz")
