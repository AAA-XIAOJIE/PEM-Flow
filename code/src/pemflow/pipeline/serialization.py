from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

LABEL_COLUMNS = [
    "frame",
    "source_track_id",
    "raw_class",
    "cx_norm",
    "cy_norm",
    "width_norm",
    "height_norm",
    "confidence",
]


def observation_digest(values, *, chunk_rows=65536):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(LABEL_COLUMNS) or chunk_rows < 1:
        raise ValueError("Expected an N by 8 observation array and a positive chunk size")
    digest = hashlib.sha256()
    header = dict(columns=LABEL_COLUMNS, shape=list(values.shape), dtype="<f8", order="C")
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(b"\n")
    for start in range(0, len(values), chunk_rows):
        part = np.ascontiguousarray(values[start : start + chunk_rows], dtype="<f8")
        digest.update(memoryview(part).cast("B"))
    return digest.hexdigest().upper()


def serialize_result(result, frame):
    if frame < 1:
        raise ValueError("Source video frames are one-based")
    if result.boxes is None or not result.boxes.is_track:
        return np.empty((0, len(LABEL_COLUMNS)), dtype=np.float64)
    boxes = result.boxes
    coordinates = boxes.xywhn.detach().cpu().tolist()
    raw = boxes.data.detach().cpu().tolist()
    records, identities = ([], set())
    for coordinate, row in zip(coordinates, raw, strict=True):
        _, _, _, _, identity, confidence, cls = row
        if not math.isfinite(identity) or identity != int(identity) or identity <= 0:
            raise ValueError("Invalid tracker identity")
        if int(identity) in identities:
            raise ValueError("Duplicate tracker identity within one frame")
        identities.add(int(identity))
        if not math.isfinite(cls) or cls != int(cls) or cls < 0:
            raise ValueError("Invalid detector class")
        if len(coordinate) != 4 or not all(
            (math.isfinite(v) and 0 <= v <= 1 for v in [*coordinate, confidence])
        ):
            raise ValueError("Invalid normalized detection output")
        if coordinate[2] <= 0 or coordinate[3] <= 0:
            raise ValueError("Detection boxes must have positive area")
        old_format = [int(cls), *coordinate, confidence, int(identity)]
        rounded = [float(f"{number:g}") for number in old_format]
        records.append([float(frame), rounded[-1], rounded[0], *rounded[1:6]])
    values = np.asarray(records, dtype=np.float64).reshape(-1, len(LABEL_COLUMNS))
    if len(values) and len(np.unique(values[:, 1])) != len(values):
        raise ValueError("Serialized tracker identities collide")
    return values


class _ChunkWriter:
    def __init__(self, directory, chunk_rows):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.buffer = np.empty((chunk_rows, len(LABEL_COLUMNS)), dtype=np.float64)
        self.used = 0
        self.rows = 0
        self.paths = []

    def append(self, values):
        start = 0
        while start < len(values):
            count = min(len(values) - start, len(self.buffer) - self.used)
            self.buffer[self.used : self.used + count] = values[start : start + count]
            self.used += count
            self.rows += count
            start += count
            if self.used == len(self.buffer):
                self.flush()

    def flush(self):
        if self.used:
            path = self.directory / f"chunk_{len(self.paths):06d}.npy"
            with path.open("xb") as handle:
                np.save(handle, self.buffer[: self.used], allow_pickle=False)
            self.paths.append(path)
            self.used = 0

    def finalize(self, destination):
        self.flush()
        destination = Path(destination)
        mapped_path = self.directory / "observations.npy"
        if self.rows:
            values = np.lib.format.open_memmap(
                mapped_path, mode="w+", dtype=np.float64, shape=(self.rows, len(LABEL_COLUMNS))
            )
            start = 0
            for path in self.paths:
                part = np.load(path, allow_pickle=False)
                values[start : start + len(part)] = part
                start += len(part)
            values.flush()
        else:
            values = np.empty((0, len(LABEL_COLUMNS)), dtype=np.float64)
        try:
            array_digest = observation_digest(values)
            temporary = destination.with_name(destination.name + ".tmp")
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, observations=values)
            temporary.replace(destination)
        finally:
            if isinstance(values, np.memmap):
                values._mmap.close()
        for path in [*self.paths, mapped_path]:
            if path.is_file():
                if path.resolve().parent != self.directory.resolve():
                    raise ValueError("Temporary output escaped its directory")
                path.unlink()
        self.directory.rmdir()
        return array_digest
