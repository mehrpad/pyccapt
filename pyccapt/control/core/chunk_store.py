"""Atomic chunk persistence, manifests, validation, and quarantine."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MANIFEST_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_manifest_record(chunk_dir: Path, stream_name: str, record: Mapping[str, Any]) -> None:
    manifest = chunk_dir / f"manifest.{stream_name}.jsonl"
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(manifest, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o666)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_chunk_group(
    chunk_dir: str | Path,
    *,
    stream_name: str,
    chunk_id: int,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish an aligned group, then append its complete record."""
    directory = Path(chunk_dir).resolve()
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    normalized = {name: np.asarray(values) for name, values in arrays.items()}
    lengths = {int(values.shape[0]) if values.ndim else 1 for values in normalized.values()}
    if len(lengths) > 1:
        raise ValueError(f"Chunk {stream_name}:{chunk_id} fields have mismatched rows: {sorted(lengths)}")

    fields: dict[str, dict[str, Any]] = {}
    published: list[Path] = []
    try:
        for stem, values in normalized.items():
            target = directory / f"{stem}_chunk_{int(chunk_id)}.npy"
            temporary = directory / f".{target.name}.{os.getpid()}.tmp"
            with temporary.open("wb") as stream:
                np.save(stream, values)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            published.append(target)
            fields[stem] = {
                "file": target.name,
                "rows": int(values.shape[0]) if values.ndim else 1,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sha256": _sha256(target),
            }
    except Exception:
        for path in directory.glob(f".*_chunk_{int(chunk_id)}.npy.*.tmp"):
            path.unlink(missing_ok=True)
        raise

    record = {
        "manifest_version": MANIFEST_VERSION,
        "stream": stream_name,
        "chunk_id": int(chunk_id),
        "status": "complete",
        "rows": next(iter(lengths), 0),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "write_latency_ms": (time.perf_counter() - started) * 1000.0,
        "fields": fields,
    }
    _append_manifest_record(directory, stream_name, record)
    return record


def read_manifest_records(chunk_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(chunk_dir).resolve()
    records: list[dict[str, Any]] = []
    for manifest in sorted(directory.glob("manifest*.jsonl")):
        with manifest.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    records.append(
                        {
                            "status": "invalid",
                            "manifest": manifest.name,
                            "line": line_number,
                            "error": f"invalid JSON: {exc}",
                            "fields": {},
                        }
                    )
                    continue
                # Upgrade the pre-schema manifest emitted by PyCCAPT 0.2.x in
                # memory. Those records were appended only after atomic .npy
                # publication, but carried ``length`` rather than file/shape/
                # checksum fields. Accepting them keeps interrupted older runs
                # recoverable after installing this upgrade.
                fields = record.get("fields", {})
                if (
                    record.get("manifest_version") is None
                    and record.get("chunk_id") is not None
                    and fields
                    and all("length" in metadata for metadata in fields.values())
                ):
                    chunk_id = int(record["chunk_id"])
                    upgraded_fields = {}
                    for stem, metadata in fields.items():
                        length = int(metadata["length"])
                        upgraded_fields[stem] = {
                            "file": f"{stem}_chunk_{chunk_id}.npy",
                            "rows": length,
                            "shape": [length],
                            "dtype": str(metadata.get("dtype", "")),
                        }
                    record = {
                        **record,
                        "manifest_version": 0,
                        "stream": "legacy",
                        "status": "complete",
                        "rows": next(iter(item["rows"] for item in upgraded_fields.values()), 0),
                        "fields": upgraded_fields,
                    }
                records.append(record)
    return records


def latest_manifest_record(chunk_dir: str | Path) -> dict[str, Any] | None:
    """Read only the final non-empty record from the newest manifest file."""
    manifests = sorted(Path(chunk_dir).resolve().glob("manifest*.jsonl"), key=lambda path: path.stat().st_mtime_ns)
    if not manifests:
        return None
    with manifests[-1].open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size == 0:
            return None
        read_size = min(size, 64 * 1024)
        stream.seek(-read_size, os.SEEK_END)
        lines = stream.read(read_size).decode("utf-8", errors="replace").splitlines()
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def quarantine_record(chunk_dir: str | Path, record: Mapping[str, Any], reason: str) -> Path:
    """Move inconsistent files into a preserved quarantine directory."""
    directory = Path(chunk_dir).resolve()
    label = f"{record.get('stream', 'unknown')}-{record.get('chunk_id', 'unknown')}"
    target_dir = directory / "quarantine" / label
    target_dir.mkdir(parents=True, exist_ok=True)
    for metadata in record.get("fields", {}).values():
        filename = metadata.get("file")
        if not filename:
            continue
        source = directory / filename
        if source.is_file():
            shutil.move(str(source), str(target_dir / source.name))
    (target_dir / "reason.json").write_text(
        json.dumps({"reason": reason, "record": record}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target_dir


def validate_manifest_records(chunk_dir: str | Path, *, quarantine: bool = True) -> tuple[list[dict], list[dict]]:
    """Return valid and invalid records, preserving invalid files in quarantine."""
    directory = Path(chunk_dir).resolve()
    valid: list[dict] = []
    invalid: list[dict] = []
    for record in read_manifest_records(directory):
        error = ""
        if record.get("status") != "complete":
            error = str(record.get("error", "record is not complete"))
        for stem, metadata in record.get("fields", {}).items():
            path = directory / str(metadata.get("file", ""))
            if not path.is_file():
                error = f"missing file for {stem}: {path.name}"
                break
            try:
                values = np.load(path, mmap_mode="r")
            except Exception as exc:
                error = f"unreadable file {path.name}: {exc}"
                break
            if str(values.dtype) != str(metadata.get("dtype")):
                error = f"dtype mismatch for {path.name}"
            elif list(values.shape) != list(metadata.get("shape", [])):
                error = f"shape mismatch for {path.name}"
            elif int(record.get("manifest_version", 0)) >= 1 and not metadata.get("sha256"):
                error = f"missing checksum for {path.name}"
            elif metadata.get("sha256") and _sha256(path) != metadata.get("sha256"):
                error = f"checksum mismatch for {path.name}"
            del values
            if error:
                break
        if error:
            bad = dict(record)
            bad["error"] = error
            if quarantine:
                bad["quarantine"] = str(quarantine_record(directory, record, error))
            invalid.append(bad)
        else:
            valid.append(record)
    return valid, invalid


def validated_files_for_stem(chunk_dir: str | Path, stem: str) -> list[Path] | None:
    """Return manifest-validated files, or None for a legacy manifest-less run."""
    directory = Path(chunk_dir).resolve()
    if not any(directory.glob("manifest*.jsonl")):
        return None
    valid, _ = validate_manifest_records(directory, quarantine=True)
    files: list[tuple[int, Path]] = []
    for record in valid:
        metadata = record.get("fields", {}).get(stem)
        if metadata:
            files.append((int(record["chunk_id"]), directory / metadata["file"]))
    return [path for _, path in sorted(files)]
