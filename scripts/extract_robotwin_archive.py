#!/usr/bin/env python3
"""Safely extract a RoboTwin ZIP while dropping macOS resource forks.

Use only after verifying the archive SHA against DynamicWAM's pinned manifest.
The command refuses an existing destination and emits a provenance JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from zipfile import ZipFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archive, destination = args.archive.resolve(), args.destination.resolve()
    actual_sha = sha256(archive)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"archive SHA mismatch: {actual_sha}")
    if destination.exists():
        raise FileExistsError(destination)
    kept = skipped = size = 0
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}-") as temporary:
        root = Path(temporary)
        with ZipFile(archive) as source:
            for member in source.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename or "\0" in member.filename:
                    raise ValueError(f"unsafe archive member: {member.filename!r}")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"archive symlink forbidden: {member.filename!r}")
                if "__MACOSX" in relative.parts:
                    skipped += int(not member.is_dir())
                    continue
                target = root.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                kept += 1
                size += member.file_size
        candidate = root / destination.name
        if not candidate.is_dir():
            raise RuntimeError(f"archive lacks expected top-level {destination.name}")
        candidate.replace(destination)
    report = {
        "archive": str(archive),
        "archive_sha256": actual_sha,
        "destination": str(destination),
        "kept_files": kept,
        "kept_size_bytes": size,
        "skipped_macos_files": skipped,
        "reason": "DynamicWAM manifest counts a cleaned tree but pre-extraction validator sees macOS resource forks",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

