#!/usr/bin/env python3
"""Download the raw data and check it against metadata/checksums.sha256.

Every source is listed in metadata/sources.tsv (URL, provider, license, the provider's checksum). This script
downloads each file into data/raw/, checks it against the provider's MD5 and against our SHA-256 in
metadata/checksums.sha256, and makes it read-only: raw data is read, never edited. A file that is already there
with the right checksum is not downloaded again.

    python scripts/download_data.py           # download and check
    python scripts/download_data.py --check   # only check what is there
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import stat
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "metadata" / "sources.tsv"
CHECKSUMS = ROOT / "metadata" / "checksums.sha256"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_sha256() -> dict[str, str]:
    """`shasum -a 256` format: '<hex>  <path relative to the repository root>'."""
    out = {}
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            hexdigest, path = line.split(maxsplit=1)
            out[path.strip()] = hexdigest
    return out


def verify(path: Path, row: dict, sha256: str | None) -> list[str]:
    problems = []
    algorithm, _, value = row["provider_checksum"].partition(":")
    if algorithm in ("md5", "sha256") and digest(path, algorithm) != value:
        problems.append(f"{algorithm} differs from the provider's ({value})")
    if sha256 is None:
        problems.append("no SHA-256 in metadata/checksums.sha256")
    elif digest(path, "sha256") != sha256:
        problems.append(f"sha256 differs from metadata/checksums.sha256 ({sha256})")
    return problems


def download(url: str, dest: Path, tries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as fh:
                while block := response.read(1 << 20):
                    fh.write(block)
            part.replace(dest)
            return
        except OSError as err:  # network errors: retry with backoff
            if attempt == tries:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"  {err}; retrying in {wait} s", flush=True)
            time.sleep(wait)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only check files that are already downloaded")
    args = parser.parse_args()
    sha = expected_sha256()
    rows = list(csv.DictReader(SOURCES.open(), delimiter="\t"))
    failed = False
    for row in rows:
        path = ROOT / row["file"]
        if path.exists() and not verify(path, row, sha.get(row["file"])):
            print(f"ok          {row['file']}")
            continue
        if args.check:
            print(f"MISSING/BAD {row['file']}")
            failed = True
            continue
        if path.exists():  # wrong content: replace it (it is read-only)
            path.chmod(stat.S_IWUSR | stat.S_IRUSR)
            path.unlink()
        print(f"downloading {row['file']} ({int(row['bytes']) / 1e6:.1f} MB) from {row['url']}", flush=True)
        download(row["url"], path)
        problems = verify(path, row, sha.get(row["file"]))
        if problems:
            print(f"BAD         {row['file']}: " + "; ".join(problems))
            failed = True
            continue
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # raw data is read-only
        print(f"ok          {row['file']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
