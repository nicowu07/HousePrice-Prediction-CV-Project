"""Download and extract the course Kaggle competition data.

Prerequisites
-------------
1. Join the competition on Kaggle (via the Canvas link) and accept the rules.
   The API returns 403 until the rules are accepted in the browser.
2. Have a Kaggle API token in ~/.kaggle/ (kaggle.json or access_token).

Usage
-----
    python src/download_data.py --slug <competition-slug>

Everything lands in data/raw/. Images are never committed to git.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download(slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] competition={slug} -> {dest}")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", slug, "-p", str(dest)],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        if "403" in result.stderr:
            print(
                "\n403 usually means the competition rules have not been accepted. "
                "Open the competition page in a browser, click 'Join Competition', "
                "then re-run this script.",
                file=sys.stderr,
            )
        sys.exit(1)

    archive = dest / f"{slug}.zip"
    if not archive.exists():
        zips = sorted(dest.glob("*.zip"))
        if not zips:
            print(f"[error] no zip archive found in {dest}", file=sys.stderr)
            sys.exit(1)
        archive = zips[-1]
    return archive


def extract(archive: Path, dest: Path) -> None:
    """Extract the competition archive, including any nested zips it contains."""
    print(f"[extract] {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    # Competition bundles sometimes ship train/test images as inner archives.
    for inner in sorted(dest.rglob("*.zip")):
        if inner == archive:
            continue
        target = inner.with_suffix("")
        if target.exists():
            continue
        print(f"[extract] nested {inner.relative_to(dest)}")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True, help="Kaggle competition slug from the page URL")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    archive = download(args.slug, args.dest)
    extract(archive, args.dest)
    print(f"\n[done] data in {args.dest}\nNext: python src/inspect_data.py")


if __name__ == "__main__":
    main()
