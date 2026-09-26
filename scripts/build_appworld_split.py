#!/usr/bin/env python3
"""Build the official AppWorld train/dev/test-normal manifests used by EviSkill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmsm.appworld_manifests import build_official_appworld_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--appworld-data-dir",
        required=True,
        help="AppWorld data directory containing tasks/ and datasets/.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/appworld_official",
        help="Destination directory for train/dev/test_normal manifests.",
    )
    args = parser.parse_args()
    summary = build_official_appworld_manifests(
        data_dir=args.appworld_data_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
