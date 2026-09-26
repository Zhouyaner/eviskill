#!/usr/bin/env python3
"""Reproduce the exact 96/48/full ScienceWorld split used by the main experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


# SkillNet row indices released with the main experiment split. Keeping these
# explicit avoids changes caused by Python, NumPy, or dataframe samplers.
MAIN_TRAIN_INDICES = [
    1386, 1271, 1400, 105, 334, 1408, 368, 1422, 897, 1286, 1188, 621,
    25, 1035, 142, 1343, 792, 1109, 1283, 1468, 1111, 180, 889, 361,
    1124, 224, 914, 226, 640, 1282, 747, 1099, 729, 1122, 619, 1209,
    276, 450, 328, 292, 110, 1148, 196, 493, 514, 1341, 766, 238,
    1030, 1203, 574, 645, 618, 1351, 1472, 1313, 1365, 326, 331, 899,
    1186, 147, 1375, 1327, 1340, 1121, 79, 847, 958, 1092, 1239, 904,
    1170, 471, 1242, 174, 392, 1290, 760, 1369, 202, 320, 699, 171,
    81, 1345, 923, 1211, 1221, 956, 1015, 436, 353, 622, 1219, 953,
]

MAIN_VAL_INDICES = [
    154, 78, 82, 54, 65, 185, 137, 89, 100, 62, 136, 109, 86, 46, 49,
    193, 150, 119, 13, 67, 157, 190, 171, 91, 64, 47, 156, 167, 51, 33,
    36, 143, 177, 126, 165, 73, 61, 176, 116, 153, 149, 188, 160, 41,
    20, 68, 123, 135,
]

EXPECTED_SOURCE_SHA256 = {
    "train": "79f6065d410bb54643b2499003c4419dbbd85be0e66171503b9466e230c229af",
    "dev": "f6c6c72f09d27cdbac5027a871e04984838c8cc996e7c7cfab060a57511d4c52",
    "test": "7afbb7b5559f4f7e8714197406f297cefb9bfd188017e4b166da259dd6accfa5",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_by_skillnet_index(
    rows: list[dict[str, Any]], indices: list[int], split: str
) -> list[dict[str, Any]]:
    by_index = {int(row["skillnet_index"]): dict(row) for row in rows}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"{split} source is missing SkillNet indices: {missing[:8]}")
    return [by_index[index] for index in indices]


def family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("task_family") or "") for row in rows)
    return dict(sorted(counts.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_paths = {
        "train": source_dir / "train.json",
        "dev": source_dir / "dev.json",
        "test": source_dir / "test.json",
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"ScienceWorld source manifest does not exist: {path}")
        actual_hash = sha256(path)
        expected_hash = EXPECTED_SOURCE_SHA256[name]
        if not args.allow_source_mismatch and actual_hash != expected_hash:
            raise ValueError(
                f"Unexpected {name}.json fingerprint: {actual_hash}. "
                "Use the released SkillNet-derived source manifests or pass "
                "--allow-source-mismatch after verifying row identities."
            )

    train_source = load_json(source_paths["train"])
    dev_source = load_json(source_paths["dev"])
    test = load_json(source_paths["test"])
    if not all(isinstance(rows, list) for rows in (train_source, dev_source, test)):
        raise ValueError("ScienceWorld source manifests must be JSON lists")

    train = select_by_skillnet_index(train_source, MAIN_TRAIN_INDICES, "train")
    val = select_by_skillnet_index(dev_source, MAIN_VAL_INDICES, "val")
    splits = {"train": train, "val": val, "test": test}
    expected_caps = {"train": 4, "val": 2}
    for name, cap in expected_caps.items():
        counts = family_counts(splits[name])
        if len(counts) != 24 or set(counts.values()) != {cap}:
            raise ValueError(f"Unexpected {name} family coverage: {counts}")

    for name, rows in splits.items():
        save_json(output_dir / name / "items.json", rows)

    manifest = {
        "benchmark": "ScienceWorld",
        "manifest_type": "stratified_train96_val48_testfull",
        "source": "SkillNet-derived full manifests",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "strategy": {
            "train": "released four-example cohort for each of 24 task families",
            "val": "released two-example cohort for each of 24 task families",
            "test": "full SkillNet test manifest, unchanged",
        },
        "family_counts": {name: family_counts(rows) for name, rows in splits.items()},
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "manifest_sha256": {
            name: sha256(output_dir / name / "items.json") for name in splits
        },
    }
    save_json(output_dir / "split_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing SkillNet-derived train.json, dev.json, and test.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/scienceworld_main",
        help="Destination directory for train/val/test manifests.",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Allow a source fingerprint mismatch while selecting by skillnet_index.",
    )
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
