#!/usr/bin/env python3
"""Build the 96/48/full ALFWorld split used by the main experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


TASK_TYPES = {
    "pick_and_place_simple": "pick_and_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_clean_then_place_in_recep": "clean",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(data_root: Path, source_split: str) -> list[dict[str, str]]:
    split_root = data_root / "json_2.1.1" / source_split
    if not split_root.is_dir():
        raise FileNotFoundError(f"ALFWorld split does not exist: {split_root}")
    rows = []
    for gamefile in sorted(split_root.rglob("game.tw-pddl")):
        traj_path = gamefile.with_name("traj_data.json")
        if not traj_path.is_file():
            continue
        task_type = str((load_json(traj_path) or {}).get("task_type") or "")
        if task_type in TASK_TYPES:
            rows.append(
                {
                    "gamefile": gamefile.relative_to(data_root).as_posix(),
                    "task_type": task_type,
                    "task_family": TASK_TYPES[task_type],
                }
            )
    return rows


def sample(rows: list[dict[str, str]], per_type: int, seed: int, split: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)
    rng = random.Random(f"alfworld-stratified:{seed}:{split}")
    selected = []
    for task_type in TASK_TYPES:
        by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sorted(grouped[task_type], key=lambda item: item["gamefile"]):
            parts = Path(row["gamefile"]).parts
            scenario = parts[2] if len(parts) > 2 else row["gamefile"]
            by_scenario[scenario].append(row)
        if len(by_scenario) < per_type:
            raise ValueError(f"Not enough {task_type} scenarios for {split}: {len(by_scenario)}")
        for scenario in rng.sample(sorted(by_scenario), per_type):
            selected.append(rng.choice(by_scenario[scenario]))
    rng.shuffle(selected)
    return [
        {"id": f"{split}:{index:04d}", **row}
        for index, row in enumerate(selected)
    ]


def canonical_test(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        gamefile = str(row.get("gamefile") or "")
        task_type = str(row.get("task_type") or "")
        if not gamefile or task_type not in TASK_TYPES:
            raise ValueError(f"Invalid ALFWorld test row {index}: {row!r}")
        output.append(
            {
                "id": f"test:{index:04d}",
                "gamefile": gamefile,
                "task_type": task_type,
                "task_family": TASK_TYPES[task_type],
            }
        )
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train = sample(collect(data_root, "train"), args.train_per_task_type, args.seed, "train")
    validation = sample(collect(data_root, "valid_seen"), args.val_per_task_type, args.seed, "val")
    test = canonical_test(collect(data_root, "valid_unseen"))
    split_rows = {"train": train, "val": validation, "test": test}
    ids = {name: {row["gamefile"] for row in rows} for name, rows in split_rows.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ValueError(f"ALFWorld split overlap {left}/{right}: {sorted(overlap)[:3]}")
    for name, rows in split_rows.items():
        save_json(output_dir / name / "items.json", rows)
    manifest = {
        "benchmark": "ALFWorld",
        "manifest_type": "stratified_train96_val48_testfull",
        "source_root": "$ALFWORLD_DATA/json_2.1.1",
        "seed": args.seed,
        "counts": {name: len(rows) for name, rows in split_rows.items()},
        "train_per_task_type": args.train_per_task_type,
        "val_per_task_type": args.val_per_task_type,
        "manifest_sha256": {
            name: hashlib.sha256((output_dir / name / "items.json").read_bytes()).hexdigest()
            for name in split_rows
        },
    }
    save_json(output_dir / "split_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="ALFWORLD_DATA root containing json_2.1.1/")
    parser.add_argument("--output-dir", default="data/splits/alfworld_main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-task-type", type=int, default=16)
    parser.add_argument("--val-per-task-type", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
