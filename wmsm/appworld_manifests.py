import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


OFFICIAL_APPWORLD_SPLIT_COUNTS = {
    "train": 90,
    "dev": 57,
    "test_normal": 168,
    "test_challenge": 417,
}


def resolve_appworld_data_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "tasks").is_dir() and (path / "datasets").is_dir():
        return path
    if (path / "data" / "tasks").is_dir() and (path / "data" / "datasets").is_dir():
        return path / "data"
    raise FileNotFoundError(
        "AppWorld data directory must contain tasks/ and datasets/, either directly "
        f"or below data/: {path}"
    )


def _task_parts(task_id: str) -> tuple[str, int]:
    try:
        family, raw_variant = str(task_id).rsplit("_", 1)
        variant = int(raw_variant)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid AppWorld task ID: {task_id!r}") from None
    if not family or variant not in {1, 2, 3}:
        raise ValueError(f"Invalid AppWorld task ID: {task_id!r}")
    return family, variant


def load_official_appworld_split(data_dir: str, split: str) -> List[Dict[str, Any]]:
    data_path = resolve_appworld_data_dir(data_dir)
    split_name = str(split or "").strip()
    split_path = data_path / "datasets" / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"AppWorld split file does not exist: {split_path}")
    task_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"AppWorld split contains duplicate task IDs: {split_path}")
    rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        family, variant = _task_parts(task_id)
        task_dir = data_path / "tasks" / task_id
        if not task_dir.is_dir() or not (task_dir / "specs.json").is_file():
            raise FileNotFoundError(f"AppWorld task directory/specs missing: {task_dir}")
        rows.append(
            {
                "dataset": "appworld",
                "task_id": task_id,
                "task_family": family,
                "variant": variant,
                "split": split_name,
            }
        )
    return rows


def inspect_appworld_manifest(
    manifest_path: str,
    *,
    data_dir: str,
    expected_split: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"AppWorld manifest must contain a JSON list: {path}")
    data_path = resolve_appworld_data_dir(data_dir)
    task_ids: List[str] = []
    split_names = set()
    family_to_variants: Dict[str, set[int]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"AppWorld manifest row {index} is not an object: {path}")
        if str(raw.get("dataset") or "").lower() != "appworld":
            raise ValueError(f"AppWorld manifest row {index} has invalid dataset: {raw.get('dataset')!r}")
        task_id = str(raw.get("task_id") or "")
        family, variant = _task_parts(task_id)
        if str(raw.get("task_family") or family) != family or int(raw.get("variant", variant)) != variant:
            raise ValueError(f"AppWorld manifest row {index} has inconsistent family/variant: {raw}")
        split_name = str(raw.get("split") or "")
        if expected_split and split_name != expected_split:
            raise ValueError(
                f"AppWorld manifest {path} must use split={expected_split!r}, got {split_name!r}"
            )
        task_dir = data_path / "tasks" / task_id
        if not task_dir.is_dir() or not (task_dir / "specs.json").is_file():
            raise FileNotFoundError(f"AppWorld task directory/specs missing: {task_dir}")
        task_ids.append(task_id)
        split_names.add(split_name)
        family_to_variants.setdefault(family, set()).add(variant)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"AppWorld manifest contains duplicate task IDs: {path}")
    if expected_split:
        official_task_ids = [
            str(row["task_id"])
            for row in load_official_appworld_split(str(data_path), expected_split)
        ]
        if task_ids != official_task_ids:
            mismatch_index = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(task_ids, official_task_ids)
                    )
                    if actual != expected
                ),
                min(len(task_ids), len(official_task_ids)),
            )
            actual = task_ids[mismatch_index] if mismatch_index < len(task_ids) else "<missing>"
            expected = (
                official_task_ids[mismatch_index]
                if mismatch_index < len(official_task_ids)
                else "<extra>"
            )
            raise ValueError(
                f"AppWorld manifest {path} does not exactly match official {expected_split} "
                f"order at index {mismatch_index}: got {actual!r}, expected {expected!r}"
            )
    incomplete = {
        family: sorted(variants)
        for family, variants in family_to_variants.items()
        if variants != {1, 2, 3}
    }
    if incomplete:
        raise ValueError(
            f"AppWorld manifest must contain all three variants per scenario: {list(incomplete.items())[:5]}"
        )
    ordered = "\n".join(task_ids).encode("utf-8")
    return {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ordered_task_id_fingerprint": hashlib.sha256(ordered).hexdigest(),
        "task_count": len(task_ids),
        "scenario_count": len(family_to_variants),
        "official_order_match": bool(expected_split),
        "task_ids": task_ids,
        "split_names": sorted(split_names),
    }


def preflight_appworld_splits(
    *,
    data_dir: str,
    train_manifest: str,
    validation_manifest: Optional[str],
    test_manifest: Optional[str],
) -> Dict[str, Any]:
    paths = {
        "train": (train_manifest, "train"),
        "validation": (validation_manifest, "dev"),
        "test": (test_manifest, "test_normal"),
    }
    splits = {
        name: inspect_appworld_manifest(
            str(path), data_dir=data_dir, expected_split=expected_split
        )
        for name, (path, expected_split) in paths.items()
        if str(path or "").strip()
    }
    overlaps: Dict[str, List[str]] = {}
    names = list(splits)
    for index, left in enumerate(names):
        left_ids = set(splits[left]["task_ids"])
        for right in names[index + 1 :]:
            overlap = sorted(left_ids.intersection(splits[right]["task_ids"]))
            overlaps[f"{left}:{right}"] = overlap
            if overlap:
                raise ValueError(
                    f"AppWorld task IDs overlap between {left} and {right}: {overlap[:5]}"
                )
            left_families = {task_id.rsplit("_", 1)[0] for task_id in left_ids}
            right_families = {
                task_id.rsplit("_", 1)[0] for task_id in splits[right]["task_ids"]
            }
            family_overlap = sorted(left_families.intersection(right_families))
            if family_overlap:
                raise ValueError(
                    f"AppWorld scenarios overlap between {left} and {right}: {family_overlap[:5]}"
                )
    return {
        "enabled": True,
        "data_dir": str(resolve_appworld_data_dir(data_dir)),
        "splits": splits,
        "overlaps": overlaps,
    }


def build_official_appworld_manifests(data_dir: str, output_dir: str) -> Dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    all_ids: Dict[str, set[str]] = {}
    summary: Dict[str, Any] = {"data_dir": str(resolve_appworld_data_dir(data_dir)), "splits": {}}
    for split, expected_count in OFFICIAL_APPWORLD_SPLIT_COUNTS.items():
        rows = load_official_appworld_split(data_dir, split)
        if len(rows) != expected_count:
            raise ValueError(
                f"Official AppWorld {split} split must contain {expected_count} tasks, got {len(rows)}"
            )
        ids = {str(row["task_id"]) for row in rows}
        families = {str(row["task_family"]) for row in rows}
        for other_split, other_ids in all_ids.items():
            overlap = sorted(ids.intersection(other_ids))
            if overlap:
                raise ValueError(
                    f"Official AppWorld splits overlap between {other_split} and {split}: {overlap[:5]}"
                )
            other_families = {task_id.rsplit("_", 1)[0] for task_id in other_ids}
            family_overlap = sorted(families.intersection(other_families))
            if family_overlap:
                raise ValueError(
                    f"Official AppWorld scenarios overlap between {other_split} and {split}: {family_overlap[:5]}"
                )
        all_ids[split] = ids
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / "items.json"
        manifest_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary["splits"][split] = {
            "manifest": str(manifest_path),
            "tasks": len(rows),
            "scenarios": len(families),
        }
    return summary
