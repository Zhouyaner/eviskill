import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .alfworld_runner import canonical_alfworld_gamefile, resolve_alfworld_gamefile


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_split(gamefile: str) -> str:
    parts = Path(canonical_alfworld_gamefile(gamefile)).parts
    return parts[1] if len(parts) > 1 and parts[0] == "json_2.1.1" else "unknown"


def _load_path_manifest(path: Path) -> List[Dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"ALFWorld manifest does not exist: {path}") from None
    if not isinstance(rows, list):
        raise ValueError(f"ALFWorld manifest must contain a JSON list: {path}")
    invalid = [index for index, row in enumerate(rows) if not isinstance(row, dict) or not row.get("gamefile")]
    if invalid:
        raise ValueError(
            f"ALFWorld path manifest rows must contain gamefile: {path}; invalid rows={invalid[:5]}"
        )
    return [dict(row) for row in rows]


def inspect_alfworld_path_manifest(
    manifest_path: str,
    *,
    expected_source_split: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    rows = _load_path_manifest(path)
    canonical_gamefiles: List[str] = []
    resolved_gamefiles: List[str] = []
    task_ids: List[str] = []
    task_types: List[str] = []
    for row in rows:
        canonical = canonical_alfworld_gamefile(str(row.get("gamefile") or ""))
        resolved = resolve_alfworld_gamefile(str(row.get("gamefile") or ""))
        canonical_gamefiles.append(canonical)
        resolved_gamefiles.append(resolved)
        task_ids.append(str(row.get("task_id") or row.get("id") or ""))
        task_types.append(str(row.get("task_type") or row.get("task_family") or row.get("family") or "other"))

    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"ALFWorld manifest contains duplicate task IDs: {path}")
    if len(set(canonical_gamefiles)) != len(canonical_gamefiles):
        raise ValueError(f"ALFWorld manifest contains duplicate gamefiles: {path}")
    source_splits = sorted({_source_split(gamefile) for gamefile in canonical_gamefiles})
    if expected_source_split and source_splits != [expected_source_split]:
        raise ValueError(
            f"ALFWorld manifest {path} must come from {expected_source_split}, got {source_splits}"
        )

    ordered_identity = "\n".join(
        f"{task_id}\t{gamefile}"
        for task_id, gamefile in zip(task_ids, canonical_gamefiles)
    ).encode("utf-8")
    return {
        "manifest_path": str(path),
        "manifest_sha256": _sha256_bytes(path.read_bytes()),
        "task_count": len(rows),
        "ordered_task_gamefile_fingerprint": _sha256_bytes(ordered_identity),
        "task_ids": task_ids,
        "canonical_gamefiles": canonical_gamefiles,
        "resolved_gamefiles": resolved_gamefiles,
        "source_splits": source_splits,
        "task_type_counts": dict(sorted(Counter(task_types).items())),
    }


def _gamefile_set(info: Dict[str, Any]) -> set:
    return {str(value) for value in info.get("canonical_gamefiles") or []}


def preflight_alfworld_path_splits(
    *,
    train_manifest: str,
    validation_manifest: Optional[str],
    test_manifest: Optional[str],
) -> Dict[str, Any]:
    paths = {
        "train": train_manifest,
        "validation": validation_manifest,
        "test": test_manifest,
    }
    available = {name: value for name, value in paths.items() if str(value or "").strip()}
    if not available:
        return {"enabled": False, "reason": "no manifests"}

    # Legacy ALFWorld manifests may identify reset-search tasks without exact
    # gamefiles. Only the SkillOPT-style path format is handled by this audit.
    first_path = Path(str(next(iter(available.values())))).expanduser()
    try:
        first_rows = json.loads(first_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        first_rows = None
    if not isinstance(first_rows, list) or not first_rows or not isinstance(first_rows[0], dict) or not first_rows[0].get("gamefile"):
        return {"enabled": False, "reason": "non-path ALFWorld manifest"}

    official_source_layout = all(available.get(name) for name in ("train", "validation", "test"))
    strict_skillopt_layout = official_source_layout and all(
        "alfworld_path_split" in Path(str(value)).parts for value in available.values()
    )
    expected = {
        "train": "train",
        "validation": "valid_seen",
        "test": "valid_unseen",
    }
    splits = {
        name: inspect_alfworld_path_manifest(
            str(value),
            expected_source_split=expected[name] if official_source_layout else None,
        )
        for name, value in available.items()
    }

    overlaps: Dict[str, List[str]] = {}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(_gamefile_set(splits[left]).intersection(_gamefile_set(splits[right])))
            overlaps[f"{left}:{right}"] = overlap
            if overlap:
                raise ValueError(
                    f"ALFWorld split gamefiles overlap between {left} and {right}: {overlap[:5]}"
                )

    return {
        "enabled": True,
        "official_source_layout": official_source_layout,
        "strict_skillopt_layout": strict_skillopt_layout,
        "splits": splits,
        "overlaps": overlaps,
        "dropped_or_missing_gamefiles": 0,
    }
