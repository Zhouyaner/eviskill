import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


WORKING_BRANCH_STATE_SCHEMA_VERSION = 1
WORKING_BRANCH_EVIDENCE_SCHEMA_VERSION = 5
WORKING_BRANCH_COMPARISON_TASK_LIMIT = 20
WORKING_BRANCH_POLICY_VERSION = 2
DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP = 4
DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP = 8


def default_working_branch_policy() -> Dict[str, int]:
    return {
        "version": WORKING_BRANCH_POLICY_VERSION,
        "epoch_edit_budget": 4,
        "provisional_cap": DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
        "post_reject_replay_cap": DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
        "l2_replay_ranges_per_edit": 2,
    }


def skill_fingerprint(skill_md: str) -> str:
    return hashlib.sha256(str(skill_md or "").encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(str(text), encoding="utf-8")
    temporary.replace(path)


def atomic_save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def initial_working_branch_state(
    dataset: str,
    skill_md: str,
    *,
    replay_mode: str = "standard",
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fingerprint = skill_fingerprint(skill_md)
    return {
        "state_schema_version": WORKING_BRANCH_STATE_SCHEMA_VERSION,
        "evidence_schema_version": WORKING_BRANCH_EVIDENCE_SCHEMA_VERSION,
        "mode": "evidence_working_branch",
        "enabled": True,
        "replay_mode": str(replay_mode or "standard"),
        "working_branch_policy": dict(policy or default_working_branch_policy()),
        "dataset": str(dataset or "").lower(),
        "validated_revision": "B0000",
        "working_revision": "W0000",
        "validated_revision_counter": 0,
        "working_revision_counter": 0,
        "validated_skill_fingerprint": fingerprint,
        "working_skill_fingerprint": fingerprint,
        "next_edit_counter": 1,
        "provisional_edits": [],
        "comparison_queue": [],
        "comparison_history": [],
        "replay_cache_keys": {},
        "stage": {"epoch_index": -1, "name": "initialized"},
    }


def validate_working_branch_state(
    state: Dict[str, Any],
    *,
    dataset: str,
    validated_skill_md: str,
    working_skill_md: str,
    replay_mode: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("working_branch_state.json must contain a JSON object")
    if int(state.get("state_schema_version", 0) or 0) != WORKING_BRANCH_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            "working branch state schema mismatch: "
            f"expected={WORKING_BRANCH_STATE_SCHEMA_VERSION} "
            f"found={state.get('state_schema_version')!r}"
        )
    if int(state.get("evidence_schema_version", 0) or 0) != WORKING_BRANCH_EVIDENCE_SCHEMA_VERSION:
        raise RuntimeError(
            "working branch evidence schema mismatch: "
            f"expected={WORKING_BRANCH_EVIDENCE_SCHEMA_VERSION} "
            f"found={state.get('evidence_schema_version')!r}"
        )
    if state.get("mode") != "evidence_working_branch" or not bool(state.get("enabled")):
        raise RuntimeError("output directory does not contain an enabled evidence working branch")
    if str(state.get("dataset") or "").lower() != str(dataset or "").lower():
        raise RuntimeError(
            "working branch dataset mismatch: "
            f"training={state.get('dataset')!r} requested={str(dataset or '').lower()!r}"
        )
    if replay_mode is not None:
        stored_replay_mode = str(state.get("replay_mode") or "standard")
        expected_replay_mode = str(replay_mode or "standard")
        if stored_replay_mode != expected_replay_mode:
            raise RuntimeError(
                "working branch replay mode mismatch: "
                f"training={stored_replay_mode!r} requested={expected_replay_mode!r}; "
                "use the original replay mode or a new output directory"
            )
    if policy is not None:
        stored_policy = state.get("working_branch_policy")
        expected_policy = dict(policy)
        if not isinstance(stored_policy, dict) or stored_policy != expected_policy:
            raise RuntimeError(
                "working branch policy differs from the existing run: "
                f"training={stored_policy!r} requested={expected_policy!r}; "
                "use the original working-branch parameters or a new output directory"
            )
    expected_validated = str(state.get("validated_skill_fingerprint") or "")
    expected_working = str(state.get("working_skill_fingerprint") or "")
    actual_validated = skill_fingerprint(validated_skill_md)
    actual_working = skill_fingerprint(working_skill_md)
    if expected_validated != actual_validated:
        raise RuntimeError(
            "evolving_skill.md does not match working_branch_state.json; "
            "restore the validated checkpoint or start a new output directory"
        )
    if expected_working != actual_working:
        raise RuntimeError(
            "working_skill.md does not match working_branch_state.json; "
            "restore the working checkpoint or start a new output directory"
        )


def persist_working_branch_state(
    state_path: Path,
    state: Dict[str, Any],
    *,
    validated_skill_md: str,
    working_skill_md: str,
) -> None:
    state["validated_skill_fingerprint"] = skill_fingerprint(validated_skill_md)
    state["working_skill_fingerprint"] = skill_fingerprint(working_skill_md)
    atomic_save_json(state_path, state)


def next_working_revision(state: Dict[str, Any]) -> str:
    counter = int(state.get("working_revision_counter", 0) or 0) + 1
    state["working_revision_counter"] = counter
    revision = f"W{counter:04d}"
    state["working_revision"] = revision
    return revision


def next_validated_revision(state: Dict[str, Any]) -> str:
    counter = int(state.get("validated_revision_counter", 0) or 0) + 1
    state["validated_revision_counter"] = counter
    revision = f"B{counter:04d}"
    state["validated_revision"] = revision
    return revision


def assign_working_edit_ids(
    state: Dict[str, Any], edits: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    counter = int(state.get("next_edit_counter", 1) or 1)
    assigned: List[Dict[str, Any]] = []
    for raw in edits or []:
        edit = dict(raw or {})
        edit_id = str(edit.get("_working_edit_id") or "")
        if not edit_id:
            edit_id = f"M{counter:06d}"
            counter += 1
        edit["_working_edit_id"] = edit_id
        assigned.append(edit)
    state["next_edit_counter"] = counter
    return assigned


def edit_payload(edit: Dict[str, Any]) -> Dict[str, Any]:
    allowed = (
        "op",
        "target",
        "content",
        "evidence_ids",
        "source_edit_ids",
        "source_type",
        "support_count",
        "merge_level",
        "failure_types",
        "patterns",
    )
    return {
        key: edit.get(key)
        for key in allowed
        if key in edit and edit.get(key) not in (None, "", [])
    }


def working_edit_identity(edit: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return the concrete Markdown identity used by the working branch."""
    return (
        str((edit or {}).get("op") or "").strip().lower(),
        str((edit or {}).get("target") or "").strip(),
        str((edit or {}).get("content") or "").strip(),
    )


def _merge_unique_values(*values: Any) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in values:
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


def _merge_working_edit_metadata(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["evidence_ids"] = _merge_unique_values(
        target.get("evidence_ids"), source.get("evidence_ids")
    )
    source_ids = _merge_unique_values(
        target.get("source_edit_ids"),
        source.get("source_edit_ids"),
        target.get("_candidate_edit_id"),
        source.get("_candidate_edit_id"),
    )
    if source_ids:
        target["source_edit_ids"] = source_ids
    if "support_count" in target or "support_count" in source:
        target["support_count"] = max(
            int(target.get("support_count", 0) or 0),
            int(source.get("support_count", 0) or 0),
        )
    if "merge_level" in target or "merge_level" in source:
        target["merge_level"] = max(
            int(target.get("merge_level", 0) or 0),
            int(source.get("merge_level", 0) or 0),
        )
    for key in ("failure_types", "patterns"):
        values = _merge_unique_values(target.get(key), source.get(key))
        if values:
            target[key] = values
    source_types = _merge_unique_values(target.get("source_type"), source.get("source_type"))
    if len(source_types) > 1:
        target["source_type"] = "mixed"
    elif source_types:
        target["source_type"] = source_types[0]


def coalesce_working_edits(edits: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coalesce equivalent edits while retaining all supporting provenance."""
    result: List[Dict[str, Any]] = []
    index_by_identity: Dict[Tuple[str, str, str], int] = {}
    for raw in edits or []:
        edit = dict(raw or {})
        identity = working_edit_identity(edit)
        if not identity[0]:
            result.append(edit)
            continue
        existing_index = index_by_identity.get(identity)
        if existing_index is None:
            index_by_identity[identity] = len(result)
            result.append(edit)
            continue
        existing = result[existing_index]
        merged_from = _merge_unique_values(
            existing.get("merged_from"),
            existing.get("_working_edit_id"),
            existing.get("_candidate_edit_id"),
            edit.get("merged_from"),
            edit.get("_working_edit_id"),
            edit.get("_candidate_edit_id"),
        )
        _merge_working_edit_metadata(existing, edit)
        if merged_from:
            existing["merged_from"] = merged_from
    return result


def _coalesce_provisional_records(
    records: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    index_by_identity: Dict[Tuple[str, str, str], int] = {}
    for raw in records or []:
        record = dict(raw or {})
        identity = working_edit_identity(record.get("edit") or {})
        if not identity[0]:
            result.append(record)
            continue
        existing_index = index_by_identity.get(identity)
        if existing_index is None:
            index_by_identity[identity] = len(result)
            result.append(record)
            continue
        existing = result[existing_index]
        existing_edit = dict(existing.get("edit") or {})
        source_edit = dict(record.get("edit") or {})
        _merge_working_edit_metadata(existing_edit, source_edit)
        merged_from = _merge_unique_values(
            existing.get("merged_from"),
            existing.get("edit_id"),
            record.get("merged_from"),
            record.get("edit_id"),
        )
        existing["edit"] = existing_edit
        if merged_from:
            existing["merged_from"] = merged_from
        for key in ("support_events", "comparison_count"):
            existing[key] = int(existing.get(key, 0) or 0) + int(record.get(key, 0) or 0)
        for key in ("non_improvement_streak", "reflection_attempts"):
            existing[key] = max(
                int(existing.get(key, 0) or 0),
                int(record.get(key, 0) or 0),
            )
        if str(record.get("status") or "") in {"unresolved", "needs_correction"}:
            existing["status"] = record["status"]
    return result


def provisional_evidence_ids(state: Optional[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    seen = set()
    for record in (state or {}).get("provisional_edits") or []:
        if str(record.get("status") or "") in {"superseded", "removed", "validated"}:
            continue
        edit = record.get("edit") or {}
        for value in edit.get("evidence_ids") or record.get("evidence_ids") or []:
            evidence_id = str(value or "")
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                ids.append(evidence_id)
    return ids


def origin_edit_ids_from_evidence(
    edit: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]
) -> List[str]:
    origin_ids: List[str] = []
    seen = set()
    for evidence_id in edit.get("evidence_ids") or []:
        card = evidence_by_id.get(str(evidence_id or "")) or {}
        for value in card.get("origin_edit_ids") or []:
            origin_id = str(value or "")
            if origin_id and origin_id not in seen:
                seen.add(origin_id)
                origin_ids.append(origin_id)
    return origin_ids


def make_provisional_record(
    edit: Dict[str, Any],
    *,
    epoch_idx: int,
    order: int,
    status: str = "provisional",
) -> Dict[str, Any]:
    edit_id = str(edit.get("_working_edit_id") or edit.get("_merged_edit_id") or "")
    return {
        "edit_id": edit_id,
        "edit": edit_payload(edit),
        "status": status,
        "introduced_epoch": int(epoch_idx),
        "introduced_order": int(order),
        "support_events": 0,
        "non_improvement_streak": 0,
        "reflection_attempts": 0,
        "comparison_count": 0,
    }


def rebuild_working_skill(
    validated_skill_md: str,
    provisional_records: Iterable[Dict[str, Any]],
    *,
    apply_edit: Callable[[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    current = str(validated_skill_md or "")
    surviving: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    ordered = sorted(
        _coalesce_provisional_records(provisional_records),
        key=lambda row: (
            int(row.get("introduced_epoch", 0) or 0),
            int(row.get("introduced_order", 0) or 0),
            str(row.get("edit_id") or ""),
        ),
    )
    for record in ordered:
        if str(record.get("status") or "") in {"superseded", "removed", "validated"}:
            continue
        updated, report = apply_edit(current, dict(record.get("edit") or {}))
        if bool((report or {}).get("applied")):
            current = updated
            surviving.append(record)
        else:
            failed.append({"record": record, "report": report})
    return current, surviving, failed


def replay_cache_key(
    candidate_skill_md: str,
    edit: Dict[str, Any],
    *,
    evidence_id: str,
    range_index: int,
    trigger: Dict[str, Any],
) -> str:
    payload = {
        "skill": skill_fingerprint(candidate_skill_md),
        "edit": edit_payload(edit),
        "evidence_id": str(evidence_id or ""),
        "range_index": int(range_index),
        "trigger": trigger,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def select_comparison_tasks(
    edits: Iterable[Dict[str, Any]],
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
    parent_epoch_index: int,
    parent_skill_revision: str,
    working_skill_revision: str,
    seed: int,
    limit: int = WORKING_BRANCH_COMPARISON_TASK_LIMIT,
) -> List[Dict[str, Any]]:
    candidates_by_edit: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    for raw_edit in edits or []:
        edit = dict(raw_edit or {})
        edit_id = str(edit.get("_working_edit_id") or edit.get("edit_id") or "")
        jobs: List[Dict[str, Any]] = []
        for evidence_id in edit.get("evidence_ids") or []:
            card = evidence_by_id.get(str(evidence_id or "")) or {}
            support = int(card.get("support_count", 1) or 1)
            family_by_task = {
                str(item.get("task_id") or ""): str(item.get("task_family") or "")
                for item in card.get("supporting_tasks") or []
                if isinstance(item, dict)
            }
            for trigger in card.get("trigger_ranges") or []:
                if not isinstance(trigger, dict):
                    continue
                task_id = str(trigger.get("task_id") or "")
                if not task_id:
                    continue
                jobs.append(
                    {
                        "task_id": task_id,
                        "task_family": family_by_task.get(task_id, ""),
                        "evidence_id": str(evidence_id or ""),
                        "trigger_range": dict(trigger),
                        "support_count": support,
                    }
                )
        jobs.sort(
            key=lambda row: (
                -int(row.get("support_count", 0) or 0),
                str(row.get("task_family") or ""),
                str(row.get("task_id") or ""),
            )
        )
        candidates_by_edit.append((edit_id, edit, jobs))

    selected: Dict[str, Dict[str, Any]] = {}
    edit_covered = set()
    for edit_id, edit, jobs in candidates_by_edit:
        for job in jobs:
            task_id = str(job.get("task_id") or "")
            if task_id in selected:
                selected[task_id]["origin_edit_ids"] = list(
                    dict.fromkeys(selected[task_id]["origin_edit_ids"] + [edit_id])
                )
                selected[task_id].setdefault("origin_edits", {})[edit_id] = edit_payload(edit)
                edit_covered.add(edit_id)
                break
            if len(selected) >= max(0, int(limit)):
                break
            selected[task_id] = {
                **job,
                "origin_edit_ids": [edit_id],
                "origin_edits": {edit_id: edit_payload(edit)},
            }
            edit_covered.add(edit_id)
            break

    remaining: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for edit_id, edit, jobs in candidates_by_edit:
        for job in jobs:
            remaining.append((edit_id, edit, job))
    rng = random.Random(int(seed))
    rng.shuffle(remaining)
    remaining.sort(
        key=lambda item: (
            0 if item[0] not in edit_covered else 1,
            -int(item[2].get("support_count", 0) or 0),
            str(item[2].get("task_family") or ""),
            str(item[2].get("task_id") or ""),
            item[0],
        )
    )
    for edit_id, edit, job in remaining:
        if len(selected) >= max(0, int(limit)):
            break
        task_id = str(job.get("task_id") or "")
        if task_id in selected:
            row = selected[task_id]
            if edit_id not in row["origin_edit_ids"]:
                row["origin_edit_ids"].append(edit_id)
                row.setdefault("origin_edits", {})[edit_id] = edit_payload(edit)
            continue
        selected[task_id] = {
            **job,
            "origin_edit_ids": [edit_id],
            "origin_edits": {edit_id: edit_payload(edit)},
        }

    queue = []
    for task_id, row in selected.items():
        queue.append(
            {
                **row,
                "task_id": task_id,
                "parent_epoch_index": int(parent_epoch_index),
                "parent_skill_revision": str(parent_skill_revision or ""),
                "working_skill_revision": str(working_skill_revision or ""),
            }
        )
    return queue[: max(0, int(limit))]


def comparison_type(parent_success: bool, working_success: bool) -> str:
    if not parent_success and working_success:
        return "improved"
    if parent_success and not working_success:
        return "regressed"
    if not parent_success and not working_success:
        return "persistent_fail"
    return "stable_success"


def reactivate_evidence(rows: Iterable[Dict[str, Any]], evidence_ids: Iterable[str]) -> List[str]:
    target = {str(value or "") for value in evidence_ids or [] if str(value or "")}
    changed: List[str] = []
    for row in rows or []:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id not in target:
            continue
        row["archived"] = False
        row["l2_status"] = "active"
        row.pop("archive_reason", None)
        row.pop("consume_reason", None)
        row["defer_reason"] = "origin_edit_removed"
        changed.append(evidence_id)
    return sorted(set(changed))
