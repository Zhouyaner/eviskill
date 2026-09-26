import re
from typing import Any, Dict, List, Optional


LONGITUDINAL_MEMORY_START = "<!-- LONGITUDINAL_MEMORY_START -->"
LONGITUDINAL_MEMORY_END = "<!-- LONGITUDINAL_MEMORY_END -->"


def has_longitudinal_memory_field(skill_md: str) -> bool:
    return LONGITUDINAL_MEMORY_START in skill_md and LONGITUDINAL_MEMORY_END in skill_md


def inject_empty_longitudinal_memory_field(skill_md: str) -> str:
    if has_longitudinal_memory_field(skill_md):
        return skill_md
    return skill_md.rstrip() + f"\n\n{LONGITUDINAL_MEMORY_START}\n{LONGITUDINAL_MEMORY_END}\n"


def extract_longitudinal_memory_field(skill_md: str) -> str:
    start = skill_md.find(LONGITUDINAL_MEMORY_START)
    end = skill_md.find(LONGITUDINAL_MEMORY_END)
    if start == -1 or end == -1 or end < start:
        return ""
    inner_start = start + len(LONGITUDINAL_MEMORY_START)
    return strip_longitudinal_memory_markers(skill_md[inner_start:end]).strip()


def _strip_all_longitudinal_memory_fields(skill_md: str) -> str:
    text = str(skill_md or "")
    while True:
        start = text.find(LONGITUDINAL_MEMORY_START)
        if start == -1:
            break
        end = text.find(LONGITUDINAL_MEMORY_END, start)
        if end == -1:
            text = text[:start] + text[start + len(LONGITUDINAL_MEMORY_START):]
            break
        text = text[:start] + text[end + len(LONGITUDINAL_MEMORY_END):]
    text = text.replace(LONGITUDINAL_MEMORY_END, "")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.rstrip()


def replace_longitudinal_memory_field(skill_md: str, new_content: str) -> str:
    base = _strip_all_longitudinal_memory_fields(skill_md)
    content = strip_longitudinal_memory_markers(str(new_content or "")).strip()
    return base + f"\n\n{LONGITUDINAL_MEMORY_START}\n{content}\n{LONGITUDINAL_MEMORY_END}\n"


def is_in_longitudinal_memory_region(skill_md: str, target: str) -> bool:
    if not target:
        return False
    start = skill_md.find(LONGITUDINAL_MEMORY_START)
    end = skill_md.find(LONGITUDINAL_MEMORY_END)
    if start == -1 or end == -1 or end < start:
        return False
    target_idx = skill_md.find(target)
    if target_idx == -1:
        return False
    return start <= target_idx < end + len(LONGITUDINAL_MEMORY_END)


def strip_longitudinal_memory_markers(text: str) -> str:
    return (
        str(text or "")
        .replace(LONGITUDINAL_MEMORY_START, "")
        .replace(LONGITUDINAL_MEMORY_END, "")
    )


def append_before_longitudinal_memory(skill_md: str, content: str) -> str:
    content = str(content or "").strip()
    start = skill_md.find(LONGITUDINAL_MEMORY_START)
    if start == -1:
        return skill_md.rstrip() + "\n\n" + content + "\n"
    before = skill_md[:start].rstrip()
    after = skill_md[start:].lstrip("\n")
    return before + "\n\n" + content + "\n\n" + after


def _result_task_id(row: Dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("id") or row.get("task") or "")


def _item_task_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("task_id") or item.get("id") or item.get("task") or item.get("instruction") or "")
    return str(item or "")


def _item_instruction(item: Any, result: Optional[Dict[str, Any]] = None) -> str:
    if result:
        text = str(result.get("instruction") or result.get("task") or "").strip()
        if text:
            return text
    if isinstance(item, dict):
        return str(item.get("instruction") or item.get("task") or item.get("task_name") or item.get("question") or "").strip()
    return str(item or "").strip()


def _item_family(item: Any, result: Optional[Dict[str, Any]] = None) -> str:
    if result and result.get("task_family"):
        return str(result.get("task_family") or "")
    if isinstance(item, dict):
        return str(item.get("task_family") or item.get("family") or "")
    return ""


def _success(row: Dict[str, Any]) -> bool:
    return bool(row.get("success")) or float(row.get("score", row.get("final_reward", 0.0)) or 0.0) >= 1.0


def _score(row: Dict[str, Any]) -> float:
    return float(row.get("score", row.get("final_reward", 0.0)) or 0.0)


def _steps(row: Dict[str, Any]) -> int:
    return int(row.get("steps", len(row.get("trajectory", []) or [])) or 0)


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text[: max(0, int(limit))]


def format_compact_trajectory(
    result: Dict[str, Any],
    *,
    max_chars: int = 3000,
    full_context: bool = False,
) -> str:
    lines: List[str] = []
    for row in result.get("trajectory", []) or []:
        if not isinstance(row, dict):
            continue
        step = row.get("step", "?")
        thought = str(row.get("thought") or "") if full_context else _clip(row.get("thought"), 300)
        action = str(row.get("action") or "") if full_context else _clip(row.get("action"), 200)
        obs = str(row.get("observation") or "") if full_context else _clip(row.get("observation"), 700)
        next_obs = str(row.get("next_observation") or "") if full_context else _clip(row.get("next_observation"), 700)
        if thought:
            lines.append(f"[step {step} think] {thought}")
        lines.append(f"[step {step} obs] {obs}")
        lines.append(f"[step {step} action] {action}")
        if next_obs:
            lines.append(f"[step {step} next_obs] {next_obs}")
    text = "\n".join(lines) or "(empty trajectory)"
    max_chars = int(max_chars or 0)
    if not full_context and max_chars > 0 and len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]
    return text


def build_longitudinal_comparison_pairs(
    *,
    items: List[Any],
    previous_results: List[Dict[str, Any]],
    current_results: List[Dict[str, Any]],
    max_trajectory_chars: int = 3000,
    full_context: bool = False,
) -> List[Dict[str, Any]]:
    prev_by_id = {_result_task_id(row): row for row in previous_results or []}
    curr_by_id = {_result_task_id(row): row for row in current_results or []}
    pairs: List[Dict[str, Any]] = []
    for item in items or []:
        tid = _item_task_id(item)
        prev = prev_by_id.get(tid, {})
        curr = curr_by_id.get(tid, {})
        if not prev and not curr:
            continue
        prev_ok = _success(prev)
        curr_ok = _success(curr)
        if not prev_ok and curr_ok:
            category = "improved"
        elif prev_ok and not curr_ok:
            category = "regressed"
        elif not prev_ok and not curr_ok:
            category = "persistent_fail"
        else:
            category = "stable_success"
        representative = curr or prev
        pairs.append(
            {
                "id": tid or _result_task_id(representative),
                "task": _item_instruction(item, representative),
                "task_family": _item_family(item, representative),
                "category": category,
                "prev": {
                    "success": bool(prev_ok),
                    "score": _score(prev),
                    "steps": _steps(prev),
                },
                "curr": {
                    "success": bool(curr_ok),
                    "score": _score(curr),
                    "steps": _steps(curr),
                },
                "prev_trajectory": format_compact_trajectory(
                    prev,
                    max_chars=max_trajectory_chars,
                    full_context=full_context,
                ) if prev else "",
                "curr_trajectory": format_compact_trajectory(
                    curr,
                    max_chars=max_trajectory_chars,
                    full_context=full_context,
                ) if curr else "",
            }
        )
    return pairs


def _category_counts(pairs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"improved": 0, "regressed": 0, "persistent_fail": 0, "stable_success": 0}
    for pair in pairs or []:
        category = str(pair.get("category") or "")
        counts[category] = counts.get(category, 0) + 1
    return counts


def format_longitudinal_comparison_text(pairs: List[Dict[str, Any]]) -> str:
    counts = _category_counts(pairs)
    parts = [
        "## Longitudinal Comparison Summary\n"
        f"Total samples: {len(pairs or [])}\n"
        f"- Improved (fail -> success): {counts.get('improved', 0)}\n"
        f"- Regressed (success -> fail): {counts.get('regressed', 0)}\n"
        f"- Persistent failures (fail -> fail): {counts.get('persistent_fail', 0)}\n"
        f"- Stable successes (success -> success): {counts.get('stable_success', 0)}\n"
    ]
    sections = [
        ("regressed", "Regressions (success -> fail) - HIGHEST PRIORITY", True),
        ("persistent_fail", "Persistent Failures (fail -> fail)", True),
        ("improved", "Improvements (fail -> success)", True),
        ("stable_success", "Stable Successes (success -> success)", False),
    ]
    for category, title, show_trajectory in sections:
        rows = [pair for pair in pairs or [] if pair.get("category") == category]
        if not rows:
            parts.append(f"### {title}\n(none)\n")
            continue
        lines = [f"### {title}"]
        for pair in rows:
            prev = pair.get("prev") or {}
            curr = pair.get("curr") or {}
            family = str(pair.get("task_family") or "").strip()
            family_text = f"\nTask family: {family}" if family else ""
            lines.append(
                f"\n#### Task {pair.get('id')}: {str(pair.get('task') or '')[:300]}"
                f"{family_text}\n"
                f"- Previous epoch: {'PASS' if prev.get('success') else 'FAIL'} "
                f"(score={float(prev.get('score', 0.0) or 0.0):.2f}, steps={int(prev.get('steps', 0) or 0)})\n"
                f"- Current epoch: {'PASS' if curr.get('success') else 'FAIL'} "
                f"(score={float(curr.get('score', 0.0) or 0.0):.2f}, steps={int(curr.get('steps', 0) or 0)})"
            )
            if show_trajectory:
                if pair.get("prev_trajectory"):
                    lines.append(f"\n**Previous epoch trajectory:**\n```\n{pair['prev_trajectory']}\n```")
                if pair.get("curr_trajectory"):
                    lines.append(f"\n**Current epoch trajectory:**\n```\n{pair['curr_trajectory']}\n```")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def format_skill_update_summary(committed_edits: List[Dict[str, Any]]) -> str:
    if not committed_edits:
        return "No skill edits were committed during this epoch."
    parts = ["The following skill edits were committed during this epoch."]
    for idx, edit in enumerate(committed_edits, start=1):
        op = str((edit or {}).get("op") or "")
        target = (edit or {}).get("target")
        target_text = "null" if target in (None, "") else str(target)
        content = str((edit or {}).get("content") or "")
        parts.append(
            f"{idx}. op: {op}\n"
            f"target: {target_text}\n"
            f"content:\n{content if content else '(empty)'}"
        )
    return "\n\n".join(parts)


def build_longitudinal_memory_input(
    *,
    previous_skill_md: str,
    current_skill_md: str,
    previous_memory: str,
    skill_update_summary: str,
    longitudinal_comparison: str,
) -> str:
    memory_text = str(previous_memory or "").strip() or "(No previous longitudinal memory.)"
    return (
        f"## Previous Epoch's Skill\n{previous_skill_md}\n\n"
        f"## Current Epoch's Skill\n{current_skill_md}\n\n"
        f"## Previous Longitudinal Memory\n{memory_text}\n\n"
        f"## Skill Update Summary\n{skill_update_summary}\n\n"
        f"## Longitudinal Comparison\n{longitudinal_comparison}"
    )


def safe_memory_dir_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return text[:80] or "epoch"
