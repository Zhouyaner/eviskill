import copy
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .action_runtime import (
    ALFWORLD_ACTION_HISTORY_LENGTH,
    APPWORLD_ACTION_HISTORY_LENGTH,
    SCIENCEWORLD_ACTION_HISTORY_LENGTH,
)
from .bundles import (
    get_task_family,
    get_task_id,
    get_task_ref,
    limit_manifest,
    mixed_family_epoch_bundle_mode,
    sample_family_epoch_bundle,
    sample_mixed_family_epoch_bundle,
    sample_random_bundle,
    sample_shuffle_epoch_bundle,
)
from .alfworld_manifests import preflight_alfworld_path_splits
from .appworld_manifests import preflight_appworld_splits
from .evidence_embeddings import (
    DEFAULT_EVIDENCE_EMBEDDING_MODEL,
    encode_evidence_cards,
)
from .json_utils import load_json, load_jsonl, save_json, save_jsonl
from .llm_clients import build_llm_client
from .longitudinal_memory import (
    LONGITUDINAL_MEMORY_START,
    append_before_longitudinal_memory,
    build_longitudinal_comparison_pairs,
    build_longitudinal_memory_input,
    extract_longitudinal_memory_field,
    format_longitudinal_comparison_text,
    format_skill_update_summary,
    has_longitudinal_memory_field,
    inject_empty_longitudinal_memory_field,
    is_in_longitudinal_memory_region,
    replace_longitudinal_memory_field,
    safe_memory_dir_name,
    strip_longitudinal_memory_markers,
)
from .progress import progress_iter
from .records import normalize_task_results, task_records_payload
from .runtime import FixedJsonActionLLM, build_alfworld_runner, build_appworld_runner, build_scienceworld_runner
from .skill_markdown import MarkdownSkillBank
from .prompts import prompts_for_dataset
from .working_branch import (
    DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
    DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
    WORKING_BRANCH_COMPARISON_TASK_LIMIT,
    WORKING_BRANCH_POLICY_VERSION,
    assign_working_edit_ids,
    atomic_write_text,
    coalesce_working_edits,
    comparison_type,
    edit_payload as _working_edit_payload,
    initial_working_branch_state,
    make_provisional_record,
    next_validated_revision,
    next_working_revision,
    origin_edit_ids_from_evidence,
    persist_working_branch_state,
    provisional_evidence_ids,
    reactivate_evidence,
    rebuild_working_skill,
    replay_cache_key,
    select_comparison_tasks,
    skill_fingerprint,
    validate_working_branch_state,
)

SELECTION_FEEDBACK_HISTORY = 5
SELECTION_FEEDBACK_MAX_POSITIVE = 3
SELECTION_FEEDBACK_MAX_NEGATIVE = 3
SELECTION_FEEDBACK_STEP_DELTA = 5
MIN_NEW_EVIDENCE_AFTER_SELECTION_REJECT = 4
L2_MAX_REJECTS_PER_EVIDENCE = 2
DEFAULT_SELECTION_SUCCESS_MARGIN = 0.0
DEFAULT_SELECTION_ACCEPT_TIE = False
DEFAULT_SELECTION_SCORE_TIEBREAK_MARGIN = 0.01
DEFAULT_SELECTION_STEP_TIEBREAK_DELTA = 1.0
DEFAULT_EPOCH_EDIT_BUDGET = 4
DEFAULT_EPOCH_MERGE_MAX_CANDIDATES = 24
DEFAULT_EPOCH_MERGE_MAX_CONTENT_CHARS = 1200
DEFAULT_EPOCH_MERGE_SKILL_MAX_CHARS = 12000
DEFAULT_EPOCH_FINAL_REPLAY_RANGES_PER_EDIT = 2
DEFAULT_EPOCH_SKILL_UPDATE_MODE = "cluster_step"
DEFAULT_LLM_MAX_OBSERVATION_CHARS = 500
DEFAULT_LLM_MAX_INPUT_CHARS = 30000
EVIDENCE_SCHEMA_VERSION = 5
L1_MAX_PROPOSED_EDITS = 4
L1_TRAJECTORY_MINIBATCH_SIZE = 4
L1_CRITICAL_FEEDBACK_MIN_CHARS = 160
L1_CRITICAL_FEEDBACK_MAX_CHARS = 1500
APPWORLD_L1_CODE_MIN_CHARS = 120
APPWORLD_L1_CODE_MAX_CHARS = 600
APPWORLD_L1_OUTPUT_MIN_CHARS = 120


class SelectionEvaluationError(RuntimeError):
    """Validation could not produce a complete, comparable task result set."""


class L2ProposerError(RuntimeError):
    """The L2 proposer failed before producing a trustworthy candidate set."""


APPWORLD_L2_CODE_MAX_CHARS = 400
APPWORLD_L2_OUTPUT_MAX_CHARS = 600
APPWORLD_L2_STEP_LIMIT = 8
L2_REPLAY_FAILURE_TYPES = {"rule_missing", "rule_wrong", "rule_ignored"}
# Kept for callers that imported the AppWorld-era name before ScienceWorld shared attribution.
APPWORLD_REPLAY_FAILURE_TYPES = L2_REPLAY_FAILURE_TYPES
PROPOSER_EVIDENCE_MAX_TASK_CHARS = 150
PROPOSER_EVIDENCE_MAX_EDIT_TARGET_CHARS = 220
PROPOSER_EVIDENCE_MAX_EDIT_CONTENT_CHARS = 520
PROPOSER_EVIDENCE_MAX_REASON_CHARS = 140
VALID_FAILURE_TYPES = {
    "rule_missing",
    "rule_wrong",
    "rule_ignored",
    "invalid_action",
    "exploration_gap",
    "experiment_error",
    "premature_finish",
    "loop",
    "other",
}

EVIDENCE_STATUS_ACTIVE = "active"
EVIDENCE_STATUS_STALE = "stale"
EVIDENCE_STATUS_ARCHIVED = "archived"
EVIDENCE_STATUS_CONSUMED = "consumed"
WORKING_BRANCH_POLICY_DEFER_REASONS = frozenset(
    {
        "l2_proposer_not_covered",
        "l2_proposer_omitted",
        "global_merge_not_covered",
        "global_merge_omitted",
        "coverage_repair_not_covered",
        "coverage_repair_omitted",
        "rule_ignored",
    }
)
RECOVERY_RULES_SECTION = "## Evidence-Derived Recovery Rules"

EPOCH_JSONL_FILENAMES = {
    "l1_skill_manager": "l1_skill_manager.jsonl",
    "l1_skill_judge": "l1_skill_judge.jsonl",
    "l2_skill_manager": "l2_skill_manager.jsonl",
    "l2_skill_judge": "l2_skill_judge.jsonl",
    "evidence_cards": "evidence_cards.jsonl",
    "evidence_replay_sources": "evidence_replay_sources.jsonl",
    "l2_skill_candidates": "l2_skill_candidates.jsonl",
    "replay_label_traces": "replay_label_traces.jsonl",
    "selection_feedback_cards": "selection_feedback_cards.jsonl",
}

TRAIN_ROLLOUT_BUNDLES_DIRNAME = "train_rollout_bundles"
TRAIN_ROLLOUT_CHECKPOINT_VERSION = 1

LEGACY_EPOCH_JSONL_PATHS = {
    key: filename for key, filename in EPOCH_JSONL_FILENAMES.items()
}


def _is_scienceworld_dataset(dataset: Any) -> bool:
    return str(dataset or "").strip().lower() in {"scienceworld", "science_world"}


def _is_appworld_dataset(dataset: Any) -> bool:
    return str(dataset or "").strip().lower() == "appworld"


def _is_alfworld_dataset(dataset: Any) -> bool:
    return str(dataset or "").strip().lower() == "alfworld"


def _uses_l2_replay_attribution(dataset: Any) -> bool:
    return str(dataset or "").strip().lower() in {
        "alfworld",
        "scienceworld",
        "science_world",
        "appworld",
    }


def _evidence_working_branch_enabled(args: Any) -> bool:
    return bool(getattr(args, "use_evidence_working_branch", False))


def _working_branch_defer_uncovered_enabled(dataset: Any, args: Any) -> bool:
    return bool(
        _is_appworld_dataset(dataset)
        and _evidence_working_branch_enabled(args)
        and _resolve_epoch_skill_update_mode(args) == "cluster_step"
        and getattr(args, "working_branch_defer_uncovered_evidence", False)
    )


def _defer_rule_ignored_configured(args: Any) -> bool:
    """Return the single effective rule_ignored accept/defer setting.

    ``--working-branch-defer-rule-ignored`` predates the global setting and is
    intentionally retained as a compatibility alias. Both names feed this one
    decision; neither changes the replay stage or working-branch mode.
    """
    return bool(
        getattr(args, "defer_rule_ignored", False)
        or getattr(args, "working_branch_defer_rule_ignored", False)
    )


def _defer_rule_ignored_enabled(dataset: Any, args: Any) -> bool:
    """Whether a replay-attributed ``rule_ignored`` label should be deferred."""
    return bool(
        (
            _is_appworld_dataset(dataset)
            or _is_scienceworld_dataset(dataset)
            or _is_alfworld_dataset(dataset)
        )
        and _defer_rule_ignored_configured(args)
    )


def _working_branch_defer_rule_ignored_enabled(dataset: Any, args: Any) -> bool:
    """Backward-compatible name for callers that used the old helper."""
    return _defer_rule_ignored_enabled(dataset, args)


def _no_replay_enabled(args: Any) -> bool:
    return bool(getattr(args, "no_replay", False))


def _replay_mode(args: Any) -> str:
    return "disabled" if _no_replay_enabled(args) else "standard"


def _working_branch_policy(args: Any) -> Dict[str, Any]:
    policy: Dict[str, Any] = {
        "version": WORKING_BRANCH_POLICY_VERSION,
        "epoch_edit_budget": max(
            0,
            _int_arg(getattr(args, "epoch_edit_budget", DEFAULT_EPOCH_EDIT_BUDGET), DEFAULT_EPOCH_EDIT_BUDGET),
        ),
        "provisional_cap": max(
            0,
            _int_arg(
                getattr(args, "working_branch_provisional_cap", DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP),
                DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
            ),
        ),
        "post_reject_replay_cap": _int_arg(
            getattr(
                args,
                "post_reject_replay_cap",
                DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
            ),
            DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
        ),
        "l2_replay_ranges_per_edit": _int_arg(
            getattr(args, "l2_replay_ranges_per_edit", 4),
            4,
        ),
    }
    if bool(getattr(args, "working_branch_defer_uncovered_evidence", False)):
        policy["defer_uncovered_evidence"] = True
    if _defer_rule_ignored_configured(args):
        policy["defer_rule_ignored"] = True
    policy_defer_limit = _int_arg(
        getattr(args, "working_branch_max_policy_defer_epochs", -1), -1
    )
    if policy_defer_limit != -1:
        policy["max_policy_defer_epochs"] = policy_defer_limit
    return policy


def _validate_evidence_working_branch_args(args: Any) -> None:
    epoch_comparison_tasks = _int_arg(
        getattr(args, "epoch_trajectory_comparison_tasks", 0), 0
    )
    if epoch_comparison_tasks < 0:
        raise ValueError("--epoch-trajectory-comparison-tasks must be >= 0")
    defer_uncovered = bool(
        getattr(args, "working_branch_defer_uncovered_evidence", False)
    )
    defer_rule_ignored = _defer_rule_ignored_configured(args)
    max_policy_defer_epochs = _int_arg(
        getattr(args, "working_branch_max_policy_defer_epochs", -1), -1
    )
    if max_policy_defer_epochs == 0 or max_policy_defer_epochs < -1:
        raise ValueError(
            "--working-branch-max-policy-defer-epochs must be -1 or a positive integer"
        )
    working_branch_policy_configured = (
        defer_uncovered or max_policy_defer_epochs != -1
    )
    if not _evidence_working_branch_enabled(args):
        if epoch_comparison_tasks > 0:
            raise ValueError(
                "--epoch-trajectory-comparison-tasks requires "
                "--use-evidence-working-branch"
            )
        if defer_uncovered or max_policy_defer_epochs != -1:
            raise ValueError(
                "working-branch defer options require --use-evidence-working-branch"
            )
        if defer_rule_ignored and not (
            _is_appworld_dataset(getattr(args, "dataset", ""))
            or _is_scienceworld_dataset(getattr(args, "dataset", ""))
            or _is_alfworld_dataset(getattr(args, "dataset", ""))
        ):
            raise ValueError(
                "--defer-rule-ignored is only supported for AppWorld, "
                "ScienceWorld, and ALFWorld"
            )
        return
    dataset = getattr(args, "dataset", "")
    if defer_uncovered and not _is_appworld_dataset(dataset):
        raise ValueError(
            "--working-branch-defer-uncovered-evidence is only supported for AppWorld"
        )
    if defer_rule_ignored and not (
        _is_appworld_dataset(dataset)
        or _is_scienceworld_dataset(dataset)
        or _is_alfworld_dataset(dataset)
    ):
        raise ValueError(
            "--defer-rule-ignored is only supported for "
            "AppWorld, ScienceWorld, and ALFWorld"
        )
    if (
        max_policy_defer_epochs != -1
        and not _is_appworld_dataset(dataset)
        and not (
            (_is_scienceworld_dataset(dataset) or _is_alfworld_dataset(dataset))
            and defer_rule_ignored
        )
    ):
        raise ValueError(
            "--working-branch-max-policy-defer-epochs requires a supported "
            "policy defer mechanism"
        )
    if (
        working_branch_policy_configured
        and _resolve_epoch_skill_update_mode(args) != "cluster_step"
    ):
        raise ValueError(
            "working-branch defer options require --epoch-skill-update-mode cluster_step"
        )
    conflicts = []
    if bool(getattr(args, "use_longitudinal_memory", False)):
        conflicts.append("--use-longitudinal-memory")
    if bool(getattr(args, "no_epoch_validation", False)):
        conflicts.append("--no-epoch-validation")
    if str(getattr(args, "reuse_evidence_dir", "") or "").strip():
        conflicts.append("--reuse-evidence-dir")
    if getattr(args, "limit_tasks", None) is not None:
        conflicts.append("--limit-tasks")
    if conflicts:
        raise ValueError(
            "--use-evidence-working-branch is incompatible with " + ", ".join(conflicts)
        )
    strategy = str(getattr(args, "bundle_sampling_strategy", "shuffle_epoch") or "shuffle_epoch").lower()
    if strategy == "random":
        raise ValueError(
            "--use-evidence-working-branch requires a complete train-manifest pass; "
            "--bundle-sampling-strategy=random is not allowed"
        )
    minibatch_size = _int_arg(
        getattr(args, "l1_trajectory_minibatch_size", L1_TRAJECTORY_MINIBATCH_SIZE),
        L1_TRAJECTORY_MINIBATCH_SIZE,
    )
    if minibatch_size != L1_TRAJECTORY_MINIBATCH_SIZE:
        raise ValueError(
            "--use-evidence-working-branch requires "
            f"--l1-trajectory-minibatch-size {L1_TRAJECTORY_MINIBATCH_SIZE}"
        )
    if _int_arg(
        getattr(args, "working_branch_provisional_cap", DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP),
        DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
    ) < 0:
        raise ValueError("--working-branch-provisional-cap must be >= 0")
    args.no_l2_selection_feedback = True


def _initialize_evidence_working_branch(
    *,
    out: Path,
    dataset: str,
    evolving_skill_path: Path,
    initial_skill_md: str,
    resume: bool,
    no_resume: bool,
    had_existing_generation: bool,
    enabled: bool,
    replay_mode: str = "standard",
    policy: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Path], Optional[Path], Optional[Dict[str, Any]]]:
    working_skill_path = out / "working_skill.md"
    state_path = out / "working_branch_state.json"
    if not enabled:
        if resume and state_path.exists():
            raise RuntimeError(
                "This output directory was created with --use-evidence-working-branch. "
                "Resume with the same flag or use a new output directory."
            )
        if no_resume:
            working_skill_path.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
        return None, None, None

    if resume and had_existing_generation and not state_path.exists():
        raise RuntimeError(
            "Cannot enable --use-evidence-working-branch inside an existing base-mode output. "
            "Use a new output directory."
        )

    if no_resume or not state_path.exists():
        atomic_write_text(working_skill_path, initial_skill_md)
        state = initial_working_branch_state(
            dataset,
            initial_skill_md,
            replay_mode=replay_mode,
            policy=policy,
        )
        persist_working_branch_state(
            state_path,
            state,
            validated_skill_md=initial_skill_md,
            working_skill_md=initial_skill_md,
        )
        return working_skill_path, state_path, state

    if not working_skill_path.exists():
        raise RuntimeError(
            "working_branch_state.json exists but working_skill.md is missing; "
            "restore the checkpoint or use a new output directory"
        )
    state = load_json(str(state_path))
    validated_skill_md = evolving_skill_path.read_text(encoding="utf-8")
    working_skill_md = working_skill_path.read_text(encoding="utf-8")
    validate_working_branch_state(
        state,
        dataset=dataset,
        validated_skill_md=validated_skill_md,
        working_skill_md=working_skill_md,
        replay_mode=replay_mode,
        policy=policy,
    )
    return working_skill_path, state_path, state


def _l2_selection_feedback_enabled(dataset: Any, args: Any = None) -> bool:
    if bool(getattr(args, "use_evidence_working_branch", False)):
        return False
    if bool(getattr(args, "no_l2_selection_feedback", False)):
        return False
    # ScienceWorld epoch evolution is train-evidence-only: validation/dev cases
    # remain a gate and must not flow back into skill edit prompts.
    if _is_scienceworld_dataset(dataset) or _is_appworld_dataset(dataset):
        return False
    return True


def _resolve_epoch_skill_update_mode(args: Any = None) -> str:
    value = str(
        getattr(args, "epoch_skill_update_mode", DEFAULT_EPOCH_SKILL_UPDATE_MODE)
        or DEFAULT_EPOCH_SKILL_UPDATE_MODE
    ).strip().lower()
    if value not in {"merge_rank", "cluster_step"}:
        return DEFAULT_EPOCH_SKILL_UPDATE_MODE
    return value


def _compact_segment(
    rows: List[Dict[str, Any]],
    *,
    max_observation_chars: Optional[int] = None,
    action_only: bool = False,
) -> List[Dict[str, Any]]:
    out = []
    for raw in rows or []:
        obs = "" if action_only else str(raw.get("observation", ""))
        if max_observation_chars is not None and int(max_observation_chars) >= 0:
            obs = obs[: int(max_observation_chars)]
        out.append(
            {
                "step": int(raw.get("step", len(out)) or 0),
                "observation": obs,
                "action": str(raw.get("action", "")),
            }
        )
    return out


def _int_arg(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _jsonable_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _llm_input_chars(system_prompt: str, payload: Dict[str, Any]) -> int:
    return len(str(system_prompt or "")) + len(json.dumps(payload, ensure_ascii=False, default=str))


def _cap_observations_in_place(value: Any, max_chars: int) -> Tuple[int, int, int]:
    """Cap LLM-visible environment text while preserving actions and structure."""
    max_chars = int(max_chars)
    count = 0
    before_chars = 0
    after_chars = 0
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key in {"observation", "initial_observation", "feedback", "next_observation", "env_feedback"}:
                text = "" if item is None else str(item)
                before_chars += len(text)
                if max_chars >= 0:
                    text = _compress_feedback(text, max_chars, keywords=set())
                after_chars += len(text)
                value[key] = text
                count += 1
            else:
                sub_count, sub_before, sub_after = _cap_observations_in_place(item, max_chars)
                count += sub_count
                before_chars += sub_before
                after_chars += sub_after
    elif isinstance(value, list):
        for item in value:
            sub_count, sub_before, sub_after = _cap_observations_in_place(item, max_chars)
            count += sub_count
            before_chars += sub_before
            after_chars += sub_after
    return count, before_chars, after_chars


def _max_observation_chars(value: Any) -> int:
    if isinstance(value, dict):
        longest = 0
        for key, item in value.items():
            if key in {"observation", "initial_observation", "feedback", "next_observation", "env_feedback"}:
                longest = max(longest, len("" if item is None else str(item)))
            else:
                longest = max(longest, _max_observation_chars(item))
        return longest
    if isinstance(value, list):
        return max((_max_observation_chars(item) for item in value), default=0)
    return 0


def _compact_llm_payload(
    payload: Dict[str, Any],
    system_prompt: str,
    args: Any = None,
    *,
    label: str = "llm",
) -> Dict[str, Any]:
    """Build the LLM-visible payload by shrinking only observation fields.

    The raw EvidenceCards/replay sources stay untouched. Actions, trigger ranges,
    edit text, evidence reasons, and skill.md are never removed here.
    """
    compact = _jsonable_copy(payload)
    if bool(getattr(args, "llm_full_context", False)):
        return compact
    obs_cap = _int_arg(
        getattr(args, "llm_max_observation_chars", DEFAULT_LLM_MAX_OBSERVATION_CHARS)
        if args is not None
        else DEFAULT_LLM_MAX_OBSERVATION_CHARS,
        DEFAULT_LLM_MAX_OBSERVATION_CHARS,
    )
    max_input_chars = _int_arg(
        getattr(args, "llm_max_input_chars", DEFAULT_LLM_MAX_INPUT_CHARS)
        if args is not None
        else DEFAULT_LLM_MAX_INPUT_CHARS,
        DEFAULT_LLM_MAX_INPUT_CHARS,
    )
    if obs_cap >= 0:
        _cap_observations_in_place(compact, obs_cap)

    current_chars = _llm_input_chars(system_prompt, compact)
    if max_input_chars <= 0 or current_chars <= max_input_chars:
        return compact

    base = _jsonable_copy(compact)
    longest = _max_observation_chars(base)
    best_payload = _jsonable_copy(base)
    best_cap = longest
    if longest > 0:
        low = 0
        high = longest
        found = False
        while low <= high:
            mid = (low + high) // 2
            candidate = _jsonable_copy(base)
            _cap_observations_in_place(candidate, mid)
            candidate_chars = _llm_input_chars(system_prompt, candidate)
            if candidate_chars <= max_input_chars:
                best_payload = candidate
                best_cap = mid
                found = True
                low = mid + 1
            else:
                high = mid - 1
        if not found:
            best_payload = _jsonable_copy(base)
            _cap_observations_in_place(best_payload, 0)
            best_cap = 0

    final_chars = _llm_input_chars(system_prompt, best_payload)
    if final_chars > max_input_chars:
        print(
            f"[llm compact warning] {label} input_chars={current_chars}->{final_chars} "
            f"budget={max_input_chars} obs_cap=0 reason=non_observation_payload_too_large",
            flush=True,
        )
    else:
        print(
            f"[llm compact] {label} input_chars={current_chars}->{final_chars} "
            f"budget={max_input_chars} obs_cap={best_cap}",
            flush=True,
        )
    return best_payload


def _appworld_clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if max_chars < 0 or len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    marker = "\n...[omitted]...\n"
    side = max(1, (max_chars - len(marker)) // 2)
    return (text[:side] + marker + text[-side:])[:max_chars]


def _appworld_relevant_skill_sections(
    skill_md: str,
    *,
    candidate_edit: Dict[str, Any],
    linked_evidence: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    """Select complete Markdown blocks relevant to an AppWorld replay edit."""
    text = str(skill_md or "")
    if max_chars < 0 or len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""

    lines = text.splitlines(keepends=True)
    starts = [idx for idx, line in enumerate(lines) if _markdown_heading_level(line) is not None]
    if not starts:
        return _appworld_clip_text(text, max_chars)
    if starts[0] != 0:
        starts.insert(0, 0)
    blocks = [
        "".join(lines[start : starts[idx + 1] if idx + 1 < len(starts) else len(lines)]).strip()
        for idx, start in enumerate(starts)
    ]
    blocks = [block for block in blocks if block]
    if not blocks:
        return _appworld_clip_text(text, max_chars)

    target = str((candidate_edit or {}).get("target") or "").strip()
    relevance_parts = [
        target,
        str((candidate_edit or {}).get("content") or ""),
    ]
    for card in linked_evidence or []:
        relevance_parts.append(str(card.get("pattern") or ""))
        proposed = card.get("proposed_edit") or {}
        if isinstance(proposed, dict):
            relevance_parts.extend(
                [str(proposed.get("target") or ""), str(proposed.get("content") or "")]
            )
    relevance_words = {
        word.lower()
        for word in _WORD_RE.findall("\n".join(relevance_parts))
        if len(word) >= 4 and word.lower() not in _FEEDBACK_STOPWORDS
    }

    ranked: List[Tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        lower = block.lower()
        score = sum(1 for word in relevance_words if word in lower)
        if target and target in block:
            score += 10000
        if index == 0:
            score += 1
        ranked.append((score, index, block))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected: List[Tuple[int, str]] = []
    used = 0
    for _, index, block in ranked:
        separator = 2 if selected else 0
        if used + separator + len(block) <= max_chars:
            selected.append((index, block))
            used += separator + len(block)
    if not selected:
        best = ranked[0][2]
        return _appworld_clip_text(best, max_chars)
    selected.sort(key=lambda item: item[0])
    rendered = "\n\n".join(block for _, block in selected)
    if len(selected) < len(blocks):
        marker = "\n\n...[unrelated skill sections omitted]..."
        if len(rendered) + len(marker) <= max_chars:
            rendered += marker
    return rendered


def _compact_appworld_l2_step(
    step: Dict[str, Any],
    *,
    code_chars: int,
    output_chars: int,
) -> Dict[str, Any]:
    """Keep the causal code/output pair while bounding one AppWorld step."""
    out: Dict[str, Any] = {
        "step": step.get("step"),
        "action": _compress_appworld_code(str(step.get("action") or step.get("code") or ""), code_chars),
        "feedback": _appworld_clip_text(
            _compress_feedback(str(step.get("feedback") or step.get("output") or ""), output_chars, keywords=set()),
            output_chars,
        ),
    }
    for key in ("execution_ok", "task_completed", "done"):
        if key in step:
            out[key] = bool(step.get(key))
    if "observation" in step:
        out["observation"] = _appworld_clip_text(step.get("observation"), output_chars)
    return out


def _compact_appworld_l2_segment(
    segment: Any,
    *,
    step_limit: int,
    code_chars: int,
    output_chars: int,
) -> Dict[str, Any]:
    if not isinstance(segment, dict):
        return {"task_id": "", "steps": []}
    raw_steps = [step for step in (segment.get("steps") or []) if isinstance(step, dict)]
    if step_limit > 0 and len(raw_steps) > step_limit:
        head = max(1, step_limit // 2)
        tail = max(0, step_limit - head)
        selected = raw_steps[:head] + (raw_steps[-tail:] if tail else [])
        omitted = len(raw_steps) - len(selected)
    elif step_limit == 0:
        selected = []
        omitted = len(raw_steps)
    else:
        selected = raw_steps
        omitted = 0
    out: Dict[str, Any] = {
        "task_id": str(segment.get("task_id") or ""),
        "steps": [
            _compact_appworld_l2_step(
                step,
                code_chars=code_chars,
                output_chars=output_chars,
            )
            for step in selected
        ],
    }
    if segment.get("initial_observation"):
        out["initial_observation"] = _appworld_clip_text(
            segment.get("initial_observation"), output_chars
        )
    if omitted:
        out["omitted_steps"] = omitted
    return out


def _compact_appworld_l2_evidence(
    card: Dict[str, Any],
    *,
    step_limit: int,
    code_chars: int,
    output_chars: int,
    edit_chars: int,
    include_before_segments: bool = True,
    full_proposed_edit: bool = False,
    full_pattern: bool = False,
) -> Dict[str, Any]:
    allowed = (
        "evidence_schema_version",
        "evidence_id",
        "evidence_mode",
        "observed_skill_revision",
        "parent_skill_revision",
        "working_skill_revision",
        "comparison_type",
        "origin_edit_ids",
        "source_type",
        "experience_type",
        "failure_type",
        "support_count",
        "merge_level",
        "trigger_ranges",
    )
    out = {key: card.get(key) for key in allowed if key in card}
    for key in ("pattern", "evidence_quality_judge_reason"):
        if card.get(key) not in (None, ""):
            out[key] = (
                str(card.get(key) or "")
                if full_pattern and key == "pattern"
                else _appworld_clip_text(card.get(key), 700)
            )
    proposed = card.get("proposed_edit")
    if isinstance(proposed, dict):
        out["proposed_edit"] = {
            key: str(value or "")
            if full_proposed_edit and key in {"target", "content"}
            else _appworld_clip_text(value, edit_chars)
            if key in {"target", "content"}
            else value
            for key, value in proposed.items()
        }
    supporting = []
    for item in card.get("supporting_tasks") or []:
        if isinstance(item, dict):
            supporting.append(
                {
                    "task_id": item.get("task_id"),
                    "task_family": item.get("task_family"),
                }
            )
    if supporting:
        out["supporting_tasks"] = supporting
    if include_before_segments:
        out["before_segments"] = [
            _compact_appworld_l2_segment(
                segment,
                step_limit=step_limit,
                code_chars=code_chars,
                output_chars=output_chars,
            )
            for segment in (card.get("before_segments") or [])
        ]
    return out


def _compact_appworld_l2_judge_payload(
    payload: Dict[str, Any],
    system_prompt: str,
    args: Any = None,
) -> Dict[str, Any]:
    """Compact AppWorld judge context without dropping all execution output."""
    if bool(getattr(args, "llm_full_context", False)):
        return _jsonable_copy(payload)
    budget = _int_arg(
        getattr(args, "llm_max_input_chars", DEFAULT_LLM_MAX_INPUT_CHARS)
        if args is not None
        else DEFAULT_LLM_MAX_INPUT_CHARS,
        DEFAULT_LLM_MAX_INPUT_CHARS,
    )
    raw_chars = _llm_input_chars(system_prompt, payload)

    linked = payload.get("linked_evidence") or []
    replay = payload.get("replay_evidence") or []
    candidate_edit = payload.get("candidate_edit") or {}
    replay_evidence_ids = {
        str(row.get("evidence_id") or "")
        for row in replay
        if str(row.get("evidence_id") or "")
    }
    variants = [
        (-1, -1, -1, 1600),
        (APPWORLD_L2_STEP_LIMIT, APPWORLD_L2_CODE_MAX_CHARS, APPWORLD_L2_OUTPUT_MAX_CHARS, 1600),
        (6, 300, 500, 1200),
        (4, 240, 400, 900),
        (3, 180, 300, 650),
        (2, 140, 240, 450),
        (1, 220, 180, 300),
    ]
    compact = payload
    selected_variant = variants[-1]
    for step_limit, code_chars, output_chars, edit_chars in variants:
        candidate = {
            "dataset": payload.get("dataset"),
            "skill_md": payload.get("skill_md", ""),
            "candidate_edit": {
                key: str(value or "") if key in {"target", "content"} else value
                for key, value in candidate_edit.items()
            },
            "linked_evidence": [
                _compact_appworld_l2_evidence(
                    card,
                    step_limit=step_limit,
                    code_chars=code_chars,
                    output_chars=output_chars,
                    edit_chars=edit_chars,
                    include_before_segments=str(card.get("evidence_id") or "") in replay_evidence_ids,
                    full_proposed_edit=str(card.get("evidence_id") or "") in replay_evidence_ids,
                    full_pattern=str(card.get("evidence_id") or "") in replay_evidence_ids,
                )
                for card in linked
            ],
            "replay_evidence": [],
        }
        if "reflection_allowed" in payload:
            candidate["reflection_allowed"] = bool(payload.get("reflection_allowed"))
        for key in (
            "replay_mode",
            "replay_attribution_allowed",
            "judge_mode_instruction",
        ):
            if key in payload:
                candidate[key] = payload.get(key)
        if isinstance(payload.get("judge_output"), dict):
            candidate["judge_output"] = dict(payload.get("judge_output") or {})
        for row in replay:
            item = {
                "evidence_id": row.get("evidence_id"),
                "range_index": row.get("range_index"),
                "trigger_ranges": row.get("trigger_ranges") or [],
                "before_segments": [
                    _compact_appworld_l2_segment(
                        segment,
                        step_limit=step_limit,
                        code_chars=code_chars,
                        output_chars=output_chars,
                    )
                    for segment in (row.get("before_segments") or [])
                ],
                "after_segments": [
                    _compact_appworld_l2_segment(
                        segment,
                        step_limit=step_limit,
                        code_chars=code_chars,
                        output_chars=output_chars,
                    )
                    for segment in (row.get("after_segments") or [])
                ],
            }
            if row.get("error") not in (None, ""):
                item["error"] = _appworld_clip_text(row.get("error"), 500)
            if isinstance(row.get("replay_outcome"), dict):
                item["replay_outcome"] = row.get("replay_outcome")
            candidate["replay_evidence"].append(item)
        compact = candidate
        selected_variant = (step_limit, code_chars, output_chars, edit_chars)
        if budget <= 0 or _llm_input_chars(system_prompt, candidate) <= budget:
            break

    if budget > 0 and _llm_input_chars(system_prompt, compact) > budget:
        without_skill = dict(compact)
        without_skill["skill_md"] = ""
        skill_budget = max(0, budget - _llm_input_chars(system_prompt, without_skill) - 200)
        compact["skill_md"] = _appworld_relevant_skill_sections(
            str(compact.get("skill_md") or ""),
            candidate_edit=candidate_edit,
            linked_evidence=linked,
            max_chars=skill_budget,
        )
    if budget > 0 and _llm_input_chars(system_prompt, compact) > budget:
        for card in compact.get("linked_evidence") or []:
            if str(card.get("evidence_id") or "") in replay_evidence_ids:
                continue
            proposed = card.get("proposed_edit")
            if not isinstance(proposed, dict):
                continue
            for key in ("target", "content"):
                if key in proposed:
                    proposed[key] = _appworld_clip_text(proposed.get(key), 180)

    final_chars = _llm_input_chars(system_prompt, compact)
    if final_chars != raw_chars or (budget > 0 and final_chars > budget):
        print(
            f"[appworld compact] skill edit judge input_chars={raw_chars}->{final_chars} "
            f"budget={budget} steps={selected_variant[0]} code_cap={selected_variant[1]} "
            f"output_cap={selected_variant[2]}",
            flush=True,
        )
    return compact


_CRITICAL_FEEDBACK_RE = re.compile(
    r"(?:no known action|invalid|cannot|can't|could not|unable|not possible|"
    r"which do you mean|choose|option|error|fail|temperature|thermometer|degree|"
    r"\b\d+(?:\.\d+)?\s*(?:degrees?|celsius|fahrenheit|°[cf]?|kg|g|ml|l|cm|mm|"
    r"volts?|amps?)\b|solid|liquid|gas|steam|boil|melt|frozen|freeze|score|"
    r"complete|success)",
    re.IGNORECASE,
)
_PRIORITY_FEEDBACK_RE = re.compile(
    r"(?:lit|unlit|turned on|turned off|open|closed|contain|inside|connect|disconnect|"
    r"activate|deactivate|focus|found|see|notice|read)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_FEEDBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "the", "then", "this", "to", "use", "with", "you",
    "your", "task", "action", "look", "around", "examine", "read",
}


def _feedback_from_trajectory_step(trajectory: List[Dict[str, Any]], index: int) -> str:
    row = trajectory[index] if 0 <= index < len(trajectory) else {}
    for key in ("env_feedback", "next_observation"):
        value = str((row or {}).get(key) or "").strip()
        if value:
            return value
    if index + 1 < len(trajectory):
        return str((trajectory[index + 1] or {}).get("observation") or "").strip()
    return ""


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_transition_from_step(
    row: Dict[str, Any],
    *,
    previous_score: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    before = _optional_float(row.get("score_before"))
    after = _optional_float(row.get("score_after"))
    delta = _optional_float(row.get("score_delta"))

    # Schema-v3 replay sources stored only cumulative per-step score.
    if after is None and "score" in row:
        after = _optional_float(row.get("score"))
    if before is None and previous_score is not None:
        before = float(previous_score)
    if delta is None and before is not None and after is not None:
        delta = round(after - before, 10)
    return before, after, delta


def _format_score_transition(
    before: Optional[float],
    after: Optional[float],
    delta: Optional[float],
) -> str:
    if before is None or after is None or delta is None:
        return "before=unavailable after=unavailable delta=unavailable"
    return f"before={before:g} after={after:g} delta={delta:+g}"


def _feedback_keywords(task_instruction: str, action: str) -> set:
    words = {
        word.lower()
        for word in _WORD_RE.findall(f"{task_instruction} {action}")
        if len(word) >= 3 and word.lower() not in _FEEDBACK_STOPWORDS
    }
    return words


def _feedback_fragments(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    fragments: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        fragments.extend(part.strip() for part in parts if part.strip())
    return fragments or [text]


def _normalized_feedback_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _clip_feedback_fragment(fragment: str, max_chars: int, keywords: set) -> str:
    fragment = str(fragment or "").strip()
    if max_chars <= 0:
        return ""
    if len(fragment) <= max_chars:
        return fragment
    lower = fragment.lower()
    positions = [lower.find(word) for word in keywords if lower.find(word) >= 0]
    match = _CRITICAL_FEEDBACK_RE.search(fragment)
    if match:
        positions.append(match.start())
    if positions and max_chars > 40:
        center = min(positions)
        start = max(0, center - max_chars // 3)
        end = min(len(fragment), start + max_chars)
        start = max(0, end - max_chars)
        clipped = fragment[start:end].strip()
        return ("..." if start else "") + clipped + ("..." if end < len(fragment) else "")
    if max_chars <= 20:
        return fragment[:max_chars].rstrip()
    head = max_chars // 2 - 2
    tail = max_chars - head - 5
    return f"{fragment[:head].rstrip()} ... {fragment[-tail:].lstrip()}"


def _compress_feedback(
    text: str,
    max_chars: int,
    *,
    keywords: set,
    baseline_text: str = "",
) -> str:
    text = str(text or "").strip()
    if max_chars < 0 or len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    fragments = _feedback_fragments(text)
    if not fragments:
        return ""
    baseline_normalized = re.sub(r"\s+", " ", str(baseline_text or "")).strip().lower()
    scored: List[Tuple[int, int, str]] = []
    for index, fragment in enumerate(fragments):
        lower = fragment.lower()
        score = 0
        if index in {0, len(fragments) - 1}:
            score += 1
        if _CRITICAL_FEEDBACK_RE.search(fragment):
            score += 5
        elif _PRIORITY_FEEDBACK_RE.search(fragment):
            score += 3
        if any(word in lower for word in keywords):
            score += 3
        normalized_fragment = re.sub(r"\s+", " ", fragment).strip().lower()
        if baseline_normalized and normalized_fragment and normalized_fragment not in baseline_normalized:
            score += 4
        if fragment.startswith(("\t", "-", "*")):
            score += 1
        scored.append((score, index, fragment))

    selected: Dict[int, str] = {}
    remaining = max_chars
    for _, index, fragment in sorted(scored, key=lambda item: (-item[0], item[1])):
        separator = 1 if selected else 0
        if remaining <= separator:
            break
        clipped = _clip_feedback_fragment(fragment, remaining - separator, keywords)
        if not clipped:
            continue
        selected[index] = clipped
        remaining -= len(clipped) + separator
    rendered = "\n".join(selected[index] for index in sorted(selected))
    if len(rendered) < len(text) and len(rendered) + 14 <= max_chars:
        rendered += "\n...[omitted]"
    return rendered[:max_chars].rstrip()


def _is_critical_feedback_step(
    row: Dict[str, Any],
    *,
    index: int,
    trajectory_size: int,
    feedback: str,
    score_delta: Optional[float],
    ignore_bad_step: bool = False,
) -> bool:
    action = str(row.get("action") or "").strip().lower()
    score_changed = score_delta is not None and score_delta != 0.0
    return bool(
        index == trajectory_size - 1
        or row.get("done")
        or (row.get("bad_step") and not ignore_bad_step)
        or score_changed
        or action.startswith(("look around", "examine ", "read "))
        or _CRITICAL_FEEDBACK_RE.search(feedback or "")
    )


def _is_priority_feedback_step(
    row: Dict[str, Any],
    *,
    feedback: str,
    task_instruction: str,
    baseline_text: str,
) -> bool:
    action = str(row.get("action") or "").strip().lower()
    lower_feedback = str(feedback or "").lower()
    keywords = _feedback_keywords(task_instruction, action)
    baseline_normalized = re.sub(r"\s+", " ", str(baseline_text or "")).strip().lower()
    has_novel_fragment = any(
        (normalized := re.sub(r"\s+", " ", fragment).strip().lower())
        and normalized not in baseline_normalized
        for fragment in _feedback_fragments(feedback)
    ) if baseline_normalized else bool(str(feedback or "").strip())
    return bool(
        action.startswith(("look around", "examine ", "look at ", "read "))
        or _PRIORITY_FEEDBACK_RE.search(feedback or "")
        or any(word in lower_feedback for word in keywords)
        or has_novel_fragment
    )


def _render_l1_trajectories(
    results: List[Dict[str, Any]],
    *,
    ordinary_cap: int,
    priority_cap: int,
    critical_cap: int,
    appworld_code_cap: int = APPWORLD_L1_CODE_MAX_CHARS,
    dataset: str = "",
    full_context: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    if _is_appworld_dataset(dataset):
        return _render_appworld_l1_trajectories(
            results,
            ordinary_cap=ordinary_cap,
            priority_cap=priority_cap,
            critical_cap=critical_cap,
            code_cap=appworld_code_cap,
            full_context=full_context,
        )
    parts: List[str] = []
    raw_feedback_chars = 0
    visible_feedback_chars = 0
    critical_feedback_chars = 0
    priority_feedback_chars = 0
    ordinary_raw_chars = 0
    ordinary_visible_chars = 0
    raw_feedback_count = 0
    visible_feedback_count = 0
    deduplicated_steps = 0
    raw_bad_step_count = 0

    for trajectory_index, result in enumerate(results or [], start=1):
        task_id = str(result.get("task_id") or result.get("id") or "")
        instruction = str(result.get("instruction") or result.get("task_description") or result.get("task") or "")
        family = str(result.get("task_family") or result.get("task_type") or "")
        trajectory = [row for row in (result.get("trajectory") or []) if isinstance(row, dict)]
        success = _is_success_result(result)
        score = float(result.get("score", result.get("final_reward", 0.0)) or 0.0)
        header = [
            f"### Trajectory {trajectory_index} id={task_id}",
            f"Task: {instruction}",
            f"Family: {family}",
            f"Outcome: success={str(success).lower()} score={score} steps={len(trajectory)}",
        ]
        if trajectory:
            initial_raw = str(trajectory[0].get("observation") or "").strip()
            if full_context:
                initial_visible = initial_raw
            else:
                initial_keywords = _feedback_keywords(instruction, "")
                initial_visible = _compress_feedback(
                    initial_raw,
                    critical_cap,
                    keywords=initial_keywords,
                )
            header.append(f"[initial observation] {initial_visible or '[omitted]'}")

        seen_feedback_steps: Dict[str, int] = {}
        previous_score: Optional[float] = None
        for index, row in enumerate(trajectory):
            step = _coerce_int(row.get("step", index), index)
            action = str(row.get("action") or "")
            feedback = _feedback_from_trajectory_step(trajectory, index)
            score_before, score_after, score_delta = _score_transition_from_step(
                row,
                previous_score=previous_score,
            )
            raw_feedback_chars += len(feedback)
            if feedback:
                raw_feedback_count += 1
            if bool(row.get("bad_step")):
                raw_bad_step_count += 1
            baseline_text = str(row.get("observation") or "")
            feedback_for_optimizer = feedback
            baseline_for_optimizer = baseline_text
            normalized_feedback = _normalized_feedback_text(feedback_for_optimizer)
            if full_context:
                critical = _is_critical_feedback_step(
                    row,
                    index=index,
                    trajectory_size=len(trajectory),
                    feedback=feedback_for_optimizer,
                    score_delta=score_delta,
                    ignore_bad_step=False,
                )
                priority = not critical and _is_priority_feedback_step(
                    row,
                    feedback=feedback_for_optimizer,
                    task_instruction=instruction,
                    baseline_text=baseline_for_optimizer,
                )
                rendered_feedback = feedback
            else:
                if normalized_feedback and normalized_feedback in seen_feedback_steps:
                    rendered_feedback = f"[same as step {seen_feedback_steps[normalized_feedback]}]"
                    deduplicated_steps += 1
                    critical = False
                    priority = False
                else:
                    critical = _is_critical_feedback_step(
                        row,
                        index=index,
                        trajectory_size=len(trajectory),
                        feedback=feedback_for_optimizer,
                        score_delta=score_delta,
                        ignore_bad_step=False,
                    )
                    priority = not critical and _is_priority_feedback_step(
                        row,
                        feedback=feedback_for_optimizer,
                        task_instruction=instruction,
                        baseline_text=baseline_for_optimizer,
                    )
                    cap = critical_cap if critical else (priority_cap if priority else ordinary_cap)
                    rendered_feedback = _compress_feedback(
                        feedback,
                        cap,
                        keywords=_feedback_keywords(instruction, action),
                        baseline_text=baseline_text,
                    )
                    if normalized_feedback:
                        seen_feedback_steps[normalized_feedback] = step
            if rendered_feedback:
                visible_feedback_count += 1
            visible_feedback_chars += len(rendered_feedback)
            if critical:
                critical_feedback_chars += len(rendered_feedback)
            elif priority:
                priority_feedback_chars += len(rendered_feedback)
            else:
                ordinary_raw_chars += len(feedback)
                ordinary_visible_chars += len(rendered_feedback)
            header.append(f"[step {step} action] {action}")
            header.append(f"[step {step} feedback] {rendered_feedback or '[omitted]'}")
            header.append(
                f"[step {step} score] "
                f"{_format_score_transition(score_before, score_after, score_delta)}"
            )
            if bool(row.get("done")):
                header.append(
                    f"[step {step} status] "
                    f"done=true bad_step={str(bool(row.get('bad_step'))).lower()}"
                )
            elif bool(row.get("bad_step")):
                header.append(
                    f"[step {step} status] "
                    f"done=false "
                    f"bad_step={str(bool(row.get('bad_step'))).lower()}"
                )
            if score_after is not None:
                previous_score = score_after
        parts.append("\n".join(header))

    stats = {
        "raw_feedback_chars": raw_feedback_chars,
        "visible_feedback_chars": visible_feedback_chars,
        "critical_feedback_chars": critical_feedback_chars,
        "priority_feedback_chars": priority_feedback_chars,
        "deduplicated_steps": deduplicated_steps,
        "ordinary_feedback_omitted_chars": max(0, ordinary_raw_chars - ordinary_visible_chars),
        "post_action_feedback_coverage": (
            visible_feedback_count / raw_feedback_count if raw_feedback_count else 1.0
        ),
        "raw_feedback_count": raw_feedback_count,
        "visible_feedback_count": visible_feedback_count,
        "ordinary_feedback_cap": ordinary_cap,
        "priority_feedback_cap": priority_cap,
        "critical_feedback_cap": critical_cap,
        "raw_bad_step_count": raw_bad_step_count,
        "full_context": bool(full_context),
    }
    return "\n\n---\n\n".join(parts), stats


_APPWORLD_CRITICAL_OUTPUT_RE = re.compile(
    r"(?:execution failed|traceback|syntax error|timed out|no code available|"
    r"marked the active task complete|maximum number of executions)",
    re.IGNORECASE,
)
_APPWORLD_PRIORITY_OUTPUT_RE = re.compile(
    r'(?:"parameters"|"response_schemas"|"api_name"|"access_token"|'
    r'"page_index"|"message"|show_api_doc|show_api_descriptions)',
    re.IGNORECASE,
)
_APPWORLD_IMPORTANT_CODE_RE = re.compile(
    r"(?:\bapis\.[A-Za-z_][\w.]*\s*\(|\b(?:for|while|if|elif|else|try|except|with)\b|"
    r"\b(?:break|continue|return)\b|\b(?:access_token|page_index|page_limit|task_id)\b|"
    r"\b(?:print|sorted|max|min|sum|set|list|dict)\s*\()"
)


def _compress_appworld_code(code: str, max_chars: int) -> str:
    code = str(code or "").strip()
    if max_chars < 0 or len(code) <= max_chars:
        return code
    if max_chars <= 0:
        return ""

    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    if not lines:
        return ""
    scored: List[Tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        score = 0
        if "apis." in stripped:
            score += 12
        if _APPWORLD_IMPORTANT_CODE_RE.search(stripped):
            score += 7
        if "=" in stripped and not stripped.startswith("#"):
            score += 5
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            score += 4
        if index in {0, len(lines) - 1}:
            score += 2
        if stripped.startswith("#"):
            score -= 3
        scored.append((score, index, stripped))

    selected: Dict[int, str] = {}
    marker = "\n...[code omitted]"
    content_budget = max_chars - len(marker) if max_chars > len(marker) + 20 else max_chars
    remaining = content_budget
    for _, index, line in sorted(scored, key=lambda item: (-item[0], item[1])):
        separator = 1 if selected else 0
        if remaining <= separator:
            break
        clipped = _clip_feedback_fragment(line, remaining - separator, set())
        if not clipped:
            continue
        selected[index] = clipped
        remaining -= len(clipped) + separator

    rendered = "\n".join(selected[index] for index in sorted(selected))
    if len(rendered) < len(code) and len(rendered) + len(marker) <= max_chars:
        rendered += marker
    return rendered[:max_chars].rstrip()


def _render_appworld_l1_trajectories(
    results: List[Dict[str, Any]],
    *,
    ordinary_cap: int,
    priority_cap: int,
    critical_cap: int,
    code_cap: int,
    full_context: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    from .appworld_runner import redact_appworld_optimizer_value

    parts: List[str] = []
    raw_chars = 0
    visible_chars = 0
    critical_chars = 0
    priority_chars = 0
    ordinary_raw = 0
    ordinary_visible = 0
    raw_count = 0
    visible_count = 0
    deduplicated = 0
    raw_bad_steps = 0
    raw_code_chars = 0
    visible_code_chars = 0
    for trajectory_index, result in enumerate(results or [], start=1):
        task_id = str(result.get("task_id") or result.get("id") or "")
        instruction = str(result.get("instruction") or result.get("task") or "")
        family = str(result.get("task_family") or "")
        trajectory = [row for row in (result.get("trajectory") or []) if isinstance(row, dict)]
        evaluation = result.get("evaluation_summary") if isinstance(result.get("evaluation_summary"), dict) else {}
        header = [
            f"### Trajectory {trajectory_index} id={task_id}",
            f"Task: {instruction}",
            f"Family: {family}",
            (
                f"Outcome: success={str(bool(result.get('success'))).lower()} "
                f"score={float(result.get('score', 0.0) or 0.0):g} "
                f"steps={len(trajectory)} task_completed={str(bool(result.get('task_completed'))).lower()}"
            ),
        ]
        seen_outputs: Dict[str, int] = {}
        for index, row in enumerate(trajectory):
            step = _coerce_int(row.get("step", index), index)
            code = str(redact_appworld_optimizer_value(str(row.get("action") or "")))
            rendered_code = code if full_context else _compress_appworld_code(code, code_cap)
            raw_code_chars += len(code)
            visible_code_chars += len(rendered_code)
            output = str(
                redact_appworld_optimizer_value(
                    str(row.get("next_observation") or row.get("env_feedback") or "")
                )
            )
            raw_chars += len(output)
            raw_count += int(bool(output))
            raw_bad_steps += int(bool(row.get("bad_step")))
            normalized = re.sub(r"\s+", " ", output).strip()
            critical = bool(
                row.get("bad_step")
                or row.get("task_completed")
                or row.get("done")
                or _APPWORLD_CRITICAL_OUTPUT_RE.search(output)
                or index == len(trajectory) - 1
            )
            priority = not critical and bool(_APPWORLD_PRIORITY_OUTPUT_RE.search(output))
            if full_context:
                rendered = output
            elif normalized and normalized in seen_outputs:
                rendered = f"[same as step {seen_outputs[normalized]}]"
                deduplicated += 1
                critical = False
                priority = False
            else:
                cap = critical_cap if critical else (priority_cap if priority else ordinary_cap)
                rendered = _compress_feedback(
                    output,
                    cap,
                    keywords=_feedback_keywords(instruction, code),
                )
                if normalized:
                    seen_outputs[normalized] = step
            visible_chars += len(rendered)
            visible_count += int(bool(rendered))
            if critical:
                critical_chars += len(rendered)
            elif priority:
                priority_chars += len(rendered)
            else:
                ordinary_raw += len(output)
                ordinary_visible += len(rendered)
            header.extend(
                [
                    f"[step {step} code]\n{rendered_code or '[omitted]'}",
                    f"[step {step} output]\n{rendered or '[omitted]'}",
                    (
                        f"[step {step} status] "
                        f"execution_ok={str(bool(row.get('execution_ok', not row.get('bad_step')))).lower()} "
                        f"task_completed={str(bool(row.get('task_completed') or row.get('done'))).lower()}"
                    ),
                ]
            )
        num_tests = int(evaluation.get("num_tests", 0) or 0)
        passed_tests = int(evaluation.get("passed_tests", 0) or 0)
        header.append(
            f"[final evaluation] success={str(bool(result.get('success'))).lower()} "
            f"passed={passed_tests}/{num_tests}"
        )
        labels = result.get("train_evaluation_requirements")
        if isinstance(labels, list) and labels:
            anonymous = ", ".join(
                f"{str(item.get('label') or '')}={'pass' if item.get('passed') else 'fail'}"
                for item in labels
                if isinstance(item, dict)
            )
            if anonymous:
                header.append(f"[anonymous requirements] {anonymous}")
        parts.append("\n\n".join(header))
    return "\n\n---\n\n".join(parts), {
        "raw_feedback_chars": raw_chars,
        "visible_feedback_chars": visible_chars,
        "critical_feedback_chars": critical_chars,
        "priority_feedback_chars": priority_chars,
        "deduplicated_steps": deduplicated,
        "ordinary_feedback_omitted_chars": max(0, ordinary_raw - ordinary_visible),
        "post_action_feedback_coverage": visible_count / raw_count if raw_count else 1.0,
        "raw_feedback_count": raw_count,
        "visible_feedback_count": visible_count,
        "ordinary_feedback_cap": ordinary_cap,
        "priority_feedback_cap": priority_cap,
        "critical_feedback_cap": critical_cap,
        "raw_bad_step_count": raw_bad_steps,
        "raw_code_chars": raw_code_chars,
        "visible_code_chars": visible_code_chars,
        "code_cap": code_cap,
        "full_context": bool(full_context),
    }


def _build_l1_evidence_payload(
    *,
    skill_md: str,
    source_type: str,
    results: List[Dict[str, Any]],
    system_prompt: str,
    args: Any,
    dataset: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full_context = bool(getattr(args, "llm_full_context", False))
    ordinary_max = _int_arg(
        getattr(args, "llm_max_observation_chars", DEFAULT_LLM_MAX_OBSERVATION_CHARS),
        DEFAULT_LLM_MAX_OBSERVATION_CHARS,
    )
    if ordinary_max < 0:
        ordinary_max = L1_CRITICAL_FEEDBACK_MAX_CHARS
    priority_max = max(ordinary_max, 800)
    critical_max = max(L1_CRITICAL_FEEDBACK_MAX_CHARS, priority_max)
    input_budget = _int_arg(
        getattr(args, "llm_max_input_chars", DEFAULT_LLM_MAX_INPUT_CHARS),
        DEFAULT_LLM_MAX_INPUT_CHARS,
    )
    if full_context:
        input_budget = 0
    is_appworld = _is_appworld_dataset(dataset)

    def build(
        ordinary_cap: int,
        priority_cap: int,
        critical_cap: int,
        appworld_code_cap: int = APPWORLD_L1_CODE_MAX_CHARS,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        text, stats = _render_l1_trajectories(
            results,
            ordinary_cap=max(0, ordinary_cap),
            priority_cap=max(80, priority_cap),
            critical_cap=max(L1_CRITICAL_FEEDBACK_MIN_CHARS, critical_cap),
            appworld_code_cap=max(0, appworld_code_cap),
            dataset=dataset,
            full_context=full_context,
        )
        payload = {
            "skill_md": skill_md,
            "source_type": source_type,
            "max_proposed_edits": L1_MAX_PROPOSED_EDITS,
            "trajectories_text": text,
        }
        stats["input_chars"] = _llm_input_chars(system_prompt, payload)
        return payload, stats

    payload, stats = build(
        ordinary_max,
        priority_max,
        critical_max,
        APPWORLD_L1_CODE_MAX_CHARS,
    )
    if not full_context and is_appworld and input_budget > 0 and stats["input_chars"] > input_budget:
        # AppWorld Python can be much longer than its execution result. Reclaim
        # code budget first so API schemas, errors, and mutation results remain
        # visible to the analyst.
        low, high = APPWORLD_L1_CODE_MIN_CHARS, APPWORLD_L1_CODE_MAX_CHARS
        best: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        while low <= high:
            mid = (low + high) // 2
            candidate = build(ordinary_max, priority_max, critical_max, mid)
            if candidate[1]["input_chars"] <= input_budget:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        if best is not None:
            payload, stats = best
        else:
            code_cap = APPWORLD_L1_CODE_MIN_CHARS
            cap_sets = [
                (max(APPWORLD_L1_OUTPUT_MIN_CHARS, ordinary_max), priority_max, critical_max),
                (APPWORLD_L1_OUTPUT_MIN_CHARS, max(240, priority_max // 2), critical_max),
                (APPWORLD_L1_OUTPUT_MIN_CHARS, 240, max(500, critical_max // 2)),
                (80, 120, L1_CRITICAL_FEEDBACK_MIN_CHARS),
            ]
            for ordinary_cap, priority_cap, critical_cap in cap_sets:
                candidate = build(ordinary_cap, priority_cap, critical_cap, code_cap)
                payload, stats = candidate
                if candidate[1]["input_chars"] <= input_budget:
                    break
    elif not full_context and input_budget > 0 and stats["input_chars"] > input_budget:
        low, high = 0, ordinary_max
        best: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        while low <= high:
            mid = (low + high) // 2
            candidate = build(mid, priority_max, critical_max)
            if candidate[1]["input_chars"] <= input_budget:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        if best is not None:
            payload, stats = best
        else:
            low, high = 80, priority_max
            best = None
            while low <= high:
                mid = (low + high) // 2
                candidate = build(0, mid, critical_max)
                if candidate[1]["input_chars"] <= input_budget:
                    best = candidate
                    low = mid + 1
                else:
                    high = mid - 1
            if best is not None:
                payload, stats = best
            else:
                low, high = L1_CRITICAL_FEEDBACK_MIN_CHARS, critical_max
                best = None
                while low <= high:
                    mid = (low + high) // 2
                    candidate = build(0, 80, mid)
                    if candidate[1]["input_chars"] <= input_budget:
                        best = candidate
                        low = mid + 1
                    else:
                        high = mid - 1
                payload, stats = (
                    best
                    if best is not None
                    else build(0, 80, L1_CRITICAL_FEEDBACK_MIN_CHARS)
                )

    stats["input_budget"] = input_budget
    stats["essential_payload_over_budget"] = bool(
        input_budget > 0 and stats["input_chars"] > input_budget
    )
    stats["full_context"] = full_context
    return payload, stats


def _materialization_record_from_result(result: Dict[str, Any], *, dataset: str = "") -> Dict[str, Any]:
    trajectory = [row for row in (result.get("trajectory") or []) if isinstance(row, dict)]
    result_dataset = dataset or str(result.get("dataset") or "")
    if _is_appworld_dataset(result_dataset):
        from .appworld_runner import redact_appworld_optimizer_value

        steps = []
        for index, row in enumerate(trajectory):
            steps.append(
                {
                    "step": _coerce_int(row.get("step", index), index),
                    "observation": str(
                        redact_appworld_optimizer_value(str(row.get("observation") or ""))
                    ),
                    "action": str(
                        redact_appworld_optimizer_value(str(row.get("action") or ""))
                    ),
                    "feedback": str(
                        redact_appworld_optimizer_value(
                            _feedback_from_trajectory_step(trajectory, index)
                        )
                    ),
                    "execution_ok": bool(row.get("execution_ok", not row.get("bad_step"))),
                    "task_completed": bool(row.get("task_completed") or row.get("done")),
                    "done": bool(row.get("done")),
                }
            )
        return {
            "task_id": str(result.get("task_id") or result.get("id") or ""),
            "instruction": str(result.get("instruction") or result.get("task") or ""),
            "task_family": str(result.get("task_family") or ""),
            "outcome": _task_outcome_from_result(result, dataset=result_dataset),
            "initial_observation": str(
                redact_appworld_optimizer_value(
                    str((trajectory[0] if trajectory else {}).get("observation") or "")
                )
            ),
            "trajectory": steps,
        }
    steps = []
    previous_score: Optional[float] = None
    for index, row in enumerate(trajectory):
        step_row = {
            "step": _coerce_int(row.get("step", index), index),
            "observation": str(row.get("observation") or ""),
            "action": str(row.get("action") or ""),
            "feedback": _feedback_from_trajectory_step(trajectory, index),
        }
        has_score_transition = any(
            key in row for key in ("score_before", "score_after", "score_delta")
        )
        if _is_scienceworld_dataset(result_dataset) or has_score_transition:
            before, after, delta = _score_transition_from_step(row, previous_score=previous_score)
            step_row.update(
                {
                    "score_before": before,
                    "score_after": after,
                    "score_delta": delta,
                    "done": bool(row.get("done")),
                }
            )
            step_row["bad_step"] = bool(row.get("bad_step"))
            if after is not None:
                previous_score = after
        steps.append(step_row)
    return {
        "task_id": str(result.get("task_id") or result.get("id") or ""),
        "instruction": str(result.get("instruction") or result.get("task_description") or result.get("task") or ""),
        "task_family": str(result.get("task_family") or result.get("task_type") or ""),
        "outcome": _task_outcome_from_result(result, dataset=result_dataset),
        "initial_observation": str((trajectory[0] if trajectory else {}).get("observation") or ""),
        "trajectory": steps,
    }


def _evidence_segment_from_steps(
    *,
    task_id: str,
    rows: List[Dict[str, Any]],
    initial_observation: Optional[str] = None,
) -> Dict[str, Any]:
    rows = [row for row in rows or [] if isinstance(row, dict)]
    materialized_steps: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        step_row: Dict[str, Any] = {
            "step": _coerce_int(row.get("step", index), index),
            "action": str(row.get("action") or ""),
            "feedback": str(row.get("feedback") or row.get("next_observation") or row.get("observation") or ""),
        }
        for key in ("score_before", "score_after", "score_delta"):
            if key in row:
                step_row[key] = _optional_float(row.get(key))
        for key in ("done", "bad_step"):
            if key in row:
                step_row[key] = bool(row.get(key))
        for key in ("execution_ok", "task_completed"):
            if key in row:
                step_row[key] = bool(row.get(key))
        materialized_steps.append(step_row)
    return {
        "task_id": str(task_id or ""),
        "initial_observation": str(
            initial_observation
            if initial_observation is not None
            else (rows[0] if rows else {}).get("observation") or ""
        ),
        "steps": materialized_steps,
    }


def _trigger_progress_from_segments(
    trigger_ranges: List[Dict[str, Any]],
    segments: List[Any],
) -> List[Dict[str, Any]]:
    progress_rows: List[Dict[str, Any]] = []
    for index, trigger in enumerate(trigger_ranges or []):
        segment = segments[index] if index < len(segments or []) else {}
        steps = _segment_steps(segment)
        available_steps = [
            step
            for step in steps
            if all(_optional_float(step.get(key)) is not None for key in ("score_before", "score_after", "score_delta"))
        ]
        progress_available = bool(steps) and len(available_steps) == len(steps)
        score_before = _optional_float(steps[0].get("score_before")) if progress_available else None
        score_after = _optional_float(steps[-1].get("score_after")) if progress_available else None
        score_delta = (
            round(float(score_after) - float(score_before), 10)
            if score_before is not None and score_after is not None
            else None
        )
        deltas = [_optional_float(step.get("score_delta")) for step in available_steps]
        progress_rows.append(
            {
                "task_id": str(trigger.get("task_id") or ""),
                "start": _coerce_int(trigger.get("start", 0), 0),
                "end": _coerce_int(trigger.get("end", trigger.get("start", 0)), 0),
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": score_delta,
                "positive_delta_steps": sum(1 for delta in deltas if delta is not None and delta > 0),
                "negative_delta_steps": sum(1 for delta in deltas if delta is not None and delta < 0),
                "progress_available": progress_available,
            }
        )
    return progress_rows


def _segment_steps(segment: Any) -> List[Dict[str, Any]]:
    if isinstance(segment, dict):
        return [row for row in (segment.get("steps") or []) if isinstance(row, dict)]
    if isinstance(segment, list):
        return [row for row in segment if isinstance(row, dict)]
    return []


def _optimizer_evidence_segment(segment: Any, *, dataset: str = "") -> Any:
    """Return the evidence segment in the form consumed by the optimizer."""
    if _is_appworld_dataset(dataset):
        from .appworld_runner import redact_appworld_optimizer_value

        return redact_appworld_optimizer_value(_jsonable_copy(segment))
    return segment


def _optimizer_evidence_card(card: Dict[str, Any], *, dataset: str = "") -> Dict[str, Any]:
    copied = _jsonable_copy(card or {})
    if _is_appworld_dataset(dataset):
        from .appworld_runner import redact_appworld_optimizer_value

        return redact_appworld_optimizer_value(copied)
    copied["before_segments"] = [
        _optimizer_evidence_segment(segment, dataset=dataset)
        for segment in (copied.get("before_segments") or [])
    ]
    return copied


def _epoch_artifact_dir(out: Path, epoch_idx: int) -> Path:
    return out / f"epoch_{int(epoch_idx) + 1}"


def _epoch_train_rollout_checkpoint_dir(out: Path, epoch_idx: int) -> Path:
    return _epoch_artifact_dir(out, epoch_idx) / TRAIN_ROLLOUT_BUNDLES_DIRNAME


def _epoch_train_rollout_checkpoint_path(out: Path, epoch_idx: int, epoch_bundle_idx: int) -> Path:
    return _epoch_train_rollout_checkpoint_dir(out, epoch_idx) / f"bundle_{int(epoch_bundle_idx):04d}.json"


def _save_epoch_train_rollout_checkpoint(
    out: Path,
    *,
    epoch_idx: int,
    epoch_bundle_idx: int,
    task_ids: List[Any],
    results: List[Dict[str, Any]],
) -> Path:
    path = _epoch_train_rollout_checkpoint_path(out, epoch_idx, epoch_bundle_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": TRAIN_ROLLOUT_CHECKPOINT_VERSION,
        "epoch_index": int(epoch_idx),
        "epoch_bundle_index": int(epoch_bundle_idx),
        "task_ids": [get_task_id(task_id) for task_id in task_ids or []],
        "results": [
            _sanitize_reused_raw_result(row)
            for row in results or []
            if isinstance(row, dict)
        ],
    }
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path


def _load_epoch_train_rollout_checkpoints(out: Path, epoch_idx: int) -> Dict[int, Dict[str, Any]]:
    directory = _epoch_train_rollout_checkpoint_dir(out, epoch_idx)
    if not directory.exists():
        return {}
    checkpoints: Dict[int, Dict[str, Any]] = {}
    for path in sorted(directory.glob("bundle_*.json")):
        try:
            payload = load_json(str(path))
        except Exception as exc:
            raise ValueError(f"Invalid train rollout checkpoint {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid train rollout checkpoint {path}: expected a JSON object")
        checkpoint_version = _row_int(payload, "checkpoint_version", -1)
        payload_epoch = _row_int(payload, "epoch_index", -1)
        bundle_idx = _row_int(payload, "epoch_bundle_index", -1)
        if (
            checkpoint_version != TRAIN_ROLLOUT_CHECKPOINT_VERSION
            or payload_epoch != int(epoch_idx)
            or bundle_idx < 0
        ):
            raise ValueError(
                f"Invalid train rollout checkpoint {path}: "
                f"version={checkpoint_version} epoch={payload_epoch} bundle={bundle_idx}"
            )
        checkpoints[bundle_idx] = payload
    return checkpoints


def _epoch_bundle_log_rows(logs: List[Dict[str, Any]], epoch_idx: int) -> List[Dict[str, Any]]:
    rows = [
        row
        for row in logs or []
        if _row_int(row or {}, "epoch_index", -1) == int(epoch_idx)
        and _row_int(row or {}, "epoch_bundle_index", -1) >= 0
    ]
    return sorted(rows, key=lambda row: _row_int(row, "epoch_bundle_index", -1))


def _epoch_l1_is_complete(logs: List[Dict[str, Any]], epoch_idx: int) -> bool:
    return any(
        isinstance((row or {}).get("epoch_l1_summary"), dict)
        for row in _epoch_bundle_log_rows(logs, epoch_idx)
    )


def _epoch_uses_reused_evidence(logs: List[Dict[str, Any]], epoch_idx: int) -> bool:
    return any(bool((row or {}).get("reused_evidence")) for row in _epoch_bundle_log_rows(logs, epoch_idx))


def _checkpoint_task_ids(payload: Dict[str, Any]) -> List[str]:
    return [get_task_id(task_id) for task_id in (payload.get("task_ids") or [])]


def _checkpoint_results_for_logged_bundles(
    out: Path,
    *,
    epoch_idx: int,
    logs: List[Dict[str, Any]],
    bundle_limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    log_rows = _epoch_bundle_log_rows(logs, epoch_idx)
    if bundle_limit is not None:
        log_rows = [
            row
            for row in log_rows
            if _row_int(row, "epoch_bundle_index", -1) < int(bundle_limit)
        ]
    try:
        checkpoints = _load_epoch_train_rollout_checkpoints(out, epoch_idx)
    except ValueError as exc:
        return [], str(exc)
    restored_results: List[Dict[str, Any]] = []
    for log_row in log_rows:
        bundle_idx = _row_int(log_row, "epoch_bundle_index", -1)
        payload = checkpoints.get(bundle_idx)
        if payload is None:
            return [], f"missing rollout checkpoint for bundle {bundle_idx + 1}"
        expected_task_ids = [get_task_id(task_id) for task_id in (log_row.get("task_ids") or [])]
        checkpoint_task_ids = _checkpoint_task_ids(payload)
        results = [row for row in (payload.get("results") or []) if isinstance(row, dict)]
        result_task_ids = [str(row.get("task_id") or row.get("id") or "") for row in results]
        if checkpoint_task_ids != expected_task_ids:
            return [], (
                f"rollout checkpoint bundle {bundle_idx + 1} task IDs do not match its log"
            )
        if result_task_ids != expected_task_ids:
            return [], (
                f"rollout checkpoint bundle {bundle_idx + 1} results do not match its task IDs"
            )
        restored_results.extend(results)
    return restored_results, ""


def _find_resume_rollout_rewind_epoch(
    out: Path,
    *,
    logs: List[Dict[str, Any]],
) -> Tuple[Optional[int], str]:
    epoch_indices = sorted(
        {
            _row_int(row or {}, "epoch_index", -1)
            for row in logs or []
            if _row_int(row or {}, "epoch_index", -1) >= 0
        }
    )
    for epoch_idx in epoch_indices:
        if _epoch_l1_is_complete(logs, epoch_idx) or _epoch_uses_reused_evidence(logs, epoch_idx):
            continue
        _, error = _checkpoint_results_for_logged_bundles(
            out,
            epoch_idx=epoch_idx,
            logs=logs,
        )
        if error:
            return epoch_idx, error
    return None, ""


def _parse_epoch_artifact_dir(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("epoch_"):
        return None
    try:
        value = int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return value - 1 if value >= 1 else None


def _epoch_artifact_dirs(out: Path) -> List[Path]:
    dirs = []
    for path in out.glob("epoch_*"):
        idx = _parse_epoch_artifact_dir(path)
        if idx is not None and path.is_dir():
            dirs.append((idx, path))
    return [path for _, path in sorted(dirs, key=lambda item: item[0])]


def _row_epoch_index(row: Dict[str, Any]) -> Optional[int]:
    if not isinstance(row, dict):
        return None
    value = row.get("epoch_index")
    if value not in (None, ""):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _row_round_index(row: Dict[str, Any]) -> Optional[int]:
    if not isinstance(row, dict):
        return None
    value = row.get("round")
    if value not in (None, ""):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _row_belongs_to_epoch(row: Dict[str, Any], epoch_idx: int, *, filename: str = "") -> bool:
    row_epoch = _row_epoch_index(row)
    if row_epoch is not None:
        return row_epoch == int(epoch_idx)
    if filename in {
        "l2_skill_manager.jsonl",
        "l2_skill_judge.jsonl",
        "l2_skill_candidates.jsonl",
        "replay_label_traces.jsonl",
        "selection_feedback_cards.jsonl",
    }:
        row_round = _row_round_index(row)
        return row_round == int(epoch_idx)
    return False


def _filter_rows_for_epoch(rows: List[Dict[str, Any]], epoch_idx: int, *, filename: str = "") -> List[Dict[str, Any]]:
    return [
        row
        for row in rows or []
        if _row_belongs_to_epoch(row or {}, int(epoch_idx), filename=filename)
    ]


def _load_epoch_jsonl_rows(out: Path, filename: str, *, resume: bool, legacy_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    if not resume:
        return []
    rows: List[Dict[str, Any]] = []
    found_epoch_file = False
    for epoch_dir in _epoch_artifact_dirs(out):
        path = epoch_dir / filename
        if path.exists():
            found_epoch_file = True
            rows.extend(load_jsonl(str(path)))
    if found_epoch_file:
        return rows
    if legacy_path is not None and legacy_path.exists():
        return load_jsonl(str(legacy_path))
    return []


def _reuse_evidence_arg(args: Any) -> Optional[Path]:
    raw = str(getattr(args, "reuse_evidence_dir", "") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _resolve_reuse_evidence_epoch_dir(reuse_path: Optional[Path], epoch_idx: int) -> Optional[Path]:
    if reuse_path is None:
        return None
    if not reuse_path.exists():
        return None
    if _parse_epoch_artifact_dir(reuse_path) is not None:
        return reuse_path if int(epoch_idx) == 0 else None
    candidate = reuse_path / f"epoch_{int(epoch_idx) + 1}"
    return candidate if candidate.exists() else None


def _evidence_id_numeric(evidence_id: Any) -> Optional[int]:
    text = str(evidence_id or "").strip()
    if len(text) >= 2 and text[0].upper() == "E" and text[1:].isdigit():
        return int(text[1:])
    return None


def _next_evidence_counter(rows: List[Dict[str, Any]]) -> int:
    max_id = 0
    for row in rows or []:
        value = _evidence_id_numeric((row or {}).get("evidence_id"))
        if value is not None:
            max_id = max(max_id, value)
    return max(max_id + 1, len(rows or []) + 1)


def _allocate_reused_evidence_id(
    old_id: Any,
    *,
    existing_ids: set,
    evidence_counter: int,
) -> Tuple[str, int]:
    old_text = str(old_id or "").strip()
    if old_text and old_text not in existing_ids:
        existing_ids.add(old_text)
        return old_text, evidence_counter
    while True:
        candidate = f"E{int(evidence_counter):06d}"
        evidence_counter += 1
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate, evidence_counter


def _remap_epoch_runtime_fields(row: Dict[str, Any], *, epoch_idx: int, bundles_per_epoch: int) -> Dict[str, Any]:
    out = _jsonable_copy(row)
    source_bundle_idx = _row_int(out, "epoch_bundle_index", _row_int(out, "round", 0))
    source_bundle_idx = max(0, int(source_bundle_idx))
    out["epoch_index"] = int(epoch_idx)
    out["epoch_bundle_index"] = source_bundle_idx
    global_bundle_idx = int(epoch_idx) * max(1, int(bundles_per_epoch)) + source_bundle_idx
    out["global_bundle_index"] = global_bundle_idx
    out["round"] = global_bundle_idx
    return out


def _sanitize_reused_raw_result(raw_result: Any) -> Any:
    raw = _jsonable_copy(raw_result)
    if isinstance(raw, dict):
        for key in (
            "retrieved_memory",
            "retrieved_memory_text",
            "retrieved_skill_ids",
            "scienceworld_env_url",
        ):
            raw.pop(key, None)
    return raw


def _upgrade_evidence_card_v4(
    card: Dict[str, Any],
    *,
    replay_by_task_id: Dict[str, Dict[str, Any]],
    fallback_minibatch_id: str,
) -> Dict[str, Any]:
    row = _jsonable_copy(card)
    proposed_edit = _normalize_proposed_edit(row.get("proposed_edit"))
    if proposed_edit is None:
        raise ValueError(
            f"EvidenceCard {row.get('evidence_id') or '<unknown>'} is missing proposed_edit"
        )
    trigger_ranges = _normalize_trigger_ranges(row.get("trigger_ranges"))
    if not trigger_ranges:
        raise ValueError(
            f"EvidenceCard {row.get('evidence_id') or '<unknown>'} is missing trigger_ranges"
        )

    records_by_id = {
        task_id: _materialization_record_from_result(
            raw_result,
            dataset=str(raw_result.get("dataset") or "scienceworld"),
        )
        for task_id, raw_result in replay_by_task_id.items()
        if isinstance(raw_result, dict)
    }
    upgraded_segments: List[Dict[str, Any]] = []
    existing_segments = row.get("before_segments") if isinstance(row.get("before_segments"), list) else []
    supporting_tasks: List[Dict[str, Any]] = []
    seen_tasks = set()
    for range_index, trigger in enumerate(trigger_ranges):
        task_id = str(trigger.get("task_id") or "")
        record = records_by_id.get(task_id)
        if record:
            start = _coerce_int(trigger.get("start", 0), 0)
            end = _coerce_int(trigger.get("end", start), start)
            selected = [
                step
                for step in (record.get("trajectory") or [])
                if start <= _coerce_int(step.get("step", -1), -1) <= end
            ]
            upgraded_segments.append(
                _evidence_segment_from_steps(
                    task_id=task_id,
                    rows=selected,
                    initial_observation=str(record.get("initial_observation") or ""),
                )
            )
            if task_id not in seen_tasks:
                supporting_tasks.append(
                    {
                        "task_id": task_id,
                        "task_instruction": str(record.get("instruction") or ""),
                        "task_family": str(record.get("task_family") or ""),
                        "task_outcome": dict(record.get("outcome") or {}),
                    }
                )
                seen_tasks.add(task_id)
            continue

        existing = existing_segments[range_index] if range_index < len(existing_segments) else []
        if isinstance(existing, dict) and "steps" in existing:
            segment = _jsonable_copy(existing)
            segment["task_id"] = str(segment.get("task_id") or task_id)
        else:
            legacy_steps = _segment_steps(existing)
            normalized_steps = [
                {
                    "step": _coerce_int(step.get("step", index), index),
                    "observation": str(step.get("observation") or ""),
                    "action": str(step.get("action") or ""),
                    "feedback": str(step.get("feedback") or step.get("next_observation") or step.get("observation") or ""),
                }
                for index, step in enumerate(legacy_steps)
            ]
            segment = _evidence_segment_from_steps(task_id=task_id, rows=normalized_steps)
        upgraded_segments.append(segment)

    if not supporting_tasks:
        supporting_tasks = [
            dict(item)
            for item in (row.get("supporting_tasks") or [])
            if isinstance(item, dict)
        ]

    experience_type = _normalize_experience_type(
        row.get("experience_type"),
        " ".join(str(proposed_edit.get(key) or "") for key in ("op", "target", "content")),
        str(row.get("pattern") or ""),
    )
    row.update(
        {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "experience_type": experience_type,
            "source_type": _source_type_from_experience(experience_type),
            "proposed_edit": proposed_edit,
            "trigger_ranges": trigger_ranges,
            "supporting_tasks": supporting_tasks,
            "before_segments": upgraded_segments,
            "trigger_progress": _trigger_progress_from_segments(trigger_ranges, upgraded_segments),
            "support_count": _evidence_support_count_from_ranges(trigger_ranges),
            "source_minibatch_id": str(row.get("source_minibatch_id") or fallback_minibatch_id),
            "source_minibatch_task_ids": list(
                dict.fromkeys(
                    [
                        str(task_id)
                        for task_id in (row.get("source_minibatch_task_ids") or [])
                        if str(task_id)
                    ]
                    or [
                        str(trigger.get("task_id") or "")
                        for trigger in trigger_ranges
                        if str(trigger.get("task_id") or "")
                    ]
                )
            ),
            "merge_level": _coerce_int(row.get("merge_level"), 0),
        }
    )
    if experience_type == "failure_reflection":
        row["failure_type"] = _normalize_failure_type(row.get("failure_type")) or "other"
    else:
        row.pop("failure_type", None)
    row["pattern"] = str(row.get("pattern") or _fallback_evidence_pattern(row)).strip()
    for key in ("evidence_role", "evidence_reason", "task_instruction", "task_family", "task_outcome"):
        row.pop(key, None)
    return row


def _load_reuse_epoch_artifacts(source_dir: Path) -> Dict[str, Any]:
    evidence_path = source_dir / EPOCH_JSONL_FILENAMES["evidence_cards"]
    replay_path = source_dir / EPOCH_JSONL_FILENAMES["evidence_replay_sources"]
    missing = [str(path) for path in (evidence_path, replay_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Reusable evidence directory is missing required files: "
            + ", ".join(missing)
        )
    logs_path = source_dir / "dual_layer_generation_logs.json"
    logs = load_json(str(logs_path)) if logs_path.exists() else []
    if not isinstance(logs, list):
        logs = []
    evidence_rows = load_jsonl(str(evidence_path))
    missing_proposed = [
        str(row.get("evidence_id") or f"row_{idx}")
        for idx, row in enumerate(evidence_rows)
        if not isinstance(row, dict) or _normalize_proposed_edit(row.get("proposed_edit")) is None
    ]
    if missing_proposed:
        raise ValueError(
            "Reusable evidence uses the old idea-style schema or has invalid proposed_edit. "
            "Concrete Edit Evidence requires proposed_edit on every EvidenceCard. "
            f"source={source_dir} examples={missing_proposed[:5]}"
        )
    replay_source_rows = load_jsonl(str(replay_path))
    missing_replay_task = [
        str(row.get("evidence_id") or f"row_{idx}")
        for idx, row in enumerate(replay_source_rows)
        if not isinstance(row, dict) or not row.get("evidence_id") or not row.get("task_id")
    ]
    if missing_replay_task:
        raise ValueError(
            "Reusable evidence replay sources must include both evidence_id and task_id "
            f"for Concrete Edit Evidence. source={source_dir} examples={missing_replay_task[:5]}"
        )
    replay_keys = {
        (str(row.get("evidence_id") or ""), str(row.get("task_id") or ""))
        for row in replay_source_rows
        if isinstance(row, dict) and isinstance(row.get("raw_result"), dict)
    }
    required_replay_keys = {
        (str(row.get("evidence_id") or ""), str(trigger.get("task_id") or ""))
        for row in evidence_rows
        if isinstance(row, dict)
        for trigger in _normalize_trigger_ranges(row.get("trigger_ranges"))
        if str(row.get("evidence_id") or "") and str(trigger.get("task_id") or "")
    }
    missing_replay_keys = sorted(required_replay_keys - replay_keys)
    if missing_replay_keys:
        raise ValueError(
            "Reusable evidence is missing raw replay sources for cited trigger tasks. "
            f"source={source_dir} examples={missing_replay_keys[:5]}"
        )
    return {
        "source_dir": str(source_dir),
        "evidence_rows": evidence_rows,
        "replay_source_rows": replay_source_rows,
        "logs": logs,
    }


def _prepare_reused_epoch_log_rows(
    *,
    source_logs: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    source_dir: Path,
    epoch_idx: int,
    bundles_per_epoch: int,
    evidence_count: int,
    replay_source_count: int,
) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    if source_logs:
        for raw in source_logs:
            if not isinstance(raw, dict):
                continue
            row = _remap_epoch_runtime_fields(raw, epoch_idx=epoch_idx, bundles_per_epoch=bundles_per_epoch)
            row["reused_evidence"] = True
            row["reuse_evidence_source"] = str(source_dir)
            row["reuse_evidence_count"] = int(evidence_count)
            row["reuse_replay_source_count"] = int(replay_source_count)
            row["l2"] = {
                "triggered": False,
                "candidate_count": 0,
                "accepted_count": 0,
                "reason": "reused_evidence_epoch_update_pending",
                "evidence_pool": {
                    "total": int(evidence_count),
                    "active": int(evidence_count),
                    "stale": 0,
                    "archived": 0,
                    "consumed": 0,
                },
            }
            for key in list(row.keys()):
                if key.startswith("epoch_") and key not in {
                    "epoch_index",
                    "epoch_bundle_index",
                }:
                    row.pop(key, None)
            logs.append(row)
    if logs:
        return sorted(logs, key=lambda row: _row_int(row, "epoch_bundle_index", 0))

    bundle_indices = sorted(
        {
            _row_int(row, "epoch_bundle_index", -1)
            for row in evidence_rows or []
            if _row_int(row, "epoch_bundle_index", -1) >= 0
        }
    )
    if not bundle_indices:
        bundle_indices = [0]
    for bundle_idx in bundle_indices:
        global_bundle_idx = int(epoch_idx) * max(1, int(bundles_per_epoch)) + int(bundle_idx)
        logs.append(
            {
                "epoch_index": int(epoch_idx),
                "epoch_bundle_index": int(bundle_idx),
                "global_bundle_index": global_bundle_idx,
                "round": global_bundle_idx,
                "task_ids": [],
                "draft_count": 0,
                "kept_count": 0,
                "dropped_evidence": [],
                "reused_evidence": True,
                "reuse_evidence_source": str(source_dir),
                "reuse_evidence_count": int(evidence_count),
                "reuse_replay_source_count": int(replay_source_count),
                "l2": {
                    "triggered": False,
                    "candidate_count": 0,
                    "accepted_count": 0,
                    "reason": "reused_evidence_epoch_update_pending",
                    "evidence_pool": {
                        "total": int(evidence_count),
                        "active": int(evidence_count),
                        "stale": 0,
                        "archived": 0,
                        "consumed": 0,
                    },
                },
            }
        )
    return logs


def _import_reused_epoch_evidence(
    *,
    source_dir: Path,
    epoch_idx: int,
    bundles_per_epoch: int,
    evidence_rows: List[Dict[str, Any]],
    replay_source_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    logs: List[Dict[str, Any]],
    evidence_counter: int,
) -> Tuple[List[str], Dict[str, Any], int]:
    loaded = _load_reuse_epoch_artifacts(source_dir)
    source_evidence_rows = loaded["evidence_rows"]
    source_replay_rows = loaded["replay_source_rows"]
    existing_ids = {str(row.get("evidence_id") or "") for row in evidence_rows or [] if str(row.get("evidence_id") or "")}
    evidence_counter = max(int(evidence_counter), _next_evidence_counter(evidence_rows))
    id_map: Dict[str, str] = {}
    imported_ids: List[str] = []
    imported_evidence_rows: List[Dict[str, Any]] = []
    source_replay_by_evidence: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for replay_row in source_replay_rows:
        if not isinstance(replay_row, dict):
            continue
        source_replay_by_evidence.setdefault(str(replay_row.get("evidence_id") or ""), {})[
            str(replay_row.get("task_id") or "")
        ] = replay_row.get("raw_result") or {}

    for source_index, raw in enumerate(source_evidence_rows):
        if not isinstance(raw, dict):
            continue
        old_id = str(raw.get("evidence_id") or "")
        new_id, evidence_counter = _allocate_reused_evidence_id(
            old_id,
            existing_ids=existing_ids,
            evidence_counter=evidence_counter,
        )
        if old_id:
            id_map[old_id] = new_id
        upgraded = _upgrade_evidence_card_v4(
            raw,
            replay_by_task_id=source_replay_by_evidence.get(old_id, {}),
            fallback_minibatch_id=(
                f"epoch_{epoch_idx + 1}_"
                f"{str(raw.get('source_type') or _source_type_from_experience(raw.get('experience_type')))}_"
                f"reused_{source_index:03d}"
            ),
        )
        row = _remap_epoch_runtime_fields(upgraded, epoch_idx=epoch_idx, bundles_per_epoch=bundles_per_epoch)
        row["evidence_id"] = new_id
        row["archived"] = False
        row["l2_status"] = EVIDENCE_STATUS_ACTIVE
        row["l2_attempt_count"] = 0
        row["l2_reject_count"] = 0
        row["l2_accept_count"] = 0
        row["l2_last_attempt_round"] = None
        row.pop("consume_reason", None)
        _normalize_evidence_runtime_state(row)
        imported_evidence_rows.append(row)
        imported_ids.append(new_id)

    imported_replay_rows: List[Dict[str, Any]] = []
    for raw in source_replay_rows:
        if not isinstance(raw, dict):
            continue
        old_id = str(raw.get("evidence_id") or "")
        if old_id not in id_map:
            continue
        row = _remap_epoch_runtime_fields(raw, epoch_idx=epoch_idx, bundles_per_epoch=bundles_per_epoch)
        row["evidence_id"] = id_map[old_id]
        row["raw_result"] = _sanitize_reused_raw_result(row.get("raw_result"))
        imported_replay_rows.append(row)
        replay_source_by_evidence_id[_replay_source_key(row["evidence_id"], row.get("task_id"))] = row.get("raw_result")

    evidence_rows.extend(imported_evidence_rows)
    replay_source_rows.extend(imported_replay_rows)
    imported_logs = _prepare_reused_epoch_log_rows(
        source_logs=loaded["logs"],
        evidence_rows=imported_evidence_rows,
        source_dir=source_dir,
        epoch_idx=epoch_idx,
        bundles_per_epoch=bundles_per_epoch,
        evidence_count=len(imported_evidence_rows),
        replay_source_count=len(imported_replay_rows),
    )
    logs.extend(imported_logs)
    info = {
        "reused_evidence": True,
        "reuse_evidence_source": str(source_dir),
        "reuse_evidence_count": len(imported_evidence_rows),
        "reuse_replay_source_count": len(imported_replay_rows),
        "reuse_log_count": len(imported_logs),
        "reuse_id_remap_count": sum(1 for old_id, new_id in id_map.items() if old_id != new_id),
    }
    return imported_ids, info, evidence_counter


def _save_epoch_artifacts(
    out: Path,
    epoch_idx: int,
    *,
    l1_manager_rows: List[Dict[str, Any]],
    l1_judge_rows: List[Dict[str, Any]],
    l2_manager_rows: List[Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    logs: List[Dict[str, Any]],
    epoch_final_skill_md: Optional[str] = None,
    epoch_working_skill_md: Optional[str] = None,
) -> Dict[str, str]:
    epoch_dir = _epoch_artifact_dir(out, epoch_idx)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    rows_by_file = {
        EPOCH_JSONL_FILENAMES["l1_skill_manager"]: l1_manager_rows,
        EPOCH_JSONL_FILENAMES["l1_skill_judge"]: l1_judge_rows,
        EPOCH_JSONL_FILENAMES["l2_skill_manager"]: l2_manager_rows,
        EPOCH_JSONL_FILENAMES["l2_skill_judge"]: l2_judge_rows,
        EPOCH_JSONL_FILENAMES["evidence_cards"]: evidence_rows,
        EPOCH_JSONL_FILENAMES["evidence_replay_sources"]: replay_source_rows,
        EPOCH_JSONL_FILENAMES["l2_skill_candidates"]: l2_candidate_rows,
        EPOCH_JSONL_FILENAMES["replay_label_traces"]: replay_rows,
        EPOCH_JSONL_FILENAMES["selection_feedback_cards"]: selection_feedback_rows,
    }
    written: Dict[str, str] = {}
    for filename, rows in rows_by_file.items():
        path = epoch_dir / filename
        save_jsonl(
            str(path),
            _filter_rows_for_epoch(rows, epoch_idx, filename=filename),
        )
        written[filename] = str(path)
    epoch_logs = [
        row
        for row in logs or []
        if _row_epoch_index(row or {}) == int(epoch_idx)
    ]
    logs_path = epoch_dir / "dual_layer_generation_logs.json"
    save_json(str(logs_path), epoch_logs)
    written["dual_layer_generation_logs.json"] = str(logs_path)
    if epoch_final_skill_md is not None:
        skill_path = epoch_dir / "evolving_skill.md"
        temp_skill_path = epoch_dir / ".evolving_skill.md.tmp"
        temp_skill_path.write_text(str(epoch_final_skill_md), encoding="utf-8")
        temp_skill_path.replace(skill_path)
        written["evolving_skill.md"] = str(skill_path)
    if epoch_working_skill_md is not None:
        working_path = epoch_dir / "working_skill.md"
        atomic_write_text(working_path, str(epoch_working_skill_md))
        written["working_skill.md"] = str(working_path)
    return written


def _checkpoint_working_branch_epoch_l2(
    out: Path,
    epoch_idx: int,
    *,
    l2_manager_rows: List[Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    validated_skill_md: str,
    working_skill_md: str,
) -> None:
    epoch_dir = _epoch_artifact_dir(out, epoch_idx)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    rows_by_filename = {
        EPOCH_JSONL_FILENAMES["l2_skill_manager"]: l2_manager_rows,
        EPOCH_JSONL_FILENAMES["l2_skill_candidates"]: l2_candidate_rows,
        EPOCH_JSONL_FILENAMES["l2_skill_judge"]: l2_judge_rows,
        EPOCH_JSONL_FILENAMES["evidence_cards"]: evidence_rows,
        EPOCH_JSONL_FILENAMES["replay_label_traces"]: replay_rows,
        EPOCH_JSONL_FILENAMES["selection_feedback_cards"]: selection_feedback_rows,
    }
    for filename, rows in rows_by_filename.items():
        save_jsonl(
            str(epoch_dir / filename),
            _filter_rows_for_epoch(rows, epoch_idx, filename=filename),
        )
    atomic_write_text(epoch_dir / "evolving_skill.md", validated_skill_md)
    atomic_write_text(epoch_dir / "working_skill.md", working_skill_md)


def _save_all_epoch_artifacts(out: Path, *, logs: List[Dict[str, Any]], **rows_by_name: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    epoch_indices = {
        idx
        for rows in rows_by_name.values()
        for row in (rows or [])
        for idx in [_row_epoch_index(row or {})]
        if idx is not None
    }
    epoch_indices.update(
        idx
        for row in logs or []
        for idx in [_row_epoch_index(row or {})]
        if idx is not None
    )
    saved: Dict[str, Dict[str, str]] = {}
    for epoch_idx in sorted(epoch_indices):
        saved[f"epoch_{epoch_idx + 1}"] = _save_epoch_artifacts(
            out,
            epoch_idx,
            logs=logs,
            **rows_by_name,
        )
    return saved


_EVALUATION_ERROR_SUMMARY_FIELDS = {
    "already_covered_current_evidence_count",
    "consumed_evidence_count",
    "final_committed_evidence_count",
    "final_deferred_evidence_count",
    "final_replay_accepts",
    "final_replay_bypassed",
    "final_replay_edits",
    "final_replay_rejects",
    "final_replay_skipped",
    "judge_rejects",
    "longitudinal_memory",
    "reflection_accepts",
    "reflection_attempts",
    "reflection_failures",
    "rule_ignored_accepts",
    "rule_ignored_defers",
}


def _strip_failed_epoch_l2_summary(row: Dict[str, Any]) -> None:
    for key in list(row):
        if key.startswith("epoch_") and key not in {
            "epoch_index",
            "epoch_bundle_index",
            "epoch_l1_summary",
        }:
            row.pop(key, None)
    for key in _EVALUATION_ERROR_SUMMARY_FIELDS:
        row.pop(key, None)


def _latest_prior_evidence_defer_reason(
    logs: List[Dict[str, Any]], *, evidence_id: str, before_epoch: int
) -> Optional[str]:
    prior_rows = sorted(
        (
            row
            for row in logs or []
            if _row_int(row or {}, "epoch_index", -1) < int(before_epoch)
        ),
        key=lambda row: _row_int(row or {}, "epoch_index", -1),
        reverse=True,
    )
    for row in prior_rows:
        reasons = (row or {}).get("epoch_deferred_evidence_reasons") or {}
        if evidence_id in reasons:
            return str(reasons[evidence_id] or "deferred")
    return None


def _recover_evaluation_error_epoch(
    *,
    out: Path,
    logs: List[Dict[str, Any]],
    l1_manager_rows: List[Dict[str, Any]],
    l1_judge_rows: List[Dict[str, Any]],
    l2_manager_rows: List[Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    failed_rows = [
        row
        for row in logs or []
        if str((row or {}).get("epoch_validation_decision") or "").lower()
        == "evaluation_error"
    ]
    if not failed_rows:
        return None
    failed_epoch = min(_row_int(row, "epoch_index", -1) for row in failed_rows)
    if failed_epoch < 0:
        return None
    failed_summary = next(
        row
        for row in failed_rows
        if _row_int(row or {}, "epoch_index", -1) == failed_epoch
    )
    later_commits = [
        row
        for row in logs or []
        if _row_int(row or {}, "epoch_index", -1) >= failed_epoch
        and (
            _row_int(row or {}, "epoch_committed_edit_count", 0) > 0
            or str((row or {}).get("epoch_validation_decision") or "").lower()
            == "commit"
        )
    ]
    if later_commits:
        raise RuntimeError(
            f"Cannot automatically recover validation evaluation_error in epoch "
            f"{failed_epoch + 1}: that epoch or a later epoch reports a committed skill. "
            "Restore a pre-error checkpoint or use a new output directory."
        )

    archived_ids = {
        str(evidence_id)
        for evidence_id in (failed_summary.get("epoch_archived_evidence_ids") or [])
        if str(evidence_id)
    }
    deferred_ids = {
        str(evidence_id)
        for evidence_id in (failed_summary.get("epoch_deferred_evidence_ids") or [])
        if str(evidence_id)
    }
    restored_archived = 0
    restored_deferred = 0
    for row in evidence_rows:
        row_epoch = _row_epoch_index(row or {})
        if row_epoch is not None and row_epoch > failed_epoch:
            continue
        evidence_id = str((row or {}).get("evidence_id") or "")
        if evidence_id in archived_ids and str(row.get("archive_reason") or "") == "already_covered_current":
            row["archived"] = False
            row["l2_status"] = EVIDENCE_STATUS_ACTIVE
            row.pop("archive_reason", None)
            restored_archived += 1
        if evidence_id in deferred_ids and not bool(row.get("archived")):
            row["l2_status"] = EVIDENCE_STATUS_ACTIVE
            row["archived"] = False
            prior_reason = _latest_prior_evidence_defer_reason(
                logs,
                evidence_id=evidence_id,
                before_epoch=failed_epoch,
            )
            if prior_reason:
                row["defer_reason"] = prior_reason
            else:
                row.pop("defer_reason", None)
                if row_epoch == failed_epoch:
                    row["l2_defer_count"] = 0
            restored_deferred += 1

    def keep_through_failed_epoch(row: Dict[str, Any]) -> bool:
        row_epoch = _row_epoch_index(row or {})
        return row_epoch is None or row_epoch <= failed_epoch

    def keep_before_failed_l2(row: Dict[str, Any]) -> bool:
        row_epoch = _row_epoch_index(row or {})
        if row_epoch is None:
            row_epoch = _row_round_index(row or {})
        return row_epoch is None or row_epoch < failed_epoch

    l1_manager_rows[:] = [row for row in l1_manager_rows if keep_through_failed_epoch(row)]
    l1_judge_rows[:] = [row for row in l1_judge_rows if keep_through_failed_epoch(row)]
    evidence_rows[:] = [row for row in evidence_rows if keep_through_failed_epoch(row)]
    replay_source_rows[:] = [row for row in replay_source_rows if keep_through_failed_epoch(row)]
    for rows in (
        l2_manager_rows,
        l2_candidate_rows,
        l2_judge_rows,
        replay_rows,
        selection_feedback_rows,
    ):
        rows[:] = [row for row in rows if keep_before_failed_l2(row)]

    logs[:] = [
        row
        for row in logs or []
        if _row_int(row or {}, "epoch_index", -1) <= failed_epoch
    ]
    for row in logs:
        if _row_int(row or {}, "epoch_index", -1) == failed_epoch:
            _strip_failed_epoch_l2_summary(row)

    removed_later_epoch_dirs = 0
    for epoch_dir in _epoch_artifact_dirs(out):
        epoch_dir_idx = _parse_epoch_artifact_dir(epoch_dir)
        if epoch_dir_idx is not None and epoch_dir_idx > failed_epoch:
            shutil.rmtree(epoch_dir)
            removed_later_epoch_dirs += 1
    for temp_path in out.glob("_tmp_l2*_skill.md"):
        temp_path.unlink(missing_ok=True)
    (out / "_tmp_epoch_final_candidate_skill.md").unlink(missing_ok=True)

    for epoch_idx in range(failed_epoch + 1):
        _save_epoch_artifacts(
            out,
            epoch_idx,
            l1_manager_rows=l1_manager_rows,
            l1_judge_rows=l1_judge_rows,
            l2_manager_rows=l2_manager_rows,
            l2_candidate_rows=l2_candidate_rows,
            l2_judge_rows=l2_judge_rows,
            evidence_rows=evidence_rows,
            replay_source_rows=replay_source_rows,
            replay_rows=replay_rows,
            selection_feedback_rows=selection_feedback_rows,
            logs=logs,
        )
    save_json(str(out / "dual_layer_generation_logs.json"), logs)
    return {
        "failed_epoch": failed_epoch,
        "preserved_bundle_count": len(_epoch_bundle_log_rows(logs, failed_epoch)),
        "preserved_l1": _epoch_l1_is_complete(logs, failed_epoch),
        "restored_archived_evidence": restored_archived,
        "restored_deferred_evidence": restored_deferred,
        "removed_later_epoch_dirs": removed_later_epoch_dirs,
    }


def _legacy_epoch_jsonl_paths(out: Path) -> List[Path]:
    return [out / filename for filename in LEGACY_EPOCH_JSONL_PATHS.values()]


def _remove_legacy_epoch_jsonl_files(out: Path) -> None:
    for path in _legacy_epoch_jsonl_paths(out):
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            pass


def _remove_epoch_artifact_dirs(out: Path) -> None:
    for path in _epoch_artifact_dirs(out):
        try:
            shutil.rmtree(path)
        except OSError:
            pass


def _record_by_task_id(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(x.get("task_id", "")): x for x in payload.get("task_records", []) or []}


def _raw_result_by_task_id(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(x.get("task_id", "")): x for x in results or []}


def _replay_source_key(evidence_id: Any, task_id: Any) -> str:
    return f"{str(evidence_id or '')}\t{str(task_id or '')}"


def _replay_source_lookup(
    rows_by_key: Dict[str, Dict[str, Any]],
    *,
    evidence_id: Any,
    task_id: Any,
) -> Optional[Dict[str, Any]]:
    key = _replay_source_key(evidence_id, task_id)
    row = rows_by_key.get(key)
    return row if isinstance(row, dict) else None


def _task_outcome_from_result(result: Dict[str, Any], *, dataset: str = "") -> Dict[str, Any]:
    score = float(result.get("score", result.get("final_reward", 0.0)) or 0.0)
    success = bool(result.get("success")) or score >= 1.0
    steps = int(result.get("steps", len(result.get("trajectory", []) or [])) or 0)
    max_steps_raw = result.get("max_steps")
    max_steps = _coerce_int(max_steps_raw, 0) if max_steps_raw not in (None, "") else None
    outcome = {
        "success": success,
        "final_score": score,
        "steps": steps,
        "max_steps": max_steps,
    }
    if _is_appworld_dataset(dataset or str(result.get("dataset") or "")):
        evaluation = result.get("evaluation_summary")
        outcome["task_completed"] = bool(result.get("task_completed"))
        outcome["evaluation_summary"] = (
            {
                "num_tests": int(evaluation.get("num_tests", 0) or 0),
                "passed_tests": int(evaluation.get("passed_tests", 0) or 0),
                "failed_tests": int(evaluation.get("failed_tests", 0) or 0),
            }
            if isinstance(evaluation, dict)
            else {"num_tests": 0, "passed_tests": 0, "failed_tests": 0}
        )
    return outcome


def _attach_l1_task_outcomes(
    records: List[Dict[str, Any]],
    raw_by_id: Dict[str, Dict[str, Any]],
    *,
    dataset: str = "",
) -> List[Dict[str, Any]]:
    out = []
    for record in records or []:
        row = dict(record)
        raw = raw_by_id.get(str(row.get("task_id") or ""))
        if raw:
            row["outcome"] = _task_outcome_from_result(raw, dataset=dataset)
        out.append(row)
    return out


def _average_score(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(float(x.get("score", x.get("final_reward", 0.0)) or 0.0) for x in results) / len(results)


def _average_steps(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(int(x.get("steps", 0) or 0) for x in results) / len(results)


def _success_count(results: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for row in results or []
        if _is_success_result(row)
    )


def _is_success_result(row: Dict[str, Any]) -> bool:
    return bool((row or {}).get("success")) or float((row or {}).get("score", (row or {}).get("final_reward", 0.0)) or 0.0) >= 1.0


def _success_rate(results: List[Dict[str, Any]]) -> float:
    return _success_count(results) / len(results) if results else 0.0


def _selection_eval_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = _success_count(results)
    task_count = len(results)
    return {
        "score": _average_score(results),
        "steps_avg": _average_steps(results),
        "success": success,
        "success_rate": success / task_count if task_count else 0.0,
        "task_count": task_count,
    }


def _compact_selection_task_result(result: Dict[str, Any], args: Any = None) -> Dict[str, Any]:
    max_observation_chars = getattr(args, "record_max_observation_chars", 1500) if args is not None else 1500
    action_only = bool(getattr(args, "record_action_only", False)) if args is not None else False
    return {
        "task_id": str(result.get("task_id") or ""),
        "task_instruction": str(result.get("instruction") or result.get("task") or ""),
        "task_family": str(result.get("task_family") or ""),
        "score": float(result.get("score", result.get("final_reward", 0.0)) or 0.0),
        "success": bool(result.get("success")) or float(result.get("score", result.get("final_reward", 0.0)) or 0.0) >= 1.0,
        "steps": int(result.get("steps", len(result.get("trajectory", []) or [])) or 0),
        "trajectory": _compact_segment(
            result.get("trajectory") or [],
            max_observation_chars=max_observation_chars,
            action_only=action_only,
        ),
    }


def _compact_selection_task_results(results: List[Dict[str, Any]], args: Any = None) -> List[Dict[str, Any]]:
    return [_compact_selection_task_result(row, args=args) for row in results or []]


def _selection_task_payload(item: Any) -> Dict[str, Any]:
    return {
        "task_id": get_task_id(item),
        "task_ref": get_task_ref(item),
        "task_family": get_task_family(item),
    }


def _sample_selection_tasks(
    manifest: List[Dict[str, Any]],
    *,
    size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    size = min(max(0, int(size)), len(manifest))
    if size <= 0:
        return []
    rng = random.Random(seed)
    family_to_items: Dict[str, List[Any]] = {}
    for item in manifest:
        family_to_items.setdefault(get_task_family(item), []).append(item)

    selected: List[Any] = []
    selected_ids = set()
    families = sorted(family_to_items)
    while families and len(selected) < size:
        progressed = False
        rng.shuffle(families)
        for family in list(families):
            pool = [item for item in family_to_items.get(family, []) if get_task_id(item) not in selected_ids]
            if not pool:
                families.remove(family)
                continue
            item = rng.choice(pool)
            selected.append(item)
            selected_ids.add(get_task_id(item))
            progressed = True
            if len(selected) >= size:
                break
        if not progressed:
            break

    if len(selected) < size:
        pool = [item for item in manifest if get_task_id(item) not in selected_ids]
        selected.extend(rng.sample(pool, min(size - len(selected), len(pool))))
    return [_selection_task_payload(item) for item in selected]


def _selection_size_arg(args: Any) -> int:
    try:
        return int(getattr(args, "l2_selection_size", 24))
    except (TypeError, ValueError):
        return 24


def _limit_selection_tasks(
    rows: List[Dict[str, Any]],
    *,
    size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Use the selection manifest as the pool, then choose a fixed subset."""
    if size <= 0 or len(rows) <= size:
        return rows
    return _sample_selection_tasks(rows, size=size, seed=seed)


def _load_or_create_selection_tasks(
    *,
    out: Path,
    manifest: List[Dict[str, Any]],
    args,
    resume: bool,
) -> List[Dict[str, Any]]:
    path = out / "selection_task_ids.json"
    if resume and path.exists():
        rows = load_json(str(path))
        if isinstance(rows, list):
            cached_rows = [dict(x) if isinstance(x, dict) else _selection_task_payload(x) for x in rows]
            size = _selection_size_arg(args)
            seed = int(getattr(args, "l2_selection_seed", 42) or 42)
            limited_rows = _limit_selection_tasks(cached_rows, size=size, seed=seed)
            if len(limited_rows) != len(cached_rows):
                print(
                    f"[selection] cached selection_task_ids.json has {len(cached_rows)} tasks; "
                    f"using {len(limited_rows)} from --l2-selection-size={size}",
                    flush=True,
                )
                save_json(str(path), limited_rows)
            return limited_rows
    selection_manifest = str(getattr(args, "selection_manifest", "") or "").strip()
    if not selection_manifest and str(getattr(args, "dataset", "") or "").lower() in {"scienceworld", "science_world"}:
        manifest_path = Path(str(getattr(args, "manifest", "") or ""))
        candidates: List[Path] = []
        if manifest_path.name == "items.json" and manifest_path.parent.name == "train":
            candidates.append(manifest_path.parent.parent / "val" / "items.json")
        if manifest_path.name == "train.json":
            candidates.extend([manifest_path.with_name("dev.json"), manifest_path.with_name("val.json")])
        candidates.append(manifest_path.with_name("dev.json"))
        for candidate in candidates:
            if candidate.exists():
                selection_manifest = str(candidate)
                break
    if selection_manifest:
        manifest_rows = [_selection_task_payload(item) for item in load_json(selection_manifest)]
        size = _selection_size_arg(args)
        seed = int(getattr(args, "l2_selection_seed", 42) or 42)
        rows = _limit_selection_tasks(manifest_rows, size=size, seed=seed)
        if len(rows) != len(manifest_rows):
            print(
                f"[selection] source={selection_manifest} source_tasks={len(manifest_rows)} "
                f"selected_tasks={len(rows)} seed={seed}",
                flush=True,
            )
        save_json(str(path), rows)
        return rows
    rows = _sample_selection_tasks(
        manifest,
        size=_selection_size_arg(args),
        seed=int(getattr(args, "l2_selection_seed", 42) or 42),
    )
    save_json(str(path), rows)
    return rows


def _exclude_selection_tasks(
    manifest: List[Dict[str, Any]],
    selection_tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selection_ids = {str(row.get("task_id") or "") for row in selection_tasks or []}
    return [item for item in manifest if get_task_id(item) not in selection_ids]


def _evidence_int(row: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default) or 0)
    except (TypeError, ValueError):
        return int(default)


def _row_int(row: Dict[str, Any], key: str, default: int = -1) -> int:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _evidence_primary_task_id(row: Dict[str, Any]) -> str:
    ranges = row.get("trigger_ranges")
    if isinstance(ranges, list) and ranges and isinstance(ranges[0], dict):
        return str(ranges[0].get("task_id") or "")
    return ""


def _normalize_evidence_runtime_state(row: Dict[str, Any]) -> Dict[str, Any]:
    if _coerce_int(row.get("evidence_schema_version"), 0) >= EVIDENCE_SCHEMA_VERSION:
        row["evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
    experience_type = _normalize_experience_type(
        row.get("experience_type"),
        _evidence_edit_text(row),
        str(row.get("pattern") or row.get("evidence_reason") or row.get("l1_skilljudge_reason") or ""),
    )
    row["experience_type"] = experience_type
    row["source_type"] = str(row.get("source_type") or _source_type_from_experience(experience_type))
    if "evidence_mode" in row:
        evidence_mode = str(row.get("evidence_mode") or "trajectory").strip().lower()
        row["evidence_mode"] = evidence_mode if evidence_mode in {"trajectory", "contrastive"} else "trajectory"
    if experience_type == "failure_reflection":
        row["failure_type"] = _normalize_failure_type(row.get("failure_type") or _failure_type_from_legacy_role(row.get("evidence_role"))) or "other"
    else:
        row["failure_type"] = None
    if not str(row.get("pattern") or "").strip():
        row["pattern"] = _fallback_evidence_pattern(row)
    trigger_ranges = _normalize_trigger_ranges(row.get("trigger_ranges"))
    before_segments = row.get("before_segments") if isinstance(row.get("before_segments"), list) else []
    if trigger_ranges and before_segments:
        row["trigger_progress"] = _trigger_progress_from_segments(trigger_ranges, before_segments)
    supporting_task_ids = {
        str(task.get("task_id") or "")
        for task in (row.get("supporting_tasks") or [])
        if isinstance(task, dict) and str(task.get("task_id") or "")
    }
    range_task_ids = {
        str(trigger.get("task_id") or "")
        for trigger in trigger_ranges
        if str(trigger.get("task_id") or "")
    }
    if supporting_task_ids or range_task_ids:
        row["support_count"] = len(supporting_task_ids | range_task_ids)
    else:
        row["support_count"] = max(1, _coerce_int(row.get("support_count"), 1))
    row["merge_level"] = _coerce_int(row.get("merge_level"), 0)
    if bool(row.get("archived")):
        row["l2_status"] = EVIDENCE_STATUS_ARCHIVED
    else:
        status = str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).strip().lower()
        if status not in {
            EVIDENCE_STATUS_ACTIVE,
            EVIDENCE_STATUS_STALE,
            EVIDENCE_STATUS_ARCHIVED,
            EVIDENCE_STATUS_CONSUMED,
        }:
            status = EVIDENCE_STATUS_ACTIVE
        row["l2_status"] = status
        row["archived"] = status == EVIDENCE_STATUS_ARCHIVED
    row["l2_attempt_count"] = _evidence_int(row, "l2_attempt_count")
    row["l2_reject_count"] = _evidence_int(row, "l2_reject_count")
    row["l2_accept_count"] = _evidence_int(row, "l2_accept_count")
    if row.get("l2_last_attempt_round") in ("", None):
        row["l2_last_attempt_round"] = None
    else:
        row["l2_last_attempt_round"] = _evidence_int(row, "l2_last_attempt_round")
    return row


def _normalize_evidence_runtime_states(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in rows or []:
        _normalize_evidence_runtime_state(row)
    return rows


def _evidence_pool_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(rows or [])
    active = 0
    stale = 0
    archived = 0
    consumed = 0
    for row in rows or []:
        status = str(row.get("l2_status") or "").lower()
        if bool(row.get("archived")) or status == EVIDENCE_STATUS_ARCHIVED:
            archived += 1
        elif status == EVIDENCE_STATUS_CONSUMED:
            consumed += 1
        elif status == EVIDENCE_STATUS_STALE:
            stale += 1
        else:
            active += 1
    return {"total": total, "active": active, "stale": stale, "archived": archived, "consumed": consumed}


def _active_evidence_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _normalize_evidence_runtime_states(rows)
    return [
        row
        for row in rows or []
        if not bool(row.get("archived"))
        and str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower() == EVIDENCE_STATUS_ACTIVE
    ]


def _l2_evidence_priority(row: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
    experience_type = _normalize_experience_type(row.get("experience_type"))
    failure_type = _normalize_failure_type(row.get("failure_type"))
    if experience_type == "failure_reflection" and failure_type == "rule_missing":
        evidence_rank = 0
    elif experience_type == "failure_reflection" and failure_type == "rule_wrong":
        evidence_rank = 1
    elif experience_type == "failure_reflection" and failure_type in {"rule_ignored", "invalid_action", "loop"}:
        evidence_rank = 2
    elif experience_type == "failure_reflection":
        evidence_rank = 3
    elif experience_type == "success_experience":
        evidence_rank = 4
    else:
        evidence_rank = 5
    return (
        _evidence_int(row, "l2_attempt_count"),
        evidence_rank,
        _evidence_int(row, "l2_reject_count"),
        -_evidence_int(row, "round", -1),
        str(row.get("evidence_id") or ""),
    )


def _resolve_l2_evidence_window_grouping(dataset: str, grouping: str = "auto") -> str:
    value = str(grouping or "auto").strip().lower()
    if value not in {"auto", "card", "round", "semantic"}:
        value = "auto"
    if value == "auto":
        return "round" if str(dataset or "").lower() in {"scienceworld", "science_world"} else "card"
    return value


def _evidence_selection_counts(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    role_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    for row in rows or []:
        role = _normalize_failure_type(row.get("failure_type")) or str(row.get("source_type") or _source_type_from_experience(row.get("experience_type")))
        family = str(row.get("task_family") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
    return role_counts, family_counts


def _empty_evidence_selection_info(
    evidence_pool: List[Dict[str, Any]],
    active_rows: List[Dict[str, Any]],
    stale_count: int,
    *,
    grouping: str,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "grouping": grouping,
        "pool_size": len(evidence_pool or []),
        "active_count": len(active_rows or []),
        "stale_count": stale_count,
        "selected_count": 0,
        "selected_evidence_ids": [],
        "selected_group_count": 0,
        "group_sizes": [],
        "truncated_group_count": 0,
        "role_counts": {},
        "family_counts": {},
    }
    if grouping == "semantic":
        info["selected_semantic_group_count"] = 0
        info["semantic_group_sizes"] = []
    elif grouping == "round":
        info["selected_round_group_count"] = 0
        info["round_group_sizes"] = []
    return info


def _l2_evidence_group_key(row: Dict[str, Any], *, grouping: str) -> Tuple[str, str]:
    if grouping == "round" and row.get("round") not in (None, ""):
        return ("round", str(_evidence_int(row, "round")))
    return ("evidence", str(row.get("evidence_id") or ""))


def _l2_active_rows(
    evidence_pool: List[Dict[str, Any]],
    *,
    exclude_evidence_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    exclude_ids = {str(eid) for eid in (exclude_evidence_ids or []) if str(eid)}
    return [
        row
        for row in evidence_pool or []
        if not bool(row.get("archived"))
        and str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower() == EVIDENCE_STATUS_ACTIVE
        and str(row.get("evidence_id") or "") not in exclude_ids
    ]


def _l2_stale_count(evidence_pool: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for row in evidence_pool or []
        if not bool(row.get("archived"))
        and str(row.get("l2_status") or "").lower() == EVIDENCE_STATUS_STALE
    )


def _l2_selection_info_for_groups(
    *,
    evidence_pool: List[Dict[str, Any]],
    active_rows: List[Dict[str, Any]],
    stale_count: int,
    selected_groups: List[List[Dict[str, Any]]],
    grouping: str,
    truncated_group_count: int = 0,
    planned_window_index: Optional[int] = None,
    planned_window_count: Optional[int] = None,
    selected_group_evidence_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    selected = [row for group in selected_groups or [] for row in group]
    role_counts, family_counts = _evidence_selection_counts(selected)
    info: Dict[str, Any] = {
        "grouping": grouping,
        "pool_size": len(evidence_pool or []),
        "active_count": len(active_rows or []),
        "stale_count": stale_count,
        "selected_count": len(selected),
        "selected_evidence_ids": [str(row.get("evidence_id") or "") for row in selected],
        "selected_group_count": len(selected_groups or []),
        "group_sizes": [len(group) for group in selected_groups or []],
        "truncated_group_count": truncated_group_count,
        "role_counts": role_counts,
        "family_counts": family_counts,
    }
    if grouping == "semantic":
        info["selected_semantic_group_count"] = len(selected_groups or [])
        info["semantic_group_sizes"] = [len(group) for group in selected_groups or []]
    elif grouping == "round":
        info["selected_round_group_count"] = len(selected_groups or [])
        info["round_group_sizes"] = [len(group) for group in selected_groups or []]
    if grouping in {"round", "semantic"}:
        info["selected_group_evidence_id_groups"] = [
            [
                str(row.get("evidence_id") or "")
                for row in group
                if str(row.get("evidence_id") or "")
            ]
            for group in selected_groups or []
        ]
        if selected_group_evidence_ids is None:
            selected_group_evidence_ids = [
                str(row.get("evidence_id") or "")
                for group in selected_groups or []
                for row in group
                if str(row.get("evidence_id") or "")
            ]
        info["selected_group_evidence_ids"] = sorted(set(selected_group_evidence_ids))
    if planned_window_index is not None:
        info["planned_window_index"] = planned_window_index
    if planned_window_count is not None:
        info["planned_window_count"] = planned_window_count
    return info


def _select_l2_evidence_window(
    evidence_pool: List[Dict[str, Any]],
    *,
    max_window: int,
    max_cards_per_task: int = 2,
    exclude_evidence_ids: Optional[List[str]] = None,
    grouping: str = "card",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    max_window = max(0, int(max_window))
    grouping = "round" if str(grouping or "").strip().lower() == "round" else "card"
    active_rows = _l2_active_rows(evidence_pool, exclude_evidence_ids=exclude_evidence_ids)
    stale_count = _l2_stale_count(evidence_pool)
    if max_window <= 0 or not active_rows:
        return [], _empty_evidence_selection_info(
            evidence_pool,
            active_rows,
            stale_count,
            grouping=grouping,
        )

    if grouping == "round":
        round_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in sorted(active_rows, key=_l2_evidence_priority):
            round_groups.setdefault(_l2_evidence_group_key(row, grouping="round"), []).append(row)

        groups = [
            sorted(rows, key=_l2_evidence_priority)
            for rows in round_groups.values()
            if rows
        ]
        groups.sort(key=lambda rows: _l2_evidence_priority(rows[0]))
        selected_groups: List[List[Dict[str, Any]]] = []
        truncated_group_count = 0

        for group in groups:
            if len(group) > max_window:
                if not selected_groups:
                    selected_groups.append(group[:max_window])
                    truncated_group_count += 1
                    break
                continue
            if sum(len(rows) for rows in selected_groups) + len(group) <= max_window:
                selected_groups.append(group)

        if not selected_groups and groups:
            selected_groups.append(groups[0][:max_window])
            truncated_group_count += int(len(groups[0]) > max_window)

        selected = [row for group in selected_groups for row in group]
        selected_group_evidence_ids: List[str] = []
        for group in selected_groups:
            if not group:
                continue
            key = _l2_evidence_group_key(group[0], grouping="round")
            selected_group_evidence_ids.extend(
                str(row.get("evidence_id") or "")
                for row in round_groups.get(key, group)
                if str(row.get("evidence_id") or "")
            )
        info = _l2_selection_info_for_groups(
            evidence_pool=evidence_pool,
            active_rows=active_rows,
            stale_count=stale_count,
            selected_groups=selected_groups,
            grouping="round",
            truncated_group_count=truncated_group_count,
            selected_group_evidence_ids=selected_group_evidence_ids,
        )
        return selected, info

    def cohort_key(row: Dict[str, Any]) -> Optional[Tuple[int, str]]:
        if row.get("round") in (None, ""):
            return None
        return (
            _evidence_int(row, "round"),
            str(row.get("task_family") or "unknown"),
        )

    family_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    ordered_active_rows = sorted(active_rows, key=_l2_evidence_priority)
    cohort_to_rows: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in ordered_active_rows:
        family = str(row.get("task_family") or "unknown")
        family_to_rows.setdefault(family, []).append(row)
        key = cohort_key(row)
        if key is not None:
            cohort_to_rows.setdefault(key, []).append(row)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    task_counts: Dict[str, int] = {}

    def can_select(row: Dict[str, Any]) -> bool:
        eid = str(row.get("evidence_id") or "")
        task_id = _evidence_primary_task_id(row)
        if eid in selected_ids:
            return False
        if task_id and task_counts.get(task_id, 0) >= int(max_cards_per_task):
            return False
        return True

    def add_selected(row: Dict[str, Any]) -> bool:
        if len(selected) >= max_window or not can_select(row):
            return False
        selected.append(row)
        selected_ids.add(str(row.get("evidence_id") or ""))
        task_id = _evidence_primary_task_id(row)
        if task_id:
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        return True

    def add_cohort_rows(row: Dict[str, Any]) -> None:
        key = cohort_key(row)
        if key is None:
            return
        for candidate in cohort_to_rows.get(key) or []:
            if len(selected) >= max_window:
                break
            add_selected(candidate)

    families = sorted(
        family_to_rows,
        key=lambda family: _l2_evidence_priority((family_to_rows.get(family) or [{}])[0]),
    )
    while families and len(selected) < max_window:
        progressed = False
        for family in list(families):
            rows = family_to_rows.get(family) or []
            picked = None
            while rows:
                candidate = rows.pop(0)
                eid = str(candidate.get("evidence_id") or "")
                task_id = _evidence_primary_task_id(candidate)
                if eid in selected_ids:
                    continue
                if task_id and task_counts.get(task_id, 0) >= int(max_cards_per_task):
                    continue
                picked = candidate
                break
            if not rows:
                families.remove(family)
            if picked is None:
                continue
            progressed = add_selected(picked) or progressed
            add_cohort_rows(picked)
            if len(selected) >= max_window:
                break
        if not progressed:
            break

    role_counts, family_counts = _evidence_selection_counts(selected)
    return selected, {
        "grouping": "card",
        "pool_size": len(evidence_pool or []),
        "active_count": len(active_rows),
        "stale_count": stale_count,
        "selected_count": len(selected),
        "selected_evidence_ids": [str(row.get("evidence_id") or "") for row in selected],
        "selected_group_count": 0,
        "group_sizes": [],
        "truncated_group_count": 0,
        "role_counts": role_counts,
        "family_counts": family_counts,
    }


def _semantic_evidence_kind(row: Dict[str, Any]) -> str:
    experience_type = _normalize_experience_type(row.get("experience_type"))
    failure_type = _normalize_failure_type(row.get("failure_type") or _failure_type_from_legacy_role(row.get("evidence_role")))
    if experience_type == "failure_reflection" and failure_type == "rule_missing":
        return "failure_rule_missing"
    if experience_type == "failure_reflection" and failure_type == "rule_wrong":
        return "failure_rule_wrong"
    if experience_type == "failure_reflection" and failure_type in {"rule_ignored", "invalid_action", "loop"}:
        return "failure_execution"
    if experience_type == "failure_reflection":
        return "failure_other"
    if experience_type == "success_experience":
        return "success"
    return "other"


def _semantic_cluster_utility(rows: List[Dict[str, Any]]) -> float:
    kinds = [_semantic_evidence_kind(row) for row in rows or []]
    count_missing = sum(1 for kind in kinds if kind == "failure_rule_missing")
    count_wrong = sum(1 for kind in kinds if kind == "failure_rule_wrong")
    count_execution = sum(1 for kind in kinds if kind == "failure_execution")
    count_other_failure = sum(1 for kind in kinds if kind == "failure_other")
    has_failure = any(kind.startswith("failure_") for kind in kinds)
    has_success = any(kind == "success" for kind in kinds)
    return (
        (100.0 if count_missing else 0.0)
        + (90.0 if count_wrong else 0.0)
        + 12.0 * count_missing
        + 10.0 * count_wrong
        + 8.0 * count_execution
        + 4.0 * count_other_failure
        + (3.0 if has_failure and has_success else 0.0)
        + 0.5 * min(len(rows or []), 8)
    )


def _semantic_is_high_value_small_cluster(rows: List[Dict[str, Any]]) -> bool:
    return any(
        _semantic_evidence_kind(row) in {"failure_rule_missing", "failure_rule_wrong"}
        for row in rows or []
    )


def _semantic_dot(left: List[float], right: List[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left or [], right or []))


def _semantic_normalize_vector(vector: List[float]) -> List[float]:
    norm = sum(float(value) * float(value) for value in vector or []) ** 0.5
    if norm <= 0:
        return [float(value) for value in vector or []]
    return [float(value) / norm for value in vector or []]


def _semantic_centroid(rows: List[Dict[str, Any]], embeddings_by_id: Dict[str, List[float]]) -> List[float]:
    vectors = [
        embeddings_by_id.get(str(row.get("evidence_id") or ""))
        for row in rows or []
        if embeddings_by_id.get(str(row.get("evidence_id") or "")) is not None
    ]
    if not vectors:
        return []
    dim = len(vectors[0])
    centroid = [0.0] * dim
    for vector in vectors:
        for idx, value in enumerate(vector[:dim]):
            centroid[idx] += float(value)
    norm = sum(value * value for value in centroid) ** 0.5
    if norm <= 0:
        return centroid
    return [value / norm for value in centroid]


def _semantic_group_similarity(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    embeddings_by_id: Dict[str, List[float]],
) -> float:
    return _semantic_dot(
        _semantic_centroid(left, embeddings_by_id),
        _semantic_centroid(right, embeddings_by_id),
    )


def _semantic_pairwise_stats(
    rows: List[Dict[str, Any]],
    embeddings_by_id: Dict[str, List[float]],
) -> Dict[str, float]:
    values: List[float] = []
    for left_idx, left in enumerate(rows or []):
        left_vec = embeddings_by_id.get(str(left.get("evidence_id") or ""))
        if left_vec is None:
            continue
        for right in (rows or [])[left_idx + 1 :]:
            right_vec = embeddings_by_id.get(str(right.get("evidence_id") or ""))
            if right_vec is None:
                continue
            values.append(_semantic_dot(left_vec, right_vec))
    if not values:
        return {"similarity_min": 1.0, "similarity_avg": 1.0, "similarity_max": 1.0}
    return {
        "similarity_min": round(min(values), 4),
        "similarity_avg": round(sum(values) / len(values), 4),
        "similarity_max": round(max(values), 4),
    }


def _semantic_similarity_matrix(rows: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[List[float]]:
    size = len(rows or [])
    matrix = [[0.0] * size for _ in range(size)]
    for idx in range(size):
        matrix[idx][idx] = 1.0
    for left in range(size):
        for right in range(left + 1, size):
            value = _semantic_dot(embeddings[left], embeddings[right])
            matrix[left][right] = value
            matrix[right][left] = value
    return matrix


def _semantic_connected_components(
    rows: List[Dict[str, Any]],
    matrix: List[List[float]],
    *,
    threshold: float,
    top_k: int,
) -> List[List[int]]:
    size = len(rows or [])
    top_k = max(1, int(top_k))
    neighbor_sets: List[set] = []
    for idx in range(size):
        neighbors = [
            other
            for other in range(size)
            if other != idx and matrix[idx][other] >= float(threshold)
        ]
        neighbors.sort(key=lambda other: (-matrix[idx][other], str(rows[other].get("evidence_id") or "")))
        neighbor_sets.append(set(neighbors[:top_k]))

    graph: List[set] = [set() for _ in range(size)]
    for left in range(size):
        for right in neighbor_sets[left]:
            if left in neighbor_sets[right]:
                graph[left].add(right)
                graph[right].add(left)

    seen = set()
    components: List[List[int]] = []
    for idx in range(size):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        component: List[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for other in sorted(graph[current]):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components.append(component)
    return components


def _coerce_auto_int(value: Any) -> Optional[int]:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _semantic_hdbscan_components(
    rows: List[Dict[str, Any]],
    embeddings: List[List[float]],
    *,
    max_window: int,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    size = len(rows or [])
    if size <= 1:
        return [[idx] for idx in range(size)], {
            "semantic_hdbscan_min_cluster_size": 1,
            "semantic_hdbscan_min_samples": 1,
            "semantic_hdbscan_noise_count": 0,
        }

    resolved_min_cluster_size = int(
        min_cluster_size
        if min_cluster_size is not None
        else max(4, min(8, max(2, int(max_window)) // 4))
    )
    resolved_min_cluster_size = max(2, min(resolved_min_cluster_size, size))
    resolved_min_samples = int(
        min_samples
        if min_samples is not None
        else max(2, min(4, resolved_min_cluster_size // 2))
    )
    resolved_min_samples = max(1, min(resolved_min_samples, resolved_min_cluster_size))

    try:
        from sklearn.cluster import HDBSCAN
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Semantic HDBSCAN clustering requires scikit-learn with sklearn.cluster.HDBSCAN. "
            "Use --semantic-clustering-method graph to fall back to threshold graph clustering."
        ) from exc

    labels = HDBSCAN(
        min_cluster_size=resolved_min_cluster_size,
        min_samples=resolved_min_samples,
        metric="euclidean",
    ).fit_predict(embeddings)

    label_to_indices: Dict[int, List[int]] = {}
    noise_components: List[List[int]] = []
    for idx, label in enumerate(labels):
        label = int(label)
        if label < 0:
            noise_components.append([idx])
        else:
            label_to_indices.setdefault(label, []).append(idx)
    components = [indices for _, indices in sorted(label_to_indices.items())] + noise_components
    if not components:
        components = [[idx] for idx in range(size)]
    return components, {
        "semantic_hdbscan_min_cluster_size": resolved_min_cluster_size,
        "semantic_hdbscan_min_samples": resolved_min_samples,
        "semantic_hdbscan_noise_count": len(noise_components),
    }


def _semantic_embedding_options_from_args(args: Any, out: Path) -> Dict[str, Any]:
    return {
        "model_name": str(getattr(args, "evidence_embedding_model", DEFAULT_EVIDENCE_EMBEDDING_MODEL)),
        "device": str(getattr(args, "evidence_embedding_device", "auto") or "auto"),
        "batch_size": int(getattr(args, "evidence_embedding_batch_size", 32) or 32),
        "cache_path": Path(out) / "evidence_embeddings.jsonl",
        "clustering_method": str(getattr(args, "semantic_clustering_method", "hdbscan") or "hdbscan").lower(),
        "cluster_threshold": float(getattr(args, "semantic_cluster_threshold", 0.68) or 0.68),
        "cluster_top_k": int(getattr(args, "semantic_cluster_top_k", 8) or 8),
        "hdbscan_min_cluster_size": _coerce_auto_int(getattr(args, "semantic_hdbscan_min_cluster_size", "auto")),
        "hdbscan_min_samples": _coerce_auto_int(getattr(args, "semantic_hdbscan_min_samples", "auto")),
        "min_window_size": int(getattr(args, "semantic_cluster_min_window_size", 4) or 4),
        "merge_threshold": float(getattr(args, "semantic_cluster_merge_threshold", 0.58) or 0.58),
        "max_reason_chars": int(getattr(args, "semantic_cluster_max_reason_chars", 300) or 300),
        "max_actions": int(getattr(args, "semantic_cluster_max_actions", 12) or 12),
    }


def _semantic_embeddings_for_rows(
    rows: List[Dict[str, Any]],
    semantic_options: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    semantic_options = dict(semantic_options or {})
    by_id = semantic_options.get("embeddings_by_evidence_id")
    if isinstance(by_id, dict):
        embeddings_by_id = {
            str(eid): _semantic_normalize_vector([float(value) for value in vector])
            for eid, vector in by_id.items()
        }
        return embeddings_by_id, {
            "embedding_model": str(semantic_options.get("model_name") or "test"),
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "embedding_count": len(rows or []),
        }

    vectors, stats = encode_evidence_cards(
        rows,
        model_name=str(semantic_options.get("model_name") or DEFAULT_EVIDENCE_EMBEDDING_MODEL),
        cache_path=semantic_options.get("cache_path"),
        device=str(semantic_options.get("device") or "auto"),
        batch_size=int(semantic_options.get("batch_size") or 32),
        max_reason_chars=int(semantic_options.get("max_reason_chars") or 300),
        max_actions=int(semantic_options.get("max_actions") or 12),
        encoder=semantic_options.get("encoder"),
    )
    return {
        str(row.get("evidence_id") or ""): vector
        for row, vector in zip(rows or [], vectors)
        if str(row.get("evidence_id") or "")
    }, stats


def _plan_semantic_l2_evidence_windows(
    evidence_pool: List[Dict[str, Any]],
    active_rows: List[Dict[str, Any]],
    stale_count: int,
    *,
    max_window: int,
    max_windows: int,
    semantic_options: Optional[Dict[str, Any]] = None,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    semantic_options = dict(semantic_options or {})
    if not active_rows:
        return []
    embeddings_by_id, embedding_stats = _semantic_embeddings_for_rows(active_rows, semantic_options)
    embeddings = [embeddings_by_id.get(str(row.get("evidence_id") or ""), []) for row in active_rows]
    if any(not vector for vector in embeddings):
        missing = [
            str(row.get("evidence_id") or "")
            for row, vector in zip(active_rows, embeddings)
            if not vector
        ]
        raise RuntimeError(f"Missing semantic evidence embeddings for: {', '.join(missing[:10])}")

    clustering_method = str(semantic_options.get("clustering_method") or "hdbscan").lower()
    if clustering_method not in {"graph", "hdbscan"}:
        clustering_method = "hdbscan"
    threshold = float(semantic_options.get("cluster_threshold", 0.68))
    top_k = int(semantic_options.get("cluster_top_k", 8))
    min_window_size = min(max_window, max(1, int(semantic_options.get("min_window_size", 4))))
    merge_threshold = float(semantic_options.get("merge_threshold", 0.58))

    matrix = _semantic_similarity_matrix(active_rows, embeddings)
    method_info: Dict[str, Any] = {"semantic_clustering_method": clustering_method}
    if clustering_method == "hdbscan":
        components, hdbscan_info = _semantic_hdbscan_components(
            active_rows,
            embeddings,
            max_window=max_window,
            min_cluster_size=semantic_options.get("hdbscan_min_cluster_size"),
            min_samples=semantic_options.get("hdbscan_min_samples"),
        )
        method_info.update(hdbscan_info)
    else:
        components = _semantic_connected_components(active_rows, matrix, threshold=threshold, top_k=top_k)
    raw_clusters: List[List[Dict[str, Any]]] = []
    for component in components:
        rows = [active_rows[idx] for idx in component]
        rows.sort(key=lambda row: (_l2_evidence_priority(row), -_semantic_dot(embeddings_by_id[str(row.get("evidence_id") or "")], _semantic_centroid(rows, embeddings_by_id))))
        raw_clusters.append(rows)
    raw_clusters.sort(key=lambda rows: (-_semantic_cluster_utility(rows), _l2_evidence_priority(rows[0])))

    semantic_groups: List[Dict[str, Any]] = []
    truncated_group_count = 0
    for cluster_idx, cluster in enumerate(raw_clusters):
        centroid = _semantic_centroid(cluster, embeddings_by_id)
        ordered = sorted(
            cluster,
            key=lambda row: (
                _l2_evidence_priority(row),
                -_semantic_dot(embeddings_by_id[str(row.get("evidence_id") or "")], centroid),
            ),
        )
        if len(ordered) > max_window:
            truncated_group_count += 1
            for split_idx in range(0, len(ordered), max_window):
                chunk = ordered[split_idx : split_idx + max_window]
                semantic_groups.append(
                    {
                        "rows": chunk,
                        "source_cluster_index": cluster_idx,
                        "split": True,
                        "merged": False,
                    }
                )
        else:
            semantic_groups.append(
                {
                    "rows": ordered,
                    "source_cluster_index": cluster_idx,
                    "split": False,
                    "merged": False,
                }
            )

    semantic_groups.sort(key=lambda group: (-_semantic_cluster_utility(group["rows"]), _l2_evidence_priority(group["rows"][0])))

    windows: List[Dict[str, Any]] = []
    pending_small: List[Dict[str, Any]] = []
    for group in semantic_groups:
        rows = group["rows"]
        if len(rows) < min_window_size:
            pending_small.append(group)
        else:
            windows.append({"groups": [group], "merged_small_group_count": 0})

    forced_merge_group_count = 0
    rebalanced_evidence_count = 0
    orphan_small: List[Dict[str, Any]] = []

    def window_rows(window: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [row for item in window["groups"] for row in item["rows"]]

    def window_size(window: Dict[str, Any]) -> int:
        return len(window_rows(window))

    def prune_empty_groups(window: Dict[str, Any]) -> None:
        window["groups"] = [
            group
            for group in (window.get("groups") or [])
            if group.get("rows")
        ]

    def best_merge_target(group: Dict[str, Any], *, require_threshold: bool) -> Optional[int]:
        best_idx: Optional[int] = None
        best_similarity = -1.0
        for idx, window in enumerate(windows):
            current_rows = window_rows(window)
            if len(current_rows) + len(group["rows"]) > max_window:
                continue
            similarity = _semantic_group_similarity(group["rows"], current_rows, embeddings_by_id)
            if require_threshold and similarity < merge_threshold:
                continue
            if similarity > best_similarity:
                best_idx = idx
                best_similarity = similarity
        return best_idx

    for group in pending_small:
        best_idx = best_merge_target(group, require_threshold=True)
        forced = False
        if best_idx is None:
            best_idx = best_merge_target(group, require_threshold=False)
            forced = best_idx is not None
        if best_idx is None:
            orphan_small.append(group)
        else:
            group["merged"] = True
            windows[best_idx]["groups"].append(group)
            windows[best_idx]["merged_small_group_count"] += 1
            if forced:
                forced_merge_group_count += 1

    packed_groups: List[Dict[str, Any]] = []
    packed_size = 0

    def flush_packed_groups() -> None:
        nonlocal packed_groups, packed_size
        if not packed_groups:
            return
        for group in packed_groups:
            group["merged"] = len(packed_groups) > 1
        windows.append(
            {
                "groups": list(packed_groups),
                "merged_small_group_count": max(0, len(packed_groups) - 1),
            }
        )
        packed_groups = []
        packed_size = 0

    for group in orphan_small:
        group_size = len(group["rows"])
        if packed_groups and packed_size + group_size > max_window:
            flush_packed_groups()
        packed_groups.append(group)
        packed_size += group_size
    flush_packed_groups()

    def best_capacity_target_for_rows(rows: List[Dict[str, Any]], *, exclude_idx: int) -> Optional[int]:
        best_idx: Optional[int] = None
        best_similarity = -1.0
        for idx, window in enumerate(windows):
            if idx == exclude_idx:
                continue
            current_rows = window_rows(window)
            if len(current_rows) + len(rows) > max_window:
                continue
            similarity = _semantic_group_similarity(rows, current_rows, embeddings_by_id)
            if similarity > best_similarity:
                best_idx = idx
                best_similarity = similarity
        return best_idx

    def borrow_rows_into_window(target_idx: int) -> bool:
        nonlocal rebalanced_evidence_count, forced_merge_group_count
        target = windows[target_idx]
        target_size = window_size(target)
        if target_size >= min_window_size:
            return True
        capacity = max_window - target_size
        if capacity <= 0:
            return False
        deficit = min_window_size - target_size
        donor_indices = sorted(
            [idx for idx in range(len(windows)) if idx != target_idx],
            key=lambda idx: window_size(windows[idx]),
            reverse=True,
        )
        for donor_idx in donor_indices:
            donor = windows[donor_idx]
            donor_size = window_size(donor)
            max_take = min(deficit, capacity, donor_size - min_window_size)
            if max_take <= 0:
                continue
            donor_groups = donor.get("groups") or []
            for group_idx in range(len(donor_groups) - 1, -1, -1):
                group = donor_groups[group_idx]
                rows = group.get("rows") or []
                if not rows:
                    continue
                take_count = min(max_take, len(rows))
                moved_rows = rows[-take_count:]
                group["rows"] = rows[:-take_count]
                target["groups"].append(
                    {
                        "rows": moved_rows,
                        "source_cluster_index": group.get("source_cluster_index"),
                        "split": True,
                        "merged": True,
                        "rebalanced": True,
                    }
                )
                target["merged_small_group_count"] += 1
                rebalanced_evidence_count += take_count
                forced_merge_group_count += 1
                prune_empty_groups(donor)
                return True
        return False

    while True:
        undersized_indices = [
            idx
            for idx, window in enumerate(windows)
            if 0 < window_size(window) < min_window_size
        ]
        if not undersized_indices:
            break
        changed = False
        for small_idx in undersized_indices:
            if small_idx >= len(windows):
                continue
            small_window = windows[small_idx]
            if not (0 < window_size(small_window) < min_window_size):
                continue
            small_rows = window_rows(small_window)
            target_idx = best_capacity_target_for_rows(small_rows, exclude_idx=small_idx)
            if target_idx is not None:
                windows[target_idx]["groups"].extend(small_window.get("groups") or [])
                windows[target_idx]["merged_small_group_count"] += max(1, len(small_window.get("groups") or []))
                forced_merge_group_count += max(1, len(small_window.get("groups") or []))
                windows.pop(small_idx)
                changed = True
                break
            if borrow_rows_into_window(small_idx):
                changed = True
                break
        if not changed:
            break

    natural_window_count = len(windows)
    total_active_cards = len(active_rows or [])
    capacity_floor_window_count = max(1, (total_active_cards + max_window - 1) // max_window)
    requested_window_count = int(max_windows)
    if requested_window_count <= 0:
        target_window_count = natural_window_count
        window_budget_reason = "all"
    else:
        target_window_count = min(
            natural_window_count,
            max(capacity_floor_window_count, requested_window_count),
        )
        window_budget_reason = (
            "capacity_floor"
            if capacity_floor_window_count > requested_window_count
            else "target"
        )

    def window_utility(window: Dict[str, Any]) -> float:
        return _semantic_cluster_utility(window_rows(window))

    def merge_window_pair(left_idx: int, right_idx: int) -> None:
        nonlocal forced_merge_group_count
        if left_idx == right_idx:
            return
        if right_idx < left_idx:
            left_idx, right_idx = right_idx, left_idx
        left = windows[left_idx]
        right = windows[right_idx]
        moved_groups = right.get("groups") or []
        for group in moved_groups:
            group["merged"] = True
        left["groups"].extend(moved_groups)
        left["merged_small_group_count"] += max(1, len(moved_groups))
        forced_merge_group_count += max(1, len(moved_groups))
        windows.pop(right_idx)

    def best_direct_window_pair() -> Optional[Tuple[int, int]]:
        best_pair: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[float, int, float]] = None
        for left_idx in range(len(windows)):
            left_rows = window_rows(windows[left_idx])
            for right_idx in range(left_idx + 1, len(windows)):
                right_rows = window_rows(windows[right_idx])
                combined_size = len(left_rows) + len(right_rows)
                if combined_size > max_window:
                    continue
                similarity = _semantic_group_similarity(left_rows, right_rows, embeddings_by_id)
                utility = window_utility(windows[left_idx]) + window_utility(windows[right_idx])
                key = (similarity, -combined_size, utility)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = (left_idx, right_idx)
        return best_pair

    def best_capacity_target_for_rebalanced_rows(
        rows: List[Dict[str, Any]],
        *,
        exclude_idx: int,
        require_full_fit: bool,
    ) -> Optional[int]:
        best_idx: Optional[int] = None
        best_similarity = -1.0
        for idx, window in enumerate(windows):
            if idx == exclude_idx:
                continue
            capacity = max_window - window_size(window)
            if capacity <= 0:
                continue
            if require_full_fit and len(rows) > capacity:
                continue
            current_rows = window_rows(window)
            sample_rows = rows[: max(1, min(len(rows), capacity))]
            similarity = _semantic_group_similarity(sample_rows, current_rows, embeddings_by_id)
            if similarity > best_similarity:
                best_idx = idx
                best_similarity = similarity
        return best_idx

    def eliminate_window_by_rebalancing(source_idx: int) -> bool:
        nonlocal rebalanced_evidence_count, forced_merge_group_count
        source = windows[source_idx]
        source_groups = list(source.get("groups") or [])
        available_capacity = sum(
            max(0, max_window - window_size(window))
            for idx, window in enumerate(windows)
            if idx != source_idx
        )
        source_size = window_size(source)
        if source_size <= 0:
            windows.pop(source_idx)
            return True
        if available_capacity < source_size:
            return False

        for group in source_groups:
            rows = list(group.get("rows") or [])
            if not rows:
                continue
            target_idx = best_capacity_target_for_rebalanced_rows(
                rows,
                exclude_idx=source_idx,
                require_full_fit=True,
            )
            if target_idx is not None:
                group["merged"] = True
                windows[target_idx]["groups"].append(group)
                windows[target_idx]["merged_small_group_count"] += 1
                forced_merge_group_count += 1
                continue

            remaining_rows = rows
            while remaining_rows:
                target_idx = best_capacity_target_for_rebalanced_rows(
                    remaining_rows,
                    exclude_idx=source_idx,
                    require_full_fit=False,
                )
                if target_idx is None:
                    return False
                capacity = max_window - window_size(windows[target_idx])
                take_count = min(capacity, len(remaining_rows))
                moved_rows = remaining_rows[:take_count]
                remaining_rows = remaining_rows[take_count:]
                windows[target_idx]["groups"].append(
                    {
                        "rows": moved_rows,
                        "source_cluster_index": group.get("source_cluster_index"),
                        "split": True,
                        "merged": True,
                        "rebalanced": True,
                    }
                )
                windows[target_idx]["merged_small_group_count"] += 1
                rebalanced_evidence_count += take_count
                forced_merge_group_count += 1

        windows.pop(source_idx)
        return True

    while len(windows) > target_window_count:
        pair = best_direct_window_pair()
        if pair is not None:
            merge_window_pair(*pair)
            continue

        source_indices = sorted(
            range(len(windows)),
            key=lambda idx: (
                window_size(windows[idx]),
                window_utility(windows[idx]),
                idx,
            ),
        )
        changed = False
        for source_idx in source_indices:
            if eliminate_window_by_rebalancing(source_idx):
                changed = True
                break
        if not changed:
            break

    underfilled_window_count = sum(
        1 for window in windows if 0 < window_size(window) < min_window_size
    )

    windows.sort(
        key=lambda window: (
            -_semantic_cluster_utility([row for group in window["groups"] for row in group["rows"]]),
            _l2_evidence_priority(window["groups"][0]["rows"][0]),
        )
    )
    selected_windows = windows

    planned: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    planned_window_count = len(selected_windows)
    raw_cluster_sizes = sorted((len(cluster) for cluster in raw_clusters), reverse=True)
    final_window_sizes = [window_size(window) for window in selected_windows]
    for window_idx, window in enumerate(selected_windows):
        selected_groups = [group["rows"] for group in window["groups"]]
        selected = [row for group in selected_groups for row in group]
        info = _l2_selection_info_for_groups(
            evidence_pool=evidence_pool,
            active_rows=active_rows,
            stale_count=stale_count,
            selected_groups=selected_groups,
            grouping="semantic",
            truncated_group_count=truncated_group_count,
            planned_window_index=window_idx,
            planned_window_count=planned_window_count,
        )
        info.update(embedding_stats)
        info.update(_semantic_pairwise_stats(selected, embeddings_by_id))
        info.update(method_info)
        info.update(
            {
                "semantic_cluster_count": len(raw_clusters),
                "semantic_cluster_sizes": raw_cluster_sizes,
                "semantic_window_split_count": sum(1 for group in window["groups"] if group.get("split")),
                "semantic_window_merged_group_count": int(window.get("merged_small_group_count") or 0),
                "semantic_window_forced_merge_group_count": forced_merge_group_count,
                "semantic_window_rebalanced_evidence_count": rebalanced_evidence_count,
                "semantic_underfilled_window_count": underfilled_window_count,
                "semantic_natural_window_count": natural_window_count,
                "semantic_target_window_count": requested_window_count,
                "semantic_capacity_floor_window_count": capacity_floor_window_count,
                "semantic_final_window_count": len(selected_windows),
                "semantic_window_budget_reason": window_budget_reason,
                "semantic_final_window_sizes": final_window_sizes,
                "semantic_dropped_small_group_count": 0,
                "semantic_dropped_small_evidence_count": 0,
                "semantic_cluster_threshold": threshold,
                "semantic_cluster_top_k": top_k,
                "semantic_cluster_min_window_size": min_window_size,
                "semantic_cluster_merge_threshold": merge_threshold,
            }
        )
        planned.append((selected, info))
    return planned


def _plan_epoch_l2_evidence_windows(
    evidence_pool: List[Dict[str, Any]],
    *,
    max_window: int,
    max_windows: int,
    grouping: str,
    semantic_options: Optional[Dict[str, Any]] = None,
    exclude_evidence_ids: Optional[List[str]] = None,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    max_window = max(0, int(max_window))
    max_windows = int(max_windows)
    raw_grouping = str(grouping or "").strip().lower()
    grouping = raw_grouping if raw_grouping in {"card", "round", "semantic"} else "card"
    active_rows = _l2_active_rows(evidence_pool, exclude_evidence_ids=exclude_evidence_ids)
    stale_count = _l2_stale_count(evidence_pool)
    if max_window <= 0 or not active_rows:
        return []

    if grouping == "semantic":
        return _plan_semantic_l2_evidence_windows(
            evidence_pool,
            active_rows,
            stale_count,
            max_window=max_window,
            max_windows=max_windows,
            semantic_options=semantic_options,
        )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in sorted(active_rows, key=_l2_evidence_priority):
        grouped.setdefault(_l2_evidence_group_key(row, grouping=grouping), []).append(row)

    groups: List[List[Dict[str, Any]]] = []
    for rows in grouped.values():
        ordered = sorted(rows, key=_l2_evidence_priority)
        if grouping == "round" and len(ordered) > max_window:
            groups.extend(ordered[idx : idx + max_window] for idx in range(0, len(ordered), max_window))
        else:
            groups.append(ordered)
    groups = [group for group in groups if group]
    groups.sort(key=lambda group: _l2_evidence_priority(group[0]))
    if not groups:
        return []

    total_cards = sum(len(group) for group in groups)
    natural_window_count = max(1, (total_cards + max_window - 1) // max_window)
    window_count = natural_window_count if max_windows <= 0 else min(max_windows, natural_window_count)
    windows: List[List[List[Dict[str, Any]]]] = []
    remaining_groups = list(groups)

    for window_idx in range(window_count):
        remaining_slots = window_count - window_idx
        remaining_cards = sum(len(group) for group in remaining_groups)
        if remaining_cards <= 0:
            break
        target_size = min(max_window, max(1, (remaining_cards + remaining_slots - 1) // remaining_slots))
        current_groups: List[List[Dict[str, Any]]] = []
        current_size = 0

        while remaining_groups and current_size < max_window:
            preferred_idx: Optional[int] = None
            for idx, group in enumerate(remaining_groups):
                group_size = len(group)
                if current_size + group_size <= target_size:
                    preferred_idx = idx
                    break
            if preferred_idx is None:
                if current_groups:
                    break
                for idx, group in enumerate(remaining_groups):
                    if current_size + len(group) <= max_window:
                        preferred_idx = idx
                        break
            if preferred_idx is None:
                break
            group = remaining_groups.pop(preferred_idx)
            current_groups.append(group)
            current_size += len(group)

        if current_groups:
            windows.append(current_groups)

    planned: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    planned_window_count = len(windows)
    for window_idx, selected_groups in enumerate(windows):
        selected = [row for group in selected_groups for row in group]
        info = _l2_selection_info_for_groups(
            evidence_pool=evidence_pool,
            active_rows=active_rows,
            stale_count=stale_count,
            selected_groups=selected_groups,
            grouping=grouping,
            planned_window_index=window_idx,
            planned_window_count=planned_window_count,
        )
        planned.append((selected, info))
    return planned


def _plan_source_split_l2_evidence_windows(
    evidence_pool: List[Dict[str, Any]],
    *,
    max_window: int,
    max_windows: int,
    grouping: str,
    semantic_options: Optional[Dict[str, Any]] = None,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if str(grouping or "").lower() != "semantic":
        return _plan_epoch_l2_evidence_windows(
            evidence_pool,
            max_window=max_window,
            max_windows=max_windows,
            grouping=grouping,
            semantic_options=semantic_options,
        )

    active_rows = _l2_active_rows(evidence_pool)
    failure_rows = [
        row
        for row in active_rows
        if str(row.get("source_type") or _source_type_from_experience(row.get("experience_type"))) == "failure"
    ]
    success_rows = [
        row
        for row in active_rows
        if str(row.get("source_type") or _source_type_from_experience(row.get("experience_type"))) == "success"
    ]
    total = len(failure_rows) + len(success_rows)
    if total <= 0:
        return []

    if max_windows > 0 and failure_rows and success_rows:
        failure_target = max(1, round(max_windows * (len(failure_rows) / max(1, total))))
        failure_target = min(max_windows - 1, failure_target)
        success_target = max(1, max_windows - failure_target)
    elif max_windows > 0 and failure_rows:
        failure_target, success_target = max_windows, 0
    elif max_windows > 0 and success_rows:
        failure_target, success_target = 0, max_windows
    else:
        failure_target, success_target = 0, 0

    planned: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    for source_type, rows, target in [
        ("failure", failure_rows, failure_target),
        ("success", success_rows, success_target),
    ]:
        if not rows:
            continue
        source_plans = _plan_semantic_l2_evidence_windows(
            rows,
            rows,
            stale_count=0,
            max_window=max_window,
            max_windows=target,
            semantic_options=semantic_options,
        )
        for selected, info in source_plans:
            info["source_type"] = source_type
            info["source_pool_size"] = len(rows)
            planned.append((selected, info))

    for idx, (_, info) in enumerate(planned):
        info["planned_window_index"] = idx
        info["planned_window_count"] = len(planned)
    return planned


def _mark_evidence_attempts(
    rows: List[Dict[str, Any]],
    evidence_ids: List[str],
    *,
    round_idx: int,
    outcome: str,
    max_rejects: int = L2_MAX_REJECTS_PER_EVIDENCE,
    count_attempt: bool = True,
) -> Dict[str, List[str]]:
    id_set = {str(eid) for eid in evidence_ids or [] if str(eid)}
    updated = {"attempted": [], "accepted": [], "rejected": [], "staled": []}
    if not id_set:
        return updated
    for row in rows or []:
        eid = str(row.get("evidence_id") or "")
        if eid not in id_set:
            continue
        _normalize_evidence_runtime_state(row)
        if bool(row.get("archived")) or row.get("l2_status") == EVIDENCE_STATUS_ARCHIVED:
            continue
        if count_attempt:
            row["l2_attempt_count"] = _evidence_int(row, "l2_attempt_count") + 1
            updated["attempted"].append(eid)
        row["l2_last_attempt_round"] = int(round_idx)
        if outcome == "accept":
            row["l2_accept_count"] = _evidence_int(row, "l2_accept_count") + 1
            updated["accepted"].append(eid)
        elif outcome == "reject":
            row["l2_reject_count"] = _evidence_int(row, "l2_reject_count") + 1
            updated["rejected"].append(eid)
            if _evidence_int(row, "l2_reject_count") >= int(max_rejects):
                row["l2_status"] = EVIDENCE_STATUS_STALE
                updated["staled"].append(eid)
    return {key: sorted(set(value)) for key, value in updated.items()}


def _mark_l2_window_attempts(
    rows: List[Dict[str, Any]],
    evidence_ids: List[str],
    *,
    round_idx: int,
) -> List[str]:
    id_set = {str(eid) for eid in evidence_ids or [] if str(eid)}
    updated: List[str] = []
    if not id_set:
        return updated
    for row in rows or []:
        eid = str(row.get("evidence_id") or "")
        if eid not in id_set:
            continue
        _normalize_evidence_runtime_state(row)
        if bool(row.get("archived")) or row.get("l2_status") == EVIDENCE_STATUS_ARCHIVED:
            continue
        row["l2_attempt_count"] = _evidence_int(row, "l2_attempt_count") + 1
        row["l2_last_attempt_round"] = int(round_idx)
        updated.append(eid)
    return sorted(set(updated))


def _mark_uncited_selected_evidence(
    rows: List[Dict[str, Any]],
    *,
    selected_evidence_ids: List[str],
    cited_evidence_ids: List[str],
    round_idx: int,
    max_rejects: int = L2_MAX_REJECTS_PER_EVIDENCE,
) -> Dict[str, List[str]]:
    selected = {str(eid) for eid in selected_evidence_ids or [] if str(eid)}
    cited = {str(eid) for eid in cited_evidence_ids or [] if str(eid)}
    uncited = sorted(selected - cited)
    if not uncited:
        return {"attempted": [], "accepted": [], "rejected": [], "staled": []}
    return _mark_evidence_attempts(
        rows,
        uncited,
        round_idx=round_idx,
        outcome="reject",
        max_rejects=max_rejects,
        count_attempt=False,
    )


def _archive_evidence_rows(rows: List[Dict[str, Any]], evidence_ids: List[str]) -> None:
    id_set = {str(eid) for eid in evidence_ids or [] if str(eid)}
    if not id_set:
        return
    for row in rows or []:
        if str(row.get("evidence_id") or "") in id_set:
            row["archived"] = True
            row["l2_status"] = EVIDENCE_STATUS_ARCHIVED


def _archive_evidence_rows_with_reason(
    rows: List[Dict[str, Any]], evidence_ids: List[str], *, reason: str
) -> List[str]:
    id_set = {str(eid) for eid in evidence_ids or [] if str(eid)}
    archived: List[str] = []
    for row in rows or []:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id not in id_set:
            continue
        row["archived"] = True
        row["l2_status"] = EVIDENCE_STATUS_ARCHIVED
        row["archive_reason"] = str(reason or "covered_by_committed_skill")
        row.pop("defer_reason", None)
        row.pop("consume_reason", None)
        archived.append(evidence_id)
    return sorted(set(archived))


def _consume_evidence_rows(rows: List[Dict[str, Any]], evidence_ids: List[str], *, reason: str) -> List[str]:
    id_set = {str(eid) for eid in evidence_ids or [] if str(eid)}
    consumed: List[str] = []
    if not id_set:
        return consumed
    for row in rows or []:
        eid = str(row.get("evidence_id") or "")
        if eid not in id_set:
            continue
        _normalize_evidence_runtime_state(row)
        if bool(row.get("archived")) or row.get("l2_status") == EVIDENCE_STATUS_ARCHIVED:
            continue
        row["l2_status"] = EVIDENCE_STATUS_CONSUMED
        row["consume_reason"] = str(reason or "consumed")
        row["archived"] = False
        consumed.append(eid)
    return sorted(set(consumed))


def _defer_evidence_rows(rows: List[Dict[str, Any]], evidence_ids: List[str], *, reason: str) -> List[str]:
    id_set = {str(eid) for eid in evidence_ids or [] if str(eid)}
    deferred: List[str] = []
    if not id_set:
        return deferred
    for row in rows or []:
        eid = str(row.get("evidence_id") or "")
        if eid not in id_set:
            continue
        _normalize_evidence_runtime_state(row)
        status = str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower()
        if bool(row.get("archived")) or status in {EVIDENCE_STATUS_ARCHIVED, EVIDENCE_STATUS_STALE}:
            continue
        row["l2_status"] = EVIDENCE_STATUS_ACTIVE
        row["archived"] = False
        row["defer_reason"] = str(reason or "deferred")
        row["l2_defer_count"] = _evidence_int(row, "l2_defer_count") + 1
        row.pop("consume_reason", None)
        deferred.append(eid)
    return sorted(set(deferred))


def _apply_working_branch_policy_defer_limit(
    rows: List[Dict[str, Any]],
    deferred_fates: Dict[str, str],
    *,
    dataset: str,
    args: Any,
    epoch_idx: int,
    validation_decision: str,
) -> Dict[str, Any]:
    limit = _int_arg(
        getattr(args, "working_branch_max_policy_defer_epochs", -1), -1
    )
    enabled = bool(
        (
            _is_appworld_dataset(dataset)
            or _defer_rule_ignored_enabled(dataset, args)
        )
        and _evidence_working_branch_enabled(args)
        and _resolve_epoch_skill_update_mode(args) == "cluster_step"
        and limit > 0
        and str(validation_decision or "").lower() != "evaluation_error"
    )
    info = {
        "enabled": enabled,
        "max_policy_defer_epochs": limit,
        "counted_evidence_ids": [],
        "staled_evidence_ids": [],
    }
    if not enabled:
        return info

    by_id = {
        str(row.get("evidence_id") or ""): row
        for row in rows or []
        if str(row.get("evidence_id") or "")
    }
    counted: List[str] = []
    staled: List[str] = []
    for evidence_id, reason in sorted((deferred_fates or {}).items()):
        if str(reason or "") not in WORKING_BRANCH_POLICY_DEFER_REASONS:
            continue
        row = by_id.get(str(evidence_id))
        if row is None:
            continue
        _normalize_evidence_runtime_state(row)
        status = str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower()
        if bool(row.get("archived")) or status in {
            EVIDENCE_STATUS_ARCHIVED,
            EVIDENCE_STATUS_STALE,
            EVIDENCE_STATUS_CONSUMED,
        }:
            continue
        last_epoch = _row_int(
            row, "working_branch_policy_last_defer_epoch", -1
        )
        if last_epoch != int(epoch_idx):
            row["working_branch_policy_defer_count"] = (
                _evidence_int(row, "working_branch_policy_defer_count") + 1
            )
            row["working_branch_policy_last_defer_epoch"] = int(epoch_idx)
            counted.append(str(evidence_id))
        if _evidence_int(row, "working_branch_policy_defer_count") >= limit:
            row["l2_status"] = EVIDENCE_STATUS_STALE
            row["archived"] = False
            row["stale_reason"] = "working_branch_policy_defer_limit"
            staled.append(str(evidence_id))

    info["counted_evidence_ids"] = sorted(set(counted))
    info["staled_evidence_ids"] = sorted(set(staled))
    return info


def _run_selection_eval(
    *,
    runner,
    skill_path: Path,
    task_rows: List[Dict[str, Any]],
    desc: str,
    args: Any = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    task_refs = [row.get("task_ref", row.get("task_id")) for row in task_rows]
    results = runner.run_task_set(task_refs, MarkdownSkillBank(str(skill_path)), {}, desc=desc)
    stats = _selection_eval_stats(results)
    stats["task_results"] = _compact_selection_task_results(results, args=args)
    print(
        f"[{desc}] metrics score_avg={float(stats.get('score', 0.0) or 0.0):.4f} "
        f"success_rate={float(stats.get('success_rate', 0.0) or 0.0):.4f} "
        f"success={int(stats.get('success', 0) or 0)}/{int(stats.get('task_count', 0) or 0)} "
        f"steps_avg={float(stats.get('steps_avg', 0.0) or 0.0):.2f}",
        flush=True,
    )
    return stats, results


def _load_selection_baseline(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    row = load_json(str(path))
    if not isinstance(row, dict):
        return None
    if "score" not in row or "steps_avg" not in row:
        return None
    if not isinstance(row.get("task_results"), list):
        return None
    return row


def _selection_task_ids(rows: List[Dict[str, Any]]) -> List[str]:
    return [str(row.get("task_id") or "") for row in rows or []]


def _baseline_matches_selection_tasks(
    baseline: Optional[Dict[str, Any]],
    selection_tasks: List[Dict[str, Any]],
) -> bool:
    if not baseline:
        return False
    expected_ids = _selection_task_ids(selection_tasks)
    if int(baseline.get("task_count", -1) or -1) != len(expected_ids):
        return False
    baseline_ids = [str(x) for x in (baseline.get("task_ids") or []) if str(x)]
    if not baseline_ids:
        baseline_ids = [
            str(row.get("task_id") or "")
            for row in (baseline.get("task_results") or [])
            if isinstance(row, dict) and str(row.get("task_id") or "")
        ]
    return baseline_ids == expected_ids


def _save_selection_baseline(
    path: Path,
    *,
    stats: Dict[str, Any],
    round_idx: int,
    source: str,
    task_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    row = {
        "score": float(stats.get("score", 0.0) or 0.0),
        "steps_avg": float(stats.get("steps_avg", 0.0) or 0.0),
        "success": int(stats.get("success", 0) or 0),
        "success_rate": float(stats.get("success_rate", 0.0) or 0.0),
        "task_count": int(stats.get("task_count", 0) or 0),
        "task_ids": list(task_ids or []),
        "updated_round": int(round_idx),
        "source": source,
        "task_results": list(stats.get("task_results") or []),
    }
    save_json(str(path), row)
    return row


def _selection_result_by_task(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("task_id") or ""): row for row in results or [] if str(row.get("task_id") or "")}


def _is_selection_success(row: Dict[str, Any]) -> bool:
    return bool(row.get("success")) or float(row.get("score", 0.0) or 0.0) >= 1.0


def _selection_delta_cases(
    baseline_results: List[Dict[str, Any]],
    candidate_results: List[Dict[str, Any]],
    *,
    max_positive: int = SELECTION_FEEDBACK_MAX_POSITIVE,
    max_negative: int = SELECTION_FEEDBACK_MAX_NEGATIVE,
    step_delta: int = SELECTION_FEEDBACK_STEP_DELTA,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline_by_id = _selection_result_by_task(baseline_results)
    candidate_by_id = _selection_result_by_task(candidate_results)
    positive: List[Dict[str, Any]] = []
    negative: List[Dict[str, Any]] = []
    for task_id, base in baseline_by_id.items():
        cand = candidate_by_id.get(task_id)
        if cand is None:
            continue
        base_score = float(base.get("score", 0.0) or 0.0)
        cand_score = float(cand.get("score", 0.0) or 0.0)
        base_steps = int(base.get("steps", 0) or 0)
        cand_steps = int(cand.get("steps", 0) or 0)
        base_success = _is_selection_success(base)
        cand_success = _is_selection_success(cand)
        why = ""
        polarity = ""
        if (not base_success) and cand_success:
            polarity = "positive"
            why = "baseline failed; candidate succeeded"
        elif base_success and not cand_success:
            polarity = "negative"
            why = "baseline succeeded; candidate failed"
        elif cand_score > base_score:
            polarity = "positive"
            why = "candidate score improved"
        elif cand_score < base_score:
            polarity = "negative"
            why = "candidate score regressed"
        elif base_steps - cand_steps >= int(step_delta):
            polarity = "positive"
            why = "candidate used noticeably fewer steps"
        elif cand_steps - base_steps >= int(step_delta):
            polarity = "negative"
            why = "candidate used noticeably more steps"
        if not polarity:
            continue
        case = {
            "task_instruction": str(cand.get("task_instruction") or base.get("task_instruction") or ""),
            "task_family": str(cand.get("task_family") or base.get("task_family") or ""),
            "why_selected": why,
            "baseline_trace": base.get("trajectory") or [],
            "candidate_trace": cand.get("trajectory") or [],
        }
        if polarity == "positive":
            positive.append(case)
        else:
            negative.append(case)
    return positive[: max(0, int(max_positive))], negative[: max(0, int(max_negative))]


def _recent_selection_feedback(rows: List[Dict[str, Any]], limit: int = SELECTION_FEEDBACK_HISTORY) -> List[str]:
    feedback = []
    for row in reversed(rows or []):
        text = str(row.get("feedback") or "").strip()
        if text:
            feedback.append(text)
        if len(feedback) >= int(limit):
            break
    return list(reversed(feedback))


def _selection_feedback_id(rows: List[Dict[str, Any]]) -> str:
    return f"SF{len(rows) + 1:06d}"


def _extract_selection_feedback(
    *,
    teacher,
    dataset: str,
    selection_source: str,
    candidate_edits: List[Dict[str, Any]],
    positive_cases: List[Dict[str, Any]],
    negative_cases: List[Dict[str, Any]],
) -> str:
    if not positive_cases and not negative_cases:
        return ""
    payload = {
        "selection_source": selection_source,
        "candidate_edits": candidate_edits,
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
    }
    try:
        raw = _teacher_chat_json(teacher, prompts_for_dataset(dataset)["selection_feedback_extractor"], payload)
    except Exception:
        return ""
    return str((raw or {}).get("feedback") or "").strip()


def _extract_rejected_edit_feedback(
    *,
    teacher,
    dataset: str,
    rejected_cases: List[Dict[str, Any]],
) -> str:
    if not rejected_cases:
        return ""
    payload = {
        "selection_source": "skill_edit_judge_reject",
        "candidate_edits": [case.get("candidate_edit") for case in rejected_cases if case.get("candidate_edit")],
        "positive_cases": [],
        "negative_cases": [],
        "rejected_edit_cases": rejected_cases,
    }
    try:
        raw = _teacher_chat_json(teacher, prompts_for_dataset(dataset)["selection_feedback_extractor"], payload)
    except Exception:
        return ""
    return str((raw or {}).get("feedback") or "").strip()


def _compact_feedback_evidence_card(card: Dict[str, Any]) -> Dict[str, Any]:
    public = _public_evidence_card(card)
    return {
        key: public.get(key)
        for key in [
            "evidence_id",
            "task_instruction",
            "task_family",
            "source_type",
            "experience_type",
            "failure_type",
            "pattern",
            "proposed_edit",
            "support_count",
            "merge_level",
            "evidence_quality_judge_reason",
        ]
        if public.get(key) not in (None, "")
    }


def _compact_feedback_replay_evidence(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact_rows = []
    for row in rows or []:
        compact_rows.append(
            {
                "evidence_id": row.get("evidence_id"),
                "range_index": row.get("range_index"),
                "trigger_ranges": row.get("trigger_ranges") or [],
                "before_segments": row.get("before_segments") or [],
                "after_segments": row.get("after_segments") or [],
                "error": row.get("error"),
            }
        )
    return compact_rows


def _append_selection_feedback(
    *,
    rows: List[Dict[str, Any]],
    feedback: str,
    round_idx: int,
    source: str,
) -> Optional[Dict[str, Any]]:
    text = str(feedback or "").strip()
    if not text:
        return None
    row = {
        "feedback_id": _selection_feedback_id(rows),
        "round": int(round_idx),
        "source": source,
        "feedback": text,
    }
    rows.append(row)
    return row


def _selection_gate(
    *,
    runner,
    out: Path,
    current_skill_path: Path,
    candidate_skill_md: str,
    selection_tasks: List[Dict[str, Any]],
    round_idx: int,
    enabled: bool,
    args: Any = None,
) -> Dict[str, Any]:
    candidate_path = out / "_tmp_l2_selection_candidate_skill.md"
    baseline_path = out / "selection_baseline.json"
    if not enabled:
        candidate_path.write_text(candidate_skill_md, encoding="utf-8")
        return {
            "enabled": False,
            "task_count": len(selection_tasks),
            "decision": "commit",
            "skill_path": str(candidate_path),
        }
    if not selection_tasks:
        return {
            "enabled": True,
            "task_count": 0,
            "decision": "reject",
            "reason": "selection set is empty",
            "skill_path": str(candidate_path),
        }
    candidate_path.write_text(candidate_skill_md, encoding="utf-8")
    try:
        baseline = _load_selection_baseline(baseline_path)
        baseline_source = "cache"
        if baseline is not None and not _baseline_matches_selection_tasks(baseline, selection_tasks):
            print(
                f"[l2 selection round {round_idx}] cached baseline does not match current "
                f"selection set; recomputing current baseline",
                flush=True,
            )
            baseline = None
        if baseline is None:
            current_stats, _ = _run_selection_eval(
                runner=runner,
                skill_path=current_skill_path,
                task_rows=selection_tasks,
                desc=f"l2 selection round {round_idx} current",
                args=args,
            )
            baseline = _save_selection_baseline(
                baseline_path,
                stats=current_stats,
                round_idx=round_idx,
                source="current",
                task_ids=_selection_task_ids(selection_tasks),
            )
            baseline_source = "current_eval"

        candidate_stats, _ = _run_selection_eval(
            runner=runner,
            skill_path=candidate_path,
            task_rows=selection_tasks,
            desc=f"l2 selection round {round_idx} candidate",
            args=args,
        )
        current_score = float(baseline.get("score", 0.0) or 0.0)
        current_steps = float(baseline.get("steps_avg", 0.0) or 0.0)
        candidate_score = float(candidate_stats.get("score", 0.0) or 0.0)
        candidate_steps = float(candidate_stats.get("steps_avg", 0.0) or 0.0)
        current_success = int(baseline.get("success", 0) or 0)
        current_success_rate = float(
            baseline.get(
                "success_rate",
                current_success / len(selection_tasks) if selection_tasks else 0.0,
            )
            or 0.0
        )
        candidate_success = int(candidate_stats.get("success", 0) or 0)
        candidate_success_rate = float(candidate_stats.get("success_rate", 0.0) or 0.0)
        success_margin = float(
            getattr(args, "selection_success_margin", DEFAULT_SELECTION_SUCCESS_MARGIN)
            if args is not None
            else DEFAULT_SELECTION_SUCCESS_MARGIN
        )
        score_margin = float(
            getattr(args, "selection_score_tiebreak_margin", DEFAULT_SELECTION_SCORE_TIEBREAK_MARGIN)
            if args is not None
            else DEFAULT_SELECTION_SCORE_TIEBREAK_MARGIN
        )
        step_delta = float(
            getattr(args, "selection_step_tiebreak_delta", DEFAULT_SELECTION_STEP_TIEBREAK_DELTA)
            if args is not None
            else DEFAULT_SELECTION_STEP_TIEBREAK_DELTA
        )
        strict_hard_gate = str(getattr(args, "dataset", "") or "").lower() in {
            "alfworld",
            "appworld",
        }
        accept_tie = bool(
            getattr(args, "selection_accept_tie", DEFAULT_SELECTION_ACCEPT_TIE)
            if args is not None
            else DEFAULT_SELECTION_ACCEPT_TIE
        )
        eps = 1e-12
        if candidate_success_rate > current_success_rate + success_margin + eps:
            decision = "commit"
            decision_reason = "success_rate_improved"
        elif candidate_success_rate + eps < current_success_rate:
            decision = "reject"
            decision_reason = "success_rate_regressed"
        elif abs(candidate_success_rate - current_success_rate) <= eps and accept_tie:
            decision = "commit"
            decision_reason = "success_rate_tie_accepted"
        elif strict_hard_gate:
            decision = "reject"
            decision_reason = "hard_success_rate_tie"
        elif candidate_score > current_score + score_margin + eps:
            decision = "commit"
            decision_reason = "soft_score_tiebreak"
        elif abs(candidate_score - current_score) <= eps and (current_steps - candidate_steps) >= step_delta - eps:
            decision = "commit"
            decision_reason = "steps_tiebreak"
        else:
            decision = "reject"
            decision_reason = "no_hard_or_tiebreak_improvement"
        print(
            f"[l2 selection round {round_idx} validation] "
            f"current_score={current_score:.4f} current_success_rate={current_success_rate:.4f} "
            f"current_success={current_success}/{len(selection_tasks)} current_steps_avg={current_steps:.2f} "
            f"candidate_score={candidate_score:.4f} candidate_success_rate={candidate_success_rate:.4f} "
            f"candidate_success={candidate_success}/{len(selection_tasks)} candidate_steps_avg={candidate_steps:.2f} "
            f"decision={decision} reason={decision_reason}",
            flush=True,
        )
        if decision == "commit":
            _save_selection_baseline(
                baseline_path,
                stats=candidate_stats,
                round_idx=round_idx,
                source="candidate",
                task_ids=_selection_task_ids(selection_tasks),
            )
        return {
            "enabled": True,
            "task_count": len(selection_tasks),
            "current_score": current_score,
            "current_steps_avg": current_steps,
            "candidate_score": candidate_score,
            "candidate_steps_avg": candidate_steps,
            "baseline_task_results": list(baseline.get("task_results") or []),
            "candidate_task_results": list(candidate_stats.get("task_results") or []),
            "decision": decision,
            "skill_path": str(candidate_path),
            "baseline_path": str(baseline_path),
            "baseline_source": baseline_source,
            "current_success": current_success,
            "current_success_rate": current_success_rate,
            "candidate_success": candidate_success,
            "candidate_success_rate": candidate_success_rate,
            "decision_reason": decision_reason,
            "gate_metric": (
                "hard_success_rate_non_regression"
                if strict_hard_gate and accept_tie
                else "hard_success_rate"
                if strict_hard_gate
                else "success_rate_non_regression_with_tiebreaks"
                if accept_tie
                else "success_rate_with_tiebreaks"
            ),
            "accept_tie": accept_tie,
            "success_margin": success_margin,
            "score_tiebreak_margin": score_margin,
            "step_tiebreak_delta": step_delta,
        }
    except Exception as exc:
        print(
            f"[l2 selection round {round_idx} validation] "
            f"evaluation_error={exc}",
            flush=True,
        )
        return {
            "enabled": True,
            "task_count": len(selection_tasks),
            "decision": "evaluation_error",
            "reason": "selection_evaluation_error",
            "error": str(exc),
            "skill_path": str(candidate_path),
        }


def _normalize_results_for_records(results, args):
    return normalize_task_results(
        results,
        max_steps=getattr(args, "record_max_steps", getattr(args, "max_steps", 50)),
        max_observation_chars=getattr(args, "record_max_observation_chars", 1500),
        trajectory_mode=getattr(args, "record_trajectory_mode", "full"),
        trajectory_head_steps=getattr(args, "record_trajectory_head_steps", 1),
        trajectory_tail_steps=getattr(args, "record_trajectory_tail_steps", 8),
        action_only=getattr(args, "record_action_only", False),
    )


def _results_to_payload(results: List[Dict[str, Any]], args) -> Tuple[Any, Dict[str, Any]]:
    records = _normalize_results_for_records(results, args)
    payload = task_records_payload(records)
    return records, payload


def _original_segment(
    result: Dict[str, Any],
    start: int,
    end: int,
    *,
    max_observation_chars: Optional[int] = None,
    action_only: bool = False,
) -> List[Dict[str, Any]]:
    rows = [
        x for x in result.get("trajectory", []) or []
        if int(start) <= int(x.get("step", -1)) <= int(end)
    ]
    return _compact_segment(rows, max_observation_chars=max_observation_chars, action_only=action_only)


def _markdown_patch_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("edits")
    if not isinstance(items, list):
        return []
    return [dict(x) for x in items if isinstance(x, dict)]


def _id_list_from_items(value: Any, keys: List[str]) -> List[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    ids: List[str] = []
    seen = set()
    for raw in raw_items:
        if isinstance(raw, dict):
            raw_id = ""
            for key in keys:
                raw_id = str(raw.get(key) or "").strip()
                if raw_id:
                    break
        else:
            raw_id = str(raw or "").strip()
        if raw_id and raw_id not in seen:
            seen.add(raw_id)
            ids.append(raw_id)
    return ids


def _omitted_evidence_ids(payload: Dict[str, Any]) -> List[str]:
    if not isinstance(payload, dict):
        return []
    values: List[str] = []
    for key in ["omitted_evidence", "omitted_evidence_ids"]:
        values.extend(_id_list_from_items(payload.get(key), ["evidence_id", "id"]))
    return _evidence_id_list(values)


def _omitted_source_edit_ids(payload: Dict[str, Any]) -> List[str]:
    if not isinstance(payload, dict):
        return []
    values: List[str] = []
    for key in ["omitted_source_edits", "omitted_source_edit_ids"]:
        values.extend(
            _id_list_from_items(
                payload.get(key),
                ["source_edit_id", "candidate_edit_id", "edit_id", "id"],
            )
        )
    return _evidence_id_list(values)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _evidence_id_list(value: Any) -> List[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    ids: List[str] = []
    seen = set()
    for raw in raw_items:
        eid = str(raw or "").strip()
        if eid and eid not in seen:
            seen.add(eid)
            ids.append(eid)
    return ids


def _normalize_proposed_edit(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    op = str(raw.get("op") or "").strip().lower()
    if op not in {"append", "insert_after", "insert_after_section", "replace", "delete"}:
        return None
    target = strip_longitudinal_memory_markers(str(raw.get("target") or "")).strip()
    content = strip_longitudinal_memory_markers(str(raw.get("content") or "")).strip()
    if op in {"insert_after", "insert_after_section", "replace", "delete"} and not target:
        return None
    if op in {"append", "insert_after", "insert_after_section", "replace"} and not content:
        return None
    edit: Dict[str, Any] = {"op": op}
    if op in {"insert_after", "insert_after_section", "replace", "delete"}:
        edit["target"] = target
    if op in {"append", "insert_after", "insert_after_section", "replace"}:
        edit["content"] = content
    return edit


def _make_standalone_contrastive_edit(
    origin_edit: Dict[str, Any],
    proposed_edit: Dict[str, Any],
    *,
    origin_is_provisional: bool = True,
) -> Optional[Dict[str, Any]]:
    """Convert a correction into a complete edit that does not depend on its origin."""
    origin = _normalize_proposed_edit(origin_edit)
    proposed = _normalize_proposed_edit(proposed_edit)
    if origin is None or proposed is None or origin.get("op") == "delete":
        return None
    origin_content = str(origin.get("content") or "").strip()
    if not origin_content:
        return None

    proposed_op = str(proposed.get("op") or "")
    proposed_target = str(proposed.get("target") or "").strip()
    proposed_content = str(proposed.get("content") or "").strip()
    origin_target = str(origin.get("target") or "").strip()

    if proposed_op == origin.get("op") and (
        proposed_op == "append" or proposed_target == origin_target
    ):
        if proposed_op == "append" and origin_content not in proposed_content:
            corrected_content = origin_content + "\n\n" + proposed_content
        else:
            corrected_content = proposed_content
    elif proposed_op == "append":
        corrected_content = origin_content + "\n\n" + proposed_content
    else:
        updated, report = _apply_markdown_edit(
            origin_content,
            {**proposed, "evidence_ids": ["__contrastive_correction__"]},
        )
        if not bool((report or {}).get("applied")):
            return None
        corrected_content = str(updated or "").strip()

    if not corrected_content or corrected_content == origin_content:
        return None
    if origin_is_provisional:
        standalone: Dict[str, Any] = {"op": str(origin.get("op") or "")}
        if origin_target:
            standalone["target"] = origin_target
        standalone["content"] = corrected_content
    else:
        standalone = {
            "op": "replace",
            "target": origin_content,
            "content": corrected_content,
        }
    return _normalize_proposed_edit(standalone)


def _evidence_edit_text(card: Dict[str, Any]) -> str:
    proposed = _normalize_proposed_edit((card or {}).get("proposed_edit"))
    if proposed:
        return " ".join(
            str(proposed.get(key) or "")
            for key in ["op", "target", "content"]
            if str(proposed.get(key) or "")
        ).strip()
    return ""


def _normalize_markdown_edit(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    op = str((raw or {}).get("op") or "").strip().lower()
    if op not in {"append", "insert_after", "insert_after_section", "replace", "delete"}:
        return None
    evidence_ids = _evidence_id_list(raw.get("evidence_ids"))
    if not evidence_ids:
        return None
    edit = {
        "op": op,
        "evidence_ids": evidence_ids,
    }
    target = strip_longitudinal_memory_markers(str(raw.get("target") or ""))
    content = strip_longitudinal_memory_markers(str(raw.get("content") or ""))
    if op in {"insert_after", "insert_after_section", "replace", "delete"}:
        edit["target"] = target
    if op in {"append", "insert_after", "insert_after_section", "replace"}:
        edit["content"] = content
    return edit


def _diagnose_markdown_edit(
    raw: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Explain why a proposer edit is or is not eligible for the current window."""
    raw = dict(raw or {})
    allowed_ops = {"append", "insert_after", "insert_after_section", "replace", "delete"}
    op = str(raw.get("op") or "").strip().lower()
    evidence_ids = _evidence_id_list(raw.get("evidence_ids"))
    known_ids = [eid for eid in evidence_ids if eid in evidence_by_id]
    unknown_ids = [eid for eid in evidence_ids if eid not in evidence_by_id]
    rejection_reasons: List[str] = []
    shape_warnings: List[str] = []
    if op not in allowed_ops:
        rejection_reasons.append("invalid_op")
    if not evidence_ids:
        rejection_reasons.append("missing_evidence_ids")
    elif not known_ids:
        rejection_reasons.append("unknown_evidence_ids")

    target = strip_longitudinal_memory_markers(str(raw.get("target") or "")).strip()
    content = strip_longitudinal_memory_markers(str(raw.get("content") or "")).strip()
    if op in {"insert_after", "insert_after_section", "replace", "delete"} and not target:
        shape_warnings.append("empty_target")
    if op in {"append", "insert_after", "insert_after_section", "replace"} and not content:
        shape_warnings.append("empty_content")

    normalized = _normalize_markdown_edit(raw)
    linked = bool(normalized and known_ids)
    if not linked and not rejection_reasons:
        rejection_reasons.append("no_linked_evidence")
    return {
        "op": op,
        "evidence_ids": evidence_ids,
        "known_evidence_ids": known_ids,
        "unknown_evidence_ids": unknown_ids,
        "normalized": normalized is not None,
        "has_linked_evidence": linked,
        "accepted": bool(normalized and linked),
        "rejection_reasons": rejection_reasons,
        "shape_warnings": shape_warnings,
        "normalized_edit": normalized,
    }


def _fallback_markdown_edit_from_evidence(card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    proposed = _normalize_proposed_edit((card or {}).get("proposed_edit"))
    evidence_id = str((card or {}).get("evidence_id") or "").strip()
    if proposed is None or not evidence_id:
        return None
    raw = dict(proposed)
    raw["evidence_ids"] = [evidence_id]
    return _normalize_markdown_edit(raw)


def _markdown_heading_level(line: str) -> Optional[int]:
    stripped = str(line or "").strip()
    if not stripped.startswith("#"):
        return None
    hashes = 0
    for ch in stripped:
        if ch == "#":
            hashes += 1
            continue
        break
    if hashes <= 0 or hashes > 6:
        return None
    if len(stripped) <= hashes or stripped[hashes] != " ":
        return None
    return hashes


def _markdown_section_end(lines: List[str], start: int) -> int:
    level = _markdown_heading_level(lines[start])
    if level is None:
        return len(lines)
    for index in range(start + 1, len(lines)):
        child_level = _markdown_heading_level(lines[index])
        if child_level is not None and child_level <= level:
            return index
    return len(lines)


def _split_markdown_insert_sections(content: str) -> Tuple[List[List[str]], List[str]]:
    lines = str(content or "").strip().splitlines()
    headings = [
        (index, _markdown_heading_level(line))
        for index, line in enumerate(lines)
        if _markdown_heading_level(line) is not None
    ]
    if not headings:
        return [], lines
    top_level = min(level for _, level in headings if level is not None)
    starts = [index for index, level in headings if level == top_level]
    prefix = lines[: starts[0]]
    sections = [
        lines[start : starts[position + 1] if position + 1 < len(starts) else len(lines)]
        for position, start in enumerate(starts)
    ]
    return sections, prefix


def _find_markdown_heading_index(lines: List[str], target: str) -> Optional[int]:
    target_level = _markdown_heading_level(target)
    for index, line in enumerate(lines):
        if line.strip() == str(target or "").strip() and _markdown_heading_level(line) == target_level:
            return index
    return None


def _merge_markdown_section_body(skill_md: str, inserted_section: List[str]) -> Tuple[str, bool]:
    if not inserted_section:
        return skill_md, False
    lines = skill_md.splitlines()
    heading = inserted_section[0].strip()
    target_index = _find_markdown_heading_index(lines, heading)
    if target_index is None:
        return skill_md, False
    section_end = _markdown_section_end(lines, target_index)
    body = list(lines[target_index + 1 : section_end])
    existing_lines = {line.strip() for line in body if line.strip()}
    additions = []
    for line in inserted_section[1:]:
        stripped = line.strip()
        if not stripped or stripped in existing_lines:
            continue
        existing_lines.add(stripped)
        additions.append(line.rstrip())
    if not additions:
        return skill_md, True
    while body and not body[-1].strip():
        body.pop()
    if body:
        body.append("")
    body.extend(additions)
    updated_lines = lines[: target_index + 1] + body + lines[section_end:]
    return "\n".join(updated_lines) + "\n", True


def _markdown_block_is_present(skill_md: str, content: str) -> bool:
    expected = [line.strip() for line in str(content or "").strip().splitlines()]
    if not expected:
        return False
    actual = str(skill_md or "").splitlines()
    width = len(expected)
    return any(
        [line.strip() for line in actual[index : index + width]] == expected
        for index in range(max(0, len(actual) - width + 1))
    )


def _dedupe_markdown_insert_content(
    skill_md: str,
    content: str,
) -> Tuple[str, str, bool]:
    """Remove an already-present insert or merge its existing same-level sections."""
    raw_content = str(content or "").strip()
    if not raw_content:
        return skill_md, "", False

    memory_start = skill_md.find(LONGITUDINAL_MEMORY_START)
    if memory_start != -1:
        editable_skill = skill_md[:memory_start].rstrip()
        memory_block = skill_md[memory_start:].lstrip("\n")
    else:
        editable_skill = skill_md
        memory_block = ""

    if _markdown_block_is_present(editable_skill, raw_content):
        return skill_md, "", True

    sections, prefix = _split_markdown_insert_sections(raw_content)
    if not sections:
        return skill_md, raw_content, False

    merged = False
    remaining: List[str] = list(prefix)
    for section in sections:
        heading = section[0].strip() if section else ""
        if _find_markdown_heading_index(editable_skill.splitlines(), heading) is None:
            remaining.extend(section)
            remaining.append("")
            continue
        editable_skill, handled = _merge_markdown_section_body(editable_skill, section)
        merged = merged or handled

    if not merged:
        return skill_md, raw_content, False
    residual = "\n".join(remaining).strip()
    if memory_block:
        updated = editable_skill.rstrip() + "\n\n" + memory_block
    else:
        updated = editable_skill
    return updated, residual, True


def _insert_after_markdown_section(skill_md: str, target_heading: str, content: str) -> Tuple[str, Dict[str, Any]]:
    target = str(target_heading or "").strip()
    if not target:
        return skill_md, {"applied": False, "reason": "missing_target"}
    target_level = _markdown_heading_level(target)
    if target_level is None:
        return skill_md, {"applied": False, "reason": "target_is_not_heading", "target": target}
    if not str(content or "").strip():
        return skill_md, {"applied": False, "reason": "empty_insert_content"}

    memory_start = skill_md.find(LONGITUDINAL_MEMORY_START)
    if memory_start != -1:
        editable_skill = skill_md[:memory_start].rstrip()
        memory_block = skill_md[memory_start:].lstrip("\n")
    else:
        editable_skill = skill_md
        memory_block = ""

    lines = editable_skill.splitlines(keepends=True)
    target_index = None
    for idx, line in enumerate(lines):
        if line.strip() == target and _markdown_heading_level(line) == target_level:
            target_index = idx
            break
    if target_index is None:
        return skill_md, {"applied": False, "reason": "target_not_found", "target": target}

    insert_at = len(lines)
    for idx in range(target_index + 1, len(lines)):
        level = _markdown_heading_level(lines[idx])
        if level is not None and level <= target_level:
            insert_at = idx
            break

    before = "".join(lines[:insert_at]).rstrip()
    after = "".join(lines[insert_at:]).lstrip("\n")
    inserted = str(content).strip()
    if after:
        updated = before + "\n\n" + inserted + "\n\n" + after
    else:
        updated = before + "\n\n" + inserted + "\n"
    if memory_block:
        updated = updated.rstrip() + "\n\n" + memory_block
    return updated, {"applied": True, "op": "insert_after_section", "target": target}


def _apply_markdown_edit(skill_md: str, edit: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    normalized = _normalize_markdown_edit(edit)
    if normalized is None:
        return skill_md, {"applied": False, "reason": "invalid_markdown_edit"}
    op = normalized["op"]
    content = str(normalized.get("content") or "")
    if op == "append":
        if not content.strip():
            return skill_md, {"applied": False, "reason": "empty_append_content"}
        deduped_skill, remaining_content, deduplicated = _dedupe_markdown_insert_content(
            skill_md, content
        )
        if deduplicated:
            if not remaining_content:
                return deduped_skill, {"applied": True, "op": op, "deduplicated": True}
            skill_md = deduped_skill
            content = remaining_content
        updated = append_before_longitudinal_memory(skill_md, content)
        report = {"applied": True, "op": op}
        if deduplicated:
            report["deduplicated"] = True
        return updated, report
    target = str(normalized.get("target") or "")
    if not target:
        return skill_md, {"applied": False, "reason": "missing_target"}
    if target not in skill_md:
        return skill_md, {"applied": False, "reason": "target_not_found", "target": target}
    if is_in_longitudinal_memory_region(skill_md, target):
        return skill_md, {"applied": False, "reason": "target_in_longitudinal_memory", "target": target}
    if op == "insert_after":
        if not content.strip():
            return skill_md, {"applied": False, "reason": "empty_insert_content"}
        deduped_skill, remaining_content, deduplicated = _dedupe_markdown_insert_content(
            skill_md, content
        )
        if deduplicated:
            if not remaining_content:
                return deduped_skill, {"applied": True, "op": op, "deduplicated": True}
            skill_md = deduped_skill
            content = remaining_content
        updated = skill_md.replace(target, target + "\n" + content.rstrip(), 1)
        report = {"applied": True, "op": op}
        if deduplicated:
            report["deduplicated"] = True
        return updated, report
    if op == "insert_after_section":
        if not content.strip():
            return skill_md, {"applied": False, "reason": "empty_insert_content"}
        deduped_skill, remaining_content, deduplicated = _dedupe_markdown_insert_content(
            skill_md, content
        )
        if deduplicated:
            if not remaining_content:
                return deduped_skill, {"applied": True, "op": op, "deduplicated": True}
            skill_md = deduped_skill
            content = remaining_content
        updated, report = _insert_after_markdown_section(skill_md, target, content)
        if deduplicated:
            report["deduplicated"] = True
        return updated, report
    if op == "replace":
        if not content.strip():
            return skill_md, {"applied": False, "reason": "empty_replace_content"}
        return skill_md.replace(target, content, 1), {"applied": True, "op": op}
    if op == "delete":
        return skill_md.replace(target, "", 1), {"applied": True, "op": op}
    return skill_md, {"applied": False, "reason": "invalid_op"}


def _apply_markdown_edits(skill_md: str, edits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    current = skill_md
    reports = []
    for idx, edit in enumerate(edits or []):
        current, report = _apply_markdown_edit(current, edit)
        report["edit_index"] = idx
        reports.append(report)
        if not report.get("applied"):
            break
    return current, reports


def _commit_markdown_edits_sequential(
    skill_md: str,
    edits: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    current = skill_md
    reports: List[Dict[str, Any]] = []
    committed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for idx, edit in enumerate(edits or []):
        updated, report = _apply_markdown_edit(current, edit)
        report["edit_index"] = idx
        reports.append(report)
        if report.get("applied"):
            current = updated
            committed.append(edit)
        else:
            failed.append({"edit": edit, "report": report})
    return current, reports, committed, failed


def _clean_markdown_edit_for_row(edit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (edit or {}).items()
        if not str(key).startswith("_") and key not in {"source_edit_ids"}
    }


def _drafts_from_l1_response(
    response: Dict[str, Any],
    *,
    default_experience_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    raw_items = response.get("evidence")
    if not isinstance(raw_items, list):
        return []
    drafts = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        proposed_edit = _normalize_proposed_edit(raw.get("proposed_edit"))
        trigger_ranges = _normalize_trigger_ranges(raw.get("trigger_ranges"))
        if proposed_edit is None or not trigger_ranges:
            continue
        pattern = str(raw.get("pattern") or "").strip()
        edit_text = " ".join(
            str(proposed_edit.get(key) or "")
            for key in ["op", "target", "content"]
            if str(proposed_edit.get(key) or "")
        )
        experience_type = _normalize_experience_type(
            raw.get("experience_type") or raw.get("polarity") or default_experience_type,
            edit_text,
            pattern,
        )
        failure_type = _normalize_failure_type(raw.get("failure_type"))
        if experience_type == "failure_reflection" and failure_type is None:
            failure_type = "other"
        if experience_type == "success_experience":
            failure_type = None
        if not pattern:
            pattern = _truncate_end(edit_text, 240)
        draft = {
            "experience_type": experience_type,
            "failure_type": failure_type,
            "pattern": pattern,
            "proposed_edit": proposed_edit,
            "trigger_ranges": trigger_ranges,
        }
        drafts.append(draft)
        if len(drafts) >= L1_MAX_PROPOSED_EDITS:
            break
    return drafts


def _normalize_trigger_ranges(value: Any) -> List[Dict[str, Any]]:
    raw_ranges = [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []
    ranges: List[Dict[str, Any]] = []
    for raw in raw_ranges:
        row_task_id = str(raw.get("task_id") or "").strip()
        if not row_task_id:
            continue
        start = _coerce_int(raw.get("start", 0), 0)
        ranges.append(
            {
                "task_id": row_task_id,
                "start": start,
                "end": _coerce_int(raw.get("end", raw.get("start", 0)), start),
            }
        )
    return ranges


def _normalize_evidence_role(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "new": "new_skill_gap",
        "gap": "new_skill_gap",
        "skill_gap": "new_skill_gap",
        "new_skill_gap": "new_skill_gap",
        "mismatch": "skill_rule_mismatch",
        "rule_mismatch": "skill_rule_mismatch",
        "skill_mismatch": "skill_rule_mismatch",
        "skill_rule_mismatch": "skill_rule_mismatch",
        "skill_conflict": "skill_rule_mismatch",
        "contradicted_skill": "skill_rule_mismatch",
        "contradicted_rule": "skill_rule_mismatch",
        "stale_rule": "skill_rule_mismatch",
        "wrong_rule": "skill_rule_mismatch",
        "execution": "execution_failure",
        "execution_failure": "execution_failure",
        "existing_rule_failure": "execution_failure",
        "rule_violation": "execution_failure",
        "not_followed": "execution_failure",
    }
    if raw in aliases:
        return aliases[raw]
    return "new_skill_gap"


def _normalize_failure_type(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw or raw in {"none", "null", "success", "success_experience"}:
        return None
    aliases = {
        "new_skill_gap": "rule_missing",
        "skill_gap": "rule_missing",
        "missing_rule": "rule_missing",
        "rule_missing": "rule_missing",
        "skill_rule_mismatch": "rule_wrong",
        "rule_mismatch": "rule_wrong",
        "skill_conflict": "rule_wrong",
        "contradicted_skill": "rule_wrong",
        "contradicted_rule": "rule_wrong",
        "wrong_rule": "rule_wrong",
        "stale_rule": "rule_wrong",
        "rule_wrong": "rule_wrong",
        "execution_failure": "rule_ignored",
        "rule_violation": "rule_ignored",
        "not_followed": "rule_ignored",
        "rule_ignored": "rule_ignored",
        "invalid": "invalid_action",
        "invalid_action": "invalid_action",
        "exploration": "exploration_gap",
        "exploration_gap": "exploration_gap",
        "experiment": "experiment_error",
        "experiment_error": "experiment_error",
        "premature": "premature_finish",
        "premature_finish": "premature_finish",
        "loop": "loop",
        "other": "other",
    }
    if raw in aliases:
        return aliases[raw]
    return raw if raw in VALID_FAILURE_TYPES else "other"


def _failure_type_from_legacy_role(value: Any) -> Optional[str]:
    role = _normalize_evidence_role(value)
    if role == "skill_rule_mismatch":
        return "rule_wrong"
    if role == "execution_failure":
        return "rule_ignored"
    if role == "new_skill_gap":
        return "rule_missing"
    return None


def _fallback_evidence_pattern(row: Dict[str, Any]) -> str:
    proposed = _normalize_proposed_edit((row or {}).get("proposed_edit")) or {}
    parts = [
        str(row.get("task_family") or "").strip(),
        str(proposed.get("target") or "").strip(),
        str(proposed.get("content") or "").strip(),
    ]
    text = " ".join(part for part in parts if part).strip()
    return _truncate_end(text, 240) if text else "unspecified reusable skill edit"


def _source_type_from_experience(experience_type: Any) -> str:
    return "failure" if _normalize_experience_type(experience_type) == "failure_reflection" else "success"


def _evidence_support_count_from_ranges(trigger_ranges: List[Dict[str, Any]]) -> int:
    task_ids = {str(row.get("task_id") or "") for row in trigger_ranges or [] if str(row.get("task_id") or "")}
    return max(1, len(task_ids))


def _normalize_experience_type(value: Any, candidate: str = "", reason: str = "") -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "success": "success_experience",
        "success_pattern": "success_experience",
        "positive": "success_experience",
        "positive_pattern": "success_experience",
        "success_experience": "success_experience",
        "failure": "failure_reflection",
        "failure_pattern": "failure_reflection",
        "failure_reflection": "failure_reflection",
        "negative": "failure_reflection",
        "mistake": "failure_reflection",
        "reflection": "failure_reflection",
    }
    if raw in aliases:
        return aliases[raw]
    text = f"{candidate} {reason}".lower()
    failure_markers = (
        "avoid",
        "do not",
        "don't",
        "never",
        "wrong",
        "fail",
        "failure",
        "mistake",
        "loop",
        "redundant",
        "waste",
        "distractor",
        "undo",
        "instead of",
        "unnecessary",
        "mismatch",
        "contradict",
        "conflict",
        "stale",
        "wrong rule",
    )
    if any(marker in text for marker in failure_markers):
        return "failure_reflection"
    return "success_experience"


def _materialize_evidence_candidate(
    *,
    draft: Dict[str, Any],
    evidence_id: str,
    records_by_id: Dict[str, Dict[str, Any]],
    source_minibatch_id: str = "",
    source_minibatch_task_ids: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    trigger_ranges = _normalize_trigger_ranges(draft.get("trigger_ranges"))
    if not trigger_ranges:
        return None, "missing trigger_ranges"
    proposed_edit = _normalize_proposed_edit(draft.get("proposed_edit"))
    if proposed_edit is None:
        return None, "missing proposed_edit"
    selected_segments = []
    supporting_tasks: List[Dict[str, Any]] = []
    seen_task_ids = set()
    for trigger in trigger_ranges:
        task_id = str(trigger.get("task_id") or "")
        record = records_by_id.get(task_id)
        if not record:
            return None, f"unknown task_id: {task_id}"
        steps = {int(x.get("step", -1)): x for x in record.get("trajectory", []) or []}
        start = _coerce_int(trigger.get("start", 0), 0)
        end = _coerce_int(trigger.get("end", start), start)
        if end < start:
            return None, "trigger_ranges end is before start"
        selected = []
        for step in range(start, end + 1):
            if step not in steps:
                return None, f"trigger_ranges step {step} is not visible"
            selected.append(steps[step])
        selected_segments.append(
            _evidence_segment_from_steps(
                task_id=task_id,
                rows=selected,
                initial_observation=(
                    str(record.get("initial_observation") or "")
                    if "initial_observation" in record
                    else None
                ),
            )
        )
        if task_id not in seen_task_ids:
            seen_task_ids.add(task_id)
            task_row = {
                "task_id": task_id,
                "task_instruction": str(record.get("instruction") or ""),
                "task_family": str(record.get("task_family") or ""),
            }
            if isinstance(record.get("outcome"), dict):
                task_row["task_outcome"] = dict(record.get("outcome") or {})
            supporting_tasks.append(task_row)
    primary_record = records_by_id.get(str(trigger_ranges[0].get("task_id") or "")) or {}
    experience_type = _normalize_experience_type(
        draft.get("experience_type"),
        " ".join(str(proposed_edit.get(key) or "") for key in ["op", "target", "content"]),
        str(draft.get("pattern") or ""),
    )
    failure_type = _normalize_failure_type(draft.get("failure_type"))
    if experience_type == "failure_reflection" and failure_type is None:
        failure_type = "other"
    if experience_type == "success_experience":
        failure_type = None
    card = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "source_type": _source_type_from_experience(experience_type),
        "experience_type": experience_type,
        "failure_type": failure_type,
        "pattern": str(
            draft.get("pattern")
            or _fallback_evidence_pattern(
                {
                    "proposed_edit": proposed_edit,
                    "task_family": str(primary_record.get("task_family") or ""),
                }
            )
        ).strip(),
        "proposed_edit": proposed_edit,
        "trigger_ranges": trigger_ranges,
        "supporting_tasks": supporting_tasks,
        "before_segments": selected_segments,
        "trigger_progress": _trigger_progress_from_segments(trigger_ranges, selected_segments),
        "support_count": _evidence_support_count_from_ranges(trigger_ranges),
        "source_minibatch_id": str(source_minibatch_id or ""),
        "source_minibatch_task_ids": [
            str(task_id)
            for task_id in (source_minibatch_task_ids or [])
            if str(task_id)
        ],
        "merge_level": 0,
    }
    if experience_type == "success_experience":
        card.pop("failure_type", None)
    return card, ""


def _linked_evidence(edit: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        evidence_by_id[eid]
        for eid in _evidence_id_list((edit or {}).get("evidence_ids"))
        if eid in evidence_by_id
    ]


def _public_evidence_card(card: Dict[str, Any]) -> Dict[str, Any]:
    allowed = [
        "evidence_schema_version",
        "evidence_id",
        "evidence_mode",
        "observed_skill_revision",
        "parent_skill_revision",
        "working_skill_revision",
        "comparison_type",
        "origin_edit_ids",
        "source_type",
        "experience_type",
        "failure_type",
        "pattern",
        "task_outcome",
        "proposed_edit",
        "trigger_ranges",
        "supporting_tasks",
        "before_segments",
        "trigger_progress",
        "support_count",
        "source_minibatch_id",
        "source_minibatch_task_ids",
        "merge_level",
    ]
    public = {key: card.get(key) for key in allowed if key in card}
    judge_reason = card.get("evidence_quality_judge_reason", card.get("l1_skilljudge_reason"))
    if judge_reason not in (None, ""):
        public["evidence_quality_judge_reason"] = judge_reason
    public["experience_type"] = _normalize_experience_type(
        public.get("experience_type"),
        _evidence_edit_text(card),
        str(card.get("pattern") or card.get("evidence_quality_judge_reason") or card.get("l1_skilljudge_reason") or ""),
    )
    modern_schema = any(key in card for key in ("source_type", "failure_type", "pattern", "support_count", "merge_level"))
    if modern_schema:
        public["source_type"] = str(public.get("source_type") or _source_type_from_experience(public.get("experience_type")))
        if public["experience_type"] == "success_experience":
            public.pop("failure_type", None)
        else:
            public["failure_type"] = _normalize_failure_type(public.get("failure_type")) or "other"
    if "support_count" in public:
        public["support_count"] = _coerce_int(public.get("support_count"), 1)
    if "merge_level" in public:
        public["merge_level"] = _coerce_int(public.get("merge_level"), 0)
    return public


def _public_evidence_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_public_evidence_card(card) for card in cards or []]


def _truncate_end(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    max_chars = int(max_chars)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= 20:
        return value[:max_chars].rstrip()
    return value[: max_chars - 15].rstrip() + " ...[truncated]"


def _public_proposed_edit_for_proposer(card: Dict[str, Any]) -> Dict[str, Any]:
    proposed = _normalize_proposed_edit((card or {}).get("proposed_edit"))
    if proposed is None:
        return {}
    public: Dict[str, Any] = {"op": proposed.get("op")}
    if "target" in proposed:
        public["target"] = str(proposed.get("target") or "")
    if "content" in proposed:
        public["content"] = str(proposed.get("content") or "")
    return {key: value for key, value in public.items() if value not in (None, "")}


def _public_supporting_tasks_for_proposer(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = card.get("supporting_tasks")
    if not isinstance(rows, list):
        trigger_ranges = card.get("trigger_ranges") if isinstance(card.get("trigger_ranges"), list) else []
        first_trigger = trigger_ranges[0] if trigger_ranges and isinstance(trigger_ranges[0], dict) else {}
        rows = [
            {
                "task_id": card.get("task_id") or first_trigger.get("task_id") or "",
                "task_instruction": card.get("task_instruction"),
                "task_family": card.get("task_family"),
            }
        ]
    public_rows: List[Dict[str, Any]] = []
    seen = set()
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("task_id") or "")
        key = task_id or json.dumps(raw, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        row = {
            "task_id": task_id,
            "task_instruction": _truncate_end(raw.get("task_instruction"), PROPOSER_EVIDENCE_MAX_TASK_CHARS),
            "task_family": str(raw.get("task_family") or ""),
        }
        public_rows.append({k: v for k, v in row.items() if v not in (None, "")})
    return public_rows


def _task_families_for_proposer(card: Dict[str, Any]) -> List[str]:
    families: List[str] = []
    seen = set()
    rows = card.get("supporting_tasks")
    if isinstance(rows, list):
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            family = str(raw.get("task_family") or "").strip()
            if family and family not in seen:
                seen.add(family)
                families.append(family)
    fallback = str(card.get("task_family") or "").strip()
    if fallback and fallback not in seen:
        families.append(fallback)
    return families


def _public_evidence_card_for_proposer(
    card: Dict[str, Any],
    *,
    full_context: bool = False,
) -> Dict[str, Any]:
    experience_type = _normalize_experience_type(
        card.get("experience_type"),
        _evidence_edit_text(card),
        str(card.get("pattern") or card.get("evidence_quality_judge_reason") or card.get("l1_skilljudge_reason") or ""),
    )
    public: Dict[str, Any] = {
        "evidence_id": str(card.get("evidence_id") or ""),
        "source_type": str(card.get("source_type") or _source_type_from_experience(experience_type)),
        "pattern": (
            str(card.get("pattern") or "")
            if full_context
            else _truncate_end(card.get("pattern"), PROPOSER_EVIDENCE_MAX_TASK_CHARS)
        ),
        "proposed_edit": _public_proposed_edit_for_proposer(card),
        "support_count": _coerce_int(card.get("support_count"), 1),
        "merge_level": _coerce_int(card.get("merge_level"), 0),
        "source_minibatch_id": str(card.get("source_minibatch_id") or ""),
        "trigger_progress": [
            dict(row)
            for row in (card.get("trigger_progress") or [])
            if isinstance(row, dict)
        ],
    }
    failure_type = _normalize_failure_type(card.get("failure_type") or _failure_type_from_legacy_role(card.get("evidence_role")))
    if experience_type == "failure_reflection":
        public["failure_type"] = failure_type or "other"
    task_families = _task_families_for_proposer(card)
    if task_families:
        public["task_families"] = task_families
    if card.get("evidence_mode") not in (None, ""):
        public["evidence_mode"] = str(card.get("evidence_mode"))
    for key in (
        "observed_skill_revision",
        "parent_skill_revision",
        "working_skill_revision",
        "comparison_type",
        "origin_edit_ids",
    ):
        if card.get(key) not in (None, "", []):
            public[key] = card.get(key)
    return {key: value for key, value in public.items() if value not in (None, "")}


def _public_evidence_cards_for_proposer(
    cards: List[Dict[str, Any]],
    *,
    full_context: bool = False,
) -> List[Dict[str, Any]]:
    return [
        _public_evidence_card_for_proposer(card, full_context=full_context)
        for card in cards or []
    ]


def _public_evidence_card_groups_for_proposer(
    cards: List[Dict[str, Any]],
    *,
    grouping: str = "card",
    group_evidence_ids: Optional[List[List[str]]] = None,
    full_context: bool = False,
) -> List[List[Dict[str, Any]]]:
    grouping = str(grouping or "").lower()
    if grouping == "semantic":
        by_id = {str(card.get("evidence_id") or ""): card for card in cards or []}
        groups: List[List[Dict[str, Any]]] = []
        if group_evidence_ids:
            for raw_group in group_evidence_ids:
                group = [
                    _public_evidence_card_for_proposer(by_id[eid], full_context=full_context)
                    for eid in [str(value) for value in raw_group or []]
                    if eid in by_id
                ]
                if group:
                    groups.append(group)
        if groups:
            return groups
        return [
            _public_evidence_cards_for_proposer(cards, full_context=full_context)
        ] if cards else []

    if grouping != "round":
        return []
    groups: List[List[Dict[str, Any]]] = []
    index_by_key: Dict[Tuple[str, str], int] = {}
    for card in cards or []:
        if card.get("round") in (None, ""):
            key = ("evidence", str(card.get("evidence_id") or ""))
        else:
            key = ("round", str(_evidence_int(card, "round")))
        if key not in index_by_key:
            index_by_key[key] = len(groups)
            groups.append([])
        groups[index_by_key[key]].append(
            _public_evidence_card_for_proposer(card, full_context=full_context)
        )
    return groups


def _public_evidence_card_groups(
    cards: List[Dict[str, Any]],
    *,
    grouping: str = "card",
    group_evidence_ids: Optional[List[List[str]]] = None,
) -> List[List[Dict[str, Any]]]:
    grouping = str(grouping or "").lower()
    if grouping == "semantic":
        by_id = {str(card.get("evidence_id") or ""): card for card in cards or []}
        groups: List[List[Dict[str, Any]]] = []
        if group_evidence_ids:
            for raw_group in group_evidence_ids:
                group = [
                    _public_evidence_card(by_id[eid])
                    for eid in [str(value) for value in raw_group or []]
                    if eid in by_id
                ]
                if group:
                    groups.append(group)
        if groups:
            return groups
        return [[_public_evidence_card(card) for card in cards or []]] if cards else []

    if grouping != "round":
        return []
    groups: List[List[Dict[str, Any]]] = []
    index_by_key: Dict[Tuple[str, str], int] = {}
    for card in cards or []:
        if card.get("round") in (None, ""):
            key = ("evidence", str(card.get("evidence_id") or ""))
        else:
            key = ("round", str(_evidence_int(card, "round")))
        if key not in index_by_key:
            index_by_key[key] = len(groups)
            groups.append([])
        groups[index_by_key[key]].append(_public_evidence_card(card))
    return groups


def _truncate_middle(text: Any, max_chars: int) -> str:
    value = str(text or "")
    max_chars = int(max_chars)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars < 80:
        return value[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 32
    return value[:head].rstrip() + "\n\n...[truncated]...\n\n" + value[-tail:].lstrip()


def _ensure_recovery_rules_section(skill_md: str) -> str:
    text = str(skill_md or "")
    if RECOVERY_RULES_SECTION in text:
        return text
    return append_before_longitudinal_memory(
        text,
        RECOVERY_RULES_SECTION + "\n",
    )


def _epoch_edit_priority(edit: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> Tuple[int, int, int, int, str]:
    linked = _linked_evidence(edit, evidence_by_id)
    if linked:
        best_evidence_rank = min(_l2_replay_priority(card) for card in linked)
    else:
        best_evidence_rank = 9
    families = {
        family
        for card in linked
        for family in _task_families_for_proposer(card)
        if family
    }
    content_len = len(str(edit.get("content") or ""))
    return (
        best_evidence_rank,
        -len(_evidence_id_list(edit.get("evidence_ids"))),
        -len(families),
        content_len,
        json.dumps(_normalize_markdown_edit(edit) or edit, sort_keys=True),
    )


def _epoch_edit_is_corrective(edit: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> bool:
    for card in _linked_evidence(edit, evidence_by_id):
        experience_type = _normalize_experience_type(
            card.get("experience_type"),
            _evidence_edit_text(card),
            str(card.get("pattern") or card.get("l1_skilljudge_reason") or ""),
        )
        if experience_type == "failure_reflection":
            return True
        if str(card.get("source_type") or "") == "failure":
            return True
    return False


def _limit_working_branch_epoch_edits(
    edits: List[Dict[str, Any]],
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
    args: Any,
    info: Optional[Dict[str, Any]] = None,
    budget: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply the epoch edit budget only to the opt-in working branch."""
    if (
        not _evidence_working_branch_enabled(args)
        or _resolve_epoch_skill_update_mode(args) != "cluster_step"
    ):
        return list(edits or []), []
    budget = max(
        0,
        _int_arg(
            budget
            if budget is not None
            else getattr(args, "epoch_edit_budget", DEFAULT_EPOCH_EDIT_BUDGET),
            DEFAULT_EPOCH_EDIT_BUDGET,
        ),
    )
    ordered = [dict(edit) for edit in edits or []]
    if not bool((info or {}).get("rank_used_llm", False)):
        ordered.sort(key=lambda edit: _epoch_edit_priority(edit, evidence_by_id))
    selected = ordered[:budget]
    dropped = ordered[budget:]
    if info is not None:
        info.update(
            {
                "budget_applied": True,
                "epoch_edit_budget": budget,
                "selected_edit_count": len(selected),
                "dropped_edit_count": len(dropped),
                "dropped_merged_edit_ids": [
                    str(edit.get("_merged_edit_id") or edit.get("_candidate_edit_id") or "")
                    for edit in dropped
                    if str(edit.get("_merged_edit_id") or edit.get("_candidate_edit_id") or "")
                ],
            }
        )
    return selected, dropped


def _split_working_branch_correction_edits(
    edits: List[Dict[str, Any]],
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep merged correction edits attributable to exactly one origin."""
    split: List[Dict[str, Any]] = []
    for raw_edit in edits or []:
        edit = dict(raw_edit or {})
        evidence_ids = _evidence_id_list(edit.get("evidence_ids"))
        correction_cards = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
            and str(evidence_by_id[evidence_id].get("evidence_mode") or "") == "contrastive"
            and len(_evidence_id_list(evidence_by_id[evidence_id].get("origin_edit_ids"))) == 1
        ]
        origins: List[str] = []
        for card in correction_cards:
            origin_id = _evidence_id_list(card.get("origin_edit_ids"))[0]
            if origin_id not in origins:
                origins.append(origin_id)
        if not origins:
            split.append(edit)
            continue

        correction_evidence_ids = {
            str(card.get("evidence_id") or "") for card in correction_cards
        }
        ordinary_evidence_ids = [
            evidence_id for evidence_id in evidence_ids if evidence_id not in correction_evidence_ids
        ]
        if ordinary_evidence_ids:
            ordinary = dict(edit)
            ordinary["evidence_ids"] = ordinary_evidence_ids
            split.append(
                _annotate_edit_metadata(
                    ordinary,
                    _linked_evidence(ordinary, evidence_by_id),
                    merge_level=_coerce_int(edit.get("merge_level"), 0),
                )
            )

        for origin_index, origin_id in enumerate(origins):
            cards = [
                card
                for card in correction_cards
                if _evidence_id_list(card.get("origin_edit_ids")) == [origin_id]
            ]
            origin_evidence_ids = [
                str(card.get("evidence_id") or "")
                for card in cards
                if str(card.get("evidence_id") or "")
            ]
            proposed = _normalize_proposed_edit((cards[0] if cards else {}).get("proposed_edit"))
            if proposed is None or not origin_evidence_ids:
                continue
            correction = {
                **edit,
                **proposed,
                "evidence_ids": origin_evidence_ids,
            }
            merged_id = str(edit.get("_merged_edit_id") or "")
            if merged_id:
                correction["_merged_edit_id"] = f"{merged_id}_O{origin_index + 1:02d}"
            split.append(
                _annotate_edit_metadata(
                    correction,
                    cards,
                    merge_level=_coerce_int(edit.get("merge_level"), 0),
                )
            )
    return split


def _epoch_edit_class(edit: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> str:
    return "corrective" if _epoch_edit_is_corrective(edit, evidence_by_id) else "success_only"


def _epoch_edit_mentions_recovery_rules(edit: Dict[str, Any]) -> bool:
    target = str((edit or {}).get("target") or "")
    content = str((edit or {}).get("content") or "")
    return RECOVERY_RULES_SECTION in target or RECOVERY_RULES_SECTION in content


def _epoch_edit_is_replace_or_generalize(edit: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> bool:
    op = str((edit or {}).get("op") or "").lower()
    if op in {"replace", "delete"}:
        return True
    content = str((edit or {}).get("content") or "").lower()
    if any(word in content for word in ["generalize", "soften", "replace the", "instead of always"]):
        return True
    return any(
        _normalize_failure_type(card.get("failure_type")) == "rule_wrong"
        for card in _linked_evidence(edit, evidence_by_id)
    )


def _edit_source_type_from_linked(linked: List[Dict[str, Any]]) -> str:
    sources = {str(card.get("source_type") or _source_type_from_experience(card.get("experience_type"))) for card in linked}
    if not sources:
        return "unknown"
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _edit_support_count_from_linked(linked: List[Dict[str, Any]]) -> int:
    task_ids = {
        str(trigger.get("task_id") or "")
        for card in linked
        for trigger in _iter_card_trigger_ranges(card)
        if str(trigger.get("task_id") or "")
    }
    if task_ids:
        return len(task_ids)
    return len({str(card.get("evidence_id") or "") for card in linked if str(card.get("evidence_id") or "")})


def _annotate_edit_metadata(
    edit: Dict[str, Any],
    linked: List[Dict[str, Any]],
    *,
    merge_level: Optional[int] = None,
) -> Dict[str, Any]:
    out = dict(edit or {})
    out["source_type"] = _edit_source_type_from_linked(linked)
    out["support_count"] = max(1, _edit_support_count_from_linked(linked))
    out["merge_level"] = _coerce_int(merge_level if merge_level is not None else out.get("merge_level"), 0)
    failure_types = sorted(
        {
            ft
            for ft in (_normalize_failure_type(card.get("failure_type")) for card in linked)
            if ft
        }
    )
    if failure_types:
        out["failure_types"] = failure_types
    patterns = sorted({str(card.get("pattern") or "").strip() for card in linked if str(card.get("pattern") or "").strip()})
    if patterns:
        out["patterns"] = patterns[:5]
    return out


def _dedupe_epoch_edits(
    edits: List[Dict[str, Any]],
    evidence_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Evidence is support metadata, not part of edit identity. Multiple
    # trajectories can independently justify the same concrete Markdown edit.
    index_by_identity: Dict[Tuple[Any, ...], int] = {}
    deduped: List[Dict[str, Any]] = []
    for raw in sorted(edits or [], key=lambda edit: _epoch_edit_priority(edit, evidence_by_id)):
        edit = _normalize_markdown_edit(raw)
        if edit is None or not _linked_evidence(edit, evidence_by_id):
            continue
        key = (
            edit.get("op"),
            str(edit.get("target") or ""),
            str(edit.get("content") or ""),
        )
        existing_index = index_by_identity.get(key)
        if existing_index is not None:
            existing = deduped[existing_index]
            existing["evidence_ids"] = _evidence_id_list(
                _evidence_id_list(existing.get("evidence_ids"))
                + _evidence_id_list(edit.get("evidence_ids"))
            )

            # Preserve all candidate/source provenance when duplicate edits
            # came from different windows or merge branches.
            source_ids = _source_ids_for_edit(existing)
            source_ids.extend(_source_ids_for_edit(raw))
            source_ids = _evidence_id_list(source_ids)
            if source_ids:
                existing["source_edit_ids"] = source_ids

            merge_level = max(
                _coerce_int(existing.get("merge_level"), 0),
                _coerce_int(edit.get("merge_level"), 0),
                _coerce_int(raw.get("merge_level"), 0),
            )
            deduped[existing_index] = _annotate_edit_metadata(
                existing,
                _linked_evidence(existing, evidence_by_id),
                merge_level=merge_level,
            )
            continue

        index_by_identity[key] = len(deduped)
        for meta_key in [
            "_candidate_edit_id",
            "_merged_edit_id",
            "_source_window_index",
            "_source_decision_reason",
            "source_edit_ids",
            "source_type",
            "support_count",
            "merge_level",
            "failure_types",
            "patterns",
        ]:
            if meta_key in raw:
                edit[meta_key] = raw.get(meta_key)
        edit = _annotate_edit_metadata(edit, _linked_evidence(edit, evidence_by_id), merge_level=_coerce_int(edit.get("merge_level"), 0))
        deduped.append(edit)
    return deduped


def _source_ids_for_edit(edit: Dict[str, Any]) -> List[str]:
    source_ids = _evidence_id_list((edit or {}).get("source_edit_ids"))
    if source_ids:
        return source_ids
    candidate_id = str((edit or {}).get("_candidate_edit_id") or "").strip()
    return [candidate_id] if candidate_id else []


def _covered_source_edit_ids(edits: List[Dict[str, Any]]) -> List[str]:
    covered: List[str] = []
    for edit in edits or []:
        covered.extend(_source_ids_for_edit(edit))
    return _evidence_id_list(covered)


def _covered_evidence_ids(edits: List[Dict[str, Any]]) -> List[str]:
    covered: List[str] = []
    for edit in edits or []:
        covered.extend(_evidence_id_list((edit or {}).get("evidence_ids")))
    return _evidence_id_list(covered)


def _edit_with_source_coverage(
    edit: Dict[str, Any],
    source_ids: List[str],
    id_to_edit: Dict[str, Dict[str, Any]],
    evidence_by_id: Dict[str, Dict[str, Any]],
    *,
    merge_level: int,
) -> Optional[Dict[str, Any]]:
    normalized = _normalize_markdown_edit(edit)
    if normalized is None:
        return None
    expanded_source_ids: List[str] = []
    evidence_ids = _evidence_id_list(normalized.get("evidence_ids"))
    for source_id in _evidence_id_list(source_ids):
        source = id_to_edit.get(source_id)
        if source is None:
            continue
        original_source_ids = _source_ids_for_edit(source) or [source_id]
        expanded_source_ids.extend(original_source_ids)
        evidence_ids.extend(_evidence_id_list(source.get("evidence_ids")))
    if expanded_source_ids:
        normalized["source_edit_ids"] = _evidence_id_list(expanded_source_ids)
    normalized["evidence_ids"] = _evidence_id_list(evidence_ids)
    linked = _linked_evidence(normalized, evidence_by_id)
    if not linked:
        return None
    return _annotate_edit_metadata(normalized, linked, merge_level=merge_level)


def _compact_epoch_edit_candidate(
    edit: Dict[str, Any],
    *,
    candidate_edit_id: str,
    evidence_by_id: Dict[str, Dict[str, Any]],
    max_content_chars: int,
) -> Dict[str, Any]:
    normalized = _normalize_markdown_edit(edit) or dict(edit)
    linked = _linked_evidence(normalized, evidence_by_id)
    task_families = sorted(
        {
            family
            for card in linked
            for family in _task_families_for_proposer(card)
            if family
        }
    )
    experience_types = sorted(
        {
            _normalize_experience_type(
                card.get("experience_type"),
                _evidence_edit_text(card),
                str(card.get("pattern") or card.get("l1_skilljudge_reason") or ""),
            )
            for card in linked
        }
    )
    failure_types = sorted({ft for ft in (_normalize_failure_type(card.get("failure_type")) for card in linked) if ft})
    source_type = _edit_source_type_from_linked(linked)
    support_count = _edit_support_count_from_linked(linked)
    merge_level = _coerce_int(normalized.get("merge_level"), _coerce_int(edit.get("merge_level"), 0))
    record = {
        "candidate_edit_id": candidate_edit_id,
        "edit_class": _epoch_edit_class(normalized, evidence_by_id),
        "source_type": source_type,
        "support_count": support_count,
        "merge_level": merge_level,
        "failure_types": failure_types,
        "op": normalized.get("op"),
        "target": normalized.get("target"),
        "content": _truncate_middle(normalized.get("content") or "", max_content_chars),
        "evidence_ids": _evidence_id_list(normalized.get("evidence_ids")),
        "source_edit_ids": _evidence_id_list(edit.get("source_edit_ids")),
        "task_families": task_families,
        "experience_types": experience_types,
        "source_window_index": edit.get("_source_window_index"),
        "source_decision_reason": edit.get("_source_decision_reason"),
    }
    if record["op"] == "delete":
        record.pop("content", None)
    if record["op"] == "append":
        record.pop("target", None)
    return {key: value for key, value in record.items() if value not in (None, "", [])}


def _fallback_epoch_merged_edits(
    accepted_edits: List[Dict[str, Any]],
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
    epoch_edit_budget: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    budget = max(0, int(epoch_edit_budget))
    deduped = _dedupe_epoch_edits(accepted_edits, evidence_by_id)
    selected = [dict(edit) for edit in deduped[:budget]]
    for idx, edit in enumerate(selected):
        edit.setdefault("source_edit_ids", [str(edit.get("_candidate_edit_id") or f"C{idx + 1:06d}")])
        edit.update(_annotate_edit_metadata(edit, _linked_evidence(edit, evidence_by_id), merge_level=3))
        for key in ["_candidate_edit_id", "_source_window_index", "_source_decision_reason"]:
            edit.pop(key, None)
    return selected, {
        "used_llm": False,
        "reasoning": "fallback deterministic ranking",
        "candidate_count": len(accepted_edits or []),
        "deduped_candidate_count": len(deduped),
        "selected_edit_count": len(selected),
        "epoch_edit_budget": budget,
        "selected_edit_ids": [
            str(edit.get("_candidate_edit_id") or f"C{idx + 1:06d}")
            for idx, edit in enumerate(deduped[:budget])
        ],
        "dropped_edit_ids": [
            str(edit.get("_candidate_edit_id") or f"C{idx + 1:06d}")
            for idx, edit in enumerate(deduped[budget:])
        ],
    }


def _merge_epoch_accepted_edits_no_budget(
    *,
    teacher: Any,
    dataset: str,
    current_skill_md: str,
    accepted_edits: List[Dict[str, Any]],
    deduped: List[Dict[str, Any]],
    corrective_edits: List[Dict[str, Any]],
    success_only_edits: List[Dict[str, Any]],
    evidence_by_id: Dict[str, Dict[str, Any]],
    args: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    defer_uncovered = _working_branch_defer_uncovered_enabled(dataset, args)
    full_context = bool(getattr(args, "llm_full_context", False))
    max_candidates = max(
        1,
        int(getattr(args, "epoch_merge_max_candidates", DEFAULT_EPOCH_MERGE_MAX_CANDIDATES) or DEFAULT_EPOCH_MERGE_MAX_CANDIDATES),
    )
    if full_context:
        max_candidates = max(1, len(deduped))
    if bool(getattr(args, "no_epoch_global_merge", False)):
        selected: List[Dict[str, Any]] = []
        for idx, edit in enumerate(deduped):
            edit.setdefault("source_edit_ids", [str(edit.get("_candidate_edit_id") or f"C{idx + 1:06d}")])
            clean = _edit_with_source_coverage(
                edit,
                _source_ids_for_edit(edit),
                {str(edit.get("_candidate_edit_id")): edit},
                evidence_by_id,
                merge_level=3,
            )
            if clean is not None:
                selected.append(clean)
        return selected, {
            "used_llm": False,
            "merge_used_llm": False,
            "rank_used_llm": False,
            "rank_candidate_count": 0,
            "rank_selected_count": 0,
            "rank_dropped_edit_ids": [],
            "rank_dropped_evidence_ids": [],
            "hard_rank_enabled": False,
            "reasoning": "fallback deterministic coverage merge without rank",
            "candidate_count": len(accepted_edits or []),
            "deduped_candidate_count": len(deduped),
            "sent_candidate_count": len(deduped),
            "selected_edit_count": len(selected),
            "epoch_edit_budget": int(getattr(args, "epoch_edit_budget", DEFAULT_EPOCH_EDIT_BUDGET) or 0),
            "selected_edit_ids": _covered_source_edit_ids(selected),
            "selected_merged_edit_ids": [],
            "dropped_edit_ids": [],
            "corrective_edit_count": len(corrective_edits),
            "success_only_edit_count": len(success_only_edits),
            "merged_pool_count": len(selected),
            "ranked_edit_count": len(selected),
            "recovery_rule_edit_count": sum(_epoch_edit_mentions_recovery_rules(edit) for edit in selected),
            "replace_or_generalize_edit_count": sum(
                _epoch_edit_is_replace_or_generalize(edit, evidence_by_id) for edit in selected
            ),
            "epoch_merge_chunk_count": 0,
            "identity_fallback_edit_count": 0,
            "uncovered_source_edit_count": 0,
        }

    # Coverage mode treats skill text and accepted edits as essential payload.
    # The legacy character caps remain available only to merge_rank ablations.
    max_content_chars = 0
    skill_max_chars = 0
    dataset_prompts = prompts_for_dataset(dataset)
    prompt = dataset_prompts.get("epoch_edit_merger")
    merge_error = ""
    merge_reasoning: List[str] = []
    omitted_original_source_ids: List[str] = []
    deferred_uncovered_source_ids: List[str] = []
    deferred_uncovered_evidence_ids: List[str] = []
    deferred_omitted_source_ids: List[str] = []
    deferred_omitted_evidence_ids: List[str] = []
    identity_fallback_count = 0
    chunk_count = 0

    def payload_id_for(edit: Dict[str, Any], pass_idx: int, item_idx: int) -> str:
        return str(
            edit.get("_payload_edit_id")
            or edit.get("_merged_edit_id")
            or edit.get("_candidate_edit_id")
            or f"P{pass_idx:02d}_{item_idx + 1:06d}"
        )

    def merge_once(input_edits: List[Dict[str, Any]], *, pass_idx: int) -> List[Dict[str, Any]]:
        nonlocal merge_error, identity_fallback_count, chunk_count
        merged_all: List[Dict[str, Any]] = []
        chunks = [
            input_edits[idx : idx + max_candidates]
            for idx in range(0, len(input_edits), max_candidates)
        ] or []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_count += 1
            id_to_edit: Dict[str, Dict[str, Any]] = {}
            candidate_records: List[Dict[str, Any]] = []
            for item_idx, edit in enumerate(chunk):
                payload_id = payload_id_for(edit, pass_idx, (chunk_idx * max_candidates) + item_idx)
                id_to_edit[payload_id] = edit
                candidate_records.append(
                    _compact_epoch_edit_candidate(
                        edit,
                        candidate_edit_id=payload_id,
                        evidence_by_id=evidence_by_id,
                        max_content_chars=max_content_chars,
                    )
                )
            payload = {
                "skill_md": _truncate_middle(current_skill_md, skill_max_chars),
                "accepted_candidate_edits": candidate_records,
            }
            raw: Dict[str, Any] = {}
            if prompt:
                try:
                    raw = _teacher_chat_json(teacher, prompt, payload)
                except Exception as exc:
                    merge_error = str(exc)
                    raw = {}
            if isinstance(raw, dict) and str(raw.get("reasoning") or "").strip():
                merge_reasoning.append(str(raw.get("reasoning") or ""))

            covered_payload_ids: List[str] = []
            omitted_payload_ids = _omitted_source_edit_ids(raw)
            for omitted_id in omitted_payload_ids:
                source = id_to_edit.get(omitted_id)
                if source is None:
                    continue
                original_source_ids = _source_ids_for_edit(source) or [omitted_id]
                omitted_original_source_ids.extend(original_source_ids)
                if defer_uncovered:
                    deferred_omitted_source_ids.extend(original_source_ids)
                    deferred_omitted_evidence_ids.extend(
                        _evidence_id_list(source.get("evidence_ids"))
                    )
            for raw_edit in (raw or {}).get("merged_edits") or []:
                if not isinstance(raw_edit, dict):
                    continue
                explicit_source_ids = _evidence_id_list(raw_edit.get("source_edit_ids"))
                edit = _normalize_markdown_edit(raw_edit)
                if edit is None:
                    continue
                if explicit_source_ids:
                    source_payload_ids = [source_id for source_id in explicit_source_ids if source_id in id_to_edit]
                else:
                    edit_evidence = set(_evidence_id_list(edit.get("evidence_ids")))
                    source_payload_ids = [
                        payload_id
                        for payload_id, source in id_to_edit.items()
                        if edit_evidence.intersection(_evidence_id_list(source.get("evidence_ids")))
                    ]
                clean = _edit_with_source_coverage(
                    edit,
                    source_payload_ids,
                    id_to_edit,
                    evidence_by_id,
                    merge_level=2,
                )
                if clean is None:
                    continue
                covered_payload_ids.extend(source_payload_ids)
                merged_all.append(clean)

            input_payload_ids = list(id_to_edit.keys())
            uncovered_payload_ids = sorted(
                set(input_payload_ids) - set(covered_payload_ids) - set(omitted_payload_ids)
            )
            for payload_id in uncovered_payload_ids:
                source = id_to_edit.get(payload_id)
                if source is None:
                    continue
                if defer_uncovered:
                    deferred_uncovered_source_ids.extend(
                        _source_ids_for_edit(source) or [payload_id]
                    )
                    deferred_uncovered_evidence_ids.extend(
                        _evidence_id_list(source.get("evidence_ids"))
                    )
                    continue
                fallback = _edit_with_source_coverage(
                    source,
                    [payload_id],
                    id_to_edit,
                    evidence_by_id,
                    merge_level=2,
                )
                if fallback is None:
                    continue
                identity_fallback_count += 1
                merged_all.append(fallback)
        return _dedupe_epoch_edits(merged_all, evidence_by_id)

    def recursive_merge(
        input_edits: List[Dict[str, Any]],
        *,
        pass_offset: int,
        force_once: bool = False,
    ) -> Tuple[List[Dict[str, Any]], int]:
        current_stage = list(input_edits)
        if len(current_stage) <= 1 and not force_once:
            return current_stage, 0
        stage_rounds = 0
        while current_stage:
            stage_rounds += 1
            pass_idx = pass_offset + stage_rounds
            merged_stage = merge_once(current_stage, pass_idx=pass_idx)
            if not merged_stage:
                if defer_uncovered:
                    return [], stage_rounds
                merged_stage = list(current_stage)
            if (
                len(merged_stage) <= max_candidates
                or len(merged_stage) >= len(current_stage)
                or stage_rounds >= 3
            ):
                current_stage = merged_stage
                break
            for idx, edit in enumerate(merged_stage):
                edit["_payload_edit_id"] = f"R{pass_idx:02d}_{idx + 1:06d}"
            current_stage = merged_stage
        return current_stage, stage_rounds

    failure_merged, failure_merge_rounds = recursive_merge(
        corrective_edits,
        pass_offset=0,
    )
    success_merged, success_merge_rounds = recursive_merge(
        success_only_edits,
        pass_offset=10,
    )
    combined_pool = failure_merged + success_merged
    current, conflict_merge_rounds = recursive_merge(
        combined_pool,
        pass_offset=20,
        force_once=len(combined_pool) > 1,
    )
    merge_rounds = failure_merge_rounds + success_merge_rounds + conflict_merge_rounds

    final_edits: List[Dict[str, Any]] = []
    for idx, edit in enumerate(_dedupe_epoch_edits(current, evidence_by_id)):
        clean = _edit_with_source_coverage(
            edit,
            _source_ids_for_edit(edit),
            {source_id: edit for source_id in _source_ids_for_edit(edit)},
            evidence_by_id,
            merge_level=3,
        )
        if clean is None:
            continue
        clean["_merged_edit_id"] = f"M{idx + 1:06d}"
        final_edits.append(clean)

    merged_pool = final_edits
    merged_pool_count = len(merged_pool)
    epoch_edit_budget = max(
        0,
        int(getattr(args, "epoch_edit_budget", DEFAULT_EPOCH_EDIT_BUDGET) or 0),
    )
    rank_enabled = (
        _evidence_working_branch_enabled(args)
        and _resolve_epoch_skill_update_mode(args) == "cluster_step"
        and bool(getattr(args, "working_branch_llm_rank", False))
    )
    rank_prompt = dataset_prompts.get("epoch_edit_ranker") if rank_enabled else None
    rank_raw: Dict[str, Any] = {}
    rank_error = ""
    rank_reasoning = ""
    rank_attempted = False
    rank_used_llm = False
    rank_fallback_used = False
    rank_selected_ids: List[str] = []
    ranked_edits = list(merged_pool)

    if rank_enabled:
        ranked_edits = []
        if epoch_edit_budget <= 0:
            rank_error = "epoch_edit_budget is zero"
        elif not rank_prompt:
            rank_error = "epoch_edit_ranker prompt is unavailable"
        else:
            rank_attempted = True
            rank_payload = {
                "skill_md": current_skill_md,
                "epoch_edit_budget": epoch_edit_budget,
                "merged_candidate_edits": [
                    _compact_epoch_edit_candidate(
                        edit,
                        candidate_edit_id=str(edit.get("_merged_edit_id") or ""),
                        evidence_by_id=evidence_by_id,
                        max_content_chars=0,
                    )
                    for edit in merged_pool
                ],
            }
            try:
                rank_raw = _teacher_chat_json(teacher, rank_prompt, rank_payload)
            except Exception as exc:
                rank_error = str(exc)

            rank_reasoning = str((rank_raw or {}).get("reasoning") or "")
            raw_selected_ids = (rank_raw or {}).get("selected_edit_ids")
            if not rank_error and not isinstance(raw_selected_ids, list):
                rank_error = "rank response is missing selected_edit_ids"
            if not rank_error:
                id_to_merged = {
                    str(edit.get("_merged_edit_id") or ""): edit
                    for edit in merged_pool
                    if str(edit.get("_merged_edit_id") or "")
                }
                seen_ids: set[str] = set()
                for raw_id in raw_selected_ids:
                    merged_id = str(raw_id or "").strip()
                    if not merged_id or merged_id in seen_ids:
                        continue
                    seen_ids.add(merged_id)
                    edit = id_to_merged.get(merged_id)
                    if edit is None:
                        continue
                    if _normalize_markdown_edit(edit) is None or not _linked_evidence(edit, evidence_by_id):
                        continue
                    rank_selected_ids.append(merged_id)
                    ranked_edits.append(edit)
                    if len(ranked_edits) >= epoch_edit_budget:
                        break
                if ranked_edits:
                    rank_used_llm = True
                else:
                    rank_error = "rank response contains no valid selected_edit_ids"

        if not rank_used_llm:
            rank_fallback_used = True
            ranked_edits = sorted(
                merged_pool,
                key=lambda edit: _epoch_edit_priority(edit, evidence_by_id),
            )[:epoch_edit_budget]

    selected_merged_ids = {
        str(edit.get("_merged_edit_id") or "")
        for edit in ranked_edits
        if str(edit.get("_merged_edit_id") or "")
    }
    dropped_budget_edits = (
        [
            edit
            for edit in merged_pool
            if str(edit.get("_merged_edit_id") or "") not in selected_merged_ids
        ]
        if rank_enabled
        else []
    )
    rank_dropped_edits = dropped_budget_edits if rank_used_llm else []
    selected_evidence_ids = set(_covered_evidence_ids(ranked_edits))
    rank_dropped_evidence_ids = sorted(
        set(_covered_evidence_ids(rank_dropped_edits)) - selected_evidence_ids
    )

    all_source_ids = _evidence_id_list([str(edit.get("_candidate_edit_id")) for edit in deduped])
    covered_source_ids = _covered_source_edit_ids(ranked_edits)
    uncovered_source_ids = sorted(set(all_source_ids) - set(covered_source_ids) - set(omitted_original_source_ids))
    policy_deferred_uncovered_evidence_ids = sorted(
        set(deferred_uncovered_evidence_ids) - selected_evidence_ids
    )
    policy_deferred_omitted_evidence_ids = sorted(
        set(deferred_omitted_evidence_ids) - selected_evidence_ids
    )
    info = {
        "used_llm": (bool(prompt) and not merge_error) or rank_used_llm,
        "merge_used_llm": bool(prompt) and not merge_error,
        "rank_used_llm": rank_used_llm,
        "rank_fallback_used": rank_fallback_used,
        "rank_candidate_count": merged_pool_count if rank_attempted else 0,
        "rank_selected_count": len(rank_selected_ids) if rank_used_llm else 0,
        "rank_dropped_edit_ids": [
            str(edit.get("_merged_edit_id") or "")
            for edit in rank_dropped_edits
            if str(edit.get("_merged_edit_id") or "")
        ],
        "rank_dropped_evidence_ids": rank_dropped_evidence_ids,
        "hard_rank_enabled": rank_enabled,
        "reasoning": rank_reasoning or " ".join(merge_reasoning).strip() or "coverage merge without rank",
        "merge_reasoning": " ".join(merge_reasoning).strip(),
        "rank_reasoning": rank_reasoning,
        "merge_error": merge_error,
        "rank_error": rank_error,
        "candidate_count": len(accepted_edits or []),
        "deduped_candidate_count": len(deduped),
        "sent_candidate_count": len(deduped),
        "selected_edit_count": len(ranked_edits),
        "epoch_edit_budget": epoch_edit_budget,
        "selected_edit_ids": covered_source_ids,
        "selected_merged_edit_ids": [
            str(edit.get("_merged_edit_id")) for edit in ranked_edits
        ],
        "dropped_edit_ids": _covered_source_edit_ids(dropped_budget_edits),
        "omitted_source_edit_ids": sorted(set(omitted_original_source_ids)),
        "uncovered_source_edit_ids": uncovered_source_ids,
        "corrective_edit_count": len(corrective_edits),
        "success_only_edit_count": len(success_only_edits),
        "merged_pool_count": merged_pool_count,
        "ranked_edit_count": len(ranked_edits),
        "recovery_rule_edit_count": sum(_epoch_edit_mentions_recovery_rules(edit) for edit in ranked_edits),
        "replace_or_generalize_edit_count": sum(
            _epoch_edit_is_replace_or_generalize(edit, evidence_by_id) for edit in ranked_edits
        ),
        "epoch_merge_chunk_count": chunk_count,
        "epoch_merge_recursive_rounds": merge_rounds,
        "epoch_failure_merge_rounds": failure_merge_rounds,
        "epoch_success_merge_rounds": success_merge_rounds,
        "epoch_conflict_merge_rounds": conflict_merge_rounds,
        "identity_fallback_edit_count": identity_fallback_count,
        "deferred_uncovered_source_edit_ids": sorted(
            set(deferred_uncovered_source_ids)
        ),
        "deferred_uncovered_source_evidence_ids": policy_deferred_uncovered_evidence_ids,
        "deferred_omitted_source_edit_ids": sorted(
            set(deferred_omitted_source_ids)
        ),
        "deferred_omitted_source_evidence_ids": policy_deferred_omitted_evidence_ids,
        "covered_source_edit_count": len(set(covered_source_ids)),
        "uncovered_source_edit_count": len(uncovered_source_ids),
        "budget_applied": False,
        "dropped_edit_count": len(dropped_budget_edits),
        "dropped_merged_edit_ids": [
            str(edit.get("_merged_edit_id") or "")
            for edit in dropped_budget_edits
        ],
    }
    return ranked_edits, info


def _merge_epoch_accepted_edits(
    *,
    teacher: Any,
    dataset: str,
    current_skill_md: str,
    accepted_edits: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    args: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    epoch_edit_budget = max(
        0,
        int(getattr(args, "epoch_edit_budget", DEFAULT_EPOCH_EDIT_BUDGET) or 0),
    )
    evidence_by_id = {
        str(row.get("evidence_id") or ""): _public_evidence_card(row)
        for row in evidence_rows or []
        if str(row.get("evidence_id") or "")
    }
    coverage_mode = _resolve_epoch_skill_update_mode(args) == "cluster_step"
    if not accepted_edits or (epoch_edit_budget <= 0 and not coverage_mode):
        return [], {
            "used_llm": False,
            "reasoning": "no accepted edits or epoch_edit_budget <= 0",
            "candidate_count": len(accepted_edits or []),
            "selected_edit_count": 0,
            "epoch_edit_budget": epoch_edit_budget,
        }

    deduped = _dedupe_epoch_edits(accepted_edits, evidence_by_id)
    if not deduped:
        return [], {
            "used_llm": False,
            "reasoning": "no valid accepted edits after dedupe",
            "candidate_count": len(accepted_edits or []),
            "deduped_candidate_count": 0,
            "selected_edit_count": 0,
            "epoch_edit_budget": epoch_edit_budget,
        }

    for idx, edit in enumerate(deduped):
        edit["_candidate_edit_id"] = str(edit.get("_candidate_edit_id") or f"C{idx + 1:06d}")
    corrective_edits = [edit for edit in deduped if _epoch_edit_is_corrective(edit, evidence_by_id)]
    success_only_edits = [edit for edit in deduped if not _epoch_edit_is_corrective(edit, evidence_by_id)]
    ordered_deduped = corrective_edits + success_only_edits
    if coverage_mode:
        return _merge_epoch_accepted_edits_no_budget(
            teacher=teacher,
            dataset=dataset,
            current_skill_md=current_skill_md,
            accepted_edits=accepted_edits,
            deduped=ordered_deduped,
            corrective_edits=corrective_edits,
            success_only_edits=success_only_edits,
            evidence_by_id=evidence_by_id,
            args=args,
        )
    max_candidates = max(
        1,
        int(getattr(args, "epoch_merge_max_candidates", DEFAULT_EPOCH_MERGE_MAX_CANDIDATES) or DEFAULT_EPOCH_MERGE_MAX_CANDIDATES),
    )
    if bool(getattr(args, "llm_full_context", False)):
        max_candidates = max(1, len(ordered_deduped))
    candidate_subset = ordered_deduped[:max_candidates]

    def add_epoch_merge_counts(info: Dict[str, Any], final_edits: List[Dict[str, Any]], merged_pool_count: int) -> Dict[str, Any]:
        info.update(
            {
                "corrective_edit_count": len(corrective_edits),
                "success_only_edit_count": len(success_only_edits),
                "merged_pool_count": int(merged_pool_count),
                "ranked_edit_count": len(final_edits),
                "recovery_rule_edit_count": sum(_epoch_edit_mentions_recovery_rules(edit) for edit in final_edits),
                "replace_or_generalize_edit_count": sum(
                    _epoch_edit_is_replace_or_generalize(edit, evidence_by_id) for edit in final_edits
                ),
            }
        )
        return info

    if bool(getattr(args, "no_epoch_global_merge", False)):
        fallback_edits, fallback_info = _fallback_epoch_merged_edits(
            candidate_subset,
            evidence_by_id=evidence_by_id,
            epoch_edit_budget=epoch_edit_budget,
        )
        fallback_info.update({"merge_used_llm": False, "rank_used_llm": False})
        return fallback_edits, add_epoch_merge_counts(fallback_info, fallback_edits, len(candidate_subset))

    if bool(getattr(args, "llm_full_context", False)):
        max_content_chars = -1
        skill_max_chars = 0
    else:
        max_content_chars = max(
            80,
            int(
                getattr(
                    args,
                    "epoch_merge_max_content_chars",
                    DEFAULT_EPOCH_MERGE_MAX_CONTENT_CHARS,
                )
                or DEFAULT_EPOCH_MERGE_MAX_CONTENT_CHARS
            ),
        )
        skill_max_chars = int(
            getattr(args, "epoch_merge_skill_max_chars", DEFAULT_EPOCH_MERGE_SKILL_MAX_CHARS)
            or DEFAULT_EPOCH_MERGE_SKILL_MAX_CHARS
        )
    dataset_prompts = prompts_for_dataset(dataset)
    candidate_records = [
        _compact_epoch_edit_candidate(
            edit,
            candidate_edit_id=str(edit.get("_candidate_edit_id")),
            evidence_by_id=evidence_by_id,
            max_content_chars=max_content_chars,
        )
        for edit in candidate_subset
    ]
    payload = {
        "skill_md": _truncate_middle(current_skill_md, skill_max_chars),
        "accepted_candidate_edits": candidate_records,
    }
    prompt = dataset_prompts.get("epoch_edit_merger")
    raw: Dict[str, Any] = {}
    merge_error = ""
    if prompt:
        try:
            raw = _teacher_chat_json(teacher, prompt, payload)
        except Exception as exc:
            merge_error = str(exc)

    id_to_edit = {str(edit.get("_candidate_edit_id")): edit for edit in candidate_subset}

    def infer_source_edit_ids(edit: Dict[str, Any], explicit_ids: Optional[List[str]] = None) -> List[str]:
        explicit = [str(eid) for eid in (explicit_ids or []) if str(eid)]
        if explicit:
            return explicit
        edit_evidence = set(_evidence_id_list(edit.get("evidence_ids")))
        if not edit_evidence:
            return []
        return [
            cid
            for cid, source in id_to_edit.items()
            if edit_evidence.intersection(_evidence_id_list(source.get("evidence_ids")))
        ]

    merged_pool: List[Dict[str, Any]] = []
    selected_ids = [str(eid) for eid in (raw or {}).get("selected_edit_ids") or [] if str(eid)]
    for raw_edit in (raw or {}).get("merged_edits") or []:
        if not isinstance(raw_edit, dict):
            continue
        edit = _normalize_markdown_edit(raw_edit)
        if edit is None or not _linked_evidence(edit, evidence_by_id):
            continue
        source_ids = infer_source_edit_ids(
            edit,
            [str(eid) for eid in raw_edit.get("source_edit_ids") or [] if str(eid)],
        )
        if source_ids:
            edit["source_edit_ids"] = source_ids
        merged_pool.append(edit)

    if not merged_pool and selected_ids:
        for selected_id in selected_ids:
            source = id_to_edit.get(selected_id)
            if source is None:
                continue
            edit = _normalize_markdown_edit(source)
            if edit is None or not _linked_evidence(edit, evidence_by_id):
                continue
            edit["source_edit_ids"] = [selected_id]
            merged_pool.append(edit)

    raw_had_merge_decision = isinstance(raw, dict) and "merged_edits" in raw and not merge_error
    if not merged_pool and raw_had_merge_decision:
        info = {
            "used_llm": bool(prompt),
            "merge_used_llm": bool(prompt),
            "rank_used_llm": False,
            "llm_returned_empty": True,
            "reasoning": str((raw or {}).get("reasoning") or ""),
            "candidate_count": len(accepted_edits or []),
            "deduped_candidate_count": len(deduped),
            "sent_candidate_count": len(candidate_subset),
            "selected_edit_count": 0,
            "epoch_edit_budget": epoch_edit_budget,
            "selected_edit_ids": selected_ids,
            "dropped_edit_ids": [str(edit.get("_candidate_edit_id")) for edit in candidate_subset],
        }
        return [], add_epoch_merge_counts(info, [], 0)

    if not merged_pool:
        fallback_edits, fallback_info = _fallback_epoch_merged_edits(
            candidate_subset,
            evidence_by_id=evidence_by_id,
            epoch_edit_budget=epoch_edit_budget,
        )
        fallback_info.update(
            {
                "used_llm": bool(prompt) and not merge_error,
                "merge_used_llm": bool(prompt) and not merge_error,
                "rank_used_llm": False,
                "llm_returned_empty": True,
                "merge_error": merge_error,
                "raw_reasoning": str((raw or {}).get("reasoning") or ""),
                "candidate_count": len(accepted_edits or []),
                "deduped_candidate_count": len(deduped),
                "sent_candidate_count": len(candidate_subset),
            }
        )
        return fallback_edits, add_epoch_merge_counts(fallback_info, fallback_edits, len(candidate_subset))

    merged_pool = _dedupe_epoch_edits(merged_pool, evidence_by_id)
    for idx, edit in enumerate(merged_pool):
        edit["_merged_edit_id"] = str(edit.get("_merged_edit_id") or f"M{idx + 1:06d}")
        annotated = _annotate_edit_metadata(edit, _linked_evidence(edit, evidence_by_id), merge_level=2)
        edit.update(annotated)

    rank_prompt = dataset_prompts.get("epoch_edit_ranker")
    rank_raw: Dict[str, Any] = {}
    rank_error = ""
    ranked_edits: List[Dict[str, Any]] = []
    ranked_ids: List[str] = []
    if rank_prompt:
        rank_payload = {
            "skill_md": _truncate_middle(current_skill_md, skill_max_chars),
            "epoch_edit_budget": epoch_edit_budget,
            "merged_candidate_edits": [
                _compact_epoch_edit_candidate(
                    edit,
                    candidate_edit_id=str(edit.get("_merged_edit_id")),
                    evidence_by_id=evidence_by_id,
                    max_content_chars=max_content_chars,
                )
                for edit in merged_pool
            ],
        }
        try:
            rank_raw = _teacher_chat_json(teacher, rank_prompt, rank_payload)
        except Exception as exc:
            rank_error = str(exc)
        raw_ranked_ids = (rank_raw or {}).get("selected_edit_ids")
        if raw_ranked_ids is None:
            raw_ranked_ids = (rank_raw or {}).get("ranked_edit_ids")
        ranked_ids = [str(eid) for eid in raw_ranked_ids or [] if str(eid)]
        id_to_merged = {str(edit.get("_merged_edit_id")): edit for edit in merged_pool}
        for merged_id in ranked_ids:
            edit = id_to_merged.get(merged_id)
            if edit is not None:
                ranked_edits.append(edit)
            if len(ranked_edits) >= epoch_edit_budget:
                break

    raw_rank_had_decision = (
        isinstance(rank_raw, dict)
        and ("selected_edit_ids" in rank_raw or "ranked_edit_ids" in rank_raw)
        and not rank_error
    )
    if not ranked_edits and raw_rank_had_decision and not ranked_ids:
        info = {
            "used_llm": (bool(prompt) and not merge_error) or bool(rank_prompt),
            "merge_used_llm": bool(prompt) and not merge_error,
            "rank_used_llm": bool(rank_prompt),
            "rank_returned_empty": True,
            "reasoning": str((rank_raw or {}).get("reasoning") or (raw or {}).get("reasoning") or ""),
            "merge_reasoning": str((raw or {}).get("reasoning") or ""),
            "rank_reasoning": str((rank_raw or {}).get("reasoning") or ""),
            "candidate_count": len(accepted_edits or []),
            "deduped_candidate_count": len(deduped),
            "sent_candidate_count": len(candidate_subset),
            "selected_edit_count": 0,
            "epoch_edit_budget": epoch_edit_budget,
            "selected_edit_ids": ranked_ids,
            "dropped_edit_ids": [str(edit.get("_candidate_edit_id")) for edit in candidate_subset],
        }
        return [], add_epoch_merge_counts(info, [], len(merged_pool))

    if not ranked_edits:
        ranked_edits = sorted(merged_pool, key=lambda edit: _epoch_edit_priority(edit, evidence_by_id))[:epoch_edit_budget]
        ranked_ids = [str(edit.get("_merged_edit_id")) for edit in ranked_edits]

    final_edits: List[Dict[str, Any]] = []
    for edit in ranked_edits[:epoch_edit_budget]:
        clean = _normalize_markdown_edit(edit)
        if clean is None or not _linked_evidence(clean, evidence_by_id):
            continue
        source_ids = infer_source_edit_ids(
            clean,
            [str(eid) for eid in edit.get("source_edit_ids") or [] if str(eid)],
        )
        if source_ids:
            clean["source_edit_ids"] = source_ids
        clean = _annotate_edit_metadata(clean, _linked_evidence(clean, evidence_by_id), merge_level=3)
        final_edits.append(clean)

    selected_source_ids = sorted(
        {
            str(source_id)
            for edit in final_edits
            for source_id in (edit.get("source_edit_ids") or [])
            if str(source_id)
        }
    )
    dropped_ids = [
        str(edit.get("_candidate_edit_id"))
        for edit in candidate_subset
        if str(edit.get("_candidate_edit_id")) not in set(selected_source_ids)
    ]
    info = {
        "used_llm": (bool(prompt) and not merge_error) or (bool(rank_prompt) and not rank_error),
        "merge_used_llm": bool(prompt) and not merge_error,
        "rank_used_llm": bool(rank_prompt) and not rank_error,
        "reasoning": str((rank_raw or {}).get("reasoning") or (raw or {}).get("reasoning") or ""),
        "merge_reasoning": str((raw or {}).get("reasoning") or ""),
        "rank_reasoning": str((rank_raw or {}).get("reasoning") or ""),
        "merge_error": merge_error,
        "rank_error": rank_error,
        "candidate_count": len(accepted_edits or []),
        "deduped_candidate_count": len(deduped),
        "sent_candidate_count": len(candidate_subset),
        "selected_edit_count": len(final_edits),
        "epoch_edit_budget": epoch_edit_budget,
        "selected_edit_ids": selected_source_ids,
        "selected_merged_edit_ids": ranked_ids[: len(final_edits)],
        "dropped_edit_ids": dropped_ids,
    }
    return final_edits, add_epoch_merge_counts(info, final_edits, len(merged_pool))


def _run_epoch_coverage_repair_merge(
    *,
    teacher: Any,
    dataset: str,
    candidate_skill_md: str,
    uncovered_cards: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    args: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    defer_uncovered = _working_branch_defer_uncovered_enabled(dataset, args)
    evidence_by_id = {
        str(row.get("evidence_id") or ""): _public_evidence_card(row)
        for row in evidence_rows or []
        if str(row.get("evidence_id") or "")
    }
    uncovered_ids = [str(card.get("evidence_id") or "") for card in uncovered_cards or [] if str(card.get("evidence_id") or "")]
    if not uncovered_ids:
        return [], {
            "enabled": False,
            "repair_merge_input_evidence": 0,
            "repair_merge_edit_count": 0,
            "repair_fallback_evidence_ids": [],
        }

    prompt = prompts_for_dataset(dataset).get("epoch_coverage_repair_merger")
    raw: Dict[str, Any] = {}
    repair_error = ""
    if prompt:
        payload = {
            "skill_md": candidate_skill_md,
            "uncovered_evidence_cards": [
                _public_evidence_card_for_proposer(
                    card,
                    full_context=bool(getattr(args, "llm_full_context", False)),
                )
                for card in uncovered_cards
            ],
        }
        try:
            raw = _teacher_chat_json(teacher, prompt, payload)
        except Exception as exc:
            repair_error = str(exc)
            raw = {}

    raw_items = []
    if isinstance(raw, dict):
        for key in ["repair_edits", "edits", "merged_edits"]:
            if isinstance(raw.get(key), list):
                raw_items = [dict(item) for item in raw.get(key) or [] if isinstance(item, dict)]
                break
    full_by_id = {str(card.get("evidence_id") or ""): card for card in uncovered_cards or []}
    repair_edits: List[Dict[str, Any]] = []
    for raw_edit in raw_items:
        edit = _normalize_markdown_edit(raw_edit)
        if edit is None:
            continue
        linked = _linked_evidence(edit, evidence_by_id)
        if not linked:
            continue
        repair_edits.append(_annotate_edit_metadata(edit, linked, merge_level=3))

    omitted_ids = set(_omitted_evidence_ids(raw))
    covered_ids = set(_covered_evidence_ids(repair_edits))
    uncovered_not_covered_ids = sorted(
        set(uncovered_ids) - covered_ids - omitted_ids
    )
    fallback_ids: List[str] = []
    if not defer_uncovered:
        for evidence_id in uncovered_not_covered_ids:
            fallback = _fallback_markdown_edit_from_evidence(full_by_id.get(evidence_id) or {})
            if fallback is None:
                continue
            linked = _linked_evidence(fallback, evidence_by_id)
            if not linked:
                continue
            repair_edits.append(_annotate_edit_metadata(fallback, linked, merge_level=3))
            fallback_ids.append(evidence_id)

    repair_edits = _dedupe_epoch_edits(repair_edits, evidence_by_id)
    return repair_edits, {
        "enabled": True,
        "repair_merge_input_evidence": len(uncovered_ids),
        "repair_merge_edit_count": len(repair_edits),
        "repair_merge_error": repair_error,
        "repair_reasoning": str((raw or {}).get("reasoning") or ""),
        "repair_omitted_evidence_ids": sorted(omitted_ids),
        "repair_fallback_evidence_ids": sorted(fallback_ids),
        "deferred_uncovered_evidence_ids": (
            uncovered_not_covered_ids if defer_uncovered else []
        ),
        "deferred_omitted_evidence_ids": (
            sorted(set(uncovered_ids).intersection(omitted_ids))
            if defer_uncovered
            else []
        ),
        "repair_covered_evidence_ids": _covered_evidence_ids(repair_edits),
    }


def _public_evidence_card_for_coverage_audit(card: Dict[str, Any]) -> Dict[str, Any]:
    public = {
        "evidence_id": str(card.get("evidence_id") or ""),
        "pattern": str(card.get("pattern") or "").strip(),
        "proposed_edit": _public_proposed_edit_for_proposer(card),
    }
    return {key: value for key, value in public.items() if value not in (None, "", {})}


def _normalized_skill_coverage_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _evidence_has_exact_skill_coverage(skill_md: str, card: Dict[str, Any]) -> bool:
    """Check whether the normalized proposed edit is already realized verbatim."""
    proposed = _normalize_proposed_edit(card.get("proposed_edit"))
    if proposed is None:
        return False
    skill_text = _normalized_skill_coverage_text(skill_md)
    target = _normalized_skill_coverage_text(proposed.get("target"))
    content = _normalized_skill_coverage_text(proposed.get("content"))
    op = str(proposed.get("op") or "").lower()
    if op == "delete":
        return bool(target) and target not in skill_text
    if not content or content not in skill_text:
        return False
    if op == "replace" and target and target != content and target in skill_text:
        return False
    return True


def _coverage_audit_chunks(
    *,
    skill_md: str,
    public_cards: List[Dict[str, Any]],
    system_prompt: str,
    args: Any,
) -> List[List[Dict[str, Any]]]:
    max_cards = max(
        1,
        _int_arg(
            getattr(
                args,
                "l2_max_evidence_per_window",
                getattr(args, "evidence_window_size", 20),
            ),
            20,
        ),
    )
    input_budget = _int_arg(
        getattr(args, "llm_max_input_chars", DEFAULT_LLM_MAX_INPUT_CHARS),
        DEFAULT_LLM_MAX_INPUT_CHARS,
    )
    if bool(getattr(args, "llm_full_context", False)):
        input_budget = 0
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for card in public_cards:
        candidate = current + [card]
        payload = {"skill_md": skill_md, "evidence_cards": candidate}
        over_count = len(candidate) > max_cards
        over_chars = input_budget > 0 and _llm_input_chars(system_prompt, payload) > input_budget
        if current and (over_count or over_chars):
            chunks.append(current)
            current = [card]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _run_evidence_skill_coverage_audit(
    *,
    teacher: Any,
    dataset: str,
    skill_md: str,
    evidence_cards: List[Dict[str, Any]],
    args: Any,
    stage: str,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    cards_by_id = {
        str(card.get("evidence_id") or ""): card
        for card in evidence_cards or []
        if str(card.get("evidence_id") or "")
    }
    all_ids = list(cards_by_id)
    if not all_ids:
        return [], [], {
            "stage": stage,
            "coverage_mode": (
                "exact_current"
                if _is_scienceworld_dataset(dataset)
                and str(stage) == "current_skill"
                else "semantic"
            ),
            "evidence_count": 0,
            "covered_count": 0,
            "uncovered_count": 0,
            "chunk_count": 0,
            "audit_errors": [],
            "fallback_evidence_ids": [],
        }

    if (
        _is_scienceworld_dataset(dataset)
    ) and str(stage) == "current_skill":
        decisions = {
            evidence_id: (
                "covered"
                if _evidence_has_exact_skill_coverage(skill_md, cards_by_id[evidence_id])
                else "uncovered"
            )
            for evidence_id in all_ids
        }
        covered_ids = [evidence_id for evidence_id in all_ids if decisions[evidence_id] == "covered"]
        uncovered_ids = [evidence_id for evidence_id in all_ids if decisions[evidence_id] == "uncovered"]
        return covered_ids, uncovered_ids, {
            "stage": stage,
            "coverage_mode": "exact_current",
            "evidence_count": len(all_ids),
            "covered_count": len(covered_ids),
            "uncovered_count": len(uncovered_ids),
            "chunk_count": 0,
            "audit_errors": [],
            "fallback_evidence_ids": [],
            "decisions": [
                {
                    "evidence_id": evidence_id,
                    "decision": decisions[evidence_id],
                    "reason": "exact proposed-edit coverage"
                    if decisions[evidence_id] == "covered"
                    else "proposed edit is not exactly present in current skill",
                }
                for evidence_id in all_ids
            ],
        }

    prompt = prompts_for_dataset(dataset).get("evidence_coverage_auditor")
    public_cards = [_public_evidence_card_for_coverage_audit(cards_by_id[eid]) for eid in all_ids]
    chunks = (
        _coverage_audit_chunks(
            skill_md=skill_md,
            public_cards=public_cards,
            system_prompt=prompt,
            args=args,
        )
        if prompt
        else [public_cards]
    )
    decisions: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    audit_errors: List[str] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_ids = {str(card.get("evidence_id") or "") for card in chunk}
        if not prompt:
            audit_errors.append("coverage auditor prompt unavailable")
            break
        try:
            raw = _teacher_chat_json(
                teacher,
                prompt,
                {"skill_md": skill_md, "evidence_cards": chunk},
            )
        except Exception as exc:
            audit_errors.append(f"chunk {chunk_index + 1}: {exc}")
            continue
        rows = raw.get("coverage") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            audit_errors.append(f"chunk {chunk_index + 1}: missing coverage list")
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id") or "")
            decision = str(row.get("decision") or "").strip().lower()
            if evidence_id not in chunk_ids or decision not in {"covered", "uncovered"}:
                continue
            decisions[evidence_id] = decision
            reasons[evidence_id] = str(row.get("reason") or "").strip()

    fallback_ids: List[str] = []
    for evidence_id in all_ids:
        if evidence_id in decisions:
            continue
        fallback_ids.append(evidence_id)
        decisions[evidence_id] = (
            "covered"
            if _evidence_has_exact_skill_coverage(skill_md, cards_by_id[evidence_id])
            else "uncovered"
        )
        reasons[evidence_id] = "conservative exact-text fallback"

    covered_ids = [evidence_id for evidence_id in all_ids if decisions[evidence_id] == "covered"]
    uncovered_ids = [evidence_id for evidence_id in all_ids if decisions[evidence_id] != "covered"]
    info = {
        "stage": stage,
        "coverage_mode": "semantic",
        "evidence_count": len(all_ids),
        "covered_count": len(covered_ids),
        "uncovered_count": len(uncovered_ids),
        "chunk_count": len(chunks),
        "audit_errors": audit_errors,
        "fallback_evidence_ids": fallback_ids,
        "decisions": [
            {
                "evidence_id": evidence_id,
                "decision": decisions[evidence_id],
                "reason": reasons.get(evidence_id, ""),
            }
            for evidence_id in all_ids
        ],
    }
    return covered_ids, uncovered_ids, info


def _iter_card_trigger_ranges(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _normalize_trigger_ranges(card.get("trigger_ranges"))


def _l2_replay_priority(card: Dict[str, Any]) -> int:
    experience_type = _normalize_experience_type(
        card.get("experience_type"),
        _evidence_edit_text(card),
        str(card.get("pattern") or card.get("l1_skilljudge_reason") or ""),
    )
    failure_type = _normalize_failure_type(card.get("failure_type") or _failure_type_from_legacy_role(card.get("evidence_role")))
    if experience_type == "failure_reflection" and failure_type == "rule_missing":
        return 0
    if experience_type == "failure_reflection" and failure_type == "rule_wrong":
        return 1
    if experience_type == "failure_reflection" and failure_type in {"rule_ignored", "invalid_action", "loop"}:
        return 2
    if experience_type == "failure_reflection":
        return 3
    if experience_type == "success_experience":
        return 4
    return 5


def _round_robin_replay_jobs_by_family(
    jobs: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
) -> List[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
    def job_family(job: Tuple[Dict[str, Any], int, Dict[str, Any]]) -> str:
        card, _, trig = job
        task_id = str((trig or {}).get("task_id") or "")
        supporting_tasks = card.get("supporting_tasks")
        if isinstance(supporting_tasks, list):
            for row in supporting_tasks:
                if isinstance(row, dict) and str(row.get("task_id") or "") == task_id:
                    return str(row.get("task_family") or "").strip()
        return str(card.get("task_family") or "").strip()

    buckets: Dict[str, List[Tuple[Dict[str, Any], int, Dict[str, Any]]]] = {}
    family_order: List[str] = []
    for job in jobs:
        family = job_family(job)
        if family not in buckets:
            buckets[family] = []
            family_order.append(family)
        buckets[family].append(job)

    ordered: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []
    while family_order:
        next_order: List[str] = []
        for family in family_order:
            bucket = buckets.get(family) or []
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            if bucket:
                next_order.append(family)
        family_order = next_order
    return ordered


def _select_l2_replay_jobs(
    linked_evidence: List[Dict[str, Any]],
    budget: int,
) -> List[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
    jobs: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []
    seen = set()
    for card in linked_evidence or []:
        evidence_id = str(card.get("evidence_id") or "")
        for range_idx, trig in enumerate(_iter_card_trigger_ranges(card)):
            start = _coerce_int(trig.get("start", 0), 0)
            key = (
                evidence_id,
                range_idx,
                str(trig.get("task_id") or ""),
                start,
                _coerce_int(trig.get("end", trig.get("start", 0)), start),
            )
            if key in seen:
                continue
            seen.add(key)
            jobs.append((card, range_idx, trig))

    ordered: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []
    for range_group in (
        [job for job in jobs if job[1] == 0],
        [job for job in jobs if job[1] != 0],
    ):
        for priority in range(5):
            priority_jobs = [
                job for job in range_group if _l2_replay_priority(job[0]) == priority
            ]
            ordered.extend(_round_robin_replay_jobs_by_family(priority_jobs))

    if budget == 0:
        return []
    if budget < 0 or budget >= len(ordered):
        return ordered
    return ordered[:budget]


def _l2_replay_evidence_row(
    *,
    card: Dict[str, Any],
    range_idx: int,
    trig: Dict[str, Any],
    before_segment: Any,
    replay: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    args: Any = None,
    dataset: str = "",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "evidence_id": card.get("evidence_id"),
        "range_index": range_idx,
        "trigger_ranges": [trig],
        "before_segments": [_optimizer_evidence_segment(before_segment, dataset=dataset)],
    }
    if replay is not None:
        raw_after = [
            item
            for item in ((replay.get("after_segments") or [[]])[0] or [])
            if isinstance(item, dict)
        ]
        materialized_after = []
        previous_score: Optional[float] = None
        for index, item in enumerate(raw_after):
            if _is_appworld_dataset(dataset):
                from .appworld_runner import redact_appworld_optimizer_value

                materialized_after.append(
                    {
                        "step": _coerce_int(item.get("step", index), index),
                        "observation": str(
                            redact_appworld_optimizer_value(str(item.get("observation") or ""))
                        ),
                        "action": str(
                            redact_appworld_optimizer_value(str(item.get("action") or ""))
                        ),
                        "feedback": str(
                            redact_appworld_optimizer_value(
                                _feedback_from_trajectory_step(raw_after, index)
                            )
                        ),
                        "execution_ok": bool(item.get("execution_ok", not item.get("bad_step"))),
                        "task_completed": bool(item.get("task_completed") or item.get("done")),
                        "done": bool(item.get("done")),
                    }
                )
                continue
            before, after, delta = _score_transition_from_step(item, previous_score=previous_score)
            materialized_after.append(
                {
                    "step": _coerce_int(item.get("step", index), index),
                    "observation": str(item.get("observation") or ""),
                    "action": str(item.get("action") or ""),
                    "feedback": _feedback_from_trajectory_step(raw_after, index),
                    "score_before": before,
                    "score_after": after,
                    "score_delta": delta,
                    "done": bool(item.get("done")),
                }
            )
            materialized_after[-1]["bad_step"] = bool(item.get("bad_step"))
            if after is not None:
                previous_score = after
        row["after_segments"] = [
            _evidence_segment_from_steps(
                task_id=str(trig.get("task_id") or ""),
                rows=materialized_after,
            )
        ]
        if _is_appworld_dataset(dataset):
            evaluation = replay.get("evaluation_summary")
            row["replay_outcome"] = {
                "execution_ok": all(
                    bool(item.get("execution_ok", not item.get("bad_step")))
                    for item in raw_after
                ),
                "task_completed": bool(replay.get("task_completed") or replay.get("done")),
                "success": bool(replay.get("success")),
                "score": 1.0 if replay.get("success") else 0.0,
                "evaluation_summary": (
                    {
                        "num_tests": int(evaluation.get("num_tests", 0) or 0),
                        "passed_tests": int(evaluation.get("passed_tests", 0) or 0),
                        "failed_tests": int(evaluation.get("failed_tests", 0) or 0),
                    }
                    if isinstance(evaluation, dict)
                    else {"num_tests": 0, "passed_tests": 0, "failed_tests": 0}
                ),
            }
    else:
        row["after_segments"] = [
            {"task_id": str(trig.get("task_id") or ""), "initial_observation": "", "steps": []}
        ]
    if error:
        row["error"] = str(error)
    return row


def _run_l2_replay_jobs_for_edit(
    *,
    runner: Any,
    replay_jobs: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    raw_by_id: Dict[str, Dict[str, Any]],
    tmp_skill_path: Path,
    args: Any,
    progress_label: str,
    window_suffix: str,
    edit_label: str,
    dataset: str = "",
) -> List[Dict[str, Any]]:
    replay_contexts: List[Dict[str, Any]] = []
    for card, range_idx, trig in replay_jobs:
        task_id = str(trig.get("task_id") or "")
        raw_result = (
            _replay_source_lookup(
                replay_source_by_evidence_id,
                evidence_id=card.get("evidence_id"),
                task_id=task_id,
            )
            or raw_by_id.get(task_id)
        )
        if raw_result is None:
            print(
                f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
                f"replay skip evidence={card.get('evidence_id')} task={task_id} reason=no_raw_result",
                flush=True,
            )
            continue
        start = _coerce_int(trig.get("start", 0), 0)
        end = _coerce_int(trig.get("end", trig.get("start", 0)), start)
        before_segments = card.get("before_segments") or []
        current_before_segment = (
            before_segments[range_idx]
            if isinstance(before_segments, list) and range_idx < len(before_segments)
            else []
        )
        replay_contexts.append(
            {
                "card": card,
                "range_idx": range_idx,
                "trig": trig,
                "task_id": task_id,
                "raw_result": raw_result,
                "start": start,
                "end": end,
                "continue_steps": max(1, end - start + 1),
                "before_segment": current_before_segment,
            }
        )

    if not replay_contexts:
        return []

    supports_batch = bool(getattr(runner, "run_from_trace_range_set", None)) and len(replay_contexts) > 1
    print(
        f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
        f"replay batch start jobs={len(replay_contexts)} parallel={'yes' if supports_batch else 'no'}",
        flush=True,
    )

    def run_sequential(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ctx in contexts:
            card = ctx["card"]
            range_idx = int(ctx["range_idx"])
            try:
                print(
                    f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
                    f"replay start evidence={card.get('evidence_id')} task={ctx['task_id']} "
                    f"range={range_idx} steps={ctx['start']}-{ctx['end']}",
                    flush=True,
                )
                replay = runner.run_from_trace_range(
                    ctx["raw_result"],
                    start_step=ctx["start"],
                    continue_steps=ctx["continue_steps"],
                    skill_bank=MarkdownSkillBank(str(tmp_skill_path)),
                    world_knowledge={},
                )
                rows.append(
                    _l2_replay_evidence_row(
                        card=card,
                        range_idx=range_idx,
                        trig=ctx["trig"],
                        before_segment=ctx["before_segment"],
                        replay=replay,
                        args=args,
                        dataset=dataset,
                    )
                )
                print(
                    f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
                    f"replay done evidence={card.get('evidence_id')} task={ctx['task_id']} range={range_idx}",
                    flush=True,
                )
            except Exception as exc:
                rows.append(
                    _l2_replay_evidence_row(
                        card=card,
                        range_idx=range_idx,
                        trig=ctx["trig"],
                        before_segment=ctx["before_segment"],
                        error=str(exc),
                        args=args,
                        dataset=dataset,
                    )
                )
                print(
                    f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
                    f"replay error evidence={card.get('evidence_id')} task={ctx['task_id']} "
                    f"range={range_idx} error={exc}",
                    flush=True,
                )
        return rows

    if not supports_batch:
        return run_sequential(replay_contexts)

    batch_jobs = [
        {
            "original_result": ctx["raw_result"],
            "start_step": ctx["start"],
            "continue_steps": ctx["continue_steps"],
            "evidence_id": ctx["card"].get("evidence_id"),
            "task_id": ctx["task_id"],
            "range_index": ctx["range_idx"],
        }
        for ctx in replay_contexts
    ]
    try:
        batch_results = runner.run_from_trace_range_set(
            batch_jobs,
            skill_bank=MarkdownSkillBank(str(tmp_skill_path)),
            world_knowledge={},
            desc=f"{progress_label} skill edit{window_suffix} edit {edit_label} replay",
        )
    except Exception as exc:
        print(
            f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
            f"replay batch error; falling back to sequential error={exc}",
            flush=True,
        )
        return run_sequential(replay_contexts)

    rows: List[Dict[str, Any]] = []
    for ctx, item in zip(replay_contexts, list(batch_results or [])):
        card = ctx["card"]
        range_idx = int(ctx["range_idx"])
        if isinstance(item, dict) and "ok" in item:
            if item.get("ok"):
                rows.append(
                    _l2_replay_evidence_row(
                        card=card,
                        range_idx=range_idx,
                        trig=ctx["trig"],
                        before_segment=ctx["before_segment"],
                        replay=item.get("result") or {},
                        args=args,
                        dataset=dataset,
                    )
                )
            else:
                rows.append(
                    _l2_replay_evidence_row(
                        card=card,
                        range_idx=range_idx,
                        trig=ctx["trig"],
                        before_segment=ctx["before_segment"],
                        error=str(item.get("error") or "unknown replay error"),
                        args=args,
                        dataset=dataset,
                    )
                )
        else:
            rows.append(
                _l2_replay_evidence_row(
                    card=card,
                    range_idx=range_idx,
                    trig=ctx["trig"],
                    before_segment=ctx["before_segment"],
                    replay=item if isinstance(item, dict) else {},
                    args=args,
                    dataset=dataset,
                )
            )
    if len(rows) < len(replay_contexts):
        for ctx in replay_contexts[len(rows) :]:
            rows.append(
                _l2_replay_evidence_row(
                    card=ctx["card"],
                    range_idx=int(ctx["range_idx"]),
                    trig=ctx["trig"],
                    before_segment=ctx["before_segment"],
                    error="missing replay result",
                    args=args,
                    dataset=dataset,
                )
            )
    print(
        f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
        f"replay batch done replay_evidence={len(rows)}",
        flush=True,
    )
    return rows


def _teacher_chat_json(teacher, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    llm = getattr(teacher, "llm", teacher)
    if llm is None:
        return {}
    return llm.chat_json(system_prompt, payload)


def _l2_progress_label(round_idx: int, trigger_reason: str = "") -> str:
    if str(trigger_reason or "") == "epoch_boundary":
        return f"epoch {int(round_idx) + 1}"
    return f"round {int(round_idx)}"


def _build_l1_skill_judge_payload(
    *,
    skill_md: str,
    evidence_candidate: Dict[str, Any],
    dataset: str,
    args: Any = None,
) -> Dict[str, Any]:
    prompt = prompts_for_dataset(dataset)["skill_judge_l1"]
    return _compact_llm_payload(
        {
            "dataset": dataset,
            "skill_md": skill_md,
            "evidence": _optimizer_evidence_card(evidence_candidate, dataset=dataset),
        },
        prompt,
        args,
        label="evidence quality judge",
    )


def _l1_skill_judge_label(
    *,
    teacher,
    skill_md: str,
    evidence_candidate: Dict[str, Any],
    dataset: str,
    args: Any = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt = prompts_for_dataset(dataset)["skill_judge_l1"]
    if payload is None:
        payload = _build_l1_skill_judge_payload(
            skill_md=skill_md,
            evidence_candidate=evidence_candidate,
            dataset=dataset,
            args=args,
        )
    try:
        raw = _teacher_chat_json(teacher, prompt, payload)
    except Exception as exc:
        raw = {"decision": "keep", "reason": f"Evidence Quality Judge unavailable; retained: {exc}"}
    decision = str(raw.get("decision") or "").strip().lower()
    if decision not in {"keep", "drop"}:
        decision = "keep"
        raw = dict(raw)
        raw["reason"] = str(raw.get("reason") or "Evidence Quality Judge returned no valid decision; retained")
    return {
        "decision": decision,
        "reason": str(raw.get("reason") or ""),
    }


def _visible_replay_evidence_for_skill_edit_judge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visible = []
    for row in rows or []:
        item = {
            "evidence_id": row.get("evidence_id"),
            "range_index": row.get("range_index"),
            "trigger_ranges": row.get("trigger_ranges") or [],
            "after_segments": row.get("after_segments") or [],
        }
        if row.get("error") not in (None, ""):
            item["error"] = row.get("error")
        if isinstance(row.get("replay_outcome"), dict):
            item["replay_outcome"] = row.get("replay_outcome")
        visible.append(item)
    return visible


def _build_l2_skill_judge_payload(
    *,
    skill_md: str,
    edit: Dict[str, Any],
    linked_evidence: List[Dict[str, Any]],
    replay_evidence: List[Dict[str, Any]],
    dataset: str,
    args: Any = None,
    reflection_allowed: bool = False,
) -> Dict[str, Any]:
    prompt = prompts_for_dataset(dataset)["skill_judge_l2"]
    replay_attribution_allowed = (
        _uses_l2_replay_attribution(dataset) and not _no_replay_enabled(args)
    )
    payload = {
        "dataset": dataset,
        "skill_md": skill_md,
        "candidate_edit": edit,
        "linked_evidence": _public_evidence_cards(
            [_optimizer_evidence_card(card, dataset=dataset) for card in linked_evidence]
        ),
        "replay_evidence": _visible_replay_evidence_for_skill_edit_judge(replay_evidence),
    }
    if _uses_l2_replay_attribution(dataset):
        payload["reflection_allowed"] = bool(
            reflection_allowed and replay_attribution_allowed
        )
    if _no_replay_enabled(args):
        payload.update(
            {
                "replay_mode": "disabled",
                "replay_attribution_allowed": False,
                "judge_mode_instruction": (
                    "No replay was run. Judge only whether the candidate edit is "
                    "applicable and supported by the linked EvidenceCards. Return "
                    "accept or reject; do not infer replay failure attribution or request reflection."
                ),
            }
        )
    if _is_appworld_dataset(dataset):
        return _compact_appworld_l2_judge_payload(payload, prompt, args)
    return _compact_llm_payload(payload, prompt, args, label="skill edit judge")


def _normalize_l2_replay_attribution_label(
    raw: Dict[str, Any],
    *,
    reflection_allowed: bool,
    defer_rule_ignored: bool = False,
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    decision = str((raw or {}).get("decision") or "").strip().lower()
    failure_type = str((raw or {}).get("replay_failure_type") or "").strip().lower()
    if failure_type not in L2_REPLAY_FAILURE_TYPES:
        failure_type = ""

    if failure_type == "rule_ignored":
        decision = "defer" if defer_rule_ignored else "accept"
    elif failure_type in {"rule_missing", "rule_wrong"}:
        decision = "reflect" if reflection_allowed else "reject"
    elif decision not in {"accept", "reject"}:
        decision = "reject"

    return {
        "decision": decision,
        "replay_failure_type": failure_type or None,
        "reason": str((raw or {}).get("reason") or ""),
    }


def _normalize_appworld_l2_judge_label(
    raw: Dict[str, Any],
    *,
    reflection_allowed: bool,
    defer_rule_ignored: bool = False,
) -> Dict[str, Any]:
    """Backward-compatible alias for existing AppWorld callers and artifacts."""
    return _normalize_l2_replay_attribution_label(
        raw,
        reflection_allowed=reflection_allowed,
        defer_rule_ignored=defer_rule_ignored,
    )


def _l2_skill_judge_label(
    *,
    teacher,
    skill_md: str,
    edit: Dict[str, Any],
    linked_evidence: List[Dict[str, Any]],
    replay_evidence: List[Dict[str, Any]],
    dataset: str,
    args: Any = None,
    payload: Optional[Dict[str, Any]] = None,
    reflection_allowed: bool = False,
) -> Dict[str, Any]:
    prompt = prompts_for_dataset(dataset)["skill_judge_l2"]
    if payload is None:
        payload = _build_l2_skill_judge_payload(
            skill_md=skill_md,
            edit=edit,
            linked_evidence=linked_evidence,
            replay_evidence=replay_evidence,
            dataset=dataset,
            args=args,
            reflection_allowed=reflection_allowed,
        )
    try:
        raw = _teacher_chat_json(teacher, prompt, payload)
    except Exception as exc:
        raw = {"decision": "reject", "reason": f"Skill Edit Judge teacher failed: {exc}"}
    if _uses_l2_replay_attribution(dataset) and not _no_replay_enabled(args):
        return _normalize_l2_replay_attribution_label(
            raw,
            reflection_allowed=reflection_allowed,
            defer_rule_ignored=_defer_rule_ignored_enabled(dataset, args),
        )
    decision = str(raw.get("decision") or "").strip().lower()
    if decision not in {"accept", "reject"}:
        decision = "reject"
    label = {"decision": decision, "reason": str(raw.get("reason") or "")}
    if _uses_l2_replay_attribution(dataset):
        label["replay_failure_type"] = None
    return label


def _l2_replay_reflection(
    *,
    dataset: str,
    teacher: Any,
    skill_md: str,
    edit: Dict[str, Any],
    linked_evidence: List[Dict[str, Any]],
    replay_evidence: List[Dict[str, Any]],
    judge_output: Dict[str, Any],
    args: Any = None,
) -> Dict[str, Any]:
    prompt = prompts_for_dataset(dataset)["skill_reflector_l2"]
    payload = {
        "dataset": dataset,
        "skill_md": skill_md,
        "candidate_edit": edit,
        "linked_evidence": _public_evidence_cards(
            [_optimizer_evidence_card(card, dataset=dataset) for card in linked_evidence]
        ),
        "replay_evidence": _visible_replay_evidence_for_skill_edit_judge(replay_evidence),
        "judge_output": dict(judge_output or {}),
    }
    if _is_appworld_dataset(dataset):
        payload = _compact_appworld_l2_judge_payload(payload, prompt, args)
    else:
        payload = _compact_llm_payload(payload, prompt, args, label="skill edit reflection")
    try:
        raw = _teacher_chat_json(teacher, prompt, payload)
    except Exception as exc:
        raw = {"reason": f"{dataset} L2 reflection teacher failed: {exc}", "revised_edit": None}
    if not isinstance(raw, dict):
        raw = {"reason": f"{dataset} L2 reflection returned non-object JSON", "revised_edit": None}

    revised = _normalize_markdown_edit((raw or {}).get("revised_edit") or {})
    validation_reason = ""
    apply_reports: List[Dict[str, Any]] = []
    original_ids = set(_evidence_id_list((edit or {}).get("evidence_ids")))
    revised_ids = set(_evidence_id_list((revised or {}).get("evidence_ids")))
    if revised is None:
        validation_reason = "missing_or_invalid_revised_edit"
    elif not revised_ids or not revised_ids.issubset(original_ids):
        revised = None
        validation_reason = "revised_evidence_ids_not_nonempty_subset"
    elif all(
        str((revised or {}).get(key) or "") == str((edit or {}).get(key) or "")
        for key in ("op", "target", "content")
    ):
        revised = None
        validation_reason = "revised_edit_unchanged"
    else:
        _, apply_reports = _apply_markdown_edits(skill_md, [revised])
        if not apply_reports or not all(bool(row.get("applied")) for row in apply_reports):
            revised = None
            validation_reason = "revised_edit_did_not_apply"

    return {
        "accepted": revised is not None,
        "revised_edit": revised,
        "reason": str((raw or {}).get("reason") or ""),
        "validation_reason": validation_reason,
        "system_prompt": prompt,
        "input": payload,
        "output": raw,
        "apply_reports": apply_reports,
    }


def _appworld_l2_reflection(
    *,
    teacher: Any,
    skill_md: str,
    edit: Dict[str, Any],
    linked_evidence: List[Dict[str, Any]],
    replay_evidence: List[Dict[str, Any]],
    judge_output: Dict[str, Any],
    args: Any = None,
) -> Dict[str, Any]:
    """Backward-compatible AppWorld wrapper around the shared reflector."""
    return _l2_replay_reflection(
        dataset="appworld",
        teacher=teacher,
        skill_md=skill_md,
        edit=edit,
        linked_evidence=linked_evidence,
        replay_evidence=replay_evidence,
        judge_output=judge_output,
        args=args,
    )


def _run_epoch_final_replay_filter(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    epoch_idx: int,
    base_skill_md: str,
    candidate_skill_md: str,
    applied_edits: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    stage: str = "epoch_final_merged",
    progress_suffix: str = "final replay",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    budget = _int_arg(
        getattr(args, "epoch_final_replay_ranges_per_edit", DEFAULT_EPOCH_FINAL_REPLAY_RANGES_PER_EDIT),
        DEFAULT_EPOCH_FINAL_REPLAY_RANGES_PER_EDIT,
    )
    if budget == 0:
        total = len(applied_edits or [])
        return list(applied_edits or []), [], {
            "enabled": False,
            "reason": "disabled_by_zero_budget",
            "final_replay_edits": total,
            "final_replay_accepts": 0,
            "final_replay_rejects": 0,
            "final_replay_skipped": total,
            "final_replay_bypassed": total,
            "epoch_final_replay_ranges_per_edit": 0,
            "stage": stage,
        }
    evidence_by_id = {
        str(row.get("evidence_id") or ""): row
        for row in evidence_rows or []
        if str(row.get("evidence_id") or "")
    }
    l2_judge_prompt = prompts_for_dataset(dataset)["skill_judge_l2"]
    tmp_skill_path = out / "_tmp_epoch_final_candidate_skill.md"
    tmp_skill_path.write_text(candidate_skill_md, encoding="utf-8")

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    skipped = 0
    progress_label = f"epoch {epoch_idx + 1}"
    progress_suffix = str(progress_suffix or "final replay")
    total = len(applied_edits or [])
    for edit_idx, edit in enumerate(applied_edits or []):
        edit_label = f"{edit_idx + 1}/{total}"
        linked = _linked_evidence(edit, evidence_by_id)
        replay_jobs = _select_l2_replay_jobs(linked, budget)
        print(
            f"[{progress_label} {progress_suffix}] edit {edit_label} start "
            f"evidence={len(linked)} replay_jobs={len(replay_jobs)} replay_budget={budget}",
            flush=True,
        )
        replay_evidence = _run_l2_replay_jobs_for_edit(
            runner=runner,
            replay_jobs=replay_jobs,
            replay_source_by_evidence_id=replay_source_by_evidence_id,
            raw_by_id={},
            tmp_skill_path=tmp_skill_path,
            args=args,
            dataset=dataset,
            progress_label=progress_label,
            window_suffix=f" {progress_suffix}",
            edit_label=edit_label,
        )
        if not replay_jobs:
            skipped += 1
        replay_rows.append(
            {
                "dataset": dataset,
                "round": epoch_idx,
                "epoch_index": epoch_idx,
                "stage": stage,
                "edit_index": edit_idx,
                "edit": edit,
                "replay_evidence": replay_evidence,
            }
        )
        l2_judge_payload = _build_l2_skill_judge_payload(
            skill_md=base_skill_md,
            edit=edit,
            linked_evidence=linked,
            replay_evidence=replay_evidence,
            dataset=dataset,
            args=args,
        )
        l2_label = _l2_skill_judge_label(
            teacher=teacher,
            skill_md=base_skill_md,
            edit=edit,
            linked_evidence=linked,
            replay_evidence=replay_evidence,
            dataset=dataset,
            args=args,
            payload=l2_judge_payload,
        )
        l2_judge_rows.append(
            {
                "dataset": dataset,
                "round": epoch_idx,
                "epoch_index": epoch_idx,
                "stage": stage,
                "system_prompt": l2_judge_prompt,
                "input": l2_judge_payload,
                "output": l2_label,
            }
        )
        print(
            f"[{progress_label} {progress_suffix}] edit {edit_label} done "
            f"decision={l2_label.get('decision')} "
            f"replay_evidence={len(replay_evidence)} "
            f"reason={str(l2_label.get('reason') or '')[:160]}",
            flush=True,
        )
        if l2_label.get("decision") == "accept":
            accepted.append(edit)
        else:
            rejected.append(
                {
                    **edit,
                    "_final_replay_decision": str(l2_label.get("decision") or "reject"),
                    "_final_replay_failure_type": str(
                        l2_label.get("replay_failure_type") or ""
                    ),
                    "_final_replay_reject_reason": str(
                        l2_label.get("reason") or ""
                    ),
                }
            )

    return accepted, rejected, {
        "enabled": True,
        "final_replay_edits": total,
        "final_replay_accepts": len(accepted),
        "final_replay_rejects": len(rejected),
        "final_replay_skipped": skipped,
        "epoch_final_replay_ranges_per_edit": budget,
        "stage": stage,
    }


def _working_branch_epoch_results_by_id(out: Path, epoch_idx: int) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for payload in _load_epoch_train_rollout_checkpoints(out, epoch_idx).values():
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            task_id = str(result.get("task_id") or result.get("id") or "")
            if task_id:
                by_id[task_id] = result
    return by_id


def _working_branch_compact_trajectory(
    result: Dict[str, Any],
    *,
    dataset: str,
    full_context: bool = False,
) -> str:
    text, _ = _render_l1_trajectories(
        [result],
        ordinary_cap=-1 if full_context else 240,
        priority_cap=-1 if full_context else 600,
        critical_cap=-1 if full_context else 1000,
        appworld_code_cap=-1 if full_context else 400,
        dataset=dataset,
        full_context=full_context,
    )
    return text if full_context else _truncate_middle(text, 14000)


def _working_branch_ledger_by_id(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("edit_id") or ""): row
        for row in state.get("provisional_edits") or []
        if str(row.get("edit_id") or "")
    }


def _working_branch_origin_edit(state: Dict[str, Any], edit_id: str) -> Dict[str, Any]:
    for collection in ("provisional_edits", "validated_edit_history"):
        for row in state.get(collection) or []:
            if str((row or {}).get("edit_id") or "") == str(edit_id or ""):
                return dict((row or {}).get("edit") or {})
    for queued in state.get("comparison_queue") or []:
        origin = ((queued or {}).get("origin_edits") or {}).get(str(edit_id or ""))
        if isinstance(origin, dict):
            return dict(origin)
    return {}


def _family_diverse_epoch_comparison_order(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows or []:
        current_result = row.get("current_result") or {}
        family = str(current_result.get("task_family") or "") or "__unknown__"
        buckets.setdefault(family, []).append(row)
    for family_rows in buckets.values():
        family_rows.sort(key=lambda row: str(row.get("task_id") or ""))

    ordered: List[Dict[str, Any]] = []
    families = sorted(buckets)
    while families:
        next_families: List[str] = []
        for family in families:
            family_rows = buckets[family]
            if family_rows:
                ordered.append(family_rows.pop(0))
            if family_rows:
                next_families.append(family)
        families = next_families
    return ordered


def _select_epoch_trajectory_comparison_pairs(
    previous_by_id: Dict[str, Dict[str, Any]],
    current_by_id: Dict[str, Dict[str, Any]],
    *,
    excluded_task_ids: Optional[Iterable[str]] = None,
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    limit = max(0, int(limit))
    excluded = {str(task_id) for task_id in (excluded_task_ids or []) if str(task_id)}
    common_ids = sorted(set(previous_by_id) & set(current_by_id))
    candidates: Dict[str, List[Dict[str, Any]]] = {
        "regressed": [],
        "persistent_fail": [],
    }
    excluded_count = 0
    for task_id in common_ids:
        previous_result = previous_by_id[task_id]
        current_result = current_by_id[task_id]
        kind = comparison_type(
            _is_success_result(previous_result),
            _is_success_result(current_result),
        )
        if kind not in candidates:
            continue
        if task_id in excluded:
            excluded_count += 1
            continue
        candidates[kind].append(
            {
                "task_id": task_id,
                "comparison_type": kind,
                "previous_result": previous_result,
                "current_result": current_result,
            }
        )

    ordered = {
        kind: _family_diverse_epoch_comparison_order(rows)
        for kind, rows in candidates.items()
    }
    targets = {
        "regressed": (limit + 1) // 2,
        "persistent_fail": limit // 2,
    }
    selected_by_kind = {
        kind: rows[: targets[kind]]
        for kind, rows in ordered.items()
    }
    remaining = limit - sum(len(rows) for rows in selected_by_kind.values())
    if remaining > 0:
        overflow: List[Dict[str, Any]] = []
        for kind in ("regressed", "persistent_fail"):
            overflow.extend(ordered[kind][len(selected_by_kind[kind]) :])
        for row in overflow[:remaining]:
            selected_by_kind[str(row["comparison_type"])].append(row)

    selected: List[Dict[str, Any]] = []
    width = max((len(rows) for rows in selected_by_kind.values()), default=0)
    for index in range(width):
        for kind in ("regressed", "persistent_fail"):
            rows = selected_by_kind[kind]
            if index < len(rows):
                selected.append(rows[index])

    return selected[:limit], {
        "common": len(common_ids),
        "excluded_queue": excluded_count,
        "regressed": len(candidates["regressed"]),
        "persistent_fail": len(candidates["persistent_fail"]),
        "selected_regressed": len(selected_by_kind["regressed"]),
        "selected_persistent_fail": len(selected_by_kind["persistent_fail"]),
    }


def _run_epoch_trajectory_comparison_evidence(
    *,
    dataset: str,
    teacher: Any,
    args: Any,
    out: Path,
    epoch_idx: int,
    working_skill_md: str,
    working_revision: str,
    epoch_train_results: List[Dict[str, Any]],
    excluded_task_ids: Optional[Iterable[str]],
    evidence_counter: int,
    evidence_rows: List[Dict[str, Any]],
    replay_source_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l1_manager_rows: List[Dict[str, Any]],
) -> Tuple[List[str], int, Dict[str, Any]]:
    budget = _int_arg(getattr(args, "epoch_trajectory_comparison_tasks", 0), 0)
    summary: Dict[str, Any] = {
        "enabled": budget > 0,
        "budget": max(0, budget),
        "common": 0,
        "excluded_queue": 0,
        "regressed": 0,
        "persistent_fail": 0,
        "selected_regressed": 0,
        "selected_persistent_fail": 0,
        "judge_calls": 0,
        "evidence_count": 0,
        "no_evidence": 0,
        "materialization_errors": 0,
    }
    if budget <= 0 or epoch_idx <= 0:
        return [], evidence_counter, summary

    previous_by_id = _working_branch_epoch_results_by_id(out, epoch_idx - 1)
    current_by_id = {
        str(row.get("task_id") or row.get("id") or ""): row
        for row in epoch_train_results or []
        if str(row.get("task_id") or row.get("id") or "")
    }
    selected, selection_summary = _select_epoch_trajectory_comparison_pairs(
        previous_by_id,
        current_by_id,
        excluded_task_ids=excluded_task_ids,
        limit=budget,
    )
    summary.update(selection_summary)
    if not selected:
        return [], evidence_counter, summary

    prompt = prompts_for_dataset(dataset)["epoch_trajectory_comparison_judge"]
    new_evidence_ids: List[str] = []
    for comparison_idx, selected_pair in enumerate(selected):
        task_id = str(selected_pair.get("task_id") or "")
        kind = str(selected_pair.get("comparison_type") or "")
        previous_result = selected_pair.get("previous_result") or {}
        current_result = selected_pair.get("current_result") or {}
        payload = {
            "dataset": dataset,
            "comparison_type": kind,
            "working_skill_revision": str(working_revision or ""),
            "working_skill_md": working_skill_md,
            "previous_epoch_index": epoch_idx - 1,
            "current_epoch_index": epoch_idx,
            "previous_trajectory": _working_branch_compact_trajectory(
                previous_result,
                dataset=dataset,
                full_context=bool(getattr(args, "llm_full_context", False)),
            ),
            "current_trajectory": _working_branch_compact_trajectory(
                current_result,
                dataset=dataset,
                full_context=bool(getattr(args, "llm_full_context", False)),
            ),
        }
        payload = _compact_llm_payload(
            payload,
            prompt,
            args,
            label="epoch trajectory comparison judge",
        )
        summary["judge_calls"] += 1
        try:
            raw = _teacher_chat_json(teacher, prompt, payload)
        except Exception as exc:
            raw = {
                "decision": "no_evidence",
                "failure_type": None,
                "reason": f"epoch comparison judge unavailable: {exc}",
            }
        if not isinstance(raw, dict):
            raw = {
                "decision": "no_evidence",
                "failure_type": None,
                "reason": "non-object output",
            }

        decision = str(raw.get("decision") or "").strip().lower()
        if decision not in {"evidence", "no_evidence"}:
            decision = "no_evidence"
            raw["decision"] = decision
        failure_type = str(raw.get("failure_type") or "").strip().lower()
        if failure_type not in L2_REPLAY_FAILURE_TYPES:
            failure_type = ""
        if decision == "evidence" and failure_type not in {"rule_missing", "rule_wrong"}:
            decision = "no_evidence"
            raw["decision"] = decision
        evidence_payload = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
        materialize_error = ""
        card: Optional[Dict[str, Any]] = None
        if decision == "evidence" and failure_type in {"rule_missing", "rule_wrong"}:
            proposed_edit = _normalize_proposed_edit(evidence_payload.get("proposed_edit"))
            if proposed_edit is None:
                materialize_error = "missing_proposed_edit"
            else:
                _, apply_report = _apply_markdown_edit(
                    working_skill_md,
                    {**proposed_edit, "evidence_ids": ["__epoch_comparison__"]},
                )
                if not bool(apply_report.get("applied")):
                    materialize_error = (
                        "non_standalone_edit:"
                        f"{str(apply_report.get('reason') or 'apply_failed')}"
                    )
                else:
                    trigger = evidence_payload.get("trigger_range")
                    draft = {
                        "experience_type": "failure_reflection",
                        "failure_type": failure_type,
                        "pattern": str(evidence_payload.get("pattern") or "").strip(),
                        "proposed_edit": proposed_edit,
                        "trigger_ranges": [trigger] if isinstance(trigger, dict) else [],
                    }
                    evidence_id = f"E{evidence_counter:06d}"
                    evidence_counter += 1
                    materialized_record = _materialization_record_from_result(
                        current_result,
                        dataset=dataset,
                    )
                    source_minibatch_id = (
                        f"epoch_{epoch_idx + 1}_trajectory_comparison_"
                        f"{comparison_idx:03d}"
                    )
                    card, materialize_error = _materialize_evidence_candidate(
                        draft=draft,
                        evidence_id=evidence_id,
                        records_by_id={task_id: materialized_record},
                        source_minibatch_id=source_minibatch_id,
                        source_minibatch_task_ids=[task_id],
                    )
                    if card is not None:
                        card.update(
                            {
                                "evidence_mode": "contrastive",
                                "source_type": "failure",
                                "observed_skill_revision": str(working_revision or ""),
                                "working_skill_revision": str(working_revision or ""),
                                "comparison_type": kind,
                                "origin_edit_ids": [],
                                "l1_skilljudge_reason": str(raw.get("reason") or ""),
                                "round": epoch_idx,
                                "epoch_index": epoch_idx,
                                "epoch_bundle_index": comparison_idx,
                                "global_bundle_index": epoch_idx,
                                "archived": False,
                            }
                        )
                        _normalize_evidence_runtime_state(card)
                        evidence_rows.append(card)
                        new_evidence_ids.append(evidence_id)
                        replay_source_by_evidence_id[
                            _replay_source_key(evidence_id, task_id)
                        ] = current_result
                        replay_source_rows.append(
                            {
                                "evidence_id": evidence_id,
                                "round": epoch_idx,
                                "epoch_index": epoch_idx,
                                "epoch_bundle_index": comparison_idx,
                                "global_bundle_index": epoch_idx,
                                "source_type": "failure",
                                "evidence_mode": "contrastive",
                                "task_id": task_id,
                                "raw_result": current_result,
                            }
                        )

        if card is None:
            summary["no_evidence"] += 1
        if materialize_error:
            summary["materialization_errors"] += 1
        source_minibatch_id = (
            f"epoch_{epoch_idx + 1}_trajectory_comparison_{comparison_idx:03d}"
        )
        l1_manager_rows.append(
            {
                "dataset": dataset,
                "epoch_index": epoch_idx,
                "epoch_bundle_index": comparison_idx,
                "global_bundle_index": epoch_idx,
                "source_type": "failure",
                "evidence_mode": "contrastive",
                "source_minibatch_id": source_minibatch_id,
                "source_minibatch_task_ids": [task_id],
                "system_prompt": prompt,
                "input": payload,
                "output": raw,
                "materialize_error": materialize_error,
            }
        )

    summary["evidence_count"] = len(new_evidence_ids)
    return new_evidence_ids, evidence_counter, summary


def _run_working_branch_contrastive_evidence(
    *,
    dataset: str,
    teacher: Any,
    args: Any,
    out: Path,
    epoch_idx: int,
    working_skill_md: str,
    working_branch_state: Dict[str, Any],
    epoch_train_results: List[Dict[str, Any]],
    evidence_counter: int,
    evidence_rows: List[Dict[str, Any]],
    replay_source_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l1_manager_rows: List[Dict[str, Any]],
) -> Tuple[List[str], int, Dict[str, Any]]:
    queue = [row for row in working_branch_state.get("comparison_queue") or [] if isinstance(row, dict)]
    if not queue:
        return [], evidence_counter, {
            "enabled": True,
            "queued": 0,
            "compared": 0,
            "evidence_count": 0,
            "support_events": 0,
        }

    current_revision = str(working_branch_state.get("working_revision") or "")
    mismatched = [
        row for row in queue
        if str(row.get("working_skill_revision") or "") != current_revision
    ]
    if mismatched:
        raise RuntimeError(
            "working branch comparison queue revision mismatch; expected "
            f"{current_revision!r}, found {str(mismatched[0].get('working_skill_revision') or '')!r}"
        )

    current_by_id = {
        str(row.get("task_id") or row.get("id") or ""): row
        for row in epoch_train_results or []
        if str(row.get("task_id") or row.get("id") or "")
    }
    parent_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}
    prompt = prompts_for_dataset(dataset)["contrastive_evidence_judge"]
    ledger_by_id = _working_branch_ledger_by_id(working_branch_state)
    aggregates: Dict[str, Dict[str, Any]] = {}
    new_evidence_ids: List[str] = []
    history_rows: List[Dict[str, Any]] = []
    compared = 0
    missing_pairs = 0
    support_events = 0
    judge_calls = 0

    for comparison_idx, queued in enumerate(queue):
        task_id = str(queued.get("task_id") or "")
        parent_epoch_idx = _coerce_int(queued.get("parent_epoch_index"), -1)
        if parent_epoch_idx not in parent_cache:
            parent_cache[parent_epoch_idx] = _working_branch_epoch_results_by_id(out, parent_epoch_idx)
        parent_result = parent_cache[parent_epoch_idx].get(task_id)
        working_result = current_by_id.get(task_id)
        if parent_result is None or working_result is None:
            missing_pairs += 1
            continue
        compared += 1
        kind = comparison_type(
            _is_success_result(parent_result),
            _is_success_result(working_result),
        )
        origin_ids = [str(value) for value in queued.get("origin_edit_ids") or [] if str(value)]
        parent_trajectory = _working_branch_compact_trajectory(
            parent_result,
            dataset=dataset,
            full_context=bool(getattr(args, "llm_full_context", False)),
        )
        working_trajectory = _working_branch_compact_trajectory(
            working_result,
            dataset=dataset,
            full_context=bool(getattr(args, "llm_full_context", False)),
        )
        queued_origin_edits = queued.get("origin_edits") or {}
        for origin_index, edit_id in enumerate(origin_ids):
            aggregate = aggregates.setdefault(
                edit_id,
                {
                    "improved": 0,
                    "regressed": 0,
                    "persistent_fail": 0,
                    "stable_success": 0,
                    "causal_support": 0,
                    "rule_ignored": 0,
                    "correction_evidence_ids": [],
                },
            )
            aggregate[kind] += 1
            origin_edit = queued_origin_edits.get(edit_id)
            if not isinstance(origin_edit, dict):
                origin_edit = _working_branch_origin_edit(working_branch_state, edit_id)
            origin_is_provisional = edit_id in ledger_by_id
            raw: Dict[str, Any] = {
                "decision": "no_evidence",
                "failure_type": None,
                "reason": "mechanical comparison does not request a correction",
            }
            should_judge = kind in {"improved", "regressed", "persistent_fail"}
            active_record = ledger_by_id.get(edit_id)
            if not origin_edit:
                should_judge = False
                raw["reason"] = "origin edit payload is unavailable"
            elif (
                kind == "persistent_fail"
                and active_record is not None
                and int(active_record.get("reflection_attempts", 0) or 0) >= 1
            ):
                should_judge = False
                raw["reason"] = "persistent failure already used its single correction attempt"

            payload = {
                "dataset": dataset,
                "comparison_type": kind,
                "task_id": task_id,
                "parent_skill_revision": str(queued.get("parent_skill_revision") or ""),
                "working_skill_revision": current_revision,
                "origin_edit_ids": [edit_id],
                "origin_edits": {edit_id: origin_edit},
                "working_skill_md": working_skill_md,
                "parent_trajectory": parent_trajectory,
                "working_trajectory": working_trajectory,
            }
            payload = _compact_llm_payload(
                payload,
                prompt,
                args,
                label="contrastive evidence judge",
            )
            if should_judge:
                judge_calls += 1
                try:
                    raw = _teacher_chat_json(teacher, prompt, payload)
                except Exception as exc:
                    raw = {
                        "decision": "no_evidence",
                        "failure_type": None,
                        "reason": f"contrastive judge unavailable: {exc}",
                    }
            if not isinstance(raw, dict):
                raw = {
                    "decision": "no_evidence",
                    "failure_type": None,
                    "reason": "non-object output",
                }
            raw_decision = str(raw.get("decision") or "").strip().lower()
            if raw_decision not in {"support", "evidence", "no_evidence"}:
                raw_decision = "no_evidence"
                raw["decision"] = raw_decision
            if raw_decision == "support" and kind != "improved":
                raw_decision = "no_evidence"
                raw["decision"] = raw_decision
            if kind == "improved" and raw_decision == "support":
                aggregate["causal_support"] += 1
            failure_type = str(raw.get("failure_type") or "").strip().lower()
            if failure_type not in L2_REPLAY_FAILURE_TYPES:
                failure_type = ""
            if failure_type == "rule_ignored":
                aggregate["rule_ignored"] += 1

            materialize_error = ""
            card: Optional[Dict[str, Any]] = None
            evidence_payload = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
            can_materialize = (
                raw_decision == "evidence"
                and failure_type in {"rule_missing", "rule_wrong"}
                and kind in {"regressed", "persistent_fail"}
            )
            if can_materialize:
                standalone_edit = _make_standalone_contrastive_edit(
                    origin_edit,
                    evidence_payload.get("proposed_edit") or {},
                    origin_is_provisional=origin_is_provisional,
                )
                if standalone_edit is None:
                    materialize_error = "non_standalone_correction"
                else:
                    trigger = evidence_payload.get("trigger_range")
                    draft = {
                        "experience_type": "failure_reflection",
                        "failure_type": failure_type,
                        "pattern": str(evidence_payload.get("pattern") or "").strip(),
                        "proposed_edit": standalone_edit,
                        "trigger_ranges": [trigger] if isinstance(trigger, dict) else [],
                    }
                    evidence_id = f"E{evidence_counter:06d}"
                    evidence_counter += 1
                    materialized_record = _materialization_record_from_result(
                        working_result,
                        dataset=dataset,
                    )
                    source_minibatch_id = (
                        f"epoch_{epoch_idx + 1}_contrastive_"
                        f"{comparison_idx:03d}_{origin_index:02d}"
                    )
                    card, materialize_error = _materialize_evidence_candidate(
                        draft=draft,
                        evidence_id=evidence_id,
                        records_by_id={task_id: materialized_record},
                        source_minibatch_id=source_minibatch_id,
                        source_minibatch_task_ids=[task_id],
                    )
                    if card is not None:
                        card.update(
                            {
                                "evidence_mode": "contrastive",
                                "observed_skill_revision": current_revision,
                                "parent_skill_revision": str(queued.get("parent_skill_revision") or ""),
                                "working_skill_revision": current_revision,
                                "comparison_type": kind,
                                "origin_edit_ids": [edit_id],
                                "l1_skilljudge_reason": str(raw.get("reason") or ""),
                                "round": epoch_idx,
                                "epoch_index": epoch_idx,
                                "epoch_bundle_index": comparison_idx,
                                "global_bundle_index": epoch_idx,
                                "archived": False,
                            }
                        )
                        _normalize_evidence_runtime_state(card)
                        evidence_rows.append(card)
                        new_evidence_ids.append(evidence_id)
                        replay_source_by_evidence_id[
                            _replay_source_key(evidence_id, task_id)
                        ] = working_result
                        replay_source_rows.append(
                            {
                                "evidence_id": evidence_id,
                                "round": epoch_idx,
                                "epoch_index": epoch_idx,
                                "epoch_bundle_index": comparison_idx,
                                "global_bundle_index": epoch_idx,
                                "source_type": "failure",
                                "evidence_mode": "contrastive",
                                "task_id": task_id,
                                "raw_result": working_result,
                            }
                        )
                        aggregate["correction_evidence_ids"].append(evidence_id)

            source_minibatch_id = (
                f"epoch_{epoch_idx + 1}_contrastive_"
                f"{comparison_idx:03d}_{origin_index:02d}"
            )
            l1_manager_rows.append(
                {
                    "dataset": dataset,
                    "epoch_index": epoch_idx,
                    "epoch_bundle_index": comparison_idx,
                    "global_bundle_index": epoch_idx,
                    "source_type": "failure",
                    "evidence_mode": "contrastive",
                    "source_minibatch_id": source_minibatch_id,
                    "source_minibatch_task_ids": [task_id],
                    "system_prompt": prompt,
                    "input": payload,
                    "output": raw,
                    "materialize_error": materialize_error,
                }
            )
            history_rows.append(
                {
                    "epoch_index": epoch_idx,
                    "task_id": task_id,
                    "comparison_type": kind,
                    "origin_edit_ids": [edit_id],
                    "decision": raw_decision,
                    "failure_type": failure_type or None,
                    "evidence_ids": [str(card.get("evidence_id"))] if card is not None else [],
                }
            )

    for edit_id, aggregate in aggregates.items():
        record = ledger_by_id.get(edit_id)
        if record is None:
            continue
        record["comparison_count"] = int(record.get("comparison_count", 0) or 0) + sum(
            int(aggregate.get(key, 0) or 0)
            for key in ("improved", "regressed", "persistent_fail", "stable_success")
        )
        if int(aggregate.get("causal_support", 0) or 0) > 0:
            record["support_events"] = int(record.get("support_events", 0) or 0) + int(
                aggregate["causal_support"]
            )
            record["non_improvement_streak"] = 0
            record["status"] = "supported"
            support_events += int(aggregate["causal_support"])
            continue
        corrections = list(dict.fromkeys(aggregate.get("correction_evidence_ids") or []))
        if int(aggregate.get("regressed", 0) or 0) > 0 and corrections:
            record["status"] = "needs_correction"
            record["pending_correction_evidence_ids"] = corrections
        elif corrections:
            record["status"] = "unresolved"
            record["pending_correction_evidence_ids"] = corrections
            record["reflection_attempts"] = int(record.get("reflection_attempts", 0) or 0) + 1
        else:
            no_improvement = int(aggregate.get("stable_success", 0) or 0) + int(
                aggregate.get("rule_ignored", 0) or 0
            )
            if no_improvement:
                record["non_improvement_streak"] = int(record.get("non_improvement_streak", 0) or 0) + 1
                if int(record["non_improvement_streak"]) >= 2:
                    record["remove_requested"] = True

    working_branch_state.setdefault("comparison_history", []).extend(history_rows)
    working_branch_state["comparison_queue"] = []
    return new_evidence_ids, evidence_counter, {
        "enabled": True,
        "queued": len(queue),
        "compared": compared,
        "judge_calls": judge_calls,
        "missing_pairs": missing_pairs,
        "evidence_count": len(new_evidence_ids),
        "support_events": support_events,
        "comparison_types": {
            kind: sum(1 for row in history_rows if row.get("comparison_type") == kind)
            for kind in ("improved", "regressed", "persistent_fail", "stable_success")
        },
    }


def _working_branch_job_family(job: Tuple[Dict[str, Any], int, Dict[str, Any]]) -> str:
    card, _, trigger = job
    task_id = str(trigger.get("task_id") or "")
    for row in card.get("supporting_tasks") or []:
        if isinstance(row, dict) and str(row.get("task_id") or "") == task_id:
            return str(row.get("task_family") or "")
    return ""


def _working_branch_replay_jobs_for_edit(
    edit: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
    linked = _linked_evidence(edit, evidence_by_id)
    jobs = _select_l2_replay_jobs(linked, -1)
    jobs.sort(
        key=lambda job: (
            -_coerce_int(job[0].get("support_count"), 1),
            _l2_replay_priority(job[0]),
            0 if int(job[1]) == 0 else 1,
            str((job[2] or {}).get("task_id") or ""),
            int(job[1]),
        )
    )
    return jobs


def _normalize_working_branch_correction_edit(
    edit: Dict[str, Any],
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    origin_ids = origin_edit_ids_from_evidence(edit, evidence_by_id)
    if not origin_ids:
        return dict(edit)
    if len(origin_ids) != 1:
        return None
    origin_id = origin_ids[0]
    origin_edit = _working_branch_origin_edit(state, origin_id)
    if not origin_edit:
        return None
    linked_cards = [
        card
        for card in _linked_evidence(edit, evidence_by_id)
        if _evidence_id_list(card.get("origin_edit_ids")) == [origin_id]
    ]
    normalized_edit = _normalize_proposed_edit(edit)
    if normalized_edit is not None and any(
        normalized_edit == _normalize_proposed_edit(card.get("proposed_edit"))
        for card in linked_cards
    ):
        return dict(edit)
    standalone = _make_standalone_contrastive_edit(
        origin_edit,
        edit,
        origin_is_provisional=origin_id in _working_branch_ledger_by_id(state),
    )
    if standalone is None:
        return None
    return {**dict(edit), **standalone}


def _admit_working_branch_replay_candidates(
    edits: List[Dict[str, Any]],
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
    state: Dict[str, Any],
    provisional_cap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    live_by_id = {
        edit_id: record
        for edit_id, record in _working_branch_ledger_by_id(state).items()
        if str(record.get("status") or "") not in {"superseded", "removed", "validated"}
        and not bool(record.get("remove_requested"))
    }
    normalized: List[Tuple[Dict[str, Any], List[str]]] = []
    excluded: List[Dict[str, Any]] = []
    for edit in edits or []:
        clean = _normalize_working_branch_correction_edit(
            edit,
            evidence_by_id=evidence_by_id,
            state=state,
        )
        if clean is None:
            excluded.append({"edit": edit, "reason": "non_standalone_correction"})
            continue
        normalized.append((clean, origin_edit_ids_from_evidence(clean, evidence_by_id)))

    normalized.sort(
        key=lambda item: (
            0 if len(item[1]) == 1 and item[1][0] in live_by_id else 1,
            0 if item[1] else 1,
            _epoch_edit_priority(item[0], evidence_by_id),
            -_coerce_int(item[0].get("support_count"), 0),
            str(item[0].get("_working_edit_id") or ""),
        )
    )
    admitted: List[Dict[str, Any]] = []
    replaced_origins = set()
    remaining_slots = max(0, int(provisional_cap) - len(live_by_id))
    for edit, origin_ids in normalized:
        replacement_origin = (
            origin_ids[0]
            if len(origin_ids) == 1 and origin_ids[0] in live_by_id
            else ""
        )
        if replacement_origin:
            if replacement_origin in replaced_origins:
                excluded.append({"edit": edit, "reason": "duplicate_origin_correction"})
                continue
            replaced_origins.add(replacement_origin)
            admitted.append(edit)
            continue
        if remaining_slots <= 0:
            excluded.append({"edit": edit, "reason": "provisional_cap_exhausted"})
            continue
        admitted.append(edit)
        remaining_slots -= 1
    return admitted, excluded


def _allocate_working_branch_post_reject_jobs(
    edits: List[Dict[str, Any]],
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
    per_edit_limit: int = 4,
    replay_cap: int = DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
) -> Tuple[List[Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], int, Dict[str, Any]]]]], List[Dict[str, Any]]]:
    allocated: List[Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], int, Dict[str, Any]]]]] = []
    excluded: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for order, edit in enumerate(edits or []):
        jobs = _working_branch_replay_jobs_for_edit(edit, evidence_by_id)
        if not jobs:
            excluded.append({"edit": edit, "reason": "no_valid_trigger_range"})
            continue
        candidates.append(
            {
                "order": order,
                "edit": edit,
                "jobs": jobs,
                "support_count": max(
                    [_coerce_int(card.get("support_count"), 1) for card in _linked_evidence(edit, evidence_by_id)]
                    or [0]
                ),
            }
        )
    if per_edit_limit == 0 or replay_cap == 0:
        reason = (
            "post_reject_replay_disabled"
            if per_edit_limit == 0
            else "post_reject_replay_cap_exhausted"
        )
        return [], [
            {"edit": candidate["edit"], "reason": reason}
            for candidate in candidates
        ] + excluded

    admitted_candidates = candidates
    if replay_cap > 0 and len(candidates) > replay_cap:
        admitted_candidates = candidates[:replay_cap]
        excluded.extend(
            {"edit": candidate["edit"], "reason": "post_reject_replay_cap_exhausted"}
            for candidate in candidates[replay_cap:]
        )

    selected_by_order: Dict[int, List[Tuple[Dict[str, Any], int, Dict[str, Any]]]] = {}
    for candidate in admitted_candidates:
        selected_by_order[int(candidate["order"])] = [candidate["jobs"][0]]
    remaining_budget: Optional[int] = None
    if replay_cap > 0:
        remaining_budget = max(0, replay_cap - len(selected_by_order))

    def desired_count(candidate: Dict[str, Any]) -> int:
        available = len(candidate["jobs"])
        return available if per_edit_limit < 0 else min(available, per_edit_limit)

    def next_diverse_job(
        candidate: Dict[str, Any],
        selected: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
    ) -> Optional[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
        selected_keys = {
            (
                str(job[0].get("evidence_id") or ""),
                int(job[1]),
                str((job[2] or {}).get("task_id") or ""),
                _coerce_int((job[2] or {}).get("start"), 0),
                _coerce_int((job[2] or {}).get("end", (job[2] or {}).get("start")), 0),
            )
            for job in selected
        }
        remaining_jobs = [
            job
            for job in candidate["jobs"]
            if (
                str(job[0].get("evidence_id") or ""),
                int(job[1]),
                str((job[2] or {}).get("task_id") or ""),
                _coerce_int((job[2] or {}).get("start"), 0),
                _coerce_int((job[2] or {}).get("end", (job[2] or {}).get("start")), 0),
            )
            not in selected_keys
        ]
        if not remaining_jobs:
            return None
        selected_tasks = {str((job[2] or {}).get("task_id") or "") for job in selected}
        selected_families = {
            family for family in (_working_branch_job_family(job) for job in selected) if family
        }
        return next(
            (
                job for job in remaining_jobs
                if str((job[2] or {}).get("task_id") or "") not in selected_tasks
                and (
                    not _working_branch_job_family(job)
                    or _working_branch_job_family(job) not in selected_families
                )
            ),
            next(
                (
                    job for job in remaining_jobs
                    if str((job[2] or {}).get("task_id") or "") not in selected_tasks
                ),
                remaining_jobs[0],
            ),
        )

    extra_order = sorted(
        [
            candidate
            for candidate in admitted_candidates
            if int(candidate["order"]) in selected_by_order
        ],
        key=lambda row: (
            -int(row["support_count"]),
            str(row["edit"].get("_working_edit_id") or ""),
            int(row["order"]),
        ),
    )
    max_rounds = max([desired_count(candidate) for candidate in extra_order] or [1])
    for round_index in range(1, max_rounds):
        for candidate in extra_order:
            if remaining_budget == 0:
                break
            selected = selected_by_order[int(candidate["order"])]
            if len(selected) >= desired_count(candidate) or len(selected) > round_index:
                continue
            job = next_diverse_job(candidate, selected)
            if job is not None:
                selected.append(job)
                if remaining_budget is not None:
                    remaining_budget -= 1
        if remaining_budget == 0:
            break

    allocated = [
        (candidate["edit"], selected_by_order[int(candidate["order"])])
        for candidate in admitted_candidates
        if int(candidate["order"]) in selected_by_order
    ]
    return allocated, excluded


def _run_working_branch_post_reject_replay(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    epoch_idx: int,
    base_skill_md: str,
    candidate_skill_md: str,
    edits: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    working_branch_state: Dict[str, Any],
    state_path: Path,
    validated_skill_md: str,
    working_skill_md: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    evidence_by_id = {
        str(row.get("evidence_id") or ""): row
        for row in evidence_rows or []
        if str(row.get("evidence_id") or "")
    }
    provisional_cap = max(
        0,
        _int_arg(
            getattr(
                args,
                "working_branch_provisional_cap",
                DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
            ),
            DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
        ),
    )
    admitted_edits, admission_excluded = _admit_working_branch_replay_candidates(
        edits,
        evidence_by_id=evidence_by_id,
        state=working_branch_state,
        provisional_cap=provisional_cap,
    )
    replay_budget_arg = getattr(args, "l2_replay_ranges_per_edit", 4)
    post_reject_per_edit_limit = (
        1
        if _no_replay_enabled(args)
        else (4 if replay_budget_arg is None else int(replay_budget_arg))
    )
    configured_replay_cap = _int_arg(
        getattr(
            args,
            "post_reject_replay_cap",
            DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
        ),
        DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
    )
    post_reject_replay_cap = -1 if _no_replay_enabled(args) else configured_replay_cap
    allocated, excluded = _allocate_working_branch_post_reject_jobs(
        admitted_edits,
        evidence_by_id=evidence_by_id,
        per_edit_limit=post_reject_per_edit_limit,
        replay_cap=post_reject_replay_cap,
    )
    excluded = list(admission_excluded) + list(excluded)
    excluded_reasons = {
        str((row.get("edit") or {}).get("_working_edit_id") or ""): str(row.get("reason") or "")
        for row in excluded
        if str((row.get("edit") or {}).get("_working_edit_id") or "")
    }
    if _no_replay_enabled(args):
        accepted = [edit for edit, _jobs in allocated]
        return accepted, list(excluded), {
            "enabled": False,
            "mode": "no_replay",
            "stage": "working_post_reject",
            "reason": "mechanical_provisional_selection_no_replay",
            "candidate_edit_count": len(edits),
            "admitted_before_replay_count": len(admitted_edits),
            "provisional_cap": provisional_cap,
            "allocated_edit_count": len(allocated),
            "accepted_edit_count": len(accepted),
            "rejected_edit_count": len(excluded),
            "bypassed_edit_count": len(accepted),
            "action_replay_count": 0,
            "replay_cache_hits": 0,
            "judge_call_count": 0,
            "judge_cache_hits": 0,
            "reflection_attempts": 0,
            "per_edit_replay_cap": 0,
            "epoch_replay_cap": 0,
            "allocated_replay_count": 0,
            "excluded_reasons": excluded_reasons,
        }
    tmp_skill_path = out / "_tmp_evidence_working_candidate_skill.md"
    atomic_write_text(tmp_skill_path, candidate_skill_md)
    cached_rows = {
        str(row.get("working_replay_cache_key") or ""): row.get("replay_evidence") or []
        for row in replay_rows
        if str(row.get("working_replay_cache_key") or "")
    }
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = list(excluded)
    deferred: List[Dict[str, Any]] = []
    action_replays = 0
    cache_hits = 0
    judge_cache_hits = 0
    reflections = 0
    for edit_index, (edit, jobs) in enumerate(allocated):
        replay_evidence: List[Dict[str, Any]] = []
        edit_replay_cache_keys: List[str] = []
        for job in jobs:
            card, range_index, trigger = job
            cache_key = replay_cache_key(
                candidate_skill_md,
                edit,
                evidence_id=str(card.get("evidence_id") or ""),
                range_index=range_index,
                trigger=trigger,
            )
            edit_replay_cache_keys.append(cache_key)
            cached = cached_rows.get(cache_key)
            if cached:
                replay_evidence.extend(_jsonable_copy(cached))
                cache_hits += 1
                continue
            rows = _run_l2_replay_jobs_for_edit(
                runner=runner,
                replay_jobs=[job],
                replay_source_by_evidence_id=replay_source_by_evidence_id,
                raw_by_id={},
                tmp_skill_path=tmp_skill_path,
                args=args,
                dataset=dataset,
                progress_label=f"epoch {epoch_idx + 1}",
                window_suffix=" working post-reject",
                edit_label=str(edit.get("_working_edit_id") or edit_index + 1),
            )
            action_replays += 1
            replay_evidence.extend(rows)
            cache_row = {
                "dataset": dataset,
                "round": epoch_idx,
                "epoch_index": epoch_idx,
                "stage": "working_post_reject",
                "edit_index": edit_index,
                "edit": edit,
                "working_replay_cache_key": cache_key,
                "replay_evidence": rows,
            }
            replay_rows.append(cache_row)
            epoch_replay_path = (
                _epoch_artifact_dir(out, epoch_idx)
                / EPOCH_JSONL_FILENAMES["replay_label_traces"]
            )
            epoch_replay_path.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(
                str(epoch_replay_path),
                _filter_rows_for_epoch(
                    replay_rows,
                    epoch_idx,
                    filename=EPOCH_JSONL_FILENAMES["replay_label_traces"],
                ),
            )
            cached_rows[cache_key] = rows
            working_branch_state.setdefault("replay_cache_keys", {})[cache_key] = {
                "epoch_index": epoch_idx,
                "edit_id": str(edit.get("_working_edit_id") or ""),
            }
            persist_working_branch_state(
                state_path,
                working_branch_state,
                validated_skill_md=validated_skill_md,
                working_skill_md=working_skill_md,
            )

        judge_cache_key = skill_fingerprint(
            candidate_skill_md
            + json.dumps(
                {
                    "edit": _working_edit_payload(edit),
                    "replay_cache_keys": edit_replay_cache_keys,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
        judge_cache = (working_branch_state.get("post_reject_judge_cache") or {}).get(
            judge_cache_key
        )
        if isinstance(judge_cache, dict) and isinstance(judge_cache.get("label"), dict):
            label = _jsonable_copy(judge_cache["label"])
            accepted_edit = _jsonable_copy(judge_cache.get("accepted_edit") or edit)
            judge_cache_hits += 1
            cached_judge_row = judge_cache.get("judge_row")
            if isinstance(cached_judge_row, dict) and not any(
                str(row.get("working_judge_cache_key") or "") == judge_cache_key
                for row in l2_judge_rows
            ):
                l2_judge_rows.append(_jsonable_copy(cached_judge_row))
            cached_reflection_row = judge_cache.get("reflection_row")
            if isinstance(cached_reflection_row, dict) and not any(
                str(row.get("working_judge_cache_key") or "") == judge_cache_key
                and str(row.get("stage") or "") == "working_post_reject_reflection"
                for row in l2_judge_rows
            ):
                l2_judge_rows.append(_jsonable_copy(cached_reflection_row))
            if label.get("decision") == "accept":
                normalized_accepted = _normalize_working_branch_correction_edit(
                    accepted_edit,
                    evidence_by_id=evidence_by_id,
                    state=working_branch_state,
                )
                if normalized_accepted is not None:
                    accepted.append(normalized_accepted)
                else:
                    rejected.append(
                        {"edit": edit, "reason": "non_standalone_correction"}
                    )
            elif label.get("decision") == "defer":
                deferred.append({"edit": edit, "reason": "rule_ignored"})
                edit_id = str(edit.get("_working_edit_id") or "")
                if edit_id:
                    excluded_reasons[edit_id] = "rule_ignored"
            else:
                rejected.append(
                    {"edit": edit, "reason": str(label.get("reason") or "judge_reject")}
                )
            continue

        linked = _linked_evidence(edit, evidence_by_id)
        payload = _build_l2_skill_judge_payload(
            skill_md=base_skill_md,
            edit=edit,
            linked_evidence=linked,
            replay_evidence=replay_evidence,
            dataset=dataset,
            args=args,
            reflection_allowed=_uses_l2_replay_attribution(dataset),
        )
        label = _l2_skill_judge_label(
            teacher=teacher,
            skill_md=base_skill_md,
            edit=edit,
            linked_evidence=linked,
            replay_evidence=replay_evidence,
            dataset=dataset,
            args=args,
            payload=payload,
            reflection_allowed=_uses_l2_replay_attribution(dataset),
        )
        judge_row = {
            "dataset": dataset,
            "round": epoch_idx,
            "epoch_index": epoch_idx,
            "stage": "working_post_reject",
            "working_judge_cache_key": judge_cache_key,
            "system_prompt": prompts_for_dataset(dataset)["skill_judge_l2"],
            "input": payload,
            "output": label,
        }
        l2_judge_rows.append(judge_row)
        accepted_edit = edit
        reflection_row = None
        if label.get("decision") == "reflect" and _uses_l2_replay_attribution(dataset):
            reflection = _l2_replay_reflection(
                dataset=dataset,
                teacher=teacher,
                skill_md=base_skill_md,
                edit=edit,
                linked_evidence=linked,
                replay_evidence=replay_evidence,
                judge_output=label,
                args=args,
            )
            reflections += 1
            reflection_row = {
                "dataset": dataset,
                "round": epoch_idx,
                "epoch_index": epoch_idx,
                "stage": "working_post_reject_reflection",
                "working_judge_cache_key": judge_cache_key,
                "system_prompt": reflection.get("system_prompt"),
                "input": reflection.get("input"),
                "output": reflection.get("output"),
            }
            l2_judge_rows.append(reflection_row)
            if reflection.get("accepted"):
                accepted_edit = {
                    **edit,
                    **dict(reflection.get("revised_edit") or {}),
                    "_working_edit_id": edit.get("_working_edit_id"),
                }
                label = {**label, "decision": "accept", "reflection_applied": True}
            else:
                label = {**label, "decision": "reject", "reflection_applied": False}
        if label.get("decision") == "accept":
            normalized_accepted = _normalize_working_branch_correction_edit(
                accepted_edit,
                evidence_by_id=evidence_by_id,
                state=working_branch_state,
            )
            if normalized_accepted is not None:
                accepted_edit = normalized_accepted
                accepted.append(accepted_edit)
            else:
                label = {
                    **label,
                    "decision": "reject",
                    "reason": "non_standalone_correction",
                }
                rejected.append({"edit": edit, "reason": "non_standalone_correction"})
        elif label.get("decision") == "defer":
            deferred.append({"edit": edit, "reason": "rule_ignored"})
            edit_id = str(edit.get("_working_edit_id") or "")
            if edit_id:
                excluded_reasons[edit_id] = "rule_ignored"
        else:
            rejected.append({"edit": edit, "reason": str(label.get("reason") or "judge_reject")})
        working_branch_state.setdefault("post_reject_judge_cache", {})[
            judge_cache_key
        ] = {
            "epoch_index": epoch_idx,
            "edit_id": str(edit.get("_working_edit_id") or ""),
            "label": _jsonable_copy(label),
            "accepted_edit": _jsonable_copy(accepted_edit),
            "judge_row": _jsonable_copy(judge_row),
            "reflection_row": _jsonable_copy(reflection_row) if reflection_row else None,
        }
        epoch_judge_path = (
            _epoch_artifact_dir(out, epoch_idx)
            / EPOCH_JSONL_FILENAMES["l2_skill_judge"]
        )
        epoch_judge_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(
            str(epoch_judge_path),
            _filter_rows_for_epoch(
                l2_judge_rows,
                epoch_idx,
                filename=EPOCH_JSONL_FILENAMES["l2_skill_judge"],
            ),
        )
        persist_working_branch_state(
            state_path,
            working_branch_state,
            validated_skill_md=validated_skill_md,
            working_skill_md=working_skill_md,
        )

    available_replays_by_edit = {
        id(edit): len(_working_branch_replay_jobs_for_edit(edit, evidence_by_id))
        for edit in admitted_edits
    }
    info = {
        "enabled": True,
        "mode": "standard",
        "stage": "working_post_reject",
        "candidate_edit_count": len(edits),
        "admitted_before_replay_count": len(admitted_edits),
        "provisional_cap": provisional_cap,
        "allocated_edit_count": len(allocated),
        "accepted_edit_count": len(accepted),
        "rejected_edit_count": len(rejected),
        "deferred_edit_count": len(deferred),
        "rule_ignored_defers": len(deferred),
        "action_replay_count": action_replays,
        "replay_cache_hits": cache_hits,
        "judge_cache_hits": judge_cache_hits,
        "reflection_attempts": reflections,
        "per_edit_replay_cap": post_reject_per_edit_limit,
        "epoch_replay_cap": post_reject_replay_cap,
        "available_replay_count": sum(available_replays_by_edit.values()),
        "allocated_replay_count": sum(len(jobs) for _, jobs in allocated),
        "range_limited_edit_count": sum(
            1
            for edit in admitted_edits
            if available_replays_by_edit[id(edit)]
            and post_reject_per_edit_limit > 0
            and available_replays_by_edit[id(edit)] > post_reject_per_edit_limit
        ),
        "excluded_reasons": excluded_reasons,
    }
    return accepted, rejected + deferred, info


def _working_branch_queue_for_transition(
    *,
    state: Dict[str, Any],
    changed_edits: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    epoch_idx: int,
    parent_revision: str,
    working_revision: str,
    seed: int,
) -> List[Dict[str, Any]]:
    evidence_by_id = {
        str(row.get("evidence_id") or ""): row
        for row in evidence_rows or []
        if str(row.get("evidence_id") or "")
    }
    return select_comparison_tasks(
        changed_edits,
        evidence_by_id=evidence_by_id,
        parent_epoch_index=epoch_idx,
        parent_skill_revision=parent_revision,
        working_skill_revision=working_revision,
        seed=seed + (epoch_idx + 1) * 7919,
        limit=WORKING_BRANCH_COMPARISON_TASK_LIMIT,
    )


def _working_branch_commit_transition(
    *,
    state: Dict[str, Any],
    state_path: Path,
    validated_skill_path: Path,
    working_skill_path: Path,
    candidate_skill_md: str,
    changed_edits: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    epoch_idx: int,
    seed: int,
) -> Dict[str, Any]:
    changed_edits = coalesce_working_edits(changed_edits)
    previous_working_md = working_skill_path.read_text(encoding="utf-8")
    parent_revision = str(state.get("working_revision") or "")
    prior_ledger = _jsonable_copy(state.get("provisional_edits") or [])
    if candidate_skill_md != previous_working_md:
        working_revision = next_working_revision(state)
    else:
        working_revision = parent_revision
    validated_revision = next_validated_revision(state)
    atomic_write_text(validated_skill_path, candidate_skill_md)
    atomic_write_text(working_skill_path, candidate_skill_md)
    state.setdefault("validated_edit_history", []).extend(
        [
            {**record, "status": "validated", "validated_epoch": epoch_idx}
            for record in prior_ledger
        ]
        + [
            {
                **make_provisional_record(edit, epoch_idx=epoch_idx, order=order, status="validated"),
                "validated_epoch": epoch_idx,
            }
            for order, edit in enumerate(changed_edits)
        ]
    )
    state["provisional_edits"] = []
    state["comparison_queue"] = _working_branch_queue_for_transition(
        state=state,
        changed_edits=changed_edits,
        evidence_rows=evidence_rows,
        epoch_idx=epoch_idx,
        parent_revision=parent_revision,
        working_revision=working_revision,
        seed=seed,
    ) if candidate_skill_md != previous_working_md else []
    state.pop("pending_validation", None)
    state["stage"] = {"epoch_index": epoch_idx, "name": "complete", "decision": "commit"}
    state["last_transition"] = {
        "epoch_index": epoch_idx,
        "decision": "commit",
        "validated_revision": validated_revision,
        "working_revision": working_revision,
        "comparison_queue_size": len(state["comparison_queue"]),
    }
    persist_working_branch_state(
        state_path,
        state,
        validated_skill_md=candidate_skill_md,
        working_skill_md=candidate_skill_md,
    )
    return dict(state["last_transition"])


def _working_branch_reject_transition(
    *,
    state: Dict[str, Any],
    state_path: Path,
    validated_skill_path: Path,
    working_skill_path: Path,
    accepted_edits: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    epoch_idx: int,
    seed: int,
) -> Dict[str, Any]:
    validated_skill_md = validated_skill_path.read_text(encoding="utf-8")
    previous_working_md = working_skill_path.read_text(encoding="utf-8")
    parent_revision = str(state.get("working_revision") or "")
    evidence_by_id = {
        str(row.get("evidence_id") or ""): row
        for row in evidence_rows or []
        if str(row.get("evidence_id") or "")
    }
    policy = state.get("working_branch_policy") or {}
    provisional_cap = max(
        0,
        _int_arg(
            policy.get("provisional_cap", DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP),
            DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
        ),
    )
    prior_records = _jsonable_copy(state.get("provisional_edits") or [])
    non_live_statuses = {"superseded", "removed", "validated"}
    retained_records = [
        dict(record)
        for record in prior_records
        if str((record or {}).get("status") or "provisional") not in non_live_statuses
        if not bool((record or {}).get("remove_requested"))
    ]
    for record in retained_records:
        record.pop("remove_if_uncorrected", None)
    removed_records: List[Dict[str, Any]] = []
    removed_records.extend(
        dict(record)
        for record in prior_records
        if bool((record or {}).get("remove_requested"))
    )
    retained_by_id = {
        str(record.get("edit_id") or ""): record
        for record in retained_records
        if str(record.get("edit_id") or "")
    }
    replacement_records: List[Dict[str, Any]] = []
    ordinary_edits: List[Dict[str, Any]] = []
    replaced_origins = set()
    for edit in accepted_edits or []:
        origin_ids = origin_edit_ids_from_evidence(edit, evidence_by_id)
        origin_id = origin_ids[0] if len(origin_ids) == 1 else ""
        origin_record = retained_by_id.get(origin_id)
        if origin_record is None or origin_id in replaced_origins:
            ordinary_edits.append(dict(edit))
            continue
        replaced_origins.add(origin_id)
        retained_records = [
            record
            for record in retained_records
            if str(record.get("edit_id") or "") != origin_id
        ]
        retained_by_id.pop(origin_id, None)
        removed_records.append({**dict(origin_record), "status": "superseded"})
        replacement = make_provisional_record(
            edit,
            epoch_idx=_coerce_int(origin_record.get("introduced_epoch"), epoch_idx),
            order=_coerce_int(origin_record.get("introduced_order"), 0),
            status="supported",
        )
        replacement["introduced_epoch"] = _coerce_int(
            origin_record.get("introduced_epoch"), epoch_idx
        )
        replacement["introduced_order"] = _coerce_int(
            origin_record.get("introduced_order"), 0
        )
        replacement_records.append(replacement)

    retained_records.extend(replacement_records)
    live_count = len(retained_records)
    remaining_slots = max(0, provisional_cap - live_count)
    ordinary_edits = sorted(
        ordinary_edits,
        key=lambda edit: (
            0 if origin_edit_ids_from_evidence(edit, evidence_by_id) else 1,
            _epoch_edit_priority(edit, evidence_by_id),
            -_coerce_int(edit.get("support_count"), 0),
            str(edit.get("_working_edit_id") or ""),
        ),
    )
    admitted_ordinary = ordinary_edits[:remaining_slots]
    cap_excluded_edits = ordinary_edits[remaining_slots:]
    start_order = max(
        [int(row.get("introduced_order", -1) or -1) for row in retained_records] or [-1]
    ) + 1
    new_records = [
        make_provisional_record(
            edit,
            epoch_idx=epoch_idx,
            order=start_order + order,
            status=(
                "supported"
                if origin_edit_ids_from_evidence(edit, evidence_by_id)
                else "provisional"
            ),
        )
        for order, edit in enumerate(admitted_ordinary)
    ]
    rebuilt, surviving, failed = rebuild_working_skill(
        validated_skill_md,
        retained_records + new_records,
        apply_edit=_apply_markdown_edit,
    )
    if replacement_records and failed:
        rollback_evidence_ids = [
            str(evidence_id)
            for edit in accepted_edits or []
            for evidence_id in _evidence_id_list(edit.get("evidence_ids"))
        ]
        reactivated = reactivate_evidence(evidence_rows, rollback_evidence_ids)
        state["provisional_edits"] = prior_records
        state.pop("pending_validation", None)
        state["stage"] = {
            "epoch_index": epoch_idx,
            "name": "complete",
            "decision": "replacement_rollback",
        }
        state["last_transition"] = {
            "epoch_index": epoch_idx,
            "decision": "reject",
            "validated_revision": str(state.get("validated_revision") or ""),
            "working_revision": parent_revision,
            "provisional_count": len(prior_records),
            "removed_edit_ids": [],
            "reactivated_evidence_ids": reactivated,
            "rebuild_failed_count": len(failed),
            "replacement_rollback": True,
            "comparison_queue_size": len(state.get("comparison_queue") or []),
        }
        persist_working_branch_state(
            state_path,
            state,
            validated_skill_md=validated_skill_md,
            working_skill_md=previous_working_md,
        )
        return dict(state["last_transition"])

    failed_records = [dict(item.get("record") or {}) for item in failed]
    removed_records.extend(failed_records)
    removed_evidence_ids = [
        str(evidence_id)
        for record in removed_records
        for evidence_id in (record.get("edit") or {}).get("evidence_ids") or []
        if str(evidence_id)
    ]
    removed_evidence_ids.extend(
        str(evidence_id)
        for edit in cap_excluded_edits
        for evidence_id in _evidence_id_list(edit.get("evidence_ids"))
    )
    reactivated = reactivate_evidence(evidence_rows, removed_evidence_ids)
    state["provisional_edits"] = surviving
    new_edit_ids = {
        str(record.get("edit_id") or "")
        for record in replacement_records + new_records
        if str(record.get("edit_id") or "")
    }
    changed_edits = [
        {**dict(record.get("edit") or {}), "_working_edit_id": record.get("edit_id")}
        for record in surviving
        if str(record.get("edit_id") or "") in new_edit_ids
    ]
    changed_edits = coalesce_working_edits(changed_edits)
    queued_edit_ids = {
        str(edit.get("_working_edit_id") or "") for edit in changed_edits
    }
    for record in surviving:
        edit_id = str(record.get("edit_id") or "")
        if edit_id in queued_edit_ids:
            continue
        if (
            int(record.get("comparison_count", 0) or 0) > 0
            and int(record.get("non_improvement_streak", 0) or 0) <= 0
            and str(record.get("status") or "")
            not in {"unresolved", "needs_correction"}
        ):
            continue
        changed_edits.append(
            {**dict(record.get("edit") or {}), "_working_edit_id": edit_id}
        )
        queued_edit_ids.add(edit_id)
    if rebuilt != previous_working_md:
        working_revision = next_working_revision(state)
        comparison_queue = _working_branch_queue_for_transition(
            state=state,
            changed_edits=changed_edits,
            evidence_rows=evidence_rows,
            epoch_idx=epoch_idx,
            parent_revision=parent_revision,
            working_revision=working_revision,
            seed=seed,
        )
    else:
        working_revision = parent_revision
        comparison_queue = []
    atomic_write_text(working_skill_path, rebuilt)
    state["comparison_queue"] = comparison_queue
    state.pop("pending_validation", None)
    state["stage"] = {"epoch_index": epoch_idx, "name": "complete", "decision": "reject"}
    state["last_transition"] = {
        "epoch_index": epoch_idx,
        "decision": "reject",
        "validated_revision": str(state.get("validated_revision") or ""),
        "working_revision": working_revision,
        "provisional_count": len(surviving),
        "provisional_cap": provisional_cap,
        "provisional_cap_excluded_count": len(cap_excluded_edits),
        "removed_edit_ids": [str(row.get("edit_id") or "") for row in removed_records],
        "reactivated_evidence_ids": reactivated,
        "rebuild_failed_count": len(failed),
        "comparison_queue_size": len(comparison_queue),
    }
    persist_working_branch_state(
        state_path,
        state,
        validated_skill_md=validated_skill_md,
        working_skill_md=rebuilt,
    )
    return dict(state["last_transition"])


def _is_final_configured_epoch(args: Any, epoch_idx: int) -> bool:
    configured_epochs = getattr(args, "epochs", None)
    if configured_epochs in (None, ""):
        return False
    try:
        epoch_count = max(1, int(configured_epochs))
    except (TypeError, ValueError):
        return False
    return int(epoch_idx) >= epoch_count - 1


def _working_branch_terminal_reject_transition(
    *,
    state: Dict[str, Any],
    state_path: Path,
    validated_skill_path: Path,
    working_skill_path: Path,
    epoch_idx: int,
) -> Dict[str, Any]:
    validated_skill_md = validated_skill_path.read_text(encoding="utf-8")
    working_skill_md = working_skill_path.read_text(encoding="utf-8")
    state["comparison_queue"] = []
    state.pop("pending_validation", None)
    state["stage"] = {
        "epoch_index": epoch_idx,
        "name": "complete",
        "decision": "reject",
    }
    state["last_transition"] = {
        "epoch_index": epoch_idx,
        "decision": "reject",
        "reason": "final_epoch_validation_reject",
        "terminal": True,
        "validated_revision": str(state.get("validated_revision") or ""),
        "working_revision": str(state.get("working_revision") or ""),
        "provisional_count": len(state.get("provisional_edits") or []),
        "comparison_queue_size": 0,
    }
    persist_working_branch_state(
        state_path,
        state,
        validated_skill_md=validated_skill_md,
        working_skill_md=working_skill_md,
    )
    return dict(state["last_transition"])


def _ensure_working_branch_pending_manager_rows(
    l2_manager_rows: List[Dict[str, Any]],
    pending_rows: List[Dict[str, Any]],
    *,
    epoch_idx: int,
) -> None:
    existing_epoch_rows = _filter_rows_for_epoch(
        l2_manager_rows,
        epoch_idx,
        filename=EPOCH_JSONL_FILENAMES["l2_skill_manager"],
    )
    if len(existing_epoch_rows) >= len(pending_rows or []):
        return
    existing_signatures = {
        json.dumps(
            {
                "round": row.get("round"),
                "epoch_index": row.get("epoch_index"),
                "source_minibatch_id": row.get("source_minibatch_id"),
                "input": row.get("input"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for row in existing_epoch_rows
    }
    for row in pending_rows or []:
        signature = json.dumps(
            {
                "round": row.get("round"),
                "epoch_index": row.get("epoch_index"),
                "source_minibatch_id": row.get("source_minibatch_id"),
                "input": row.get("input"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if signature in existing_signatures:
            continue
        l2_manager_rows.append(_jsonable_copy(row))
        existing_signatures.add(signature)


def _filter_working_branch_epoch_manager_edits(
    l2_manager_rows: List[Dict[str, Any]],
    *,
    epoch_idx: int,
    selected_source_ids: set[str],
) -> None:
    for row in l2_manager_rows:
        if not _row_belongs_to_epoch(
            row,
            epoch_idx,
            filename=EPOCH_JSONL_FILENAMES["l2_skill_manager"],
        ):
            continue
        edits = (row.get("output") or {}).get("edits") or []
        if not selected_source_ids:
            row.setdefault("output", {})["edits"] = []
            continue
        row.setdefault("output", {})["edits"] = [
            _clean_markdown_edit_for_row(edit)
            for edit in edits
            if str(edit.get("_candidate_edit_id") or "") in selected_source_ids
        ]


def _resume_working_branch_validation_stage(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    epoch_idx: int,
    epoch_rollout_task_count: int,
    current_skill_path: Path,
    validation_skill_path: Path,
    working_branch_state: Dict[str, Any],
    working_branch_state_path: Path,
    evidence_pool: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l2_manager_rows: List[Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    reject_cooldown_pool_size: int,
) -> Optional[Tuple[Dict[str, Any], int]]:
    stage = working_branch_state.get("stage") or {}
    stage_name = str(stage.get("name") or "")
    if stage_name not in {"post_reject_replay", "validation_commit"}:
        return None
    if _row_int(stage, "epoch_index", -1) != epoch_idx:
        raise RuntimeError(
            "working branch pending validation epoch mismatch: "
            f"state={stage.get('epoch_index')!r} requested={epoch_idx}"
        )
    pending = working_branch_state.get("pending_validation")
    if not isinstance(pending, dict):
        raise RuntimeError(
            f"working branch stage {stage_name!r} is missing pending_validation"
        )
    if _coerce_int(pending.get("epoch_index"), -1) != epoch_idx:
        raise RuntimeError("working branch pending_validation belongs to another epoch")
    candidate_skill_md = str(pending.get("candidate_skill_md") or "")
    if not candidate_skill_md:
        raise RuntimeError("working branch pending_validation is missing candidate_skill_md")
    expected_fingerprint = str(pending.get("candidate_skill_fingerprint") or "")
    if expected_fingerprint != skill_fingerprint(candidate_skill_md):
        raise RuntimeError("working branch pending candidate skill fingerprint mismatch")
    selection_info = pending.get("selection")
    if not isinstance(selection_info, dict):
        raise RuntimeError("working branch pending_validation is missing selection output")
    selection_info = _jsonable_copy(selection_info)
    decision = str(selection_info.get("decision") or "").lower()
    if stage_name == "validation_commit" and decision != "commit":
        raise RuntimeError("validation_commit stage does not contain a commit decision")
    if stage_name == "post_reject_replay" and decision == "commit":
        raise RuntimeError("post_reject_replay stage contains a commit decision")

    candidate_edits = [
        dict(edit)
        for edit in pending.get("candidate_edits") or []
        if isinstance(edit, dict)
    ]
    pending_manager_rows = [
        dict(row)
        for row in pending.get("pending_l2_manager_rows") or []
        if isinstance(row, dict)
    ]
    _ensure_working_branch_pending_manager_rows(
        l2_manager_rows,
        pending_manager_rows,
        epoch_idx=epoch_idx,
    )
    context = pending.get("summary_context") or {}
    base_skill_md = str(
        pending.get("base_skill_md")
        or context.get("base_skill_md")
        or current_skill_path.read_text(encoding="utf-8")
    )
    epoch_pool_ids = {
        str(value)
        for value in context.get("epoch_pool_ids") or []
        if str(value)
    }
    if not epoch_pool_ids:
        epoch_pool_ids = {
            str(row.get("evidence_id") or "")
            for row in evidence_rows
            if str(row.get("evidence_id") or "")
            and not bool(row.get("archived"))
            and str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower()
            == EVIDENCE_STATUS_ACTIVE
        }
    archived_ids = {
        str(value) for value in context.get("archived_evidence_ids") or [] if str(value)
    }
    accepted_evidence_ids = {
        str(value) for value in context.get("accepted_evidence_ids") or [] if str(value)
    }
    rejected_evidence_ids = {
        str(value) for value in context.get("rejected_evidence_ids") or [] if str(value)
    }
    staled_evidence_ids = {
        str(value) for value in context.get("staled_evidence_ids") or [] if str(value)
    }
    deferred_fates = {
        str(key): str(value)
        for key, value in (context.get("deferred_evidence_fates") or {}).items()
        if str(key)
    }
    coverage_mode = bool(
        context.get(
            "coverage_mode",
            _resolve_epoch_skill_update_mode(args) == "cluster_step",
        )
    )
    final_covered_ids = {
        str(value)
        for value in context.get("final_candidate_covered_evidence_ids") or []
        if str(value)
    }
    baseline_covered_ids = {
        str(value)
        for value in context.get("baseline_covered_evidence_ids") or []
        if str(value)
    }
    epoch_final_replay_info = _jsonable_copy(
        context.get("epoch_final_replay")
        or {
            "enabled": False,
            "final_replay_edits": len(candidate_edits),
            "final_replay_accepts": 0,
            "final_replay_rejects": 0,
            "final_replay_skipped": len(candidate_edits),
            "final_replay_bypassed": len(candidate_edits),
            "reason": "replaced_by_working_branch_post_reject_replay",
        }
    )
    post_reject_info: Dict[str, Any] = {"enabled": False}
    provisional_edits: List[Dict[str, Any]] = []
    deferred_ids: set[str] = set()
    consumed_ids: set[str] = set()

    print(
        f"[epoch {epoch_idx + 1} working branch resume] stage={stage_name} "
        "reuse_l2=true reuse_merge=true reuse_validation=true",
        flush=True,
    )
    if decision == "commit":
        commit_covered_ids = (
            final_covered_ids if coverage_mode else set(_covered_evidence_ids(candidate_edits))
        )
        if commit_covered_ids:
            updates = _mark_evidence_attempts(
                evidence_rows,
                sorted(commit_covered_ids),
                round_idx=epoch_idx,
                outcome="accept",
                max_rejects=max(
                    1,
                    int(
                        getattr(args, "l2_max_rejects_per_evidence", L2_MAX_REJECTS_PER_EVIDENCE)
                        or L2_MAX_REJECTS_PER_EVIDENCE
                    ),
                ),
                count_attempt=False,
            )
            accepted_evidence_ids.update(updates["accepted"])
            archived_ids.update(
                _archive_evidence_rows_with_reason(
                    evidence_rows,
                    sorted(commit_covered_ids),
                    reason="covered_by_committed_skill",
                )
            )
        for evidence_id in sorted(epoch_pool_ids - archived_ids):
            reason = deferred_fates.get(evidence_id) or "accepted_not_covered"
            deferred_ids.update(
                _defer_evidence_rows(evidence_rows, [evidence_id], reason=reason)
            )
            deferred_fates[evidence_id] = reason
        selected_source_ids = {
            str(source_id)
            for edit in candidate_edits
            for source_id in edit.get("source_edit_ids") or []
            if str(source_id)
        }
        _filter_working_branch_epoch_manager_edits(
            l2_manager_rows,
            epoch_idx=epoch_idx,
            selected_source_ids=selected_source_ids,
        )
        transition_info = _working_branch_commit_transition(
            state=working_branch_state,
            state_path=working_branch_state_path,
            validated_skill_path=validation_skill_path,
            working_skill_path=current_skill_path,
            candidate_skill_md=candidate_skill_md,
            changed_edits=candidate_edits,
            evidence_rows=evidence_rows,
            epoch_idx=epoch_idx,
            seed=int(getattr(args, "seed", 42) or 42),
        )
        committed_edits = candidate_edits
        reject_cooldown_pool_size = 0
    else:
        if _is_final_configured_epoch(args, epoch_idx):
            post_reject_info.update(
                {
                    "reason": "skipped_final_epoch",
                    "candidate_edit_count": len(candidate_edits),
                    "skipped_edit_count": len(candidate_edits),
                    "action_replay_count": 0,
                }
            )
            print(
                f"[epoch {epoch_idx + 1} skill edit working post-reject] "
                f"skipped reason=final_epoch candidate_edits={len(candidate_edits)} "
                "resume=true",
                flush=True,
            )
            for edit in candidate_edits:
                for evidence_id in _evidence_id_list(edit.get("evidence_ids")):
                    deferred_ids.update(
                        _defer_evidence_rows(
                            evidence_rows,
                            [evidence_id],
                            reason="final_epoch_validation_reject",
                        )
                    )
                    deferred_fates[evidence_id] = "final_epoch_validation_reject"
            for evidence_id in sorted(epoch_pool_ids - archived_ids):
                reason = deferred_fates.get(evidence_id) or "final_epoch_validation_reject"
                deferred_ids.update(
                    _defer_evidence_rows(evidence_rows, [evidence_id], reason=reason)
                )
                deferred_fates[evidence_id] = reason
            _filter_working_branch_epoch_manager_edits(
                l2_manager_rows,
                epoch_idx=epoch_idx,
                selected_source_ids=set(),
            )
            transition_info = _working_branch_terminal_reject_transition(
                state=working_branch_state,
                state_path=working_branch_state_path,
                validated_skill_path=validation_skill_path,
                working_skill_path=current_skill_path,
                epoch_idx=epoch_idx,
            )
        else:
            provisional_edits, _, post_reject_info = _run_working_branch_post_reject_replay(
                dataset=dataset,
                teacher=teacher,
                runner=runner,
                out=out,
                args=args,
                epoch_idx=epoch_idx,
                base_skill_md=base_skill_md,
                candidate_skill_md=candidate_skill_md,
                edits=candidate_edits,
                evidence_rows=evidence_rows,
                replay_source_by_evidence_id=replay_source_by_evidence_id,
                l2_judge_rows=l2_judge_rows,
                replay_rows=replay_rows,
                working_branch_state=working_branch_state,
                state_path=working_branch_state_path,
                validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
                working_skill_md=current_skill_path.read_text(encoding="utf-8"),
            )
            surviving_ids = {
                str(edit.get("_working_edit_id") or "") for edit in provisional_edits
            }
            excluded_reasons = post_reject_info.get("excluded_reasons") or {}
            for edit in candidate_edits:
                edit_id = str(edit.get("_working_edit_id") or "")
                reason = (
                    "working_branch_provisional"
                    if edit_id in surviving_ids
                    else str(excluded_reasons.get(edit_id) or "working_branch_replay_reject")
                )
                for evidence_id in _evidence_id_list(edit.get("evidence_ids")):
                    deferred_ids.update(
                        _defer_evidence_rows(evidence_rows, [evidence_id], reason=reason)
                    )
                    deferred_fates[evidence_id] = reason
            for evidence_id in sorted(epoch_pool_ids - archived_ids):
                reason = deferred_fates.get(evidence_id) or "validation_reject"
                deferred_ids.update(
                    _defer_evidence_rows(evidence_rows, [evidence_id], reason=reason)
                )
                deferred_fates[evidence_id] = reason
            selected_source_ids = {
                str(source_id)
                for edit in provisional_edits
                for source_id in edit.get("source_edit_ids") or []
                if str(source_id)
            }
            _filter_working_branch_epoch_manager_edits(
                l2_manager_rows,
                epoch_idx=epoch_idx,
                selected_source_ids=selected_source_ids,
            )
            transition_info = _working_branch_reject_transition(
                state=working_branch_state,
                state_path=working_branch_state_path,
                validated_skill_path=validation_skill_path,
                working_skill_path=current_skill_path,
                accepted_edits=provisional_edits,
                evidence_rows=evidence_rows,
                epoch_idx=epoch_idx,
                seed=int(getattr(args, "seed", 42) or 42),
            )
        committed_edits = []

    policy_defer_limit_info = _apply_working_branch_policy_defer_limit(
        evidence_rows,
        deferred_fates,
        dataset=dataset,
        args=args,
        epoch_idx=epoch_idx,
        validation_decision=decision,
    )
    policy_staled_ids = set(
        policy_defer_limit_info.get("staled_evidence_ids") or []
    )
    staled_evidence_ids.update(policy_staled_ids)
    deferred_ids.difference_update(policy_staled_ids)
    evidence_pool[:] = _active_evidence_rows(evidence_rows)
    _normalize_evidence_runtime_states(evidence_pool)
    evidence_counts_after = _evidence_pool_counts(evidence_pool)
    window_infos = _jsonable_copy(context.get("epoch_l2_windows") or [])
    for window_info in window_infos:
        window_info["selection"] = selection_info
        window_info.pop("accepted_edits", None)
    merge_info = _jsonable_copy(context.get("epoch_global_merge") or {})
    summary = {
        "epoch_index": epoch_idx,
        "epoch_rollout_task_count": int(
            context.get("epoch_rollout_task_count", epoch_rollout_task_count)
            or epoch_rollout_task_count
        ),
        "epoch_skill_update_mode": _resolve_epoch_skill_update_mode(args),
        "epoch_evidence_counts": evidence_counts_after,
        "epoch_evidence_counts_before_l2": _jsonable_copy(
            context.get("epoch_evidence_counts_before_l2") or {}
        ),
        "epoch_l2_window_count": len(window_infos),
        "epoch_candidate_edit_count": int(context.get("epoch_candidate_edit_count", 0) or 0),
        "epoch_accepted_candidate_edit_count": int(
            context.get("epoch_accepted_edit_count", 0) or 0
        ),
        "epoch_accepted_edit_count": int(context.get("epoch_accepted_edit_count", 0) or 0),
        "rule_ignored_accepts": int(context.get("rule_ignored_accepts", 0) or 0),
        "rule_ignored_defers": (
            int(context.get("rule_ignored_defers", 0) or 0)
            + int(post_reject_info.get("rule_ignored_defers", 0) or 0)
        ),
        "reflection_attempts": int(context.get("reflection_attempts", 0) or 0),
        "reflection_accepts": int(context.get("reflection_accepts", 0) or 0),
        "reflection_failures": int(context.get("reflection_failures", 0) or 0),
        "judge_rejects": int(context.get("judge_rejects", 0) or 0),
        "epoch_global_merge": merge_info,
        "epoch_coverage_repair": _jsonable_copy(context.get("epoch_coverage_repair") or {}),
        "epoch_current_skill_coverage_audit": _jsonable_copy(
            context.get("epoch_current_skill_coverage_audit") or {}
        ),
        "epoch_candidate_coverage_audit": _jsonable_copy(
            context.get("epoch_candidate_coverage_audit") or {}
        ),
        "epoch_final_coverage_audit": _jsonable_copy(
            context.get("epoch_final_coverage_audit") or {}
        ),
        "epoch_corrective_edit_count": merge_info.get("corrective_edit_count", 0),
        "epoch_success_only_edit_count": merge_info.get("success_only_edit_count", 0),
        "epoch_merged_pool_count": merge_info.get("merged_pool_count", 0),
        "epoch_ranked_edit_count": merge_info.get("ranked_edit_count", 0),
        "epoch_recovery_rule_edit_count": merge_info.get("recovery_rule_edit_count", 0),
        "epoch_replace_or_generalize_edit_count": merge_info.get(
            "replace_or_generalize_edit_count", 0
        ),
        "epoch_final_replay": epoch_final_replay_info,
        "epoch_working_branch": transition_info,
        "epoch_working_branch_post_reject_replay": post_reject_info,
        "epoch_working_branch_policy_defer": policy_defer_limit_info,
        "epoch_provisional_edits": [
            {
                **_working_edit_payload(edit),
                "working_edit_id": str(edit.get("_working_edit_id") or ""),
            }
            for edit in provisional_edits
        ],
        "epoch_provisional_edit_count": len(provisional_edits),
        "final_replay_edits": epoch_final_replay_info.get("final_replay_edits", 0),
        "final_replay_accepts": epoch_final_replay_info.get("final_replay_accepts", 0),
        "final_replay_rejects": epoch_final_replay_info.get("final_replay_rejects", 0),
        "final_replay_skipped": epoch_final_replay_info.get("final_replay_skipped", 0),
        "final_replay_bypassed": epoch_final_replay_info.get("final_replay_bypassed", 0),
        "epoch_global_commit_reports": _jsonable_copy(
            context.get("epoch_global_commit_reports") or []
        ),
        "epoch_committed_edits": [
            _clean_markdown_edit_for_row(edit) for edit in committed_edits
        ],
        "epoch_committed_edit_count": len(committed_edits),
        "epoch_validation_decision": decision,
        "epoch_validation_current_score": selection_info.get("current_score"),
        "epoch_validation_candidate_score": selection_info.get("candidate_score"),
        "epoch_validation_current_success_rate": selection_info.get("current_success_rate"),
        "epoch_validation_candidate_success_rate": selection_info.get("candidate_success_rate"),
        "epoch_validation_current_success": selection_info.get("current_success"),
        "epoch_validation_candidate_success": selection_info.get("candidate_success"),
        "epoch_selection": selection_info,
        "epoch_l2_windows": window_infos,
        "epoch_archived_evidence_ids": sorted(archived_ids),
        "epoch_accepted_evidence_ids": sorted(accepted_evidence_ids),
        "epoch_rejected_evidence_ids": sorted(rejected_evidence_ids),
        "epoch_staled_evidence_ids": sorted(staled_evidence_ids),
        "epoch_policy_defer_staled_evidence_ids": sorted(policy_staled_ids),
        "epoch_consumed_evidence_ids": sorted(consumed_ids),
        "epoch_deferred_evidence_ids": sorted(deferred_ids),
        "epoch_deferred_evidence_reasons": {
            evidence_id: deferred_fates[evidence_id]
            for evidence_id in sorted(deferred_ids)
            if evidence_id in deferred_fates
        },
        "already_covered_current_evidence_count": len(baseline_covered_ids),
        "final_committed_evidence_count": len(final_covered_ids) if decision == "commit" else 0,
        "final_deferred_evidence_count": len(deferred_ids),
        "consumed_evidence_count": 0,
        "resumed_from_working_branch_stage": stage_name,
    }
    _checkpoint_working_branch_epoch_l2(
        out,
        epoch_idx,
        l2_manager_rows=l2_manager_rows,
        l2_candidate_rows=l2_candidate_rows,
        l2_judge_rows=l2_judge_rows,
        evidence_rows=evidence_rows,
        replay_rows=replay_rows,
        selection_feedback_rows=selection_feedback_rows,
        validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
        working_skill_md=current_skill_path.read_text(encoding="utf-8"),
    )
    working_branch_state["last_completed_epoch_summary"] = _jsonable_copy(summary)
    persist_working_branch_state(
        working_branch_state_path,
        working_branch_state,
        validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
        working_skill_md=current_skill_path.read_text(encoding="utf-8"),
    )
    return summary, reject_cooldown_pool_size


def build_teacher(args):
    if not getattr(args, "teacher_model", None):
        raise ValueError("--teacher-model is required for EviSkill evolution")
    teacher_extra_body = (
        json.loads(getattr(args, "teacher_extra_body_json", None))
        if getattr(args, "teacher_extra_body_json", None)
        else None
    )
    teacher_reasoning_effort = getattr(args, "teacher_reasoning_effort", None)
    teacher_reasoning_effort = None if str(teacher_reasoning_effort or "").lower() == "none" else teacher_reasoning_effort
    return build_llm_client(
        getattr(args, "llm_kind", "openai"),
        args.teacher_model,
        api_key=getattr(args, "teacher_api_key", None),
        base_url=getattr(args, "teacher_base_url", None),
        temperature=getattr(args, "teacher_temperature", None),
        max_tokens=getattr(args, "teacher_max_tokens", 16384),
        load_in_4bit=getattr(args, "teacher_load_in_4bit", False),
        timeout=getattr(args, "teacher_timeout", 180),
        max_retries=getattr(args, "teacher_max_retries", -1),
        retry_initial_sleep=getattr(args, "teacher_retry_initial_sleep", 5.0),
        retry_max_sleep=getattr(args, "teacher_retry_max_sleep", 60.0),
        debug_dir=getattr(args, "llm_debug_dir", None),
        thinking=getattr(args, "teacher_thinking", None),
        reasoning_effort=teacher_reasoning_effort,
        extra_body=teacher_extra_body,
        credential_env_prefix="TEACHER",
        proxy_url=getattr(args, "teacher_proxy_url", None),
    )


def build_action_llm(args):
    if getattr(args, "dry_run_action", False):
        return FixedJsonActionLLM()
    if not getattr(args, "action_model", None):
        raise ValueError("--action-model is required, or pass --dry-run-action for a code-path check.")
    action_extra_body = (
        json.loads(getattr(args, "action_extra_body_json", None))
        if getattr(args, "action_extra_body_json", None)
        else None
    )
    action_reasoning_effort = getattr(args, "action_reasoning_effort", None)
    action_reasoning_effort = None if str(action_reasoning_effort or "").lower() == "none" else action_reasoning_effort
    return build_llm_client(
        getattr(args, "action_llm_kind", None) or "openai",
        args.action_model,
        api_key=getattr(args, "action_api_key", None),
        base_url=getattr(args, "action_base_url", None),
        temperature=getattr(args, "action_temperature", None),
        max_tokens=getattr(args, "action_max_new_tokens", 512),
        load_in_4bit=getattr(args, "action_load_in_4bit", False),
        timeout=getattr(args, "action_timeout", 180),
        max_retries=getattr(args, "action_max_retries", -1),
        retry_initial_sleep=getattr(args, "action_retry_initial_sleep", 5.0),
        retry_max_sleep=getattr(args, "action_retry_max_sleep", 60.0),
        debug_dir=getattr(args, "llm_debug_dir", None),
        thinking=getattr(args, "action_thinking", None),
        reasoning_effort=action_reasoning_effort,
        extra_body=action_extra_body,
        context_length_retry=str(getattr(args, "dataset", "") or "").lower() == "appworld",
        credential_env_prefix="ACTION",
        proxy_url=getattr(args, "action_proxy_url", None),
    )


def _build_dual_layer_runner(args, dataset: str, action_llm):
    dataset = str(dataset or "alfworld").lower()
    parallel_workers = max(1, int(getattr(args, "bundle_size", 1) or 1))
    legacy_scienceworld_workers = int(getattr(args, "scienceworld_parallel_workers", 0) or 0)
    if legacy_scienceworld_workers > 0 and legacy_scienceworld_workers != parallel_workers:
        print(
            f"[rollout workers] ignoring legacy --scienceworld-parallel-workers="
            f"{legacy_scienceworld_workers}; using bundle_size={parallel_workers}",
            flush=True,
        )
    if dataset == "alfworld":
        return build_alfworld_runner(
            alfworld_path=args.alfworld_path,
            alfworld_config=args.alfworld_config,
            alfworld_split=args.alfworld_split,
            action_llm=action_llm,
            max_steps=args.max_steps,
            top_k_general_skills=getattr(args, "top_k_general_skills", 5),
            top_k_task_specific_skills=getattr(args, "top_k_task_specific_skills", 2),
            general_skill_retrieval_threshold=getattr(args, "general_skill_retrieval_threshold", 0.0),
            task_specific_skill_retrieval_threshold=getattr(args, "task_specific_skill_retrieval_threshold", 0.0),
            task_specific_retrieval_scope=getattr(args, "task_specific_retrieval_scope", "detected"),
            max_resets_to_find_task=getattr(args, "alfworld_max_resets_to_find_task", 512),
            action_history_length=getattr(
                args,
                "alfworld_action_history_length",
                ALFWORLD_ACTION_HISTORY_LENGTH,
            ),
            parallel_workers=parallel_workers,
        )
    if dataset == "appworld":
        return build_appworld_runner(
            action_llm=action_llm,
            appworld_data_dir=getattr(args, "appworld_data_dir", None),
            output_dir=str(getattr(args, "output_dir", "outputs/appworld")),
            appworld_env_url=getattr(args, "appworld_env_url", None),
            appworld_auto_server=bool(getattr(args, "appworld_auto_server", False)),
            appworld_server_python=getattr(args, "appworld_server_python", None),
            appworld_server_host=getattr(args, "appworld_server_host", "127.0.0.1"),
            appworld_server_port_start=int(getattr(args, "appworld_server_port_start", 8841) or 8841),
            appworld_env_timeout=int(getattr(args, "appworld_env_timeout", 100) or 100),
            action_history_length=int(
                getattr(args, "appworld_action_history_length", APPWORLD_ACTION_HISTORY_LENGTH)
                or APPWORLD_ACTION_HISTORY_LENGTH
            ),
            random_seed=int(getattr(args, "appworld_random_seed", 100) or 100),
            max_steps=args.max_steps,
            parallel_workers=parallel_workers,
            top_k_general_skills=getattr(args, "top_k_general_skills", 5),
            top_k_task_specific_skills=getattr(args, "top_k_task_specific_skills", 2),
            general_skill_retrieval_threshold=getattr(args, "general_skill_retrieval_threshold", 0.0),
            task_specific_skill_retrieval_threshold=getattr(args, "task_specific_skill_retrieval_threshold", 0.0),
            task_specific_retrieval_scope=getattr(args, "task_specific_retrieval_scope", "detected"),
        )
    if dataset in {"scienceworld", "science_world"}:
        return build_scienceworld_runner(
            action_llm=action_llm,
            max_steps=args.max_steps,
            scienceworld_env_url=getattr(args, "scienceworld_env_url", "http://127.0.0.1:8811"),
            scienceworld_env_timeout=getattr(args, "scienceworld_env_timeout", 180),
            scienceworld_parallel_workers=parallel_workers,
            scienceworld_auto_server=getattr(args, "scienceworld_auto_server", False),
            scienceworld_server_host=getattr(args, "scienceworld_server_host", "127.0.0.1"),
            scienceworld_server_port_start=getattr(args, "scienceworld_server_port_start", 8811),
            scienceworld_path=getattr(args, "scienceworld_path", None),
            scienceworld_jar_path=getattr(args, "scienceworld_jar_path", None),
            scienceworld_env_step_limit=getattr(args, "scienceworld_env_step_limit", None),
            scienceworld_server_python=getattr(args, "scienceworld_server_python", None),
            action_history_length=getattr(
                args,
                "scienceworld_action_history_length",
                SCIENCEWORLD_ACTION_HISTORY_LENGTH,
            ),
            use_admissible_actions=getattr(args, "scienceworld_use_admissible_actions", False),
            top_k_general_skills=getattr(args, "top_k_general_skills", 5),
            top_k_task_specific_skills=getattr(args, "top_k_task_specific_skills", 2),
            general_skill_retrieval_threshold=getattr(args, "general_skill_retrieval_threshold", 0.0),
            task_specific_skill_retrieval_threshold=getattr(args, "task_specific_skill_retrieval_threshold", 0.0),
            task_specific_retrieval_scope=getattr(args, "task_specific_retrieval_scope", "detected"),
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _sample_evolution_bundle(
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    *,
    strategy: str,
    rng: random.Random,
    seed: int,
    round_idx: int,
    family_group_frequency: int = 2,
    shuffle_epoch_seed_base: int = 1,
) -> Tuple[List[Any], str]:
    if strategy == "shuffle_epoch":
        return sample_shuffle_epoch_bundle(
            manifest,
            bundle_size,
            seed=seed,
            round_idx=round_idx,
            epoch_seed_base=shuffle_epoch_seed_base,
        ), "shuffle"
    if strategy == "family_epoch":
        return sample_family_epoch_bundle(
            manifest,
            bundle_size,
            seed=seed,
            round_idx=round_idx,
        ), "diverse"
    if strategy == "mixed_family_epoch":
        return sample_mixed_family_epoch_bundle(
            manifest,
            bundle_size,
            seed=seed,
            round_idx=round_idx,
            family_group_frequency=family_group_frequency,
        ), mixed_family_epoch_bundle_mode(round_idx, family_group_frequency)
    return sample_random_bundle(manifest, bundle_size, rng=rng), "random"


def _resume_shuffle_epoch_seed_plans(
    *,
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    seed: int,
    logs: List[Dict[str, Any]],
) -> Dict[int, Dict[str, int]]:
    """Detect legacy shuffle epochs and the first mixed-plan resume bundle."""
    bundles_per_epoch = max(1, (len(manifest) + int(bundle_size) - 1) // int(bundle_size))
    epoch_indices = sorted(
        {
            _row_int(row or {}, "epoch_index", -1)
            for row in logs or []
            if _row_int(row or {}, "epoch_index", -1) >= 0
        }
    )
    plans: Dict[int, Dict[str, int]] = {}
    for epoch_idx in epoch_indices:
        if _epoch_l1_is_complete(logs, epoch_idx) or _epoch_uses_reused_evidence(logs, epoch_idx):
            continue
        rows = _epoch_bundle_log_rows(logs, epoch_idx)
        if not rows:
            continue
        prefix_by_base: Dict[int, int] = {}
        for epoch_seed_base in (1, 0):
            matched = 0
            for row in rows:
                bundle_idx = _row_int(row, "epoch_bundle_index", -1)
                expected = sample_shuffle_epoch_bundle(
                    manifest,
                    bundle_size,
                    seed=seed,
                    round_idx=epoch_idx * bundles_per_epoch + bundle_idx,
                    epoch_seed_base=epoch_seed_base,
                )
                expected_ids = [get_task_id(item) for item in expected]
                logged_ids = [get_task_id(item) for item in (row.get("task_ids") or [])]
                if bundle_idx != matched or logged_ids != expected_ids:
                    break
                matched += 1
            prefix_by_base[epoch_seed_base] = matched
        epoch_seed_base = max((1, 0), key=lambda value: (prefix_by_base[value], value))
        plans[epoch_idx] = {
            "epoch_seed_base": epoch_seed_base,
            "matched_prefix_bundles": prefix_by_base[epoch_seed_base],
            "logged_bundle_count": len(rows),
        }
    return plans


def _truncate_incomplete_epoch_bundle_suffix(
    *,
    out: Path,
    logs: List[Dict[str, Any]],
    epoch_idx: int,
    first_bundle_idx: int,
) -> None:
    logs[:] = [
        row
        for row in logs or []
        if _row_int(row or {}, "epoch_index", -1) < int(epoch_idx)
        or (
            _row_int(row or {}, "epoch_index", -1) == int(epoch_idx)
            and _row_int(row or {}, "epoch_bundle_index", -1) < int(first_bundle_idx)
        )
    ]
    checkpoint_dir = _epoch_train_rollout_checkpoint_dir(out, epoch_idx)
    for path in checkpoint_dir.glob("bundle_*.json") if checkpoint_dir.exists() else []:
        try:
            bundle_idx = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if bundle_idx >= int(first_bundle_idx):
            path.unlink()
    for epoch_dir in _epoch_artifact_dirs(out):
        parsed_epoch = _parse_epoch_artifact_dir(epoch_dir)
        if parsed_epoch is not None and parsed_epoch > int(epoch_idx):
            shutil.rmtree(epoch_dir)


def _sample_longitudinal_memory_tasks(
    manifest: List[Dict[str, Any]],
    *,
    sample_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    items = list(manifest or [])
    if not items or int(sample_count or 0) <= 0:
        return []
    rng = random.Random(int(seed))
    rng.shuffle(items)
    return items[: min(len(items), int(sample_count))]


def _invalidate_selection_baseline(out: Path) -> None:
    baseline_path = out / "selection_baseline.json"
    if baseline_path.exists():
        try:
            baseline_path.unlink()
        except OSError:
            pass


def _run_longitudinal_memory_update(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    epoch_idx: int,
    previous_skill_md: str,
    current_skill_path: Path,
    evolution_manifest: List[Dict[str, Any]],
    committed_edits: List[Dict[str, Any]],
    selection_tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not bool(getattr(args, "use_longitudinal_memory", False)):
        return {"enabled": False}

    memory_root = out / "longitudinal_memory"
    memory_dir = memory_root / safe_memory_dir_name(f"epoch_{epoch_idx:03d}")
    memory_dir.mkdir(parents=True, exist_ok=True)
    result_path = memory_dir / "memory_result.json"
    current_skill_md = current_skill_path.read_text(encoding="utf-8")
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] start "
        f"committed_edits={len(committed_edits or [])}",
        flush=True,
    )

    if result_path.exists():
        try:
            saved = load_json(str(result_path))
        except Exception:
            saved = {}
        print(
            f"[epoch {epoch_idx + 1} longitudinal memory] resumed "
            f"action={saved.get('action', 'resumed')}",
            flush=True,
        )
        return {
            "enabled": True,
            "epoch_index": epoch_idx,
            "action": saved.get("action", "resumed"),
            "resumed": True,
            "result_path": str(result_path),
        }

    if epoch_idx <= 0:
        updated = inject_empty_longitudinal_memory_field(current_skill_md)
        changed = updated != current_skill_md
        if changed:
            current_skill_path.write_text(updated, encoding="utf-8")
            _invalidate_selection_baseline(out)
        result = {
            "enabled": True,
            "epoch_index": epoch_idx,
            "action": "inject_placeholder",
            "changed": changed,
        }
        save_json(str(result_path), result)
        print(
            f"[epoch {epoch_idx + 1} longitudinal memory] placeholder "
            f"changed={changed}",
            flush=True,
        )
        return result

    sample_count = int(getattr(args, "longitudinal_memory_samples", 20) or 20)
    seed = int(getattr(args, "seed", 42) or 42) + (epoch_idx + 1) * 2000
    sampled_tasks = _sample_longitudinal_memory_tasks(
        evolution_manifest,
        sample_count=sample_count,
        seed=seed,
    )
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] sampled tasks "
        f"count={len(sampled_tasks)} requested={sample_count}",
        flush=True,
    )
    if not sampled_tasks:
        result = {
            "enabled": True,
            "epoch_index": epoch_idx,
            "action": "skipped",
            "reason": "no sampled tasks",
        }
        save_json(str(result_path), result)
        print(
            f"[epoch {epoch_idx + 1} longitudinal memory] skipped reason=no sampled tasks",
            flush=True,
        )
        return result

    previous_skill_path = memory_dir / "previous_epoch_skill.md"
    current_snapshot_path = memory_dir / "current_epoch_skill.md"
    previous_skill_path.write_text(previous_skill_md, encoding="utf-8")
    current_snapshot_path.write_text(current_skill_md, encoding="utf-8")

    previous_results_path = memory_dir / "previous_results.json"
    current_results_path = memory_dir / "current_results.json"
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] previous rollout start "
        f"tasks={len(sampled_tasks)}",
        flush=True,
    )
    previous_results = runner.run_task_set(
        sampled_tasks,
        MarkdownSkillBank(str(previous_skill_path)),
        {},
        desc=f"longitudinal memory epoch {epoch_idx + 1} previous",
    )
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] current rollout start "
        f"tasks={len(sampled_tasks)}",
        flush=True,
    )
    current_results = runner.run_task_set(
        sampled_tasks,
        MarkdownSkillBank(str(current_snapshot_path)),
        {},
        desc=f"longitudinal memory epoch {epoch_idx + 1} current",
    )
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] rollout done "
        f"previous_score={_average_score(previous_results):.4f} "
        f"previous_success_rate={_success_rate(previous_results):.4f} "
        f"current_score={_average_score(current_results):.4f} "
        f"current_success_rate={_success_rate(current_results):.4f}",
        flush=True,
    )
    save_json(str(previous_results_path), previous_results)
    save_json(str(current_results_path), current_results)

    max_trajectory_chars = int(getattr(args, "longitudinal_memory_max_trajectory_chars", 3000) or 3000)
    pairs = build_longitudinal_comparison_pairs(
        items=sampled_tasks,
        previous_results=previous_results,
        current_results=current_results,
        max_trajectory_chars=max_trajectory_chars,
        full_context=bool(getattr(args, "llm_full_context", False)),
    )
    pair_policy = str(getattr(args, "longitudinal_memory_pair_policy", "mixed") or "mixed").strip().lower()
    if pair_policy == "changed":
        pairs = [row for row in pairs if row.get("category") in {"improved", "regressed"}]
    elif pair_policy == "unchanged":
        pairs = [row for row in pairs if row.get("category") in {"persistent_fail", "stable_success"}]
    elif pair_policy != "mixed":
        raise ValueError("--longitudinal-memory-pair-policy must be one of: mixed, changed, unchanged")
    save_json(str(memory_dir / "comparison_pairs.json"), pairs)
    category_counts: Dict[str, int] = {}
    for pair in pairs:
        category = str(pair.get("category") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] comparison ready "
        f"pairs={len(pairs)} policy={pair_policy} categories={category_counts}",
        flush=True,
    )

    comparison_text = format_longitudinal_comparison_text(pairs)
    skill_update_summary = format_skill_update_summary(committed_edits)
    formatted_input = build_longitudinal_memory_input(
        previous_skill_md=previous_skill_md,
        current_skill_md=current_skill_md,
        previous_memory=extract_longitudinal_memory_field(current_skill_md),
        skill_update_summary=skill_update_summary,
        longitudinal_comparison=comparison_text,
    )
    (memory_dir / "memory_input.txt").write_text(formatted_input, encoding="utf-8")

    payload = {"formatted_input": formatted_input}
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] writer start "
        f"input_chars={len(formatted_input)}",
        flush=True,
    )
    try:
        raw = _teacher_chat_json(teacher, prompts_for_dataset(dataset)["longitudinal_memory"], payload)
    except Exception as exc:
        raw = {"reasoning": f"Longitudinal Skill Memory writer failed: {exc}", "longitudinal_memory_content": ""}

    memory_content = str(
        (raw or {}).get("longitudinal_memory_content")
        or (raw or {}).get("memory_content")
        or ""
    ).strip()
    result: Dict[str, Any] = {
        "enabled": True,
        "epoch_index": epoch_idx,
        "sample_count": len(sampled_tasks),
        "pair_count": len(pairs),
        "pair_policy": pair_policy,
        "reasoning": str((raw or {}).get("reasoning") or ""),
        "longitudinal_memory_content": memory_content,
    }
    if not memory_content:
        result["action"] = "no_content"
        save_json(str(result_path), result)
        print(
            f"[epoch {epoch_idx + 1} longitudinal memory] writer done action=no_content",
            flush=True,
        )
        return result

    candidate_skill_md = replace_longitudinal_memory_field(current_skill_md, memory_content)
    candidate_path = memory_dir / "candidate_skill.md"
    candidate_path.write_text(candidate_skill_md, encoding="utf-8")

    if bool(getattr(args, "longitudinal_memory_gate_with_selection", False)):
        print(
            f"[epoch {epoch_idx + 1} longitudinal memory] validation start "
            f"selection_tasks={len(selection_tasks)}",
            flush=True,
        )
        selection = _selection_gate(
            runner=runner,
            out=out,
            current_skill_path=current_skill_path,
            candidate_skill_md=candidate_skill_md,
            selection_tasks=selection_tasks,
            round_idx=epoch_idx,
            enabled=True,
            args=args,
        )
        result["selection"] = {
            key: value
            for key, value in selection.items()
            if key not in {"baseline_task_results", "candidate_task_results"}
        }
        if selection.get("decision") == "commit":
            current_skill_path.write_text(candidate_skill_md, encoding="utf-8")
            result["action"] = "accept"
        else:
            result["action"] = "reject"
        save_json(str(result_path), result)
        print(
            f"[epoch {epoch_idx + 1} longitudinal memory] validation done "
            f"decision={selection.get('decision')} action={result['action']}",
            flush=True,
        )
        return result

    current_skill_path.write_text(candidate_skill_md, encoding="utf-8")
    _invalidate_selection_baseline(out)
    result["action"] = "force_accept"
    save_json(str(result_path), result)
    print(
        f"[epoch {epoch_idx + 1} longitudinal memory] writer done "
        f"action=force_accept content_chars={len(memory_content)}",
        flush=True,
    )
    return result


def _run_l2_update_window(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    round_idx: int,
    skill_md: str,
    current_skill_path: Path,
    evidence_pool: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    raw_by_id: Dict[str, Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    selection_tasks: List[Dict[str, Any]],
    l2_trigger_reason: str,
    reject_cooldown_pool_size: int = 0,
    defer_selection: bool = False,
    force_selection_enabled: Optional[bool] = None,
    exclude_evidence_ids: Optional[List[str]] = None,
    preselected_pool: Optional[List[Dict[str, Any]]] = None,
    preselected_selection_info: Optional[Dict[str, Any]] = None,
    defer_commit: bool = False,
) -> Tuple[str, Dict[str, Any], int, Dict[str, Any]]:
    max_l2_window = int(
        getattr(
            args,
            "l2_max_evidence_per_window",
            getattr(args, "evidence_window_size", 20),
        )
        or getattr(args, "evidence_window_size", 20)
    )
    max_l2_window = max(1, max_l2_window)
    l2_evidence_grouping = _resolve_l2_evidence_window_grouping(
        dataset,
        getattr(args, "l2_evidence_window_grouping", "auto"),
    )
    if preselected_pool is not None:
        selected_pool = list(preselected_pool)
        evidence_selection_info = dict(preselected_selection_info or {})
        if not evidence_selection_info:
            active_rows = _l2_active_rows(evidence_pool, exclude_evidence_ids=exclude_evidence_ids)
            evidence_selection_info = _l2_selection_info_for_groups(
                evidence_pool=evidence_pool,
                active_rows=active_rows,
                stale_count=_l2_stale_count(evidence_pool),
                selected_groups=[selected_pool],
                grouping=l2_evidence_grouping,
            )
    else:
        if l2_evidence_grouping == "semantic":
            planned = _plan_epoch_l2_evidence_windows(
                evidence_pool,
                max_window=max_l2_window,
                max_windows=1,
                grouping=l2_evidence_grouping,
                semantic_options=_semantic_embedding_options_from_args(args, out),
                exclude_evidence_ids=exclude_evidence_ids,
            )
            if planned:
                selected_pool, evidence_selection_info = planned[0]
            else:
                active_rows = _l2_active_rows(evidence_pool, exclude_evidence_ids=exclude_evidence_ids)
                evidence_selection_info = _empty_evidence_selection_info(
                    evidence_pool,
                    active_rows,
                    _l2_stale_count(evidence_pool),
                    grouping=l2_evidence_grouping,
                )
                selected_pool = []
        else:
            selected_pool, evidence_selection_info = _select_l2_evidence_window(
                evidence_pool,
                max_window=max_l2_window,
                exclude_evidence_ids=exclude_evidence_ids,
                grouping=l2_evidence_grouping,
            )
    progress_label = _l2_progress_label(round_idx, l2_trigger_reason)
    if not selected_pool:
        l2_prompt = prompts_for_dataset(dataset)["skill_manager_l2"]
        print(
            f"[{progress_label} skill edit window] skipped empty window "
            f"trigger={l2_trigger_reason} grouping={l2_evidence_grouping} "
            f"active={evidence_selection_info.get('active_count', 0)} "
            f"stale={evidence_selection_info.get('stale_count', 0)}",
            flush=True,
        )
        l2_info = {
            "triggered": False,
            "trigger_reason": l2_trigger_reason,
            "candidate_count": 0,
            "accepted_count": 0,
            "committed_count": 0,
            "reason": "empty_l2_evidence_window",
            "evidence_pool": _evidence_pool_counts(evidence_pool),
            "evidence_selection": evidence_selection_info,
        }
        return skill_md, l2_info, reject_cooldown_pool_size, {
            "dataset": dataset,
            "round": round_idx,
            **({"epoch_index": round_idx} if str(l2_trigger_reason or "") == "epoch_boundary" else {}),
            "system_prompt": l2_prompt,
            "input": {"skill_md": skill_md, "source_type": "", "evidence_cards": []},
            "output": {"reasoning": "empty skill edit evidence window", "edits": []},
        }

    public_pool = _public_evidence_cards_for_proposer(
        selected_pool,
        full_context=bool(getattr(args, "llm_full_context", False)),
    )
    full_evidence_by_id = {str(x.get("evidence_id")): x for x in selected_pool}
    feedback_enabled = _l2_selection_feedback_enabled(dataset, args)
    selection_feedback = (
        _recent_selection_feedback(
            selection_feedback_rows,
            limit=max(0, int(getattr(args, "l2_selection_feedback_history", SELECTION_FEEDBACK_HISTORY))),
        )
        if feedback_enabled
        else []
    )
    l2_prompt = prompts_for_dataset(dataset)["skill_manager_l2"]
    l2_judge_prompt = prompts_for_dataset(dataset)["skill_judge_l2"]
    coverage_mode = _resolve_epoch_skill_update_mode(args) == "cluster_step" and str(l2_trigger_reason or "") == "epoch_boundary"
    max_candidate_edits = int(getattr(args, "l2_candidate_budget", 5))
    l2_input_raw = {
        "skill_md": skill_md,
        "source_type": evidence_selection_info.get("source_type") or _edit_source_type_from_linked(selected_pool),
        "evidence_cards": public_pool,
    }
    if not coverage_mode:
        l2_input_raw["max_candidate_edits"] = max_candidate_edits
    if feedback_enabled:
        l2_input_raw["selection_feedback"] = selection_feedback
    l2_input = _compact_llm_payload(
        l2_input_raw,
        l2_prompt,
        args,
        label="skill edit proposer",
    )
    window_index = evidence_selection_info.get("planned_window_index")
    window_count = evidence_selection_info.get("planned_window_count")
    window_suffix = (
        f" window {int(window_index) + 1}/{window_count}"
        if window_index is not None and window_count is not None
        else " window"
    )
    grouping_label = str(evidence_selection_info.get("grouping") or l2_evidence_grouping or "card")
    group_log = ""
    if grouping_label == "semantic":
        group_log = (
            f" semantic_groups={evidence_selection_info.get('selected_semantic_group_count', 1)} "
            f"semantic_group_sizes={evidence_selection_info.get('semantic_group_sizes', [])}"
        )
    elif grouping_label == "round":
        group_log = (
            f" round_groups={evidence_selection_info.get('selected_round_group_count', 1)} "
            f"round_group_sizes={evidence_selection_info.get('round_group_sizes', [])}"
        )
    print(
        f"[{progress_label} skill edit{window_suffix}] proposer start "
        f"evidence={len(public_pool)}{group_log} "
            f"feedback={len(selection_feedback)} "
            f"max_candidate_edits={l2_input.get('max_candidate_edits', 'coverage')}",
        flush=True,
    )
    try:
        l2_raw = _teacher_chat_json(teacher, l2_prompt, l2_input)
    except Exception as exc:
        raise L2ProposerError(
            f"{progress_label} skill edit{window_suffix} proposer failed after "
            f"teacher retries: {exc}"
        ) from exc
    l2_manager_error = ""
    candidate_edits = []
    raw_patch_items = _markdown_patch_items(l2_raw)
    raw_edit_diagnostics: List[Dict[str, Any]] = []
    raw_rejection_counts: Dict[str, int] = {}
    raw_shape_warning_counts: Dict[str, int] = {}
    raw_valid_linked_edit_count = 0
    raw_unknown_evidence_ids = set()
    for raw_edit in raw_patch_items:
        diagnostic = _diagnose_markdown_edit(raw_edit, full_evidence_by_id)
        raw_edit_diagnostics.append(diagnostic)
        for reason in diagnostic["rejection_reasons"]:
            raw_rejection_counts[reason] = raw_rejection_counts.get(reason, 0) + 1
        for warning in diagnostic["shape_warnings"]:
            raw_shape_warning_counts[warning] = raw_shape_warning_counts.get(warning, 0) + 1
        raw_unknown_evidence_ids.update(diagnostic["unknown_evidence_ids"])
        if not diagnostic["accepted"]:
            continue
        edit = diagnostic["normalized_edit"]
        raw_valid_linked_edit_count += 1
        if edit is None:
            continue
        if not coverage_mode and len(candidate_edits) >= int(getattr(args, "l2_candidate_budget", 5)):
            continue
        candidate_edits.append(edit)
    omitted_evidence_ids = set(_omitted_evidence_ids(l2_raw))
    fallback_evidence_ids: List[str] = []
    policy_deferred_evidence_reasons: Dict[str, str] = {}
    if coverage_mode and not l2_manager_error:
        covered_ids = set(_covered_evidence_ids(candidate_edits))
        selected_ids = {str(eid) for eid in evidence_selection_info.get("selected_evidence_ids") or [] if str(eid)}
        missing_ids = sorted(selected_ids - covered_ids - omitted_evidence_ids)
        if _working_branch_defer_uncovered_enabled(dataset, args):
            policy_deferred_evidence_reasons.update(
                {evidence_id: "l2_proposer_not_covered" for evidence_id in missing_ids}
            )
            policy_deferred_evidence_reasons.update(
                {
                    evidence_id: "l2_proposer_omitted"
                    for evidence_id in sorted(selected_ids.intersection(omitted_evidence_ids))
                }
            )
        else:
            for evidence_id in missing_ids:
                card = full_evidence_by_id.get(evidence_id)
                fallback_edit = _fallback_markdown_edit_from_evidence(card or {})
                if fallback_edit is None or not _linked_evidence(fallback_edit, full_evidence_by_id):
                    continue
                candidate_edits.append(fallback_edit)
                fallback_evidence_ids.append(evidence_id)
    print(
        f"[{progress_label} skill edit{window_suffix}] proposer done "
        f"raw_edits={len(raw_patch_items)} usable_edits={len(candidate_edits)} "
        f"raw_valid={raw_valid_linked_edit_count} "
        f"raw_rejected={len(raw_patch_items) - raw_valid_linked_edit_count} "
        f"raw_rejection_reasons={json.dumps(raw_rejection_counts, sort_keys=True)} "
        f"omitted_evidence={len(omitted_evidence_ids)} "
        f"fallback_evidence={len(fallback_evidence_ids)} "
        f"deferred_uncovered_evidence={len(policy_deferred_evidence_reasons)} "
        f"error={'yes' if l2_manager_error else 'no'}",
        flush=True,
    )

    l2_candidate_row = {
        "dataset": dataset,
        "round": round_idx,
        **({"epoch_index": round_idx} if str(l2_trigger_reason or "") == "epoch_boundary" else {}),
        "input": l2_input,
        "output": {
            "reasoning": str(l2_raw.get("reasoning") or ""),
            "raw_edits": raw_patch_items,
            "raw_edit_diagnostics": raw_edit_diagnostics,
            "raw_valid_linked_edit_count": raw_valid_linked_edit_count,
            "raw_rejected_edit_count": len(raw_patch_items) - raw_valid_linked_edit_count,
            "raw_rejection_counts": raw_rejection_counts,
            "raw_shape_warning_counts": raw_shape_warning_counts,
            "raw_unknown_evidence_ids": sorted(raw_unknown_evidence_ids),
            "edits": candidate_edits,
            "omitted_evidence": l2_raw.get("omitted_evidence") if isinstance(l2_raw, dict) else [],
            "fallback_evidence_ids": fallback_evidence_ids,
            "deferred_uncovered_evidence_ids": sorted(
                evidence_id
                for evidence_id, reason in policy_deferred_evidence_reasons.items()
                if reason == "l2_proposer_not_covered"
            ),
            "deferred_omitted_evidence_ids": sorted(
                evidence_id
                for evidence_id, reason in policy_deferred_evidence_reasons.items()
                if reason == "l2_proposer_omitted"
            ),
        },
    }
    l2_candidate_rows.append(l2_candidate_row)

    accepted_edits: List[Dict[str, Any]] = []
    l2_decisions: List[Dict[str, Any]] = []
    rejected_feedback_cases: List[Dict[str, Any]] = []
    attempted_evidence_ids = set()
    accepted_evidence_ids = set()
    rejected_evidence_ids = set()
    staled_evidence_ids = set()
    deferred_evidence_ids = set()
    deferred_evidence_ids.update(policy_deferred_evidence_reasons)
    reflection_narrowed_evidence_ids = set()
    reflection_failed_evidence_ids = set()
    reflection_attempt_rows: List[Dict[str, Any]] = []
    rule_ignored_accepts = 0
    rule_ignored_defers = 0
    rule_ignored_deferred_evidence_ids = set()
    reflection_attempts = 0
    reflection_accepts = 0
    reflection_failures = 0
    judge_rejects = 0
    selected_evidence_ids = evidence_selection_info.get("selected_evidence_ids") or []
    cited_evidence_ids = set()
    max_rejects = max(
        1,
        int(
            getattr(
                args,
                "l2_max_rejects_per_evidence",
                L2_MAX_REJECTS_PER_EVIDENCE,
            )
            or L2_MAX_REJECTS_PER_EVIDENCE
        ),
    )
    if not l2_manager_error:
        attempted_evidence_ids.update(
            _mark_l2_window_attempts(
                evidence_rows,
                selected_evidence_ids,
                round_idx=round_idx,
            )
        )
    for edit_idx, edit in enumerate(candidate_edits):
        cited_evidence_ids.update(_evidence_id_list(edit.get("evidence_ids")))
        linked = _linked_evidence(edit, full_evidence_by_id)
        edit_label = f"{edit_idx + 1}/{len(candidate_edits)}"
        print(
            f"[{progress_label} skill edit{window_suffix}] edit {edit_label} apply start "
            f"op={edit.get('op')} evidence={len(linked)} "
            f"target_len={len(str(edit.get('target') or ''))} "
            f"content_len={len(str(edit.get('content') or ''))}",
            flush=True,
        )
        candidate_skill_md, apply_reports = _apply_markdown_edits(skill_md, [edit])
        apply_ok = bool(apply_reports) and all(bool(row.get("applied")) for row in apply_reports)
        replay_evidence: List[Dict[str, Any]] = []
        if apply_ok:
            tmp_skill_path = out / "_tmp_l2_candidate_skill.md"
            tmp_skill_path.write_text(candidate_skill_md, encoding="utf-8")
            replay_budget_arg = getattr(args, "l2_replay_ranges_per_edit", 4)
            replay_budget = (
                0
                if _no_replay_enabled(args)
                else (4 if replay_budget_arg is None else int(replay_budget_arg))
            )
            replay_jobs = _select_l2_replay_jobs(linked, replay_budget)
            print(
                f"[{progress_label} skill edit{window_suffix}] edit {edit_label} "
                f"apply ok replay_jobs={len(replay_jobs)} replay_budget={replay_budget}",
                flush=True,
            )
            if replay_jobs:
                replay_evidence = _run_l2_replay_jobs_for_edit(
                    runner=runner,
                    replay_jobs=replay_jobs,
                    replay_source_by_evidence_id=replay_source_by_evidence_id,
                    raw_by_id=raw_by_id,
                    tmp_skill_path=tmp_skill_path,
                    args=args,
                    dataset=dataset,
                    progress_label=progress_label,
                    window_suffix=window_suffix,
                    edit_label=edit_label,
                )
        else:
            failed_reason = "; ".join(str(row.get("reason") or "") for row in apply_reports if not row.get("applied"))
            print(
                f"[{progress_label} skill edit{window_suffix}] edit {edit_label} apply failed "
                f"reason={failed_reason or 'unknown'}",
                flush=True,
            )
        if not _no_replay_enabled(args):
            replay_rows.append(
                {
                    "dataset": dataset,
                    "round": round_idx,
                    **({"epoch_index": round_idx} if str(l2_trigger_reason or "") == "epoch_boundary" else {}),
                    "stage": "window_local",
                    "edit_index": edit_idx,
                    "edit": edit,
                    "replay_evidence": replay_evidence,
                }
            )
        if apply_ok:
            replay_attribution_allowed = (
                _uses_l2_replay_attribution(dataset) and not _no_replay_enabled(args)
            )
            print(
                f"[{progress_label} skill edit{window_suffix}] edit {edit_label} judge start "
                f"replay_evidence={len(replay_evidence)} "
                f"mode={_replay_mode(args)}",
                flush=True,
            )
            l2_judge_payload = _build_l2_skill_judge_payload(
                skill_md=skill_md,
                edit=edit,
                linked_evidence=linked,
                replay_evidence=replay_evidence,
                dataset=dataset,
                args=args,
                reflection_allowed=replay_attribution_allowed,
            )
            l2_label = _l2_skill_judge_label(
                teacher=teacher,
                skill_md=skill_md,
                edit=edit,
                linked_evidence=linked,
                replay_evidence=replay_evidence,
                dataset=dataset,
                args=args,
                payload=l2_judge_payload,
                reflection_allowed=replay_attribution_allowed,
            )
            print(
                f"[{progress_label} skill edit{window_suffix}] edit {edit_label} judge done "
                f"decision={l2_label.get('decision')} "
                f"replay_failure_type={l2_label.get('replay_failure_type')} "
                f"reason={str(l2_label.get('reason') or '')[:160]}",
                flush=True,
            )
        else:
            failed = next((row for row in apply_reports if not row.get("applied")), {})
            l2_judge_payload = _compact_llm_payload(
                {
                    "dataset": dataset,
                    "skill_md": skill_md,
                    "candidate_edit": edit,
                    "linked_evidence": _public_evidence_cards(linked),
                    "replay_evidence": [],
                },
                l2_judge_prompt,
                args,
                label="skill edit judge",
            )
            l2_label = {
                "decision": "reject",
                "reason": "markdown patch did not apply: " + str(failed.get("reason") or "unknown"),
            }
            if _uses_l2_replay_attribution(dataset):
                l2_label["replay_failure_type"] = None
        l2_judge_rows.append(
            {
                "dataset": dataset,
                "round": round_idx,
                **({"epoch_index": round_idx} if str(l2_trigger_reason or "") == "epoch_boundary" else {}),
                "system_prompt": l2_judge_prompt,
                "input": l2_judge_payload,
                "output": l2_label,
            }
        )
        final_decision = str(l2_label.get("decision") or "reject")
        effective_edit: Optional[Dict[str, Any]] = None
        reflection_result: Optional[Dict[str, Any]] = None
        if final_decision == "accept":
            effective_edit = edit
            accepted_edits.append(_annotate_edit_metadata(edit, linked, merge_level=1))
            if l2_label.get("replay_failure_type") == "rule_ignored":
                rule_ignored_accepts += 1
        elif (
            final_decision == "defer"
            and l2_label.get("replay_failure_type") == "rule_ignored"
        ):
            rule_ignored_defers += 1
            rule_ignored_deferred_evidence_ids.update(
                _evidence_id_list(edit.get("evidence_ids"))
            )
        elif (
            final_decision == "reflect"
            and _uses_l2_replay_attribution(dataset)
            and not _no_replay_enabled(args)
            and apply_ok
        ):
            reflection_attempts += 1
            print(
                f"[{progress_label} skill edit{window_suffix}] edit {edit_label} reflection start "
                f"failure_type={l2_label.get('replay_failure_type')}",
                flush=True,
            )
            reflection_result = _l2_replay_reflection(
                dataset=dataset,
                teacher=teacher,
                skill_md=skill_md,
                edit=edit,
                linked_evidence=linked,
                replay_evidence=replay_evidence,
                judge_output=l2_label,
                args=args,
            )
            reflection_attempt_row = {
                "candidate_index": edit_idx,
                "system_prompt": reflection_result.get("system_prompt"),
                "input": reflection_result.get("input"),
                "output": reflection_result.get("output"),
                "accepted": bool(reflection_result.get("accepted")),
                "validation_reason": reflection_result.get("validation_reason"),
                "apply_reports": reflection_result.get("apply_reports") or [],
            }
            reflection_attempt_rows.append(reflection_attempt_row)
            if reflection_result.get("accepted"):
                revised_edit = dict(reflection_result.get("revised_edit") or {})
                revised_linked = _linked_evidence(revised_edit, full_evidence_by_id)
                accepted_edits.append(
                    _annotate_edit_metadata(revised_edit, revised_linked, merge_level=1)
                )
                effective_edit = revised_edit
                final_decision = "accept"
                reflection_accepts += 1
                original_ids = set(_evidence_id_list(edit.get("evidence_ids")))
                revised_ids = set(_evidence_id_list(revised_edit.get("evidence_ids")))
                reflection_narrowed_evidence_ids.update(original_ids - revised_ids)
            else:
                final_decision = "reject"
                reflection_failures += 1
                reflection_failed_evidence_ids.update(_evidence_id_list(edit.get("evidence_ids")))
            print(
                f"[{progress_label} skill edit{window_suffix}] edit {edit_label} reflection done "
                f"decision={final_decision} "
                f"reason={str(reflection_result.get('validation_reason') or reflection_result.get('reason') or '')[:160]}",
                flush=True,
            )
        else:
            final_decision = "reject"
            judge_rejects += 1

        l2_decisions.append(
            {
                "edit": edit,
                **l2_label,
                "final_decision": final_decision,
                "effective_edit": effective_edit,
                "reflection": (
                    {
                        "accepted": bool(reflection_result.get("accepted")),
                        "reason": reflection_result.get("reason"),
                        "validation_reason": reflection_result.get("validation_reason"),
                        "revised_edit": reflection_result.get("revised_edit"),
                    }
                    if reflection_result is not None
                    else None
                ),
                "apply_reports": apply_reports,
            }
        )
        if final_decision == "reject" and reflection_result is None:
            edit_eids = _evidence_id_list(edit.get("evidence_ids"))
            if coverage_mode:
                deferred_evidence_ids.update(
                    _defer_evidence_rows(
                        evidence_rows,
                        edit_eids,
                        reason=(
                            "fallback_proposed_edit_rejected"
                            if set(edit_eids).issubset(set(fallback_evidence_ids))
                            else "window_local_replay_or_judge_reject"
                        ),
                    )
                )
            else:
                updates = _mark_evidence_attempts(
                    evidence_rows,
                    edit_eids,
                    round_idx=round_idx,
                    outcome="reject",
                    max_rejects=max_rejects,
                    count_attempt=False,
                )
                attempted_evidence_ids.update(updates["attempted"])
                rejected_evidence_ids.update(updates["rejected"])
                staled_evidence_ids.update(updates["staled"])
            if feedback_enabled and str(l2_label.get("reason") or "").strip():
                rejected_feedback_cases.append(
                    {
                        "candidate_edit": edit,
                        "linked_evidence": [_compact_feedback_evidence_card(card) for card in linked],
                        "replay_evidence": _compact_feedback_replay_evidence(replay_evidence),
                        "skill_edit_judge_reason": str(l2_label.get("reason") or ""),
                    }
                )

    if reflection_attempt_rows:
        l2_candidate_row["output"]["reflection_attempts"] = reflection_attempt_rows
    locally_accepted_evidence_ids = set(_covered_evidence_ids(accepted_edits))
    rule_ignored_deferred_evidence_ids.difference_update(
        locally_accepted_evidence_ids
    )
    for evidence_id in sorted(rule_ignored_deferred_evidence_ids):
        policy_deferred_evidence_reasons[evidence_id] = "rule_ignored"
    deferred_evidence_ids.update(rule_ignored_deferred_evidence_ids)
    reflection_failed_evidence_ids.difference_update(locally_accepted_evidence_ids)
    reflection_narrowed_evidence_ids.difference_update(locally_accepted_evidence_ids)
    reflection_narrowed_evidence_ids.difference_update(reflection_failed_evidence_ids)
    deferred_evidence_ids.update(
        _defer_evidence_rows(
            evidence_rows,
            sorted(reflection_failed_evidence_ids),
            reason="reflection_failed",
        )
    )
    deferred_evidence_ids.update(
        _defer_evidence_rows(
            evidence_rows,
            sorted(reflection_narrowed_evidence_ids),
            reason="reflection_narrowed",
        )
    )

    if not l2_manager_error and l2_evidence_grouping == "card":
        updates = _mark_uncited_selected_evidence(
            evidence_rows,
            selected_evidence_ids=selected_evidence_ids,
            cited_evidence_ids=sorted(cited_evidence_ids),
            round_idx=round_idx,
            max_rejects=max_rejects,
        )
        attempted_evidence_ids.update(updates["attempted"])
        rejected_evidence_ids.update(updates["rejected"])
        staled_evidence_ids.update(updates["staled"])

    rejected_feedback_row = None
    if rejected_feedback_cases and feedback_enabled:
        print(
            f"[{progress_label} skill edit{window_suffix}] rejected-edit feedback start "
            f"cases={len(rejected_feedback_cases)}",
            flush=True,
        )
        feedback_text = _extract_rejected_edit_feedback(
            teacher=teacher,
            dataset=dataset,
            rejected_cases=rejected_feedback_cases[
                : max(0, int(getattr(args, "l2_selection_feedback_max_negative", SELECTION_FEEDBACK_MAX_NEGATIVE)))
            ],
        )
        rejected_feedback_row = _append_selection_feedback(
            rows=selection_feedback_rows,
            feedback=feedback_text,
            round_idx=round_idx,
            source="l2_judge_reject",
        )
        print(
            f"[{progress_label} skill edit{window_suffix}] rejected-edit feedback done "
            f"written={'yes' if rejected_feedback_row is not None else 'no'}",
            flush=True,
        )

    selection_enabled = True if force_selection_enabled is None else bool(force_selection_enabled)
    selection_info: Dict[str, Any] = {
        "enabled": selection_enabled,
        "task_count": len(selection_tasks),
        "decision": "skipped",
        "reason": "no accepted edits",
    }
    committed_edits: List[Dict[str, Any]] = []
    updated_skill_md = skill_md
    commit_reports: List[Dict[str, Any]] = []
    if accepted_edits:
        if defer_commit:
            print(
                f"[{progress_label} skill edit{window_suffix}] commit deferred "
                f"accepted_edits={len(accepted_edits)}",
                flush=True,
            )
            selection_info = {
                "enabled": selection_enabled,
                "task_count": len(selection_tasks),
                "decision": "deferred",
                "reason": "epoch global merge pending",
            }
            committed_edits = []
            commit_reports.append(
                {
                    "applied": False,
                    "reason": "commit_deferred_for_epoch_global_merge",
                    "accepted_edit_count": len(accepted_edits),
                }
            )
        else:
            updated_skill_md, commit_reports, applied_edits, failed_commit_edits = (
                _commit_markdown_edits_sequential(skill_md, accepted_edits)
            )
            for failed in failed_commit_edits:
                edit = failed.get("edit") or {}
                updates = _mark_evidence_attempts(
                    evidence_rows,
                    _evidence_id_list(edit.get("evidence_ids")),
                    round_idx=round_idx,
                    outcome="reject",
                    max_rejects=max_rejects,
                    count_attempt=False,
                )
                attempted_evidence_ids.update(updates["attempted"])
                rejected_evidence_ids.update(updates["rejected"])
                staled_evidence_ids.update(updates["staled"])
            if applied_edits:
                if defer_selection:
                    print(
                        f"[{progress_label} skill edit{window_suffix}] selection deferred "
                        f"applied_edits={len(applied_edits)}",
                        flush=True,
                    )
                    selection_info = {
                        "enabled": selection_enabled,
                        "task_count": len(selection_tasks),
                        "decision": "deferred",
                        "reason": "epoch validation pending",
                    }
                    committed_edits = list(applied_edits)
                else:
                    selection_info = _selection_gate(
                        runner=runner,
                        out=out,
                        current_skill_path=current_skill_path,
                        candidate_skill_md=updated_skill_md,
                        selection_tasks=selection_tasks,
                        round_idx=round_idx,
                        enabled=selection_enabled,
                        args=args,
                    )
                    selection_feedback_row = None
                    positive_cases: List[Dict[str, Any]] = []
                    negative_cases: List[Dict[str, Any]] = []
                    selection_info["positive_case_count"] = len(positive_cases)
                    selection_info["negative_case_count"] = len(negative_cases)
                    if selection_feedback_row is not None:
                        selection_info["feedback_id"] = selection_feedback_row.get("feedback_id")
                    selection_info["baseline_task_result_count"] = len(selection_info.get("baseline_task_results") or [])
                    selection_info["candidate_task_result_count"] = len(selection_info.get("candidate_task_results") or [])
                    selection_info.pop("baseline_task_results", None)
                    selection_info.pop("candidate_task_results", None)
                    if selection_info.get("decision") == "commit":
                        current_skill_path.write_text(updated_skill_md, encoding="utf-8")
                        if not selection_info.get("enabled", True):
                            _invalidate_selection_baseline(out)
                        reject_cooldown_pool_size = 0
                        committed_edits = list(applied_edits)
                        for edit in committed_edits:
                            updates = _mark_evidence_attempts(
                                evidence_rows,
                                _evidence_id_list(edit.get("evidence_ids")),
                                round_idx=round_idx,
                                outcome="accept",
                                max_rejects=max_rejects,
                                count_attempt=False,
                            )
                            attempted_evidence_ids.update(updates["attempted"])
                            accepted_evidence_ids.update(updates["accepted"])
                    else:
                        updated_skill_md = skill_md
                        current_counts = _evidence_pool_counts(evidence_pool)
                        reject_cooldown_pool_size = current_counts["active"]
                        commit_reports.append(
                            {
                                "applied": False,
                                "reason": "selection_gate_rejected",
                                "selection": selection_info,
                            }
                        )
                        for edit in applied_edits:
                            updates = _mark_evidence_attempts(
                                evidence_rows,
                                _evidence_id_list(edit.get("evidence_ids")),
                                round_idx=round_idx,
                                outcome="reject",
                                max_rejects=max_rejects,
                                count_attempt=False,
                            )
                            attempted_evidence_ids.update(updates["attempted"])
                            rejected_evidence_ids.update(updates["rejected"])
                            staled_evidence_ids.update(updates["staled"])
            else:
                selection_info = {
                    "enabled": selection_enabled,
                    "task_count": len(selection_tasks),
                    "decision": "skipped",
                    "reason": "accepted edits did not apply",
                }

    l2_manager_row = {
        "dataset": dataset,
        "round": round_idx,
        **({"epoch_index": round_idx} if str(l2_trigger_reason or "") == "epoch_boundary" else {}),
        "system_prompt": l2_prompt,
        "input": l2_input,
        "output": {
            "reasoning": str(l2_raw.get("reasoning") or ""),
            "edits": accepted_edits if defer_commit else committed_edits,
        },
    }
    archived_ids = set()
    if not defer_selection:
        for edit in committed_edits:
            archived_ids.update(_evidence_id_list(edit.get("evidence_ids")))
        if archived_ids:
            _archive_evidence_rows(evidence_rows, sorted(archived_ids))
            evidence_pool[:] = [row for row in evidence_pool if str(row.get("evidence_id")) not in archived_ids]
    _normalize_evidence_runtime_states(evidence_pool)
    pool_counts_after = _evidence_pool_counts(evidence_pool)
    context_evidence_ids = sorted(set(str(eid) for eid in selected_evidence_ids) - set(str(eid) for eid in cited_evidence_ids))
    print(
        f"[{progress_label} skill edit{window_suffix}] window done "
        f"candidate_edits={len(candidate_edits)} accepted_edits={len(accepted_edits)} "
        f"committed_edits={len(committed_edits)} rejected_evidence={len(rejected_evidence_ids)} "
        f"staled_evidence={len(staled_evidence_ids)} "
        f"rule_ignored_accepts={rule_ignored_accepts} "
        f"rule_ignored_defers={rule_ignored_defers} "
        f"reflection_attempts={reflection_attempts} "
        f"reflection_accepts={reflection_accepts} "
        f"reflection_failures={reflection_failures} judge_rejects={judge_rejects}",
        flush=True,
    )
    l2_info = {
        "triggered": True,
        "trigger_reason": l2_trigger_reason,
        "candidate_count": len(candidate_edits),
        "accepted_count": len(accepted_edits),
        "committed_count": len(committed_edits) if not defer_selection else 0,
        "candidate_committed_count": len(accepted_edits) if defer_commit else (len(committed_edits) if defer_selection else len(committed_edits)),
        "commit_deferred": bool((defer_selection or defer_commit) and (committed_edits or accepted_edits)),
        "rejected_feedback_case_count": len(rejected_feedback_cases),
        "rejected_feedback_id": rejected_feedback_row.get("feedback_id") if rejected_feedback_row else None,
        "archived_evidence_ids": sorted(archived_ids),
        "evidence_pool_size_after": len(evidence_pool),
        "evidence_pool": pool_counts_after,
        "evidence_selection": evidence_selection_info,
        "l2_manager_error": l2_manager_error,
        "attempted_evidence_ids": sorted(attempted_evidence_ids),
        "accepted_evidence_ids": sorted(accepted_evidence_ids),
        "rejected_evidence_ids": sorted(rejected_evidence_ids),
        "staled_evidence_ids": sorted(staled_evidence_ids),
        "deferred_evidence_ids": sorted(deferred_evidence_ids),
        "deferred_evidence_reasons": {
            evidence_id: (
                policy_deferred_evidence_reasons[evidence_id]
                if evidence_id in policy_deferred_evidence_reasons
                else "reflection_failed"
                if evidence_id in reflection_failed_evidence_ids
                else "reflection_narrowed"
                if evidence_id in reflection_narrowed_evidence_ids
                else next(
                    (
                        str(row.get("defer_reason") or "deferred")
                        for row in evidence_rows
                        if str(row.get("evidence_id") or "") == evidence_id
                    ),
                    "deferred",
                )
            )
            for evidence_id in sorted(deferred_evidence_ids)
        },
        "rule_ignored_accepts": rule_ignored_accepts,
        "rule_ignored_defers": rule_ignored_defers,
        "rule_ignored_deferred_evidence_ids": sorted(
            rule_ignored_deferred_evidence_ids
        ),
        "reflection_attempts": reflection_attempts,
        "reflection_accepts": reflection_accepts,
        "reflection_failures": reflection_failures,
        "judge_rejects": judge_rejects,
        "cited_evidence_ids": sorted(cited_evidence_ids),
        "context_evidence_ids": context_evidence_ids,
        "omitted_evidence_ids": sorted(omitted_evidence_ids),
        "fallback_evidence_ids": sorted(fallback_evidence_ids),
        "raw_edit_count": len(raw_patch_items),
        "raw_valid_linked_edit_count": raw_valid_linked_edit_count,
        "raw_rejected_edit_count": len(raw_patch_items) - raw_valid_linked_edit_count,
        "raw_rejection_counts": raw_rejection_counts,
        "raw_shape_warning_counts": raw_shape_warning_counts,
        "raw_unknown_evidence_ids": sorted(raw_unknown_evidence_ids),
        "window_coverage_input_evidence": len(selected_evidence_ids),
        "window_coverage_covered_evidence": len(cited_evidence_ids),
        "window_coverage_omitted_evidence": len(omitted_evidence_ids),
        "window_coverage_fallback_evidence": len(fallback_evidence_ids),
        "accepted_edits": accepted_edits,
        "committed_edits": committed_edits,
        "decisions": l2_decisions,
        "commit_reports": commit_reports,
        "selection": selection_info,
    }
    return updated_skill_md, l2_info, reject_cooldown_pool_size, l2_manager_row


def _run_epoch_l2_update_merge_rank(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    epoch_idx: int,
    epoch_rollout_task_count: int,
    current_skill_path: Path,
    validation_skill_path: Optional[Path] = None,
    working_branch_state: Optional[Dict[str, Any]] = None,
    working_branch_state_path: Optional[Path] = None,
    evidence_pool: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l2_manager_rows: List[Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    selection_tasks: List[Dict[str, Any]],
    reject_cooldown_pool_size: int = 0,
) -> Tuple[Dict[str, Any], int]:
    working_branch_enabled = working_branch_state is not None
    validation_skill_path = validation_skill_path or current_skill_path
    if working_branch_enabled and working_branch_state_path is None:
        raise ValueError("working_branch_state_path is required in evidence working branch mode")
    if working_branch_enabled:
        resumed = _resume_working_branch_validation_stage(
            dataset=dataset,
            teacher=teacher,
            runner=runner,
            out=out,
            args=args,
            epoch_idx=epoch_idx,
            epoch_rollout_task_count=epoch_rollout_task_count,
            current_skill_path=current_skill_path,
            validation_skill_path=validation_skill_path,
            working_branch_state=working_branch_state,
            working_branch_state_path=working_branch_state_path,
            evidence_pool=evidence_pool,
            evidence_rows=evidence_rows,
            replay_source_by_evidence_id=replay_source_by_evidence_id,
            l2_manager_rows=l2_manager_rows,
            l2_candidate_rows=l2_candidate_rows,
            l2_judge_rows=l2_judge_rows,
            replay_rows=replay_rows,
            selection_feedback_rows=selection_feedback_rows,
            reject_cooldown_pool_size=reject_cooldown_pool_size,
        )
        if resumed is not None:
            return resumed
    coverage_mode = _resolve_epoch_skill_update_mode(args) == "cluster_step"
    configured_epoch_l2_windows = max(0, int(getattr(args, "epoch_l2_windows", 4) or 0))
    epoch_l2_windows = 0 if coverage_mode else configured_epoch_l2_windows
    epoch_evidence_counts_before = _evidence_pool_counts(evidence_pool)
    max_rejects = max(
        1,
        int(getattr(args, "l2_max_rejects_per_evidence", L2_MAX_REJECTS_PER_EVIDENCE) or L2_MAX_REJECTS_PER_EVIDENCE),
    )
    epoch_pool_ids = [
        str(row.get("evidence_id") or "")
        for row in evidence_pool or []
        if str(row.get("evidence_id") or "")
    ]
    base_skill_md = current_skill_path.read_text(encoding="utf-8")
    candidate_skill_md = base_skill_md
    pending_l2_manager_rows: List[Dict[str, Any]] = []
    epoch_window_infos: List[Dict[str, Any]] = []
    epoch_accepted_edits: List[Dict[str, Any]] = []
    tentative_committed_edits: List[Dict[str, Any]] = []
    archived_ids = set()
    accepted_evidence_ids = set()
    rejected_evidence_ids = set()
    staled_evidence_ids = set()
    consumed_evidence_ids = set()
    epoch_deferred_evidence_ids = set()
    deferred_evidence_fates: Dict[str, str] = {}
    epoch_merge_info: Dict[str, Any] = {
        "used_llm": False,
        "reasoning": "no accepted edits",
        "candidate_count": 0,
        "selected_edit_count": 0,
        "epoch_edit_budget": int(getattr(args, "epoch_edit_budget", DEFAULT_EPOCH_EDIT_BUDGET) or 0),
    }
    epoch_final_replay_budget = _int_arg(
        getattr(
            args,
            "epoch_final_replay_ranges_per_edit",
            DEFAULT_EPOCH_FINAL_REPLAY_RANGES_PER_EDIT,
        ),
        DEFAULT_EPOCH_FINAL_REPLAY_RANGES_PER_EDIT,
    )
    if _no_replay_enabled(args):
        epoch_final_replay_budget = 0
    epoch_final_replay_enabled = epoch_final_replay_budget != 0 and not working_branch_enabled
    epoch_final_replay_info: Dict[str, Any] = {
        "enabled": epoch_final_replay_enabled,
        "final_replay_edits": 0,
        "final_replay_accepts": 0,
        "final_replay_rejects": 0,
        "final_replay_skipped": 0,
        "final_replay_bypassed": 0,
        "epoch_final_replay_ranges_per_edit": epoch_final_replay_budget,
        **(
            {
                "reason": (
                    "replaced_by_working_branch_post_reject_selection_no_replay"
                    if _no_replay_enabled(args)
                    else "replaced_by_working_branch_post_reject_replay"
                )
            }
            if working_branch_enabled
            else (
                {"reason": "disabled_by_no_replay"}
                if _no_replay_enabled(args)
                else {}
            )
        ),
    }
    working_branch_transition_info: Dict[str, Any] = {
        "enabled": working_branch_enabled,
        "decision": "unchanged",
    }
    working_branch_post_reject_info: Dict[str, Any] = {
        "enabled": False,
        "mode": _replay_mode(args),
        "action_replay_count": 0,
    }
    working_branch_provisional_edits: List[Dict[str, Any]] = []
    working_branch_pending_transition = ""
    epoch_coverage_repair_info: Dict[str, Any] = {
        "enabled": bool(coverage_mode),
        "repair_merge_input_evidence": 0,
        "repair_merge_edit_count": 0,
        "repair_applied_edit_count": 0,
        "repair_failed_edit_count": 0,
        "repair_fallback_evidence_ids": [],
        "repair_covered_evidence_ids": [],
    }
    baseline_coverage_audit_info: Dict[str, Any] = {
        "stage": "current_skill",
        "evidence_count": 0,
        "covered_count": 0,
        "uncovered_count": 0,
    }
    candidate_coverage_audit_info: Dict[str, Any] = {
        "stage": "candidate_before_repair",
        "evidence_count": 0,
        "covered_count": 0,
        "uncovered_count": 0,
    }
    final_coverage_audit_info: Dict[str, Any] = {
        "stage": "candidate_after_final_replay",
        "evidence_count": 0,
        "covered_count": 0,
        "uncovered_count": 0,
    }
    baseline_covered_evidence_ids: set[str] = set()
    final_candidate_covered_evidence_ids: set[str] = set()
    global_commit_reports: List[Dict[str, Any]] = []
    deferred_evidence_ids = set()
    epoch_candidate_edit_count = 0
    epoch_accepted_edit_count = 0
    epoch_rule_ignored_accepts = 0
    epoch_rule_ignored_defers = 0
    policy_defer_limit_info: Dict[str, Any] = {
        "enabled": False,
        "max_policy_defer_epochs": _int_arg(
            getattr(args, "working_branch_max_policy_defer_epochs", -1), -1
        ),
        "counted_evidence_ids": [],
        "staled_evidence_ids": [],
    }
    epoch_reflection_attempts = 0
    epoch_reflection_accepts = 0
    epoch_reflection_failures = 0
    epoch_judge_rejects = 0
    candidate_edit_counter = 1
    protected_provisional_evidence_ids = set(
        provisional_evidence_ids(working_branch_state)
        if working_branch_enabled
        else []
    )
    max_l2_window = int(
        getattr(
            args,
            "l2_max_evidence_per_window",
            getattr(args, "evidence_window_size", 20),
        )
        or getattr(args, "evidence_window_size", 20)
    )
    l2_evidence_grouping = _resolve_l2_evidence_window_grouping(
        dataset,
        getattr(args, "l2_evidence_window_grouping", "auto"),
    )
    current_audit_pool = [
        row for row in evidence_pool
        if str(row.get("evidence_id") or "") not in protected_provisional_evidence_ids
    ]
    if coverage_mode and current_audit_pool:
        current_audit_skill_md = (
            validation_skill_path.read_text(encoding="utf-8")
            if working_branch_enabled
            else base_skill_md
        )
        baseline_covered, _, baseline_coverage_audit_info = _run_evidence_skill_coverage_audit(
            teacher=teacher,
            dataset=dataset,
            skill_md=current_audit_skill_md,
            evidence_cards=current_audit_pool,
            args=args,
            stage="current_skill",
        )
        if working_branch_enabled:
            baseline_coverage_audit_info["skill_role"] = "validated_skill"
        baseline_covered_evidence_ids.update(baseline_covered)
        archived_ids.update(
            _archive_evidence_rows_with_reason(
                evidence_rows,
                baseline_covered,
                reason="already_covered_current",
            )
        )
        if baseline_covered:
            evidence_pool[:] = [
                row
                for row in evidence_pool
                if str(row.get("evidence_id") or "") not in baseline_covered_evidence_ids
            ]
        print(
            f"[epoch {epoch_idx + 1} coverage audit] stage=current_skill "
            f"coverage_mode={baseline_coverage_audit_info.get('coverage_mode', 'semantic')} "
            f"covered={len(baseline_covered)} "
            f"uncovered={baseline_coverage_audit_info.get('uncovered_count', 0)} "
            f"fallback={len(baseline_coverage_audit_info.get('fallback_evidence_ids') or [])}",
            flush=True,
        )
    planning_pool = [
        row for row in evidence_pool
        if str(row.get("evidence_id") or "") not in protected_provisional_evidence_ids
    ]
    semantic_options = (
        _semantic_embedding_options_from_args(args, out)
        if l2_evidence_grouping == "semantic"
        else None
    )
    if working_branch_enabled:
        planned_l2_windows = []
        for evidence_mode in ("trajectory", "contrastive"):
            mode_pool = [
                row for row in planning_pool
                if str(row.get("evidence_mode") or "trajectory") == evidence_mode
            ]
            mode_plans = _plan_source_split_l2_evidence_windows(
                mode_pool,
                max_window=max_l2_window,
                max_windows=epoch_l2_windows,
                grouping=l2_evidence_grouping,
                semantic_options=semantic_options,
            )
            for _, info in mode_plans:
                info["evidence_mode"] = evidence_mode
            planned_l2_windows.extend(mode_plans)
        for plan_index, (_, info) in enumerate(planned_l2_windows):
            info["planned_window_index"] = plan_index
            info["planned_window_count"] = len(planned_l2_windows)
    else:
        planned_l2_windows = _plan_source_split_l2_evidence_windows(
            planning_pool,
            max_window=max_l2_window,
            max_windows=epoch_l2_windows,
            grouping=l2_evidence_grouping,
            semantic_options=semantic_options,
        )
    planned_sizes = [len(planned_pool) for planned_pool, _ in planned_l2_windows]
    window_limit_label = "all" if epoch_l2_windows <= 0 else str(epoch_l2_windows)
    source_window_counts: Dict[str, int] = {}
    source_evidence_counts: Dict[str, int] = {}
    for planned_pool, planned_info in planned_l2_windows:
        source = str((planned_info or {}).get("source_type") or "mixed")
        source_window_counts[source] = source_window_counts.get(source, 0) + 1
        source_evidence_counts[source] = source_evidence_counts.get(source, 0) + len(planned_pool)
    planned_group_summary = ""
    if l2_evidence_grouping == "semantic":
        planned_semantic_group_counts = [
            int((planned_info or {}).get("selected_semantic_group_count") or 0)
            for _, planned_info in planned_l2_windows
        ]
        planned_group_summary = f" semantic_group_counts={planned_semantic_group_counts}"
    elif l2_evidence_grouping == "round":
        planned_round_group_counts = [
            int((planned_info or {}).get("selected_round_group_count") or 0)
            for _, planned_info in planned_l2_windows
        ]
        planned_group_summary = f" round_group_counts={planned_round_group_counts}"
    print(
        f"[epoch {epoch_idx + 1} skill edit] start "
        f"train_tasks={epoch_rollout_task_count} "
        f"evidence_active_before_audit={epoch_evidence_counts_before.get('active', 0)} "
        f"evidence_total={epoch_evidence_counts_before.get('total', 0)} "
        f"baseline_covered={len(baseline_covered_evidence_ids)} "
        f"evidence_planned={sum(planned_sizes)} "
        f"grouping={l2_evidence_grouping} max_window={max_l2_window} "
        f"planned_windows={len(planned_l2_windows)}/{window_limit_label} "
        f"window_sizes={planned_sizes}{planned_group_summary} "
        f"source_windows={source_window_counts} source_evidence={source_evidence_counts}",
        flush=True,
    )
    if l2_evidence_grouping == "semantic" and planned_l2_windows:
        first_info = planned_l2_windows[0][1] or {}
        print(
            f"[epoch {epoch_idx + 1} skill edit] semantic clusters="
            f"{first_info.get('semantic_cluster_count', 0)} "
            f"method={first_info.get('semantic_clustering_method', 'unknown')} "
            f"model={first_info.get('embedding_model', DEFAULT_EVIDENCE_EMBEDDING_MODEL)} "
            f"cache_hit={first_info.get('embedding_cache_hits', 0)} "
            f"cache_miss={first_info.get('embedding_cache_misses', 0)} "
            f"hdbscan_min_cluster_size={first_info.get('semantic_hdbscan_min_cluster_size', 'n/a')} "
            f"hdbscan_min_samples={first_info.get('semantic_hdbscan_min_samples', 'n/a')} "
            f"noise={first_info.get('semantic_hdbscan_noise_count', 'n/a')} "
            f"cluster_sizes={first_info.get('semantic_cluster_sizes', [])}",
            flush=True,
        )
        print(
            f"[epoch {epoch_idx + 1} skill edit] semantic windows "
            f"natural_windows={first_info.get('semantic_natural_window_count', len(planned_l2_windows))} "
            f"target_windows={first_info.get('semantic_target_window_count', epoch_l2_windows)} "
            f"capacity_floor={first_info.get('semantic_capacity_floor_window_count', len(planned_l2_windows))} "
            f"final_windows={first_info.get('semantic_final_window_count', len(planned_l2_windows))} "
            f"reason={first_info.get('semantic_window_budget_reason', 'n/a')} "
            f"window_sizes={first_info.get('semantic_final_window_sizes', planned_sizes)} "
            f"underfilled_windows={first_info.get('semantic_underfilled_window_count', 0)} "
            f"dropped_evidence={first_info.get('semantic_dropped_small_evidence_count', 0)}",
            flush=True,
        )
    for window_idx, (planned_pool, planned_selection_info) in enumerate(planned_l2_windows):
        planned_selection_info = planned_selection_info or {}
        if l2_evidence_grouping == "semantic":
            window_group_summary = (
                f"source={planned_selection_info.get('source_type', 'mixed')} "
                f"semantic_groups={planned_selection_info.get('selected_semantic_group_count', 0)}"
            )
        elif l2_evidence_grouping == "round":
            window_group_summary = (
                f"round_groups={planned_selection_info.get('selected_round_group_count', 0)}"
            )
        else:
            window_group_summary = "card_window"
        print(
            f"[epoch {epoch_idx + 1} skill edit] window {window_idx + 1}/{len(planned_l2_windows)} start "
            f"evidence={len(planned_pool)} {window_group_summary} "
            f"sim_avg={planned_selection_info.get('similarity_avg', 'n/a')}",
            flush=True,
        )
        _, window_info, reject_cooldown_pool_size, l2_manager_row = _run_l2_update_window(
            dataset=dataset,
            teacher=teacher,
            runner=runner,
            out=out,
            args=args,
            round_idx=epoch_idx,
            skill_md=base_skill_md,
            current_skill_path=current_skill_path,
            evidence_pool=evidence_pool,
            evidence_rows=evidence_rows,
            replay_source_by_evidence_id=replay_source_by_evidence_id,
            raw_by_id={},
            l2_candidate_rows=l2_candidate_rows,
            l2_judge_rows=l2_judge_rows,
            replay_rows=replay_rows,
            selection_feedback_rows=selection_feedback_rows,
            selection_tasks=selection_tasks,
            l2_trigger_reason="epoch_boundary",
            reject_cooldown_pool_size=reject_cooldown_pool_size,
            defer_selection=True,
            force_selection_enabled=True,
            exclude_evidence_ids=sorted(deferred_evidence_ids),
            preselected_pool=planned_pool,
            preselected_selection_info=planned_selection_info,
            defer_commit=True,
        )
        window_info["epoch_index"] = epoch_idx
        window_info["epoch_window_index"] = window_idx
        epoch_window_infos.append(window_info)
        pending_l2_manager_rows.append(l2_manager_row)
        if not window_info.get("triggered"):
            print(
                f"[epoch {epoch_idx + 1} skill edit] window {window_idx + 1}/{len(planned_l2_windows)} "
                f"not triggered reason={window_info.get('reason')}",
                flush=True,
            )
            break
        epoch_candidate_edit_count += int(window_info.get("candidate_count") or 0)
        epoch_rule_ignored_accepts += int(window_info.get("rule_ignored_accepts") or 0)
        epoch_rule_ignored_defers += int(window_info.get("rule_ignored_defers") or 0)
        epoch_reflection_attempts += int(window_info.get("reflection_attempts") or 0)
        epoch_reflection_accepts += int(window_info.get("reflection_accepts") or 0)
        epoch_reflection_failures += int(window_info.get("reflection_failures") or 0)
        epoch_judge_rejects += int(window_info.get("judge_rejects") or 0)
        window_accepted_edits = list(window_info.get("accepted_edits") or [])
        epoch_accepted_edit_count += len(window_accepted_edits)
        rejected_evidence_ids.update(window_info.get("rejected_evidence_ids") or [])
        staled_evidence_ids.update(window_info.get("staled_evidence_ids") or [])
        window_deferred_reasons = window_info.get("deferred_evidence_reasons") or {}
        for evidence_id in window_info.get("deferred_evidence_ids") or []:
            deferred_evidence_fates[str(evidence_id)] = str(
                window_deferred_reasons.get(str(evidence_id))
                or "fallback_proposed_edit_rejected"
            )
        print(
            f"[epoch {epoch_idx + 1} skill edit] window {window_idx + 1}/{len(planned_l2_windows)} done "
            f"candidate_edits={window_info.get('candidate_count', 0)} "
            f"accepted_edits={len(window_accepted_edits)} "
            f"cited_evidence={len(window_info.get('cited_evidence_ids') or [])}",
            flush=True,
        )
        for edit in window_accepted_edits:
            edit["_candidate_edit_id"] = str(edit.get("_candidate_edit_id") or f"C{candidate_edit_counter:06d}")
            edit["_source_window_index"] = window_idx
            candidate_edit_counter += 1
            for decision in window_info.get("decisions") or []:
                if (decision.get("edit") or {}) == {k: v for k, v in edit.items() if not str(k).startswith("_")}:
                    edit["_source_decision_reason"] = decision.get("reason")
                    break
        epoch_accepted_edits.extend(window_accepted_edits)
        evidence_selection = window_info.get("evidence_selection") or {}
        deferred_evidence_ids.update(evidence_selection.get("selected_group_evidence_ids") or [])
        deferred_evidence_ids.update(evidence_selection.get("selected_evidence_ids") or [])
        for edit in window_accepted_edits:
            deferred_evidence_ids.update(_evidence_id_list(edit.get("evidence_ids")))
    if not epoch_accepted_edits:
        print(
            f"[epoch {epoch_idx + 1} skill edit] no accepted edits from planned windows",
            flush=True,
        )

    selection_info: Dict[str, Any] = {
        "enabled": True,
        "task_count": len(selection_tasks),
        "decision": "skipped",
        "reason": "no epoch candidate edits",
    }
    epoch_validation_decision = "skipped"
    if epoch_accepted_edits:
        print(
            f"[epoch {epoch_idx + 1} skill edit] global merge start "
            f"accepted_edits={len(epoch_accepted_edits)} "
            f"hard_rank={'no' if coverage_mode else 'yes'} "
            f"epoch_edit_budget={getattr(args, 'epoch_edit_budget', DEFAULT_EPOCH_EDIT_BUDGET)}",
            flush=True,
        )
        merged_edits, epoch_merge_info = _merge_epoch_accepted_edits(
            teacher=teacher,
            dataset=dataset,
            current_skill_md=base_skill_md,
            accepted_edits=epoch_accepted_edits,
            evidence_rows=evidence_rows,
            selection_feedback_rows=selection_feedback_rows,
            args=args,
        )
        if _working_branch_defer_uncovered_enabled(dataset, args):
            for evidence_id in epoch_merge_info.get(
                "deferred_uncovered_source_evidence_ids"
            ) or []:
                deferred_evidence_ids.add(str(evidence_id))
                deferred_evidence_fates[str(evidence_id)] = (
                    "global_merge_not_covered"
                )
            for evidence_id in epoch_merge_info.get(
                "deferred_omitted_source_evidence_ids"
            ) or []:
                deferred_evidence_ids.add(str(evidence_id))
                deferred_evidence_fates[str(evidence_id)] = "global_merge_omitted"
        if working_branch_enabled:
            evidence_by_id_for_budget = {
                str(row.get("evidence_id") or ""): row
                for row in evidence_rows or []
                if str(row.get("evidence_id") or "")
            }
            merged_edits = _split_working_branch_correction_edits(
                merged_edits,
                evidence_by_id=evidence_by_id_for_budget,
            )
            merged_pool_count = int(
                epoch_merge_info.get("merged_pool_count", len(merged_edits))
            )
            ranked_edit_count = int(
                epoch_merge_info.get("ranked_edit_count", len(merged_edits))
            )
            prior_dropped_edit_ids = _evidence_id_list(
                epoch_merge_info.get("dropped_edit_ids")
            )
            merged_edits, dropped_budget_edits = _limit_working_branch_epoch_edits(
                merged_edits,
                evidence_by_id=evidence_by_id_for_budget,
                args=args,
                info=epoch_merge_info,
            )
            epoch_merge_info.update(
                {
                    "merged_pool_count": merged_pool_count,
                    "ranked_edit_count": ranked_edit_count,
                    "selected_edit_count": len(merged_edits),
                    "selected_edit_ids": _covered_source_edit_ids(merged_edits),
                    "selected_merged_edit_ids": [
                        str(edit.get("_merged_edit_id") or "")
                        for edit in merged_edits
                        if str(edit.get("_merged_edit_id") or "")
                    ],
                    "dropped_edit_ids": _evidence_id_list(
                        prior_dropped_edit_ids
                        + _covered_source_edit_ids(dropped_budget_edits)
                    ),
                }
            )
        sent_candidate_count = (
            epoch_merge_info.get("rank_candidate_count")
            or epoch_merge_info.get(
                "sent_candidate_count", epoch_merge_info.get("candidate_count")
            )
        )
        print(
            f"[epoch {epoch_idx + 1} skill edit] global merge done "
            f"selected_edits={len(merged_edits)} "
            f"merged_pool={epoch_merge_info.get('merged_pool_count', 'n/a')} "
            f"ranked={epoch_merge_info.get('ranked_edit_count', len(merged_edits))} "
            f"merge_llm={epoch_merge_info.get('merge_used_llm', False)} "
            f"rank_llm={epoch_merge_info.get('rank_used_llm', False)} "
            f"sent_candidates={sent_candidate_count}",
            flush=True,
        )
        print(
            f"[epoch {epoch_idx + 1} skill edit] global commit apply start "
            f"merged_edits={len(merged_edits)}",
            flush=True,
        )
        candidate_skill_md, global_commit_reports, applied_edits, failed_global_edits = (
            _commit_markdown_edits_sequential(base_skill_md, merged_edits)
        )
        tentative_committed_edits = list(applied_edits)
        print(
            f"[epoch {epoch_idx + 1} skill edit] global commit apply done "
            f"applied_edits={len(applied_edits)} failed_edits={len(failed_global_edits)}",
            flush=True,
        )
        for failed in failed_global_edits:
            edit = failed.get("edit") or {}
            failed_eids = _evidence_id_list(edit.get("evidence_ids"))
            if coverage_mode:
                _defer_evidence_rows(evidence_rows, failed_eids, reason="global_apply_failed")
                for eid in failed_eids:
                    deferred_evidence_fates[str(eid)] = "global_apply_failed"
            else:
                updates = _mark_evidence_attempts(
                    evidence_rows,
                    failed_eids,
                    round_idx=epoch_idx,
                    outcome="reject",
                    max_rejects=max_rejects,
                    count_attempt=False,
                )
                rejected_evidence_ids.update(updates["rejected"])
                staled_evidence_ids.update(updates["staled"])
        if coverage_mode:
            evidence_by_id_full = {
                str(row.get("evidence_id") or ""): row
                for row in evidence_rows or []
                if str(row.get("evidence_id") or "")
            }
            blocked_ids = (
                set(rejected_evidence_ids)
                | set(staled_evidence_ids)
                | set(archived_ids)
                | set(deferred_evidence_fates)
            )
            auditable_cards = []
            for evidence_id in epoch_pool_ids:
                if not evidence_id or evidence_id in blocked_ids:
                    continue
                row = evidence_by_id_full.get(evidence_id)
                if not row:
                    continue
                _normalize_evidence_runtime_state(row)
                status = str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower()
                if bool(row.get("archived")) or status in {
                    EVIDENCE_STATUS_ARCHIVED,
                    EVIDENCE_STATUS_STALE,
                    EVIDENCE_STATUS_CONSUMED,
                }:
                    continue
                auditable_cards.append(row)
            _, uncovered_ids, candidate_coverage_audit_info = _run_evidence_skill_coverage_audit(
                teacher=teacher,
                dataset=dataset,
                skill_md=candidate_skill_md,
                evidence_cards=auditable_cards,
                args=args,
                stage="candidate_before_repair",
            )
            print(
                f"[epoch {epoch_idx + 1} coverage audit] stage=candidate_before_repair "
                f"covered={candidate_coverage_audit_info.get('covered_count', 0)} "
                f"uncovered={len(uncovered_ids)} "
                f"fallback={len(candidate_coverage_audit_info.get('fallback_evidence_ids') or [])}",
                flush=True,
            )
            if uncovered_ids:
                uncovered_cards = [evidence_by_id_full[eid] for eid in uncovered_ids if eid in evidence_by_id_full]
                repair_budget = (
                    max(
                        0,
                        _int_arg(
                            getattr(args, "epoch_edit_budget", DEFAULT_EPOCH_EDIT_BUDGET),
                            DEFAULT_EPOCH_EDIT_BUDGET,
                        )
                        - len(applied_edits),
                    )
                    if working_branch_enabled
                    else None
                )
                print(
                    f"[epoch {epoch_idx + 1} repair] start "
                    f"uncovered_evidence={len(uncovered_cards)} "
                    f"remaining_budget={repair_budget if repair_budget is not None else 'unlimited'}",
                    flush=True,
                )
                if repair_budget == 0:
                    repair_edits = []
                    epoch_coverage_repair_info = {
                        "enabled": False,
                        "reason": "epoch_edit_budget_exhausted",
                        "repair_merge_input_evidence": len(uncovered_cards),
                        "repair_merge_edit_count": 0,
                        "repair_fallback_evidence_ids": [],
                        "budget_applied": True,
                        "repair_budget": 0,
                        "repair_dropped_edit_count": 0,
                    }
                    for evidence_id in uncovered_ids:
                        _defer_evidence_rows(
                            evidence_rows,
                            [evidence_id],
                            reason="epoch_edit_budget_exhausted",
                        )
                        deferred_evidence_fates[str(evidence_id)] = "epoch_edit_budget_exhausted"
                else:
                    repair_edits, epoch_coverage_repair_info = _run_epoch_coverage_repair_merge(
                        teacher=teacher,
                        dataset=dataset,
                        candidate_skill_md=candidate_skill_md,
                        uncovered_cards=uncovered_cards,
                        evidence_rows=evidence_rows,
                        args=args,
                    )
                if working_branch_enabled and repair_budget:
                    repair_edits = _split_working_branch_correction_edits(
                        repair_edits,
                        evidence_by_id=evidence_by_id_full,
                    )
                    repair_edits, dropped_repair_edits = _limit_working_branch_epoch_edits(
                        repair_edits,
                        evidence_by_id=evidence_by_id_full,
                        args=args,
                        budget=repair_budget,
                    )
                    dropped_repair_ids = {
                        evidence_id
                        for edit in dropped_repair_edits
                        for evidence_id in _evidence_id_list(edit.get("evidence_ids"))
                    }
                    for evidence_id in sorted(dropped_repair_ids):
                        _defer_evidence_rows(
                            evidence_rows,
                            [evidence_id],
                            reason="epoch_edit_budget_exhausted",
                        )
                        deferred_evidence_fates[str(evidence_id)] = "epoch_edit_budget_exhausted"
                    epoch_coverage_repair_info.update(
                        {
                            "budget_applied": True,
                            "repair_budget": repair_budget,
                            "repair_dropped_edit_count": len(dropped_repair_edits),
                        }
                    )
                if _working_branch_defer_uncovered_enabled(dataset, args):
                    for evidence_id in epoch_coverage_repair_info.get(
                        "deferred_uncovered_evidence_ids"
                    ) or []:
                        deferred_evidence_ids.add(str(evidence_id))
                        deferred_evidence_fates[str(evidence_id)] = (
                            "coverage_repair_not_covered"
                        )
                    for evidence_id in epoch_coverage_repair_info.get(
                        "deferred_omitted_evidence_ids"
                    ) or []:
                        deferred_evidence_ids.add(str(evidence_id))
                        deferred_evidence_fates[str(evidence_id)] = (
                            "coverage_repair_omitted"
                        )
                else:
                    for evidence_id in epoch_coverage_repair_info.get("repair_omitted_evidence_ids") or []:
                        deferred_evidence_fates[str(evidence_id)] = "repair_omitted"
                print(
                    f"[epoch {epoch_idx + 1} repair] merge done "
                    f"repair_edits={len(repair_edits)} "
                    f"fallback_evidence={len(epoch_coverage_repair_info.get('repair_fallback_evidence_ids') or [])}",
                    flush=True,
                )
                if repair_edits:
                    repaired_skill_md, repair_reports, repair_applied_edits, failed_repair_edits = (
                        _commit_markdown_edits_sequential(candidate_skill_md, repair_edits)
                    )
                    global_commit_reports.append(
                        {
                            "stage": "epoch_coverage_repair_apply",
                            "input_evidence_count": len(uncovered_cards),
                            "repair_edit_count": len(repair_edits),
                            "applied_count": len(repair_applied_edits),
                            "failed_count": len(failed_repair_edits),
                        }
                    )
                    global_commit_reports.extend(repair_reports)
                    candidate_skill_md = repaired_skill_md
                    tentative_committed_edits.extend(repair_applied_edits)
                    for failed in failed_repair_edits:
                        edit = failed.get("edit") or {}
                        failed_eids = _evidence_id_list(edit.get("evidence_ids"))
                        _defer_evidence_rows(evidence_rows, failed_eids, reason="repair_apply_failed")
                        for eid in failed_eids:
                            deferred_evidence_fates[str(eid)] = "repair_apply_failed"
                    epoch_coverage_repair_info.update(
                        {
                            "repair_applied_edit_count": len(repair_applied_edits),
                            "repair_failed_edit_count": len(failed_repair_edits),
                        }
                    )
                    print(
                        f"[epoch {epoch_idx + 1} repair] apply done "
                        f"applied_edits={len(repair_applied_edits)} "
                        f"failed_edits={len(failed_repair_edits)}",
                        flush=True,
                    )
        if working_branch_enabled and tentative_committed_edits:
            tentative_committed_edits = assign_working_edit_ids(
                working_branch_state,
                tentative_committed_edits,
            )
        if tentative_committed_edits and epoch_final_replay_enabled:
            print(
                f"[epoch {epoch_idx + 1} final replay] start "
                f"merged_edits={len(tentative_committed_edits)}",
                flush=True,
            )
            final_accepted_edits, final_rejected_edits, epoch_final_replay_info = (
                _run_epoch_final_replay_filter(
                    dataset=dataset,
                    teacher=teacher,
                    runner=runner,
                    out=out,
                    args=args,
                    epoch_idx=epoch_idx,
                    base_skill_md=base_skill_md,
                    candidate_skill_md=candidate_skill_md,
                    applied_edits=tentative_committed_edits,
                    evidence_rows=evidence_rows,
                    replay_source_by_evidence_id=replay_source_by_evidence_id,
                    l2_judge_rows=l2_judge_rows,
                    replay_rows=replay_rows,
                )
            )
            final_accepted_evidence_ids = set(
                _covered_evidence_ids(final_accepted_edits)
            )
            for rejected_edit in final_rejected_edits:
                rejected_ids = [
                    evidence_id
                    for evidence_id in _evidence_id_list(
                        rejected_edit.get("evidence_ids")
                    )
                    if evidence_id not in final_accepted_evidence_ids
                ]
                if not rejected_ids:
                    continue
                final_replay_deferred = bool(
                    str(rejected_edit.get("_final_replay_decision") or "").lower()
                    == "defer"
                    and str(rejected_edit.get("_final_replay_failure_type") or "").lower()
                    == "rule_ignored"
                    and _defer_rule_ignored_enabled(dataset, args)
                )
                if coverage_mode or final_replay_deferred:
                    reason = (
                        "rule_ignored"
                        if final_replay_deferred
                        else "epoch_final_replay_or_judge_reject"
                    )
                    _defer_evidence_rows(
                        evidence_rows,
                        rejected_ids,
                        reason=reason,
                    )
                    for evidence_id in rejected_ids:
                        deferred_evidence_fates[str(evidence_id)] = reason
                else:
                    updates = _mark_evidence_attempts(
                        evidence_rows,
                        rejected_ids,
                        round_idx=epoch_idx,
                        outcome="reject",
                        max_rejects=max_rejects,
                        count_attempt=False,
                    )
                    rejected_evidence_ids.update(updates["rejected"])
                    staled_evidence_ids.update(updates["staled"])
            if len(final_accepted_edits) != len(tentative_committed_edits):
                candidate_skill_md, filtered_reports, filtered_applied_edits, failed_filter_edits = (
                    _commit_markdown_edits_sequential(base_skill_md, final_accepted_edits)
                )
                global_commit_reports.append(
                    {
                        "stage": "epoch_final_replay_filter",
                        "applied_count": len(filtered_applied_edits),
                        "failed_count": len(failed_filter_edits),
                    }
                )
                global_commit_reports.extend(filtered_reports)
                for failed in failed_filter_edits:
                    edit = failed.get("edit") or {}
                    failed_ids = _evidence_id_list(edit.get("evidence_ids"))
                    if coverage_mode:
                        _defer_evidence_rows(
                            evidence_rows,
                            failed_ids,
                            reason="epoch_final_rebuild_apply_failed",
                        )
                        for evidence_id in failed_ids:
                            deferred_evidence_fates[str(evidence_id)] = "epoch_final_rebuild_apply_failed"
                    else:
                        updates = _mark_evidence_attempts(
                            evidence_rows,
                            failed_ids,
                            round_idx=epoch_idx,
                            outcome="reject",
                            max_rejects=max_rejects,
                            count_attempt=False,
                        )
                        rejected_evidence_ids.update(updates["rejected"])
                        staled_evidence_ids.update(updates["staled"])
                tentative_committed_edits = list(filtered_applied_edits)
            print(
                f"[epoch {epoch_idx + 1} final replay] done "
                f"accepts={epoch_final_replay_info.get('final_replay_accepts', 0)} "
                f"rejects={epoch_final_replay_info.get('final_replay_rejects', 0)} "
                f"skipped={epoch_final_replay_info.get('final_replay_skipped', 0)}",
                flush=True,
            )
        elif tentative_committed_edits:
            bypassed_edits = len(tentative_committed_edits)
            epoch_final_replay_info.update(
                {
                    "enabled": False,
                    "final_replay_edits": bypassed_edits,
                    "final_replay_accepts": 0,
                    "final_replay_rejects": 0,
                    "final_replay_skipped": bypassed_edits,
                    "final_replay_bypassed": bypassed_edits,
                    "reason": (
                        (
                            "replaced_by_working_branch_post_reject_selection_no_replay"
                            if _no_replay_enabled(args)
                            else "replaced_by_working_branch_post_reject_replay"
                        )
                        if working_branch_enabled
                        else (
                            "disabled_by_no_replay"
                            if _no_replay_enabled(args)
                            else "disabled_by_zero_budget"
                        )
                    ),
                }
            )
            print(
                f"[epoch {epoch_idx + 1} final replay] "
                f"{'deferred_until_validation_reject' if working_branch_enabled else 'disabled'} "
                f"merged_edits={bypassed_edits} proceeding_to_validation=true",
                flush=True,
            )
        if not tentative_committed_edits:
            selection_info = {
                "enabled": True,
                "task_count": len(selection_tasks),
                "decision": "skipped",
                "reason": "epoch merged edits did not apply or failed final replay",
            }

    working_branch_revalidation = bool(
        working_branch_enabled
        and candidate_skill_md != validation_skill_path.read_text(encoding="utf-8")
        and not tentative_committed_edits
    )
    if working_branch_revalidation:
        print(
            f"[epoch {epoch_idx + 1} validation] working skill differs from validated best; "
            "validating unchanged provisional branch",
            flush=True,
        )

    if coverage_mode and (tentative_committed_edits or working_branch_revalidation):
        evidence_by_id_full = {
            str(row.get("evidence_id") or ""): row
            for row in evidence_rows or []
            if str(row.get("evidence_id") or "")
        }
        final_audit_blocked_ids = (
            set(staled_evidence_ids)
            | set(rejected_evidence_ids)
            | set(baseline_covered_evidence_ids)
            | {
                evidence_id
                for evidence_id, reason in deferred_evidence_fates.items()
                if reason
                in {
                    "fallback_proposed_edit_rejected",
                    "global_apply_failed",
                    "repair_apply_failed",
                    "epoch_final_replay_or_judge_reject",
                    "epoch_final_rebuild_apply_failed",
                    "rule_ignored",
                    "reflection_failed",
                    "reflection_narrowed",
                }
            }
        )
        final_audit_cards = [
            evidence_by_id_full[evidence_id]
            for evidence_id in epoch_pool_ids
            if evidence_id in evidence_by_id_full and evidence_id not in final_audit_blocked_ids
        ]
        final_covered, _, final_coverage_audit_info = _run_evidence_skill_coverage_audit(
            teacher=teacher,
            dataset=dataset,
            skill_md=candidate_skill_md,
            evidence_cards=final_audit_cards,
            args=args,
            stage="candidate_after_final_replay",
        )
        final_candidate_covered_evidence_ids.update(final_covered)
        print(
            f"[epoch {epoch_idx + 1} coverage audit] stage=candidate_after_final_replay "
            f"covered={len(final_covered)} "
            f"uncovered={final_coverage_audit_info.get('uncovered_count', 0)} "
            f"fallback={len(final_coverage_audit_info.get('fallback_evidence_ids') or [])}",
            flush=True,
        )

    if tentative_committed_edits or working_branch_revalidation:
        selection_feedback_row = None
        positive_cases: List[Dict[str, Any]] = []
        negative_cases: List[Dict[str, Any]] = []
        if bool(getattr(args, "no_epoch_validation", False)):
            selection_info = {
                "enabled": False,
                "validation_skipped": True,
                "task_count": len(selection_tasks),
                "decision": "commit",
                "reason": "no_epoch_validation",
                "current_score": None,
                "candidate_score": None,
                "current_success_rate": None,
                "candidate_success_rate": None,
                "current_success": None,
                "candidate_success": None,
            }
            print(
                f"[epoch {epoch_idx + 1} validation] skipped "
                f"candidate_edits={len(tentative_committed_edits)} "
                "reason=no_epoch_validation",
                flush=True,
            )
        else:
            print(
                f"[epoch {epoch_idx + 1} validation] start "
                f"candidate_edits={len(tentative_committed_edits)} "
                f"selection_tasks={len(selection_tasks)}",
                flush=True,
            )
            pending_validation = (
                (working_branch_state or {}).get("pending_validation")
                if working_branch_enabled
                else None
            )
            candidate_fingerprint = skill_fingerprint(candidate_skill_md)
            if (
                isinstance(pending_validation, dict)
                and _coerce_int(pending_validation.get("epoch_index"), -1) == epoch_idx
                and str(pending_validation.get("candidate_skill_fingerprint") or "")
                == candidate_fingerprint
                and isinstance(pending_validation.get("selection"), dict)
            ):
                selection_info = _jsonable_copy(pending_validation["selection"])
                pending_edits = pending_validation.get("candidate_edits")
                if isinstance(pending_edits, list):
                    tentative_committed_edits = [
                        dict(edit) for edit in pending_edits if isinstance(edit, dict)
                    ]
                selection_info["resumed_from_working_branch_state"] = True
                print(
                    f"[epoch {epoch_idx + 1} validation] reusing completed result "
                    "from working_branch_state.json",
                    flush=True,
                )
            else:
                selection_info = _selection_gate(
                    runner=runner,
                    out=out,
                    current_skill_path=validation_skill_path,
                    candidate_skill_md=candidate_skill_md,
                    selection_tasks=selection_tasks,
                    round_idx=epoch_idx,
                    enabled=True,
                    args=args,
                )
        selection_info["positive_case_count"] = len(positive_cases)
        selection_info["negative_case_count"] = len(negative_cases)
        if selection_feedback_row is not None:
            selection_info["feedback_id"] = selection_feedback_row.get("feedback_id")
        selection_info["baseline_task_result_count"] = len(selection_info.get("baseline_task_results") or [])
        selection_info["candidate_task_result_count"] = len(selection_info.get("candidate_task_results") or [])
        selection_info.pop("baseline_task_results", None)
        selection_info.pop("candidate_task_results", None)
        print(
            f"[epoch {epoch_idx + 1} validation] done "
            f"decision={selection_info.get('decision')} "
            f"current_score={selection_info.get('current_score')} "
            f"candidate_score={selection_info.get('candidate_score')} "
            f"current_success_rate={selection_info.get('current_success_rate')} "
            f"candidate_success_rate={selection_info.get('candidate_success_rate')}",
            flush=True,
        )
        if str(selection_info.get("decision") or "").lower() == "evaluation_error":
            raise SelectionEvaluationError(
                f"Epoch {epoch_idx + 1} validation did not complete: "
                f"{selection_info.get('error') or selection_info.get('reason') or 'unknown evaluation error'}"
            )

        if working_branch_enabled:
            working_branch_state["pending_validation"] = {
                "epoch_index": epoch_idx,
                "candidate_skill_fingerprint": skill_fingerprint(candidate_skill_md),
                "candidate_skill_md": candidate_skill_md,
                "base_skill_md": base_skill_md,
                "selection": _jsonable_copy(selection_info),
                "candidate_edits": _jsonable_copy(tentative_committed_edits),
                "pending_l2_manager_rows": _jsonable_copy(pending_l2_manager_rows),
                "summary_context": {
                    "epoch_rollout_task_count": epoch_rollout_task_count,
                    "epoch_evidence_counts_before_l2": epoch_evidence_counts_before,
                    "epoch_candidate_edit_count": epoch_candidate_edit_count,
                    "epoch_accepted_edit_count": epoch_accepted_edit_count,
                    "rule_ignored_accepts": epoch_rule_ignored_accepts,
                    "rule_ignored_defers": epoch_rule_ignored_defers,
                    "reflection_attempts": epoch_reflection_attempts,
                    "reflection_accepts": epoch_reflection_accepts,
                    "reflection_failures": epoch_reflection_failures,
                    "judge_rejects": epoch_judge_rejects,
                    "epoch_global_merge": _jsonable_copy(epoch_merge_info),
                    "epoch_coverage_repair": _jsonable_copy(epoch_coverage_repair_info),
                    "epoch_current_skill_coverage_audit": _jsonable_copy(baseline_coverage_audit_info),
                    "epoch_candidate_coverage_audit": _jsonable_copy(candidate_coverage_audit_info),
                    "epoch_final_coverage_audit": _jsonable_copy(final_coverage_audit_info),
                    "coverage_mode": coverage_mode,
                    "epoch_pool_ids": list(epoch_pool_ids),
                    "archived_evidence_ids": sorted(archived_ids),
                    "accepted_evidence_ids": sorted(accepted_evidence_ids),
                    "rejected_evidence_ids": sorted(rejected_evidence_ids),
                    "staled_evidence_ids": sorted(staled_evidence_ids),
                    "deferred_evidence_fates": _jsonable_copy(deferred_evidence_fates),
                    "epoch_l2_windows": _jsonable_copy(epoch_window_infos),
                    "epoch_final_replay": _jsonable_copy(epoch_final_replay_info),
                    "epoch_global_commit_reports": _jsonable_copy(global_commit_reports),
                    "final_candidate_covered_evidence_ids": sorted(final_candidate_covered_evidence_ids),
                    "baseline_covered_evidence_ids": sorted(baseline_covered_evidence_ids),
                },
            }
            working_branch_state["stage"] = {
                "epoch_index": epoch_idx,
                "name": (
                    "post_reject_replay"
                    if selection_info.get("decision") != "commit"
                    else "validation_commit"
                ),
            }
            _checkpoint_working_branch_epoch_l2(
                out,
                epoch_idx,
                l2_manager_rows=l2_manager_rows + pending_l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                evidence_rows=evidence_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
                working_skill_md=current_skill_path.read_text(encoding="utf-8"),
            )
            persist_working_branch_state(
                working_branch_state_path,
                working_branch_state,
                validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
                working_skill_md=current_skill_path.read_text(encoding="utf-8"),
            )

        if selection_info.get("decision") == "commit":
            if not working_branch_enabled:
                atomic_write_text(current_skill_path, candidate_skill_md)
            reject_cooldown_pool_size = 0
            commit_covered_evidence_ids = (
                set(final_candidate_covered_evidence_ids)
                if coverage_mode
                else set(_covered_evidence_ids(tentative_committed_edits))
            )
            if commit_covered_evidence_ids:
                updates = _mark_evidence_attempts(
                    evidence_rows,
                    sorted(commit_covered_evidence_ids),
                    round_idx=epoch_idx,
                    outcome="accept",
                    max_rejects=max_rejects,
                    count_attempt=False,
                )
                accepted_evidence_ids.update(updates["accepted"])
                archived_ids.update(
                    _archive_evidence_rows_with_reason(
                        evidence_rows,
                        sorted(commit_covered_evidence_ids),
                        reason="covered_by_committed_skill",
                    )
                )
            if archived_ids:
                evidence_pool[:] = [
                    row for row in evidence_pool if str(row.get("evidence_id") or "") not in archived_ids
                ]
            selected_source_ids = {
                str(source_id)
                for edit in tentative_committed_edits
                for source_id in (edit.get("source_edit_ids") or [])
                if str(source_id)
            }
            if selected_source_ids:
                for row in pending_l2_manager_rows:
                    edits = (row.get("output") or {}).get("edits") or []
                    row.setdefault("output", {})["edits"] = [
                        edit
                        for edit in edits
                        if str(edit.get("_candidate_edit_id") or "") in selected_source_ids
                    ]
            if working_branch_enabled:
                working_branch_pending_transition = "commit"
            epoch_validation_decision = "commit"
        else:
            if working_branch_enabled:
                if _is_final_configured_epoch(args, epoch_idx):
                    working_branch_post_reject_info.update(
                        {
                            "reason": "skipped_final_epoch",
                            "candidate_edit_count": len(tentative_committed_edits),
                            "skipped_edit_count": len(tentative_committed_edits),
                            "action_replay_count": 0,
                        }
                    )
                    for row in pending_l2_manager_rows:
                        row.setdefault("output", {})["edits"] = []
                    for edit in tentative_committed_edits:
                        edit_eids = _evidence_id_list(edit.get("evidence_ids"))
                        _defer_evidence_rows(
                            evidence_rows,
                            edit_eids,
                            reason="final_epoch_validation_reject",
                        )
                        for evidence_id in edit_eids:
                            deferred_evidence_fates[str(evidence_id)] = (
                                "final_epoch_validation_reject"
                            )
                    working_branch_pending_transition = "terminal_reject"
                    print(
                        f"[epoch {epoch_idx + 1} skill edit working post-reject] "
                        f"skipped reason=final_epoch candidate_edits={len(tentative_committed_edits)}",
                        flush=True,
                    )
                else:
                    working_branch_provisional_edits, post_reject_rejected, working_branch_post_reject_info = (
                        _run_working_branch_post_reject_replay(
                            dataset=dataset,
                            teacher=teacher,
                            runner=runner,
                            out=out,
                            args=args,
                            epoch_idx=epoch_idx,
                            base_skill_md=base_skill_md,
                            candidate_skill_md=candidate_skill_md,
                            edits=tentative_committed_edits,
                            evidence_rows=evidence_rows,
                            replay_source_by_evidence_id=replay_source_by_evidence_id,
                            l2_judge_rows=l2_judge_rows,
                            replay_rows=replay_rows,
                            working_branch_state=working_branch_state,
                            state_path=working_branch_state_path,
                            validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
                            working_skill_md=current_skill_path.read_text(encoding="utf-8"),
                        )
                    )
                    epoch_rule_ignored_defers += int(
                        working_branch_post_reject_info.get(
                            "rule_ignored_defers", 0
                        )
                        or 0
                    )
                    surviving_ids = {
                        str(edit.get("_working_edit_id") or "")
                        for edit in working_branch_provisional_edits
                    }
                    excluded_reasons = working_branch_post_reject_info.get("excluded_reasons") or {}
                    selected_source_ids = {
                        str(source_id)
                        for edit in working_branch_provisional_edits
                        for source_id in (edit.get("source_edit_ids") or [])
                        if str(source_id)
                    }
                    for row in pending_l2_manager_rows:
                        edits = (row.get("output") or {}).get("edits") or []
                        row.setdefault("output", {})["edits"] = [
                            edit
                            for edit in edits
                            if str(edit.get("_candidate_edit_id") or "") in selected_source_ids
                        ]
                    for edit in tentative_committed_edits:
                        edit_id = str(edit.get("_working_edit_id") or "")
                        edit_eids = _evidence_id_list(edit.get("evidence_ids"))
                        reason = (
                            "working_branch_provisional"
                            if edit_id in surviving_ids
                            else str(excluded_reasons.get(edit_id) or "working_branch_replay_reject")
                        )
                        _defer_evidence_rows(evidence_rows, edit_eids, reason=reason)
                        for evidence_id in edit_eids:
                            deferred_evidence_fates[str(evidence_id)] = reason
                    working_branch_pending_transition = "reject"
            else:
                for row in pending_l2_manager_rows:
                    row.setdefault("output", {})["edits"] = []
                for edit in tentative_committed_edits:
                    edit_eids = _evidence_id_list(edit.get("evidence_ids"))
                    if coverage_mode:
                        _defer_evidence_rows(evidence_rows, edit_eids, reason="validation_reject")
                        for eid in edit_eids:
                            deferred_evidence_fates[str(eid)] = "validation_reject"
                    else:
                        updates = _mark_evidence_attempts(
                            evidence_rows,
                            edit_eids,
                            round_idx=epoch_idx,
                            outcome="reject",
                            max_rejects=max_rejects,
                            count_attempt=False,
                        )
                        rejected_evidence_ids.update(updates["rejected"])
                        staled_evidence_ids.update(updates["staled"])
            epoch_validation_decision = str(selection_info.get("decision") or "reject")

    selected_evidence_ids = set()
    cited_evidence_ids = set()
    for window_info in epoch_window_infos:
        selected_evidence_ids.update(
            str(eid)
            for eid in ((window_info.get("evidence_selection") or {}).get("selected_evidence_ids") or [])
            if str(eid)
        )
        cited_evidence_ids.update(str(eid) for eid in (window_info.get("cited_evidence_ids") or []) if str(eid))
    if coverage_mode:
        evidence_by_id_full = {
            str(row.get("evidence_id") or ""): row
            for row in evidence_rows or []
            if str(row.get("evidence_id") or "")
        }
        accepted_candidate_evidence_ids = set(_covered_evidence_ids(epoch_accepted_edits))
        final_covered_evidence_ids = (
            set(final_candidate_covered_evidence_ids)
            if epoch_validation_decision == "commit"
            else set()
        )
        deferred_ids: List[str] = []
        for evidence_id in sorted(set(epoch_pool_ids) - set(archived_ids) - set(staled_evidence_ids) - set(rejected_evidence_ids)):
            row = evidence_by_id_full.get(evidence_id)
            if not row:
                continue
            _normalize_evidence_runtime_state(row)
            status = str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower()
            if bool(row.get("archived")) or status in {
                EVIDENCE_STATUS_ARCHIVED,
                EVIDENCE_STATUS_STALE,
                EVIDENCE_STATUS_CONSUMED,
            }:
                continue
            if epoch_validation_decision == "commit":
                if evidence_id in final_covered_evidence_ids:
                    continue
                reason = deferred_evidence_fates.get(evidence_id)
                if not reason:
                    if evidence_id in accepted_candidate_evidence_ids:
                        reason = "accepted_not_covered"
                    elif evidence_id in selected_evidence_ids and evidence_id not in cited_evidence_ids:
                        reason = "not_cited_by_l2_proposer"
                    elif evidence_id in selected_evidence_ids:
                        reason = "not_covered_by_window_merge"
                    else:
                        reason = "not_processed_by_semantic_windows"
            else:
                reason = deferred_evidence_fates.get(evidence_id)
                if not reason:
                    if tentative_committed_edits:
                        reason = "validation_reject"
                    elif epoch_accepted_edits:
                        reason = "epoch_candidate_not_committed"
                    else:
                        reason = "no_l2_edit"
            deferred_ids.extend(_defer_evidence_rows(evidence_rows, [evidence_id], reason=reason))
            deferred_evidence_fates[evidence_id] = reason
        epoch_deferred_evidence_ids.update(deferred_ids)
        policy_defer_limit_info = _apply_working_branch_policy_defer_limit(
            evidence_rows,
            deferred_evidence_fates,
            dataset=dataset,
            args=args,
            epoch_idx=epoch_idx,
            validation_decision=epoch_validation_decision,
        )
        policy_staled_ids = set(
            policy_defer_limit_info.get("staled_evidence_ids") or []
        )
        staled_evidence_ids.update(policy_staled_ids)
        epoch_deferred_evidence_ids.difference_update(policy_staled_ids)
        print(
            f"[epoch {epoch_idx + 1} evidence fate] "
            f"archived={len(archived_ids)} "
            f"deferred={len(set(deferred_ids) - policy_staled_ids)} "
            f"rejected={len(rejected_evidence_ids)} "
            f"staled={len(staled_evidence_ids)} "
            f"policy_defer_staled={len(policy_staled_ids)} "
            "consumed=0",
            flush=True,
        )
    else:
        # A globally deferred rule_ignored attribution must remain available
        # for a later epoch even when the evidence working branch is disabled.
        # The ordinary merge_rank cleanup consumes every other evidence row.
        globally_deferred_rule_ignored_ids = {
            str(evidence_id)
            for evidence_id, reason in (deferred_evidence_fates or {}).items()
            if str(reason or "") == "rule_ignored"
        }
        if _defer_rule_ignored_enabled(dataset, args):
            _defer_evidence_rows(
                evidence_rows,
                sorted(globally_deferred_rule_ignored_ids),
                reason="rule_ignored",
            )
            epoch_deferred_evidence_ids.update(globally_deferred_rule_ignored_ids)
        else:
            globally_deferred_rule_ignored_ids = set()
        if epoch_validation_decision == "commit":
            context_ids = sorted(
                (selected_evidence_ids - cited_evidence_ids)
                - archived_ids
                - globally_deferred_rule_ignored_ids
            )
            unused_ids = sorted(
                (set(epoch_pool_ids) - archived_ids)
                - set(context_ids)
                - globally_deferred_rule_ignored_ids
            )
            consumed_evidence_ids.update(_consume_evidence_rows(evidence_rows, context_ids, reason="context_only"))
            consumed_evidence_ids.update(_consume_evidence_rows(evidence_rows, unused_ids, reason="epoch_commit_unused"))
        elif epoch_pool_ids:
            reason = "epoch_reject" if epoch_accepted_edits else "no_l2_edit"
            consumed_evidence_ids.update(
                _consume_evidence_rows(
                    evidence_rows,
                    sorted(
                        (set(epoch_pool_ids) - archived_ids)
                        - globally_deferred_rule_ignored_ids
                    ),
                    reason=reason,
                )
            )
    selected_source_ids = {
        str(source_id)
        for edit in tentative_committed_edits
        for source_id in (edit.get("source_edit_ids") or [])
        if str(source_id)
    }
    for row in pending_l2_manager_rows:
        if epoch_validation_decision != "commit" and not working_branch_enabled:
            row.setdefault("output", {})["edits"] = []
            continue
        if epoch_validation_decision != "commit" and working_branch_enabled:
            selected_source_ids = {
                str(source_id)
                for edit in working_branch_provisional_edits
                for source_id in (edit.get("source_edit_ids") or [])
                if str(source_id)
            }
        else:
            selected_source_ids = {
                str(source_id)
                for edit in tentative_committed_edits
                for source_id in (edit.get("source_edit_ids") or [])
                if str(source_id)
            }
        edits = (row.get("output") or {}).get("edits") or []
        if not selected_source_ids:
            row.setdefault("output", {})["edits"] = []
            continue
        edits = [
            edit
            for edit in edits
            if str(edit.get("_candidate_edit_id") or "") in selected_source_ids
        ]
        row.setdefault("output", {})["edits"] = [_clean_markdown_edit_for_row(edit) for edit in edits]
    evidence_pool[:] = _active_evidence_rows(evidence_pool)
    l2_manager_rows.extend(pending_l2_manager_rows)
    epoch_committed_edit_count = (
        len(tentative_committed_edits) if epoch_validation_decision == "commit" else 0
    )
    if working_branch_enabled and working_branch_pending_transition == "commit":
        working_branch_transition_info = _working_branch_commit_transition(
            state=working_branch_state,
            state_path=working_branch_state_path,
            validated_skill_path=validation_skill_path,
            working_skill_path=current_skill_path,
            candidate_skill_md=candidate_skill_md,
            changed_edits=tentative_committed_edits,
            evidence_rows=evidence_rows,
            epoch_idx=epoch_idx,
            seed=int(getattr(args, "seed", 42) or 42),
        )
    elif working_branch_enabled and working_branch_pending_transition == "reject":
        working_branch_transition_info = _working_branch_reject_transition(
            state=working_branch_state,
            state_path=working_branch_state_path,
            validated_skill_path=validation_skill_path,
            working_skill_path=current_skill_path,
            accepted_edits=working_branch_provisional_edits,
            evidence_rows=evidence_rows,
            epoch_idx=epoch_idx,
            seed=int(getattr(args, "seed", 42) or 42),
        )
    elif working_branch_enabled and working_branch_pending_transition == "terminal_reject":
        working_branch_transition_info = _working_branch_terminal_reject_transition(
            state=working_branch_state,
            state_path=working_branch_state_path,
            validated_skill_path=validation_skill_path,
            working_skill_path=current_skill_path,
            epoch_idx=epoch_idx,
        )
    elif working_branch_enabled and working_branch_transition_info.get("decision") == "unchanged":
        lifecycle_change = any(
            bool(record.get("remove_requested"))
            or bool(record.get("remove_if_uncorrected"))
            for record in working_branch_state.get("provisional_edits") or []
        )
        if lifecycle_change:
            working_branch_transition_info = _working_branch_reject_transition(
                state=working_branch_state,
                state_path=working_branch_state_path,
                validated_skill_path=validation_skill_path,
                working_skill_path=current_skill_path,
                accepted_edits=[],
                evidence_rows=evidence_rows,
                epoch_idx=epoch_idx,
                seed=int(getattr(args, "seed", 42) or 42),
            )
            working_branch_transition_info["decision"] = "lifecycle_only"
        else:
            working_branch_state.pop("pending_validation", None)
            working_branch_state["stage"] = {
                "epoch_index": epoch_idx,
                "name": "complete",
                "decision": epoch_validation_decision,
            }
            persist_working_branch_state(
                working_branch_state_path,
                working_branch_state,
                validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
                working_skill_md=current_skill_path.read_text(encoding="utf-8"),
            )
    if working_branch_enabled:
        working_branch_state["stage"] = {
            "epoch_index": epoch_idx,
            "name": "complete",
            "decision": epoch_validation_decision,
        }
        persist_working_branch_state(
            working_branch_state_path,
            working_branch_state,
            validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
            working_skill_md=current_skill_path.read_text(encoding="utf-8"),
        )
    evidence_pool[:] = _active_evidence_rows(evidence_rows)
    _normalize_evidence_runtime_states(evidence_pool)
    epoch_evidence_counts_after = _evidence_pool_counts(evidence_pool)
    for window_info in epoch_window_infos:
        if epoch_validation_decision == "commit":
            selected_source_ids = {
                str(source_id)
                for edit in tentative_committed_edits
                for source_id in (edit.get("source_edit_ids") or [])
                if str(source_id)
            }
            window_info["committed_count"] = sum(
                1
                for edit in (window_info.get("accepted_edits") or [])
                if str(edit.get("_candidate_edit_id") or "") in selected_source_ids
            )
        else:
            window_info["committed_count"] = 0
            if working_branch_enabled:
                provisional_source_ids = {
                    str(source_id)
                    for edit in working_branch_provisional_edits
                    for source_id in (edit.get("source_edit_ids") or [])
                    if str(source_id)
                }
                window_info["provisional_count"] = sum(
                    1
                    for edit in (window_info.get("accepted_edits") or [])
                    if str(edit.get("_candidate_edit_id") or "") in provisional_source_ids
                )
        window_info["selection"] = selection_info
        window_info.pop("accepted_edits", None)
    print(
        f"[epoch {epoch_idx + 1} skill edit] finish "
        f"validation={epoch_validation_decision} "
        f"committed_edits={epoch_committed_edit_count} "
        f"final_replay_enabled={epoch_final_replay_info.get('enabled', True)} "
        f"final_replay_accepts={epoch_final_replay_info.get('final_replay_accepts', 0)} "
        f"final_replay_rejects={epoch_final_replay_info.get('final_replay_rejects', 0)} "
        f"final_replay_bypassed={epoch_final_replay_info.get('final_replay_bypassed', 0)} "
        f"rule_ignored_accepts={epoch_rule_ignored_accepts} "
        f"rule_ignored_defers={epoch_rule_ignored_defers} "
        f"reflection_attempts={epoch_reflection_attempts} "
        f"reflection_accepts={epoch_reflection_accepts} "
        f"reflection_failures={epoch_reflection_failures} "
        f"judge_rejects={epoch_judge_rejects} "
        f"archived_evidence={len(archived_ids)} "
        f"deferred_evidence={len(epoch_deferred_evidence_ids)} "
        f"consumed_evidence={len(consumed_evidence_ids)} "
        f"active_after={epoch_evidence_counts_after.get('active', 0)}",
        flush=True,
    )
    epoch_summary = {
        "epoch_index": epoch_idx,
        "replay_mode": _replay_mode(args),
        "epoch_rollout_task_count": epoch_rollout_task_count,
        "epoch_skill_update_mode": _resolve_epoch_skill_update_mode(args),
        "epoch_evidence_counts": epoch_evidence_counts_after,
        "epoch_evidence_counts_before_l2": epoch_evidence_counts_before,
        "epoch_l2_window_count": len(epoch_window_infos),
        "epoch_candidate_edit_count": epoch_candidate_edit_count,
        "epoch_accepted_candidate_edit_count": epoch_accepted_edit_count,
        "epoch_accepted_edit_count": epoch_accepted_edit_count,
        "rule_ignored_accepts": epoch_rule_ignored_accepts,
        "rule_ignored_defers": epoch_rule_ignored_defers,
        "reflection_attempts": epoch_reflection_attempts,
        "reflection_accepts": epoch_reflection_accepts,
        "reflection_failures": epoch_reflection_failures,
        "judge_rejects": epoch_judge_rejects,
        "epoch_global_merge": epoch_merge_info,
        "epoch_coverage_repair": epoch_coverage_repair_info,
        "epoch_current_skill_coverage_audit": baseline_coverage_audit_info,
        "epoch_candidate_coverage_audit": candidate_coverage_audit_info,
        "epoch_final_coverage_audit": final_coverage_audit_info,
        "epoch_corrective_edit_count": epoch_merge_info.get("corrective_edit_count", 0),
        "epoch_success_only_edit_count": epoch_merge_info.get("success_only_edit_count", 0),
        "epoch_merged_pool_count": epoch_merge_info.get("merged_pool_count", 0),
        "epoch_ranked_edit_count": epoch_merge_info.get("ranked_edit_count", 0),
        "epoch_recovery_rule_edit_count": epoch_merge_info.get("recovery_rule_edit_count", 0),
        "epoch_replace_or_generalize_edit_count": epoch_merge_info.get("replace_or_generalize_edit_count", 0),
        "epoch_final_replay": epoch_final_replay_info,
        "epoch_working_branch": working_branch_transition_info,
        "epoch_working_branch_post_reject_replay": working_branch_post_reject_info,
        "epoch_working_branch_policy_defer": policy_defer_limit_info,
        "epoch_provisional_edits": [
            {
                **_working_edit_payload(edit),
                "working_edit_id": str(edit.get("_working_edit_id") or ""),
            }
            for edit in working_branch_provisional_edits
        ],
        "epoch_provisional_edit_count": len(working_branch_provisional_edits),
        "final_replay_edits": epoch_final_replay_info.get("final_replay_edits", 0),
        "final_replay_accepts": epoch_final_replay_info.get("final_replay_accepts", 0),
        "final_replay_rejects": epoch_final_replay_info.get("final_replay_rejects", 0),
        "final_replay_skipped": epoch_final_replay_info.get("final_replay_skipped", 0),
        "final_replay_bypassed": epoch_final_replay_info.get("final_replay_bypassed", 0),
        "epoch_global_commit_reports": global_commit_reports,
        "epoch_committed_edits": [_clean_markdown_edit_for_row(edit) for edit in tentative_committed_edits]
        if epoch_validation_decision == "commit"
        else [],
        "epoch_committed_edit_count": epoch_committed_edit_count,
        "epoch_validation_decision": epoch_validation_decision,
        "epoch_validation_current_score": selection_info.get("current_score"),
        "epoch_validation_candidate_score": selection_info.get("candidate_score"),
        "epoch_validation_current_success_rate": selection_info.get("current_success_rate"),
        "epoch_validation_candidate_success_rate": selection_info.get("candidate_success_rate"),
        "epoch_validation_current_success": selection_info.get("current_success"),
        "epoch_validation_candidate_success": selection_info.get("candidate_success"),
        "epoch_selection": selection_info,
        "epoch_l2_windows": epoch_window_infos,
        "epoch_archived_evidence_ids": sorted(archived_ids),
        "epoch_accepted_evidence_ids": sorted(accepted_evidence_ids),
        "epoch_rejected_evidence_ids": sorted(rejected_evidence_ids),
        "epoch_staled_evidence_ids": sorted(staled_evidence_ids),
        "epoch_policy_defer_staled_evidence_ids": sorted(
            policy_defer_limit_info.get("staled_evidence_ids") or []
        ),
        "epoch_consumed_evidence_ids": sorted(consumed_evidence_ids),
        "epoch_deferred_evidence_ids": sorted(epoch_deferred_evidence_ids),
        "epoch_deferred_evidence_reasons": {
            eid: deferred_evidence_fates[eid]
            for eid in sorted(epoch_deferred_evidence_ids)
            if eid in deferred_evidence_fates
        },
        "already_covered_current_evidence_count": len(baseline_covered_evidence_ids),
        "final_committed_evidence_count": (
            len(final_candidate_covered_evidence_ids)
            if epoch_validation_decision == "commit"
            else 0
        ),
        "final_deferred_evidence_count": len(epoch_deferred_evidence_ids),
        "consumed_evidence_count": len(consumed_evidence_ids),
    }
    if working_branch_enabled:
        _checkpoint_working_branch_epoch_l2(
            out,
            epoch_idx,
            l2_manager_rows=l2_manager_rows,
            l2_candidate_rows=l2_candidate_rows,
            l2_judge_rows=l2_judge_rows,
            evidence_rows=evidence_rows,
            replay_rows=replay_rows,
            selection_feedback_rows=selection_feedback_rows,
            validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
            working_skill_md=current_skill_path.read_text(encoding="utf-8"),
        )
        working_branch_state["last_completed_epoch_summary"] = _jsonable_copy(epoch_summary)
        persist_working_branch_state(
            working_branch_state_path,
            working_branch_state,
            validated_skill_md=validation_skill_path.read_text(encoding="utf-8"),
            working_skill_md=current_skill_path.read_text(encoding="utf-8"),
        )
    return epoch_summary, reject_cooldown_pool_size


def _consume_cluster_step_window_evidence(
    *,
    evidence_rows: List[Dict[str, Any]],
    evidence_pool: List[Dict[str, Any]],
    window_info: Dict[str, Any],
) -> List[str]:
    selection = window_info.get("evidence_selection") or {}
    selected_ids = {str(eid) for eid in (selection.get("selected_evidence_ids") or []) if str(eid)}
    cited_ids = {str(eid) for eid in (window_info.get("cited_evidence_ids") or []) if str(eid)}
    archived_ids = {str(eid) for eid in (window_info.get("archived_evidence_ids") or []) if str(eid)}
    rejected_ids = {str(eid) for eid in (window_info.get("rejected_evidence_ids") or []) if str(eid)}
    deferred_rule_ignored_ids = {
        str(evidence_id)
        for evidence_id, reason in (window_info.get("deferred_evidence_reasons") or {}).items()
        if str(reason or "") == "rule_ignored"
    }
    consumed: set[str] = set()
    decision = str(((window_info.get("selection") or {}).get("decision") or "skipped")).lower()

    rejected_remaining = sorted(
        (selected_ids & rejected_ids)
        - archived_ids
        - deferred_rule_ignored_ids
    )
    consumed.update(_consume_evidence_rows(evidence_rows, rejected_remaining, reason="window_rejected_edit"))

    if decision == "commit":
        context_ids = sorted(
            (selected_ids - cited_ids)
            - archived_ids
            - consumed
            - deferred_rule_ignored_ids
        )
        consumed.update(_consume_evidence_rows(evidence_rows, context_ids, reason="context_only"))
        unused_ids = sorted(
            selected_ids
            - archived_ids
            - consumed
            - deferred_rule_ignored_ids
        )
        consumed.update(_consume_evidence_rows(evidence_rows, unused_ids, reason="window_commit_unused"))
    else:
        reason = "window_reject" if decision == "reject" else "no_l2_edit"
        remaining_ids = sorted(
            selected_ids
            - archived_ids
            - consumed
            - deferred_rule_ignored_ids
        )
        consumed.update(_consume_evidence_rows(evidence_rows, remaining_ids, reason=reason))

    evidence_pool[:] = _active_evidence_rows(evidence_pool)
    return sorted(consumed)


def _run_epoch_l2_update_cluster_step(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    epoch_idx: int,
    epoch_rollout_task_count: int,
    current_skill_path: Path,
    evidence_pool: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l2_manager_rows: List[Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    selection_tasks: List[Dict[str, Any]],
    reject_cooldown_pool_size: int = 0,
) -> Tuple[Dict[str, Any], int]:
    epoch_l2_windows = int(getattr(args, "epoch_l2_windows", 4) or 0)
    epoch_evidence_counts_before = _evidence_pool_counts(evidence_pool)
    max_l2_window = int(
        getattr(
            args,
            "l2_max_evidence_per_window",
            getattr(args, "evidence_window_size", 20),
        )
        or getattr(args, "evidence_window_size", 20)
    )
    l2_evidence_grouping = _resolve_l2_evidence_window_grouping(
        dataset,
        getattr(args, "l2_evidence_window_grouping", "auto"),
    )
    semantic_options = _semantic_embedding_options_from_args(args, out) if l2_evidence_grouping == "semantic" else None
    semantic_window_target = epoch_l2_windows if l2_evidence_grouping == "semantic" else 0
    all_planned_l2_windows = _plan_epoch_l2_evidence_windows(
        evidence_pool,
        max_window=max_l2_window,
        max_windows=semantic_window_target,
        grouping=l2_evidence_grouping,
        semantic_options=semantic_options,
    )
    planned_l2_windows = (
        all_planned_l2_windows
        if epoch_l2_windows <= 0 or l2_evidence_grouping == "semantic"
        else all_planned_l2_windows[:epoch_l2_windows]
    )
    for planned_idx, (_, planned_info) in enumerate(planned_l2_windows):
        planned_info["planned_window_index"] = planned_idx
        planned_info["planned_window_count"] = len(planned_l2_windows)
    planned_sizes = [len(planned_pool) for planned_pool, _ in planned_l2_windows]
    window_limit_label = "all" if epoch_l2_windows <= 0 else str(epoch_l2_windows)
    print(
        f"[epoch {epoch_idx + 1} skill edit] start "
        f"mode=cluster_step train_tasks={epoch_rollout_task_count} "
        f"evidence_active={epoch_evidence_counts_before.get('active', 0)} "
        f"evidence_total={epoch_evidence_counts_before.get('total', 0)} "
        f"grouping={l2_evidence_grouping} max_window={max_l2_window} "
        f"planned_windows={len(planned_l2_windows)}/{window_limit_label} "
        f"all_windows={len(all_planned_l2_windows)} window_sizes={planned_sizes}",
        flush=True,
    )
    if l2_evidence_grouping == "semantic" and all_planned_l2_windows:
        first_info = all_planned_l2_windows[0][1] or {}
        print(
            f"[epoch {epoch_idx + 1} skill edit] semantic clusters="
            f"{first_info.get('semantic_cluster_count', 0)} "
            f"method={first_info.get('semantic_clustering_method', 'unknown')} "
            f"model={first_info.get('embedding_model', DEFAULT_EVIDENCE_EMBEDDING_MODEL)} "
            f"cache_hit={first_info.get('embedding_cache_hits', 0)} "
            f"cache_miss={first_info.get('embedding_cache_misses', 0)} "
            f"hdbscan_min_cluster_size={first_info.get('semantic_hdbscan_min_cluster_size', 'n/a')} "
            f"hdbscan_min_samples={first_info.get('semantic_hdbscan_min_samples', 'n/a')} "
            f"noise={first_info.get('semantic_hdbscan_noise_count', 'n/a')} "
            f"cluster_sizes={first_info.get('semantic_cluster_sizes', [])}",
            flush=True,
        )
        print(
            f"[epoch {epoch_idx + 1} skill edit] semantic windows "
            f"natural_windows={first_info.get('semantic_natural_window_count', len(all_planned_l2_windows))} "
            f"target_windows={first_info.get('semantic_target_window_count', epoch_l2_windows)} "
            f"capacity_floor={first_info.get('semantic_capacity_floor_window_count', len(all_planned_l2_windows))} "
            f"final_windows={first_info.get('semantic_final_window_count', len(all_planned_l2_windows))} "
            f"reason={first_info.get('semantic_window_budget_reason', 'n/a')} "
            f"window_sizes={first_info.get('semantic_final_window_sizes', planned_sizes)} "
            f"underfilled_windows={first_info.get('semantic_underfilled_window_count', 0)} "
            f"dropped_evidence={first_info.get('semantic_dropped_small_evidence_count', 0)}",
            flush=True,
        )

    epoch_window_infos: List[Dict[str, Any]] = []
    epoch_candidate_edit_count = 0
    epoch_accepted_edit_count = 0
    epoch_committed_edit_count = 0
    window_validation_attempts = 0
    window_validation_commits = 0
    window_validation_rejects = 0
    archived_evidence_ids: set[str] = set()
    accepted_evidence_ids: set[str] = set()
    rejected_evidence_ids: set[str] = set()
    staled_evidence_ids: set[str] = set()
    consumed_evidence_ids: set[str] = set()
    deferred_evidence_ids: set[str] = set()
    deferred_evidence_reasons: Dict[str, str] = {}
    committed_edits: List[Dict[str, Any]] = []

    for window_idx, (planned_pool, planned_selection_info) in enumerate(planned_l2_windows):
        planned_selection_info = planned_selection_info or {}
        current_skill_md = current_skill_path.read_text(encoding="utf-8")
        grouping_label = str(planned_selection_info.get("grouping") or l2_evidence_grouping)
        if grouping_label == "semantic":
            window_group_summary = f"semantic_groups={planned_selection_info.get('selected_semantic_group_count', 0)}"
        elif grouping_label == "round":
            window_group_summary = f"round_groups={planned_selection_info.get('selected_round_group_count', 0)}"
        else:
            window_group_summary = "card_window"
        print(
            f"[epoch {epoch_idx + 1} skill edit] window {window_idx + 1}/{len(planned_l2_windows)} start "
            f"mode=cluster_step evidence={len(planned_pool)} {window_group_summary} "
            f"sim_avg={planned_selection_info.get('similarity_avg', 'n/a')}",
            flush=True,
        )
        _, window_info, reject_cooldown_pool_size, l2_manager_row = _run_l2_update_window(
            dataset=dataset,
            teacher=teacher,
            runner=runner,
            out=out,
            args=args,
            round_idx=epoch_idx,
            skill_md=current_skill_md,
            current_skill_path=current_skill_path,
            evidence_pool=evidence_pool,
            evidence_rows=evidence_rows,
            replay_source_by_evidence_id=replay_source_by_evidence_id,
            raw_by_id={},
            l2_candidate_rows=l2_candidate_rows,
            l2_judge_rows=l2_judge_rows,
            replay_rows=replay_rows,
            selection_feedback_rows=selection_feedback_rows,
            selection_tasks=selection_tasks,
            l2_trigger_reason="epoch_boundary",
            reject_cooldown_pool_size=reject_cooldown_pool_size,
            defer_selection=False,
            force_selection_enabled=not bool(getattr(args, "no_epoch_validation", False)),
            exclude_evidence_ids=[],
            preselected_pool=planned_pool,
            preselected_selection_info=planned_selection_info,
            defer_commit=False,
        )
        window_info["epoch_index"] = epoch_idx
        window_info["epoch_window_index"] = window_idx
        window_info["epoch_skill_update_mode"] = "cluster_step"
        l2_manager_row["epoch_skill_update_mode"] = "cluster_step"
        l2_manager_rows.append(l2_manager_row)

        epoch_candidate_edit_count += int(window_info.get("candidate_count") or 0)
        epoch_accepted_edit_count += int(window_info.get("accepted_count") or 0)
        epoch_committed_edit_count += int(window_info.get("committed_count") or 0)
        committed_edits.extend(window_info.get("committed_edits") or [])
        archived_evidence_ids.update(window_info.get("archived_evidence_ids") or [])
        accepted_evidence_ids.update(window_info.get("accepted_evidence_ids") or [])
        rejected_evidence_ids.update(window_info.get("rejected_evidence_ids") or [])
        staled_evidence_ids.update(window_info.get("staled_evidence_ids") or [])
        for evidence_id in window_info.get("deferred_evidence_ids") or []:
            evidence_id = str(evidence_id)
            if not evidence_id:
                continue
            deferred_evidence_ids.add(evidence_id)
            deferred_evidence_reasons[evidence_id] = str(
                (window_info.get("deferred_evidence_reasons") or {}).get(evidence_id)
                or "deferred"
            )
        rule_ignored_ids = {
            evidence_id
            for evidence_id, reason in deferred_evidence_reasons.items()
            if reason == "rule_ignored"
        }
        _defer_evidence_rows(
            evidence_rows,
            sorted(rule_ignored_ids),
            reason="rule_ignored",
        )
        selection_info = window_info.get("selection") or {}
        decision = str(selection_info.get("decision") or "skipped")
        if decision.lower() == "evaluation_error":
            raise SelectionEvaluationError(
                f"Epoch {epoch_idx + 1} window {window_idx + 1} validation did not complete: "
                f"{selection_info.get('error') or selection_info.get('reason') or 'unknown evaluation error'}"
            )
        if selection_info.get("enabled") and decision in {"commit", "reject"}:
            window_validation_attempts += 1
        if decision == "commit":
            window_validation_commits += 1
        elif decision == "reject":
            window_validation_rejects += 1
        window_consumed_ids = _consume_cluster_step_window_evidence(
            evidence_rows=evidence_rows,
            evidence_pool=evidence_pool,
            window_info=window_info,
        )
        consumed_evidence_ids.update(window_consumed_ids)
        window_info["consumed_evidence_ids"] = window_consumed_ids
        window_info["committed_edits"] = [_clean_markdown_edit_for_row(edit) for edit in (window_info.get("committed_edits") or [])]
        window_info.pop("accepted_edits", None)
        epoch_window_infos.append(window_info)
        print(
            f"[epoch {epoch_idx + 1} skill edit] window {window_idx + 1}/{len(planned_l2_windows)} done "
            f"decision={decision} candidate_edits={window_info.get('candidate_count', 0)} "
            f"accepted_edits={window_info.get('accepted_count', 0)} "
            f"committed_edits={window_info.get('committed_count', 0)} "
            f"active_after={_evidence_pool_counts(evidence_pool).get('active', 0)}",
            flush=True,
        )

    remaining_ids = [
        str(row.get("evidence_id") or "")
        for row in evidence_pool or []
        if str(row.get("evidence_id") or "")
        and str(row.get("evidence_id") or "") not in deferred_evidence_ids
    ]
    if remaining_ids:
        window_limit_truncated = epoch_l2_windows > 0 and len(planned_l2_windows) < len(all_planned_l2_windows)
        consumed_evidence_ids.update(
            _consume_evidence_rows(
                evidence_rows,
                remaining_ids,
                reason="window_limit_not_processed" if window_limit_truncated else "no_l2_edit",
            )
        )
        evidence_pool[:] = _active_evidence_rows(evidence_pool)

    epoch_evidence_counts_after = _evidence_pool_counts(evidence_pool)
    final_selection = (epoch_window_infos[-1].get("selection") if epoch_window_infos else {}) or {}
    print(
        f"[epoch {epoch_idx + 1} skill edit] finish "
        f"mode=cluster_step processed_windows={len(epoch_window_infos)} "
        f"unprocessed_windows={max(0, len(all_planned_l2_windows) - len(planned_l2_windows))} "
        f"committed_edits={epoch_committed_edit_count} "
        f"validation_commits={window_validation_commits} "
        f"validation_rejects={window_validation_rejects} "
        f"active_after={epoch_evidence_counts_after.get('active', 0)}",
        flush=True,
    )
    return {
        "epoch_index": epoch_idx,
        "epoch_rollout_task_count": epoch_rollout_task_count,
        "epoch_skill_update_mode": "cluster_step",
        "epoch_evidence_counts": epoch_evidence_counts_after,
        "epoch_evidence_counts_before_l2": epoch_evidence_counts_before,
        "epoch_l2_window_count": len(epoch_window_infos),
        "epoch_processed_window_count": len(epoch_window_infos),
        "epoch_unprocessed_window_count": max(0, len(all_planned_l2_windows) - len(planned_l2_windows)),
        "epoch_candidate_edit_count": epoch_candidate_edit_count,
        "epoch_accepted_candidate_edit_count": epoch_accepted_edit_count,
        "epoch_accepted_edit_count": epoch_accepted_edit_count,
        "epoch_committed_edit_count": epoch_committed_edit_count,
        "epoch_committed_edits": [_clean_markdown_edit_for_row(edit) for edit in committed_edits],
        "epoch_window_validation_attempts": window_validation_attempts,
        "epoch_window_validation_commits": window_validation_commits,
        "epoch_window_validation_rejects": window_validation_rejects,
        "epoch_validation_decision": (
            "commit"
            if window_validation_commits > 0 or (bool(getattr(args, "no_epoch_validation", False)) and epoch_committed_edit_count > 0)
            else ("reject" if window_validation_rejects > 0 else "skipped")
        ),
        "epoch_validation_current_score": final_selection.get("current_score"),
        "epoch_validation_candidate_score": final_selection.get("candidate_score"),
        "epoch_validation_current_success_rate": final_selection.get("current_success_rate"),
        "epoch_validation_candidate_success_rate": final_selection.get("candidate_success_rate"),
        "epoch_validation_current_success": final_selection.get("current_success"),
        "epoch_validation_candidate_success": final_selection.get("candidate_success"),
        "epoch_selection": final_selection,
        "epoch_l2_windows": epoch_window_infos,
        "epoch_archived_evidence_ids": sorted(archived_evidence_ids),
        "epoch_accepted_evidence_ids": sorted(accepted_evidence_ids),
        "epoch_rejected_evidence_ids": sorted(rejected_evidence_ids),
        "epoch_staled_evidence_ids": sorted(staled_evidence_ids),
        "epoch_consumed_evidence_ids": sorted(consumed_evidence_ids),
        "epoch_deferred_evidence_ids": sorted(deferred_evidence_ids),
        "epoch_deferred_evidence_reasons": {
            evidence_id: deferred_evidence_reasons[evidence_id]
            for evidence_id in sorted(deferred_evidence_ids)
        },
    }, reject_cooldown_pool_size


def _run_epoch_l2_update(
    *,
    dataset: str,
    teacher: Any,
    runner: Any,
    out: Path,
    args: Any,
    epoch_idx: int,
    epoch_rollout_task_count: int,
    current_skill_path: Path,
    validation_skill_path: Optional[Path] = None,
    working_branch_state: Optional[Dict[str, Any]] = None,
    working_branch_state_path: Optional[Path] = None,
    evidence_pool: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l2_manager_rows: List[Dict[str, Any]],
    l2_candidate_rows: List[Dict[str, Any]],
    l2_judge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    selection_feedback_rows: List[Dict[str, Any]],
    selection_tasks: List[Dict[str, Any]],
    reject_cooldown_pool_size: int = 0,
) -> Tuple[Dict[str, Any], int]:
    mutable_rows = (
        evidence_pool,
        evidence_rows,
        l2_manager_rows,
        l2_candidate_rows,
        l2_judge_rows,
        replay_rows,
        selection_feedback_rows,
    )
    row_snapshots = [(rows, copy.deepcopy(rows)) for rows in mutable_rows]
    skill_before = current_skill_path.read_text(encoding="utf-8")
    validation_skill_path = validation_skill_path or current_skill_path
    validated_skill_before = validation_skill_path.read_text(encoding="utf-8")
    working_branch_state_before = copy.deepcopy(working_branch_state)
    try:
        if _resolve_epoch_skill_update_mode(args) == "cluster_step":
            summary, reject_cooldown_pool_size = _run_epoch_l2_update_merge_rank(
                dataset=dataset,
                teacher=teacher,
                runner=runner,
                out=out,
                args=args,
                epoch_idx=epoch_idx,
                epoch_rollout_task_count=epoch_rollout_task_count,
                current_skill_path=current_skill_path,
                validation_skill_path=validation_skill_path,
                working_branch_state=working_branch_state,
                working_branch_state_path=working_branch_state_path,
                evidence_pool=evidence_pool,
                evidence_rows=evidence_rows,
                replay_source_by_evidence_id=replay_source_by_evidence_id,
                l2_manager_rows=l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                selection_tasks=selection_tasks,
                reject_cooldown_pool_size=reject_cooldown_pool_size,
            )
            summary["epoch_skill_update_mode"] = "cluster_step"
        else:
            summary, reject_cooldown_pool_size = _run_epoch_l2_update_merge_rank(
                dataset=dataset,
                teacher=teacher,
                runner=runner,
                out=out,
                args=args,
                epoch_idx=epoch_idx,
                epoch_rollout_task_count=epoch_rollout_task_count,
                current_skill_path=current_skill_path,
                validation_skill_path=validation_skill_path,
                working_branch_state=working_branch_state,
                working_branch_state_path=working_branch_state_path,
                evidence_pool=evidence_pool,
                evidence_rows=evidence_rows,
                replay_source_by_evidence_id=replay_source_by_evidence_id,
                l2_manager_rows=l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                selection_tasks=selection_tasks,
                reject_cooldown_pool_size=reject_cooldown_pool_size,
            )
        if str(summary.get("epoch_validation_decision") or "").lower() == "evaluation_error":
            error = (summary.get("epoch_selection") or {}).get("error")
            raise SelectionEvaluationError(
                f"Epoch {epoch_idx + 1} validation did not complete: "
                f"{error or 'selection evaluation error'}"
            )
        return summary, reject_cooldown_pool_size
    except Exception:
        preserve_post_reject_progress = bool(
            working_branch_state is not None
            and _row_int(working_branch_state.get("stage") or {}, "epoch_index", -1) == epoch_idx
            and str((working_branch_state.get("stage") or {}).get("name") or "")
            == "post_reject_replay"
            and current_skill_path.read_text(encoding="utf-8") == skill_before
            and validation_skill_path.read_text(encoding="utf-8") == validated_skill_before
        )
        if preserve_post_reject_progress:
            raise
        for rows, snapshot in row_snapshots:
            rows[:] = snapshot
        if current_skill_path.read_text(encoding="utf-8") != skill_before:
            atomic_write_text(current_skill_path, skill_before)
        if validation_skill_path.read_text(encoding="utf-8") != validated_skill_before:
            atomic_write_text(validation_skill_path, validated_skill_before)
        if working_branch_state is not None and working_branch_state_before is not None:
            working_branch_state.clear()
            working_branch_state.update(working_branch_state_before)
            if working_branch_state_path is not None:
                persist_working_branch_state(
                    working_branch_state_path,
                    working_branch_state,
                    validated_skill_md=validated_skill_before,
                    working_skill_md=skill_before,
                )
        raise


def _chunk_list(rows: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    size = max(1, int(size))
    return [rows[idx : idx + size] for idx in range(0, len(rows or []), size)]


def _partition_epoch_l1_results(
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    epoch_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match SkillOPT's deterministic failure/success minibatch ordering."""
    failure_results = [row for row in rows or [] if not _is_success_result(row)]
    success_results = [row for row in rows or [] if _is_success_result(row)]
    failure_seed = int(seed) + (int(epoch_idx) + 1) * 1000 + 1
    success_seed = failure_seed + 1
    random.Random(failure_seed).shuffle(failure_results)
    random.Random(success_seed).shuffle(success_results)
    return failure_results, success_results


def _run_epoch_l1_evidence_extraction(
    *,
    dataset: str,
    teacher: Any,
    args: Any,
    skill_md: str,
    epoch_idx: int,
    bundles_per_epoch: int,
    epoch_train_results: List[Dict[str, Any]],
    evidence_counter: int,
    evidence_rows: List[Dict[str, Any]],
    replay_source_rows: List[Dict[str, Any]],
    replay_source_by_evidence_id: Dict[str, Dict[str, Any]],
    l1_manager_rows: List[Dict[str, Any]],
    l1_judge_rows: List[Dict[str, Any]],
    manifest_task_order: Optional[List[str]] = None,
) -> Tuple[List[str], int, Dict[str, Any]]:
    # Keep the epoch rollout order. SkillOPT splits and shuffles the result list
    # it receives; sorting back to manifest order changes minibatch membership.
    ordered_results = list(epoch_train_results or [])
    result_ids = [str(row.get("task_id") or row.get("id") or "") for row in ordered_results]
    if len(set(result_ids)) != len(result_ids):
        duplicates = sorted({task_id for task_id in result_ids if result_ids.count(task_id) > 1})
        raise ValueError(f"Epoch train rollout contains duplicate task IDs: {duplicates[:5]}")
    expected_ids = [str(task_id) for task_id in (manifest_task_order or [])]
    if expected_ids and set(result_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(result_ids))
        unexpected = sorted(set(result_ids) - set(expected_ids))
        raise ValueError(
            "Epoch train rollout does not cover the evolution manifest exactly: "
            f"expected={len(expected_ids)} actual={len(result_ids)} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    failure_results, success_results = _partition_epoch_l1_results(
        ordered_results,
        seed=int(getattr(args, "seed", 42) or 42),
        epoch_idx=epoch_idx,
    )

    prompts = prompts_for_dataset(dataset)
    l1_judge_prompt = prompts["skill_judge_l1"]
    source_specs = [
        ("failure", "failure_reflection", failure_results, prompts.get("skill_manager_l1_failure") or prompts["skill_manager_l1"]),
        ("success", "success_experience", success_results, prompts.get("skill_manager_l1_success") or prompts["skill_manager_l1"]),
    ]
    epoch_evidence_ids: List[str] = []
    source_summaries: Dict[str, Any] = {}
    l1_call_index = 0
    minibatch_size = max(
        1,
        _int_arg(
            getattr(args, "l1_trajectory_minibatch_size", L1_TRAJECTORY_MINIBATCH_SIZE),
            L1_TRAJECTORY_MINIBATCH_SIZE,
        ),
    )
    failure_seed = int(getattr(args, "seed", 42) or 42) + (int(epoch_idx) + 1) * 1000 + 1
    success_seed = failure_seed + 1
    print(
        f"[epoch {epoch_idx + 1} L1 partition] "
        f"failure={len(failure_results)} seed={failure_seed} "
        f"success={len(success_results)} seed={success_seed} "
        f"minibatch_size={minibatch_size}",
        flush=True,
    )
    for source_type, default_experience_type, source_results, l1_prompt in source_specs:
        source_summaries[source_type] = {
            "trajectory_count": len(source_results),
            "minibatch_count": 0,
            "draft_count": 0,
            "kept_count": 0,
            "dropped_count": 0,
            "input_compaction": [],
        }
        for minibatch_idx, batch_results in enumerate(_chunk_list(source_results, minibatch_size)):
            if not batch_results:
                continue
            l1_call_index += 1
            source_summaries[source_type]["minibatch_count"] += 1
            raw_by_id = _raw_result_by_task_id(batch_results)
            materialization_records = [
                _materialization_record_from_result(row, dataset=dataset)
                for row in batch_results
            ]
            records_by_id = {
                str(row.get("task_id") or ""): row
                for row in materialization_records
                if str(row.get("task_id") or "")
            }
            minibatch_task_ids = [
                str(row.get("task_id") or row.get("id") or "")
                for row in batch_results
                if str(row.get("task_id") or row.get("id") or "")
            ]
            source_minibatch_id = f"epoch_{epoch_idx + 1}_{source_type}_{minibatch_idx:03d}"
            l1_input, input_compaction = _build_l1_evidence_payload(
                skill_md=skill_md,
                source_type=source_type,
                results=batch_results,
                system_prompt=l1_prompt,
                args=args,
                dataset=dataset,
            )
            input_compaction["source_minibatch_task_ids"] = minibatch_task_ids
            source_summaries[source_type]["input_compaction"].append(input_compaction)
            compact_label = "warning" if input_compaction.get("essential_payload_over_budget") else "info"
            print(
                f"[epoch {epoch_idx + 1} L1 {source_type} minibatch {minibatch_idx + 1}] "
                f"input_{compact_label} chars={input_compaction.get('input_chars')} "
                f"budget={input_compaction.get('input_budget')} "
                f"code={input_compaction.get('raw_code_chars', 0)}->"
                f"{input_compaction.get('visible_code_chars', 0)} "
                f"feedback={input_compaction.get('raw_feedback_chars')}->"
                f"{input_compaction.get('visible_feedback_chars')} "
                f"critical={input_compaction.get('critical_feedback_chars')} "
                f"deduplicated={input_compaction.get('deduplicated_steps')} "
                f"coverage={float(input_compaction.get('post_action_feedback_coverage', 0.0)):.3f} "
                f"raw_bad_step={input_compaction.get('raw_bad_step_count', 0)}",
                flush=True,
            )
            try:
                l1_raw = _teacher_chat_json(teacher, l1_prompt, l1_input)
            except Exception as exc:
                l1_raw = {"evidence": [], "error": str(exc)}
            drafts = [
                draft
                for draft in _drafts_from_l1_response(
                    l1_raw,
                    default_experience_type=default_experience_type,
                )
                if _source_type_from_experience(draft.get("experience_type")) == source_type
            ]
            source_summaries[source_type]["draft_count"] += len(drafts)

            kept_drafts: List[Dict[str, Any]] = []
            dropped_evidence: List[Dict[str, Any]] = []
            global_bundle_idx = epoch_idx * max(1, bundles_per_epoch * 2) + l1_call_index - 1
            for draft in drafts:
                eid = f"E{evidence_counter:06d}"
                evidence_counter += 1
                candidate, materialize_error = _materialize_evidence_candidate(
                    draft=draft,
                    evidence_id=eid,
                    records_by_id=records_by_id,
                    source_minibatch_id=source_minibatch_id,
                    source_minibatch_task_ids=minibatch_task_ids,
                )
                if candidate is None:
                    dropped_evidence.append({"draft": draft, "reason": materialize_error})
                    continue
                l1_judge_payload = _build_l1_skill_judge_payload(
                    skill_md=skill_md,
                    evidence_candidate=candidate,
                    dataset=dataset,
                    args=args,
                )
                l1_label = _l1_skill_judge_label(
                    teacher=teacher,
                    skill_md=skill_md,
                    evidence_candidate=candidate,
                    dataset=dataset,
                    args=args,
                    payload=l1_judge_payload,
                )
                l1_judge_rows.append(
                    {
                        "dataset": dataset,
                        "epoch_index": epoch_idx,
                        "epoch_bundle_index": l1_call_index - 1,
                        "global_bundle_index": global_bundle_idx,
                        "source_type": source_type,
                        "system_prompt": l1_judge_prompt,
                        "input": l1_judge_payload,
                        "output": l1_label,
                    }
                )
                if l1_label["decision"] != "keep":
                    dropped_evidence.append({"evidence": candidate, "reason": l1_label.get("reason")})
                    continue
                card = dict(candidate)
                card["l1_skilljudge_reason"] = l1_label.get("reason", "")
                card["round"] = global_bundle_idx
                card["epoch_index"] = epoch_idx
                card["epoch_bundle_index"] = l1_call_index - 1
                card["global_bundle_index"] = global_bundle_idx
                card["archived"] = False
                _normalize_evidence_runtime_state(card)
                kept_drafts.append(draft)
                evidence_rows.append(card)
                epoch_evidence_ids.append(str(card.get("evidence_id") or ""))
                seen_replay_task_ids = set()
                for trig in _iter_card_trigger_ranges(card):
                    task_id = str(trig.get("task_id") or "")
                    if not task_id or task_id in seen_replay_task_ids:
                        continue
                    seen_replay_task_ids.add(task_id)
                    raw_result = raw_by_id.get(task_id)
                    if raw_result is not None:
                        replay_source_by_evidence_id[_replay_source_key(card.get("evidence_id"), task_id)] = raw_result
                        replay_source_rows.append(
                            {
                                "evidence_id": card.get("evidence_id"),
                                "round": global_bundle_idx,
                                "epoch_index": epoch_idx,
                                "epoch_bundle_index": l1_call_index - 1,
                                "global_bundle_index": global_bundle_idx,
                                "source_type": source_type,
                                "task_id": task_id,
                                "raw_result": raw_result,
                            }
                        )
            source_summaries[source_type]["kept_count"] += len(kept_drafts)
            source_summaries[source_type]["dropped_count"] += len(dropped_evidence)
            l1_manager_rows.append(
                {
                    "dataset": dataset,
                    "epoch_index": epoch_idx,
                    "epoch_bundle_index": l1_call_index - 1,
                    "global_bundle_index": global_bundle_idx,
                    "source_type": source_type,
                    "source_minibatch_id": source_minibatch_id,
                    "source_minibatch_task_ids": minibatch_task_ids,
                    "input_compaction": input_compaction,
                    "system_prompt": l1_prompt,
                    "input": l1_input,
                    "output": {"evidence": kept_drafts},
                    "dropped_evidence": dropped_evidence,
                }
            )

    summary = {
        "failure_trajectory_count": len(failure_results),
        "success_trajectory_count": len(success_results),
        "l1_trajectory_minibatch_size": minibatch_size,
        "source_summaries": source_summaries,
        "evidence_count": len(epoch_evidence_ids),
    }
    print(
        f"[epoch {epoch_idx + 1} evidence] "
        f"failure_traj={len(failure_results)} success_traj={len(success_results)} "
        f"evidence={len(epoch_evidence_ids)} "
        f"failure_kept={source_summaries['failure']['kept_count']} "
        f"success_kept={source_summaries['success']['kept_count']}",
        flush=True,
    )
    return epoch_evidence_ids, evidence_counter, summary


def run_eviskill_evolution(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _validate_evidence_working_branch_args(args)
    no_resume = bool(getattr(args, "no_resume", False))
    resume = not no_resume
    had_existing_generation = bool(
        (out / "evolving_skill.md").exists()
        or (out / "dual_layer_generation_logs.json").exists()
        or any(out.glob("epoch_*"))
    )
    rng = random.Random(args.seed)
    dataset = str(getattr(args, "dataset", "alfworld") or "alfworld")
    if dataset.lower() == "alfworld":
        split_alignment = preflight_alfworld_path_splits(
            train_manifest=str(args.manifest),
            validation_manifest=getattr(args, "selection_manifest", None),
            test_manifest=getattr(args, "test_manifest", None),
        )
        save_json(str(out / "split_alignment.json"), split_alignment)
        if split_alignment.get("enabled"):
            split_counts = {
                name: int((info or {}).get("task_count", 0))
                for name, info in (split_alignment.get("splits") or {}).items()
            }
            print(
                f"[ALFWorld split preflight] counts={split_counts} "
                f"official_source_layout={bool(split_alignment.get('official_source_layout'))} "
                "overlap=0 missing_gamefiles=0",
                flush=True,
            )
    elif dataset.lower() == "appworld":
        split_alignment = preflight_appworld_splits(
            data_dir=str(getattr(args, "appworld_data_dir", None) or ""),
            train_manifest=str(args.manifest),
            validation_manifest=getattr(args, "selection_manifest", None),
            test_manifest=getattr(args, "test_manifest", None),
        )
        save_json(str(out / "split_alignment.json"), split_alignment)
        split_counts = {
            name: {
                "tasks": int((info or {}).get("task_count", 0)),
                "scenarios": int((info or {}).get("scenario_count", 0)),
            }
            for name, info in (split_alignment.get("splits") or {}).items()
        }
        print(
            f"[AppWorld split preflight] counts={split_counts} overlap=0 scenario_overlap=0",
            flush=True,
        )
    manifest = limit_manifest(load_json(args.manifest), getattr(args, "limit_tasks", None))
    skill_md_path = Path(getattr(args, "initial_skill_md", None) or args.skill_bank)
    if not skill_md_path.exists():
        raise FileNotFoundError(f"Dual-layer markdown evolution needs an initial skill.md file: {skill_md_path}")
    initial_skill_md = skill_md_path.read_text(encoding="utf-8")
    if _is_scienceworld_dataset(dataset):
        initial_skill_md = _ensure_recovery_rules_section(initial_skill_md)
    evolving_skill_path = out / "evolving_skill.md"
    if not evolving_skill_path.exists() or no_resume:
        atomic_write_text(evolving_skill_path, initial_skill_md)
    elif _is_scienceworld_dataset(dataset):
        existing_skill_md = evolving_skill_path.read_text(encoding="utf-8")
        updated_skill_md = _ensure_recovery_rules_section(existing_skill_md)
        if updated_skill_md != existing_skill_md:
            atomic_write_text(evolving_skill_path, updated_skill_md)

    working_skill_path, working_branch_state_path, working_branch_state = (
        _initialize_evidence_working_branch(
            out=out,
            dataset=dataset,
            evolving_skill_path=evolving_skill_path,
            initial_skill_md=evolving_skill_path.read_text(encoding="utf-8"),
            resume=resume,
            no_resume=no_resume,
            had_existing_generation=had_existing_generation,
            enabled=_evidence_working_branch_enabled(args),
            replay_mode=_replay_mode(args),
            policy=_working_branch_policy(args),
        )
    )
    rollout_skill_path = working_skill_path or evolving_skill_path
    if working_branch_state is not None:
        post_reject_per_edit_limit = _int_arg(
            getattr(args, "l2_replay_ranges_per_edit", 4), 4
        )
        post_reject_replay_cap = _int_arg(
            getattr(
                args,
                "post_reject_replay_cap",
                DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
            ),
            DEFAULT_WORKING_BRANCH_POST_REJECT_REPLAY_CAP,
        )
        provisional_cap = max(
            0,
            _int_arg(
                getattr(
                    args,
                    "working_branch_provisional_cap",
                    DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
                ),
                DEFAULT_WORKING_BRANCH_PROVISIONAL_CAP,
            ),
        )
        print(
            "[working branch] enabled validation_feedback=false "
            f"replay_mode={_replay_mode(args)} "
            "pre_validation_final_replay=false "
            f"post_reject_replay_per_edit={post_reject_per_edit_limit} "
            f"post_reject_replay_cap={post_reject_replay_cap} "
            f"provisional_cap={provisional_cap} "
            f"epoch_edit_budget={getattr(args, 'epoch_edit_budget', DEFAULT_EPOCH_EDIT_BUDGET)} "
            "comparison_task_cap=20 "
            f"epoch_trajectory_comparison_tasks="
            f"{getattr(args, 'epoch_trajectory_comparison_tasks', 0)}",
            flush=True,
        )
        configured_final_budget = _int_arg(
            getattr(args, "epoch_final_replay_ranges_per_edit", 0), 0
        )
        if configured_final_budget != 0 and not _no_replay_enabled(args):
            print(
                "[working branch] --epoch-final-replay-ranges-per-edit is ignored; "
                "the mode uses its fixed adaptive post-reject replay budget",
                flush=True,
            )
    if _no_replay_enabled(args):
        print(
            "[replay] mode=disabled local_action_replay=0 "
            "replay_attribution=false reflection=false "
            "working_post_reject=mechanical_selection",
            flush=True,
        )

    action_llm = build_action_llm(args)
    runner = _build_dual_layer_runner(args, dataset, action_llm)
    teacher = build_teacher(args)

    l1_manager_path = out / EPOCH_JSONL_FILENAMES["l1_skill_manager"]
    l1_judge_path = out / EPOCH_JSONL_FILENAMES["l1_skill_judge"]
    l2_manager_path = out / EPOCH_JSONL_FILENAMES["l2_skill_manager"]
    l2_judge_path = out / EPOCH_JSONL_FILENAMES["l2_skill_judge"]
    evidence_path = out / EPOCH_JSONL_FILENAMES["evidence_cards"]
    replay_source_path = out / EPOCH_JSONL_FILENAMES["evidence_replay_sources"]
    l2_candidates_path = out / EPOCH_JSONL_FILENAMES["l2_skill_candidates"]
    replay_trace_path = out / EPOCH_JSONL_FILENAMES["replay_label_traces"]
    selection_feedback_path = out / EPOCH_JSONL_FILENAMES["selection_feedback_cards"]
    logs_path = out / "dual_layer_generation_logs.json"

    if not resume:
        _remove_epoch_artifact_dirs(out)
        _remove_legacy_epoch_jsonl_files(out)
    l1_manager_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["l1_skill_manager"], resume=resume, legacy_path=l1_manager_path)
    l1_judge_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["l1_skill_judge"], resume=resume, legacy_path=l1_judge_path)
    l2_manager_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["l2_skill_manager"], resume=resume, legacy_path=l2_manager_path)
    l2_judge_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["l2_skill_judge"], resume=resume, legacy_path=l2_judge_path)
    evidence_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["evidence_cards"], resume=resume, legacy_path=evidence_path)
    replay_source_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["evidence_replay_sources"], resume=resume, legacy_path=replay_source_path)
    l2_candidate_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["l2_skill_candidates"], resume=resume, legacy_path=l2_candidates_path)
    replay_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["replay_label_traces"], resume=resume, legacy_path=replay_trace_path)
    selection_feedback_rows = _load_epoch_jsonl_rows(out, EPOCH_JSONL_FILENAMES["selection_feedback_cards"], resume=resume, legacy_path=selection_feedback_path)
    logs = load_json(str(logs_path)) if resume and logs_path.exists() else []
    replay_by_evidence: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for replay_row in replay_source_rows:
        if not isinstance(replay_row, dict):
            continue
        replay_by_evidence.setdefault(str(replay_row.get("evidence_id") or ""), {})[
            str(replay_row.get("task_id") or "")
        ] = replay_row.get("raw_result") or {}
    for index, evidence_row in enumerate(list(evidence_rows)):
        if not isinstance(evidence_row, dict):
            continue
        if _coerce_int(evidence_row.get("evidence_schema_version"), 0) >= EVIDENCE_SCHEMA_VERSION:
            continue
        if _normalize_proposed_edit(evidence_row.get("proposed_edit")) is None:
            raise ValueError(
                "Existing evidence uses the old idea-style schema. Start a fresh run or reuse "
                "an evidence directory whose cards contain proposed_edit."
            )
        evidence_id = str(evidence_row.get("evidence_id") or "")
        evidence_rows[index] = _upgrade_evidence_card_v4(
            evidence_row,
            replay_by_task_id=replay_by_evidence.get(evidence_id, {}),
            fallback_minibatch_id=f"resume_{evidence_id or index}",
        )
    _normalize_evidence_runtime_states(evidence_rows)
    if working_branch_state is not None and resume:
        completed_summary = working_branch_state.get("last_completed_epoch_summary")
        completed_stage = working_branch_state.get("stage") or {}
        if (
            isinstance(completed_summary, dict)
            and str(completed_stage.get("name") or "") == "complete"
        ):
            completed_epoch = _coerce_int(completed_summary.get("epoch_index"), -1)
            epoch_logs = [
                row for row in logs
                if _row_int(row or {}, "epoch_index", -1) == completed_epoch
            ]
            has_summary = any(
                "epoch_validation_decision" in row or "epoch_l2_window_count" in row
                for row in epoch_logs
            )
            if completed_epoch >= 0 and epoch_logs and not has_summary:
                epoch_logs[-1].update(_jsonable_copy(completed_summary))
                save_json(str(logs_path), logs)
                save_json(
                    str(_epoch_artifact_dir(out, completed_epoch) / "dual_layer_generation_logs.json"),
                    epoch_logs,
                )
                print(
                    f"[working branch resume] recovered completed epoch "
                    f"{completed_epoch + 1} summary from working_branch_state.json",
                    flush=True,
                )
    selection_tasks = _load_or_create_selection_tasks(
        out=out,
        manifest=manifest,
        args=args,
        resume=resume,
    )
    evolution_manifest = _exclude_selection_tasks(manifest, selection_tasks)
    if _evidence_working_branch_enabled(args) and len(evolution_manifest) != len(manifest):
        raise ValueError(
            "--use-evidence-working-branch requires the complete train manifest for rollout. "
            "Use a disjoint --selection-manifest instead of sampling validation tasks from train."
        )
    bundle_sampling_strategy = str(
        getattr(args, "bundle_sampling_strategy", "shuffle_epoch") or "shuffle_epoch"
    ).strip().lower()
    if bundle_sampling_strategy not in {"random", "shuffle_epoch", "family_epoch", "mixed_family_epoch"}:
        raise ValueError(
            "--bundle-sampling-strategy must be one of: random, shuffle_epoch, family_epoch, mixed_family_epoch; "
            f"got {bundle_sampling_strategy!r}"
        )
    if int(args.bundle_size) > 0 and len(evolution_manifest) < int(args.bundle_size):
        raise ValueError(
            "Not enough evolution tasks after excluding the L2 selection set: "
            f"evolution={len(evolution_manifest)} bundle_size={args.bundle_size} "
            f"selection={len(selection_tasks)} manifest={len(manifest)}"
        )

    epochs = max(1, int(getattr(args, "epochs", 1) or 1))
    bundles_per_epoch = max(1, (len(evolution_manifest) + int(args.bundle_size) - 1) // int(args.bundle_size))
    total_rounds = epochs * bundles_per_epoch

    if resume and logs:
        recovery_info = _recover_evaluation_error_epoch(
            out=out,
            logs=logs,
            l1_manager_rows=l1_manager_rows,
            l1_judge_rows=l1_judge_rows,
            l2_manager_rows=l2_manager_rows,
            l2_candidate_rows=l2_candidate_rows,
            l2_judge_rows=l2_judge_rows,
            evidence_rows=evidence_rows,
            replay_source_rows=replay_source_rows,
            replay_rows=replay_rows,
            selection_feedback_rows=selection_feedback_rows,
        )
        if recovery_info is not None:
            failed_epoch = int(recovery_info["failed_epoch"])
            print(
                f"[resume validation recovery] epoch {failed_epoch + 1} evaluation_error "
                f"preserved_bundles={recovery_info['preserved_bundle_count']} "
                f"preserved_l1={str(bool(recovery_info['preserved_l1'])).lower()} "
                f"restored_archived_evidence={recovery_info['restored_archived_evidence']} "
                f"restored_deferred_evidence={recovery_info['restored_deferred_evidence']} "
                f"removed_later_epoch_dirs={recovery_info['removed_later_epoch_dirs']}; "
                "rerunning epoch-boundary L2 and validation",
                flush=True,
            )

    resume_shuffle_epoch_seed_bases: Dict[int, int] = {}
    if resume and logs and bundle_sampling_strategy == "shuffle_epoch":
        shuffle_plans = _resume_shuffle_epoch_seed_plans(
            manifest=evolution_manifest,
            bundle_size=int(args.bundle_size),
            seed=int(args.seed),
            logs=logs,
        )
        for plan_epoch_idx, plan in shuffle_plans.items():
            seed_base = int(plan.get("epoch_seed_base", 1))
            matched_prefix = int(plan.get("matched_prefix_bundles", 0))
            logged_count = int(plan.get("logged_bundle_count", 0))
            resume_shuffle_epoch_seed_bases[plan_epoch_idx] = seed_base
            if matched_prefix < logged_count:
                _truncate_incomplete_epoch_bundle_suffix(
                    out=out,
                    logs=logs,
                    epoch_idx=plan_epoch_idx,
                    first_bundle_idx=matched_prefix,
                )
                save_json(str(logs_path), logs)
                print(
                    f"[resume shuffle migration] epoch {plan_epoch_idx + 1} "
                    f"seed_base={seed_base} preserved_bundles={matched_prefix} "
                    f"discarded_mixed_suffix={logged_count - matched_prefix}",
                    flush=True,
                )
            elif seed_base == 0:
                print(
                    f"[resume shuffle migration] epoch {plan_epoch_idx + 1} "
                    f"continuing legacy seed plan from {logged_count} completed bundle(s)",
                    flush=True,
                )

    if resume and logs:
        rewind_epoch, rewind_reason = _find_resume_rollout_rewind_epoch(
            out,
            logs=logs,
        )
        if rewind_epoch is not None:
            affected_logs = [
                row
                for row in logs
                if _row_int(row or {}, "epoch_index", -1) >= int(rewind_epoch)
            ]
            has_committed_skill = any(
                _row_int(row or {}, "epoch_committed_edit_count", 0) > 0
                or str((row or {}).get("epoch_validation_decision") or "").lower() == "commit"
                for row in affected_logs
            )
            if has_committed_skill:
                raise RuntimeError(
                    f"Cannot automatically rewind incomplete epoch {rewind_epoch + 1}: "
                    "a later log reports a committed skill update. Start a fresh output directory "
                    "or restore the skill checkpoint for that epoch."
                )

            def keep_before_rewind(row: Dict[str, Any]) -> bool:
                row_epoch = _row_epoch_index(row or {})
                return row_epoch is None or row_epoch < int(rewind_epoch)

            logs[:] = [row for row in logs if keep_before_rewind(row)]
            for rows in (
                l1_manager_rows,
                l1_judge_rows,
                l2_manager_rows,
                l2_candidate_rows,
                l2_judge_rows,
                evidence_rows,
                replay_source_rows,
                replay_rows,
                selection_feedback_rows,
            ):
                rows[:] = [row for row in rows if keep_before_rewind(row)]
            for epoch_dir in _epoch_artifact_dirs(out):
                epoch_dir_idx = _parse_epoch_artifact_dir(epoch_dir)
                if epoch_dir_idx is not None and epoch_dir_idx >= int(rewind_epoch):
                    shutil.rmtree(epoch_dir)
            save_json(str(logs_path), logs)
            print(
                f"[resume recovery] epoch {rewind_epoch + 1} has rollout logs but no "
                f"recoverable train trajectories ({rewind_reason}); rerunning that epoch "
                "from bundle 1 instead of entering L2 with an empty evidence pool",
                flush=True,
            )

    completed_bundle_count = len(logs) if resume else 0
    evidence_counter = _next_evidence_counter(evidence_rows)
    reject_cooldown_pool_size = 0
    for row in reversed(logs if isinstance(logs, list) else []):
        l2_row = (row or {}).get("l2") or {}
        if l2_row.get("triggered") and (l2_row.get("selection") or {}).get("decision") == "reject":
            reject_cooldown_pool_size = int(
                ((l2_row.get("evidence_pool") or {}).get("active"))
                or l2_row.get("evidence_pool_size_after")
                or 0
            )
            break
        if int(l2_row.get("committed_count") or 0) > 0:
            break
    replay_source_by_evidence_id = {
        _replay_source_key(row.get("evidence_id"), row.get("task_id")): row.get("raw_result")
        for row in replay_source_rows
        if isinstance(row, dict) and row.get("evidence_id") and row.get("task_id")
    }

    reuse_evidence_path = _reuse_evidence_arg(args)
    reuse_evidence_missing = bool(reuse_evidence_path is not None and not reuse_evidence_path.exists())
    if reuse_evidence_missing:
        print(
            f"[reuse evidence] path missing {reuse_evidence_path}; running from scratch",
            flush=True,
        )
    start_epoch = min(epochs, completed_bundle_count // bundles_per_epoch)
    start_epoch_bundle = completed_bundle_count % bundles_per_epoch
    resume_epoch_boundary_l2 = False
    if (
        resume
        and logs
        and completed_bundle_count > 0
        and completed_bundle_count % bundles_per_epoch == 0
    ):
        last_full_epoch = min(epochs - 1, completed_bundle_count // bundles_per_epoch - 1)
        last_epoch_logs = [
            row
            for row in logs
            if _row_int(row or {}, "epoch_index", -1) == last_full_epoch
        ]
        last_epoch_has_l2_summary = any(
            "epoch_l2_window_count" in (row or {})
            or "epoch_validation_decision" in (row or {})
            for row in last_epoch_logs
        )
        if len(last_epoch_logs) >= bundles_per_epoch and not last_epoch_has_l2_summary:
            start_epoch = last_full_epoch
            start_epoch_bundle = bundles_per_epoch
            resume_epoch_boundary_l2 = True
            print(
                f"[resume] epoch {last_full_epoch + 1} train bundles are complete "
                "but epoch skill edit summary is missing; resuming at epoch-boundary skill edit",
                flush=True,
            )

    if bundle_sampling_strategy == "random":
        for _ in range(completed_bundle_count):
            sample_random_bundle(evolution_manifest, args.bundle_size, rng=rng)

    epoch_bar = progress_iter(
        range(start_epoch, epochs),
        total=max(0, epochs - start_epoch),
        desc="dual-layer-md epochs",
        disable=getattr(args, "no_progress", False),
    )
    for epoch_idx in epoch_bar:
        shuffle_epoch_seed_base = resume_shuffle_epoch_seed_bases.get(epoch_idx, 1)
        if bundle_sampling_strategy == "shuffle_epoch":
            rollout_shuffle_seed = int(args.seed) + (
                int(epoch_idx) + int(shuffle_epoch_seed_base)
            ) * 1000
            print(
                f"[epoch {epoch_idx + 1} sampling] strategy=shuffle_epoch "
                f"seed={rollout_shuffle_seed} tasks={len(evolution_manifest)} "
                f"bundles={bundles_per_epoch} coverage=all_once",
                flush=True,
            )
        epoch_start_skill_md = rollout_skill_path.read_text(encoding="utf-8")
        epoch_start_working_revision = (
            str((working_branch_state or {}).get("working_revision") or "")
            if working_branch_state is not None
            else ""
        )
        bundle_start = start_epoch_bundle if epoch_idx == start_epoch else 0
        epoch_evidence_ids = [
            str(row.get("evidence_id") or "")
            for row in evidence_rows
            if _row_int(row, "epoch_index", -1) == epoch_idx
            and str(row.get("evidence_id") or "")
            and str(row.get("l2_status") or EVIDENCE_STATUS_ACTIVE).lower() == EVIDENCE_STATUS_ACTIVE
            and not bool(row.get("archived"))
        ]
        epoch_train_results: List[Dict[str, Any]] = []
        if (
            bundle_start > 0
            and not _epoch_l1_is_complete(logs, epoch_idx)
            and not _epoch_uses_reused_evidence(logs, epoch_idx)
        ):
            restored_results, restore_error = _checkpoint_results_for_logged_bundles(
                out,
                epoch_idx=epoch_idx,
                logs=logs,
                bundle_limit=bundle_start,
            )
            if restore_error:
                raise RuntimeError(
                    f"Epoch {epoch_idx + 1} rollout checkpoint recovery failed after startup "
                    f"validation: {restore_error}"
                )
            epoch_train_results.extend(restored_results)
            print(
                f"[resume] epoch {epoch_idx + 1} restored train trajectories "
                f"bundles={bundle_start} tasks={len(restored_results)}; "
                "remaining rollout/L1 will use the complete epoch trajectory set",
                flush=True,
            )
        reused_epoch_info: Optional[Dict[str, Any]] = None
        if (
            reuse_evidence_path is not None
            and not reuse_evidence_missing
            and bundle_start == 0
            and not epoch_evidence_ids
        ):
            reuse_epoch_dir = _resolve_reuse_evidence_epoch_dir(reuse_evidence_path, epoch_idx)
            if reuse_epoch_dir is not None:
                epoch_evidence_ids, reused_epoch_info, evidence_counter = _import_reused_epoch_evidence(
                    source_dir=reuse_epoch_dir,
                    epoch_idx=epoch_idx,
                    bundles_per_epoch=bundles_per_epoch,
                    evidence_rows=evidence_rows,
                    replay_source_rows=replay_source_rows,
                    replay_source_by_evidence_id=replay_source_by_evidence_id,
                    logs=logs,
                    evidence_counter=evidence_counter,
                )
                if bundle_sampling_strategy == "random":
                    for _ in range(bundles_per_epoch):
                        sample_random_bundle(evolution_manifest, args.bundle_size, rng=rng)
                bundle_start = bundles_per_epoch
                print(
                    f"[reuse evidence epoch {epoch_idx + 1}] source={reuse_epoch_dir} "
                    f"evidence={reused_epoch_info.get('reuse_evidence_count', 0)} "
                    f"replay_sources={reused_epoch_info.get('reuse_replay_source_count', 0)} "
                    "skip_rollout=yes",
                    flush=True,
                )
                _save_epoch_artifacts(
                    out,
                    epoch_idx,
                    l1_manager_rows=l1_manager_rows,
                    l1_judge_rows=l1_judge_rows,
                    l2_manager_rows=l2_manager_rows,
                    l2_candidate_rows=l2_candidate_rows,
                    l2_judge_rows=l2_judge_rows,
                    evidence_rows=evidence_rows,
                    replay_source_rows=replay_source_rows,
                    replay_rows=replay_rows,
                    selection_feedback_rows=selection_feedback_rows,
                    logs=logs,
                )
                save_json(str(logs_path), logs)
            elif _parse_epoch_artifact_dir(reuse_evidence_path) is None:
                print(
                    f"[reuse evidence epoch {epoch_idx + 1}] source missing "
                    f"{reuse_evidence_path / f'epoch_{epoch_idx + 1}'}; running rollout/L1",
                    flush=True,
                )
        bundle_bar = progress_iter(
            range(bundle_start, bundles_per_epoch),
            total=max(0, bundles_per_epoch - bundle_start),
            desc=f"epoch {epoch_idx + 1}/{epochs} bundles",
            disable=getattr(args, "no_progress", False),
        )
        for epoch_bundle_idx in bundle_bar:
            global_bundle_idx = epoch_idx * bundles_per_epoch + epoch_bundle_idx
            skill_md = rollout_skill_path.read_text(encoding="utf-8")
            task_ids, bundle_mode = _sample_evolution_bundle(
                evolution_manifest,
                args.bundle_size,
                strategy=bundle_sampling_strategy,
                rng=rng,
                seed=int(args.seed),
                round_idx=global_bundle_idx,
                family_group_frequency=int(getattr(args, "family_group_frequency", 2) or 0),
                shuffle_epoch_seed_base=shuffle_epoch_seed_base,
            )
            bundle_results = runner.run_task_set(
                task_ids,
                MarkdownSkillBank(str(rollout_skill_path)),
                {},
                desc=(
                    f"epoch {epoch_idx + 1}/{epochs} "
                    f"bundle {epoch_bundle_idx + 1}/{bundles_per_epoch} rollout"
                ),
            )
            _save_epoch_train_rollout_checkpoint(
                out,
                epoch_idx=epoch_idx,
                epoch_bundle_idx=epoch_bundle_idx,
                task_ids=task_ids,
                results=bundle_results,
            )
            epoch_train_results.extend(bundle_results)
            print(
                f"[epoch {epoch_idx + 1}/{epochs} train running] "
                f"bundles={epoch_bundle_idx + 1}/{bundles_per_epoch} "
                f"tasks={len(epoch_train_results)} "
                f"score_avg={_average_score(epoch_train_results):.4f} "
                f"success_rate={_success_rate(epoch_train_results):.4f} "
                f"success={_success_count(epoch_train_results)}/{len(epoch_train_results)} "
                f"steps_avg={_average_steps(epoch_train_results):.2f}",
                flush=True,
            )
            epoch_evidence_pool = [
                row
                for row in evidence_rows
                if str(row.get("evidence_id") or "") in set(epoch_evidence_ids)
            ]
            l2_info = {
                "triggered": False,
                "candidate_count": 0,
                "accepted_count": 0,
                "reason": "epoch_update_deferred",
                "evidence_pool": _evidence_pool_counts(epoch_evidence_pool),
            }

            logs.append(
                {
                    "epoch_index": epoch_idx,
                    "epoch_bundle_index": epoch_bundle_idx,
                    "global_bundle_index": global_bundle_idx,
                    "round": global_bundle_idx,
                    "bundle_sampling_strategy": bundle_sampling_strategy,
                    "bundle_mode": bundle_mode,
                    "task_ids": task_ids,
                    "draft_count": 0,
                    "kept_count": 0,
                    "dropped_evidence": [],
                    "l1_deferred_until_epoch_end": True,
                    "evidence_pool_size": len(epoch_evidence_pool),
                    "evidence_pool": _evidence_pool_counts(epoch_evidence_pool),
                    "l2": l2_info,
                }
            )

            _save_epoch_artifacts(
                out,
                epoch_idx,
                l1_manager_rows=l1_manager_rows,
                l1_judge_rows=l1_judge_rows,
                l2_manager_rows=l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                evidence_rows=evidence_rows,
                replay_source_rows=replay_source_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                logs=logs,
            )
            save_json(str(logs_path), logs)
            if hasattr(bundle_bar, "set_postfix"):
                counts = _evidence_pool_counts(epoch_evidence_pool)
                bundle_bar.set_postfix(
                    {
                        "active": counts["active"],
                        "consumed": counts["consumed"],
                        "total": counts["total"],
                    }
                )

        if epoch_train_results:
            epoch_l1_skill_md = epoch_start_skill_md
            new_epoch_evidence_ids, evidence_counter, epoch_l1_summary = _run_epoch_l1_evidence_extraction(
                dataset=dataset,
                teacher=teacher,
                args=args,
                skill_md=epoch_l1_skill_md,
                epoch_idx=epoch_idx,
                bundles_per_epoch=bundles_per_epoch,
                epoch_train_results=epoch_train_results,
                evidence_counter=evidence_counter,
                evidence_rows=evidence_rows,
                replay_source_rows=replay_source_rows,
                replay_source_by_evidence_id=replay_source_by_evidence_id,
                l1_manager_rows=l1_manager_rows,
                l1_judge_rows=l1_judge_rows,
                manifest_task_order=[get_task_id(item) for item in evolution_manifest],
            )
            epoch_evidence_ids.extend(new_epoch_evidence_ids)
            if working_branch_state is not None:
                new_id_set = set(new_epoch_evidence_ids)
                for card in evidence_rows:
                    if str(card.get("evidence_id") or "") not in new_id_set:
                        continue
                    card["evidence_mode"] = "trajectory"
                    card["observed_skill_revision"] = epoch_start_working_revision
                    card["working_skill_revision"] = epoch_start_working_revision
            if logs:
                logs[-1]["epoch_l1_summary"] = epoch_l1_summary
            _save_epoch_artifacts(
                out,
                epoch_idx,
                l1_manager_rows=l1_manager_rows,
                l1_judge_rows=l1_judge_rows,
                l2_manager_rows=l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                evidence_rows=evidence_rows,
                replay_source_rows=replay_source_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                logs=logs,
            )
            save_json(str(logs_path), logs)

        epoch_comparison_budget = _int_arg(
            getattr(args, "epoch_trajectory_comparison_tasks", 0), 0
        )
        queued_comparison_task_ids = {
            str(row.get("task_id") or "")
            for row in ((working_branch_state or {}).get("comparison_queue") or [])
            if str(row.get("task_id") or "")
        }
        comparison_train_results = epoch_train_results
        if (
            working_branch_state is not None
            and not comparison_train_results
            and (
                (working_branch_state.get("comparison_queue") or [])
                or (epoch_comparison_budget > 0 and epoch_idx > 0)
            )
        ):
            comparison_train_results = list(
                _working_branch_epoch_results_by_id(out, epoch_idx).values()
            )
        if working_branch_state is not None and comparison_train_results:
            contrastive_ids, evidence_counter, contrastive_summary = (
                _run_working_branch_contrastive_evidence(
                    dataset=dataset,
                    teacher=teacher,
                    args=args,
                    out=out,
                    epoch_idx=epoch_idx,
                    working_skill_md=epoch_start_skill_md,
                    working_branch_state=working_branch_state,
                    epoch_train_results=comparison_train_results,
                    evidence_counter=evidence_counter,
                    evidence_rows=evidence_rows,
                    replay_source_rows=replay_source_rows,
                    replay_source_by_evidence_id=replay_source_by_evidence_id,
                    l1_manager_rows=l1_manager_rows,
                )
            )
            epoch_evidence_ids.extend(contrastive_ids)
            epoch_comparison_summary: Optional[Dict[str, Any]] = None
            if epoch_comparison_budget > 0 and epoch_idx > 0:
                (
                    epoch_comparison_ids,
                    evidence_counter,
                    epoch_comparison_summary,
                ) = _run_epoch_trajectory_comparison_evidence(
                    dataset=dataset,
                    teacher=teacher,
                    args=args,
                    out=out,
                    epoch_idx=epoch_idx,
                    working_skill_md=epoch_start_skill_md,
                    working_revision=epoch_start_working_revision,
                    epoch_train_results=comparison_train_results,
                    excluded_task_ids=queued_comparison_task_ids,
                    evidence_counter=evidence_counter,
                    evidence_rows=evidence_rows,
                    replay_source_rows=replay_source_rows,
                    replay_source_by_evidence_id=replay_source_by_evidence_id,
                    l1_manager_rows=l1_manager_rows,
                )
                epoch_evidence_ids.extend(epoch_comparison_ids)
            working_branch_state["stage"] = {
                "epoch_index": epoch_idx,
                "name": "l1_complete",
            }
            persist_working_branch_state(
                working_branch_state_path,
                working_branch_state,
                validated_skill_md=evolving_skill_path.read_text(encoding="utf-8"),
                working_skill_md=rollout_skill_path.read_text(encoding="utf-8"),
            )
            if logs:
                logs[-1]["epoch_contrastive_evidence"] = contrastive_summary
                if epoch_comparison_summary is not None:
                    logs[-1]["epoch_trajectory_comparison_evidence"] = (
                        epoch_comparison_summary
                    )
            print(
                f"[epoch {epoch_idx + 1} contrastive evidence] "
                f"queued={contrastive_summary.get('queued', 0)} "
                f"compared={contrastive_summary.get('compared', 0)} "
                f"evidence={contrastive_summary.get('evidence_count', 0)} "
                f"support_events={contrastive_summary.get('support_events', 0)}",
                flush=True,
            )
            if epoch_comparison_summary is not None:
                print(
                    f"[epoch {epoch_idx + 1} trajectory comparison] "
                    f"budget={epoch_comparison_summary.get('budget', 0)} "
                    f"common={epoch_comparison_summary.get('common', 0)} "
                    f"excluded_queue={epoch_comparison_summary.get('excluded_queue', 0)} "
                    f"regressed={epoch_comparison_summary.get('regressed', 0)} "
                    f"persistent_fail={epoch_comparison_summary.get('persistent_fail', 0)} "
                    f"selected_regressed="
                    f"{epoch_comparison_summary.get('selected_regressed', 0)} "
                    f"selected_persistent_fail="
                    f"{epoch_comparison_summary.get('selected_persistent_fail', 0)} "
                    f"judge_calls={epoch_comparison_summary.get('judge_calls', 0)} "
                    f"evidence={epoch_comparison_summary.get('evidence_count', 0)} "
                    f"no_evidence={epoch_comparison_summary.get('no_evidence', 0)}",
                    flush=True,
                )
            _save_epoch_artifacts(
                out,
                epoch_idx,
                l1_manager_rows=l1_manager_rows,
                l1_judge_rows=l1_judge_rows,
                l2_manager_rows=l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                evidence_rows=evidence_rows,
                replay_source_rows=replay_source_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                logs=logs,
            )
            save_json(str(logs_path), logs)

        # Deferred evidence is intentionally carried across epochs. L2 sees every
        # still-active train EvidenceCard, not only cards created in this epoch.
        epoch_evidence_pool = _active_evidence_rows(evidence_rows)
        epoch_rollout_task_count = sum(
            len((row or {}).get("task_ids") or [])
            for row in logs or []
            if _row_int(row or {}, "epoch_index", -1) == epoch_idx
        )
        if (
            working_branch_state is not None
            and epoch_rollout_task_count != len(evolution_manifest)
        ):
            raise RuntimeError(
                f"Epoch {epoch_idx + 1} working branch requires one complete train pass: "
                f"logged_tasks={epoch_rollout_task_count} expected={len(evolution_manifest)}"
            )
        should_run_epoch_l2 = bool(logs) and (
            bundle_start < bundles_per_epoch
            or (resume_epoch_boundary_l2 and epoch_idx == start_epoch)
            or reused_epoch_info is not None
        )
        if should_run_epoch_l2:
            epoch_summary, reject_cooldown_pool_size = _run_epoch_l2_update(
                dataset=dataset,
                teacher=teacher,
                runner=runner,
                out=out,
                args=args,
                epoch_idx=epoch_idx,
                epoch_rollout_task_count=epoch_rollout_task_count,
                current_skill_path=rollout_skill_path,
                validation_skill_path=evolving_skill_path,
                working_branch_state=working_branch_state,
                working_branch_state_path=working_branch_state_path,
                evidence_pool=epoch_evidence_pool,
                evidence_rows=evidence_rows,
                replay_source_by_evidence_id=replay_source_by_evidence_id,
                l2_manager_rows=l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                selection_tasks=selection_tasks,
                reject_cooldown_pool_size=reject_cooldown_pool_size,
            )
            memory_info = _run_longitudinal_memory_update(
                dataset=dataset,
                teacher=teacher,
                runner=runner,
                out=out,
                args=args,
                epoch_idx=epoch_idx,
                previous_skill_md=epoch_start_skill_md,
                current_skill_path=rollout_skill_path,
                evolution_manifest=evolution_manifest,
                committed_edits=epoch_summary.get("epoch_committed_edits") or [],
                selection_tasks=selection_tasks,
            )
            epoch_summary["longitudinal_memory"] = memory_info
            logs[-1].update(epoch_summary)
            epoch_update_mode = str(
                epoch_summary.get("epoch_skill_update_mode")
                or DEFAULT_EPOCH_SKILL_UPDATE_MODE
            )
            update_detail = (
                f"merged_edits={(epoch_summary.get('epoch_global_merge') or {}).get('selected_edit_count', 0)} "
                f"validation={epoch_summary.get('epoch_validation_decision')}"
            )
            print(
                "[epoch "
                f"{epoch_idx + 1}/{epochs} L2] windows={epoch_summary.get('epoch_l2_window_count', 0)} "
                f"mode={epoch_update_mode} "
                f"candidate_edits={epoch_summary.get('epoch_candidate_edit_count', 0)} "
                f"accepted_edits={epoch_summary.get('epoch_accepted_edit_count', 0)} "
                f"{update_detail} "
                f"current={epoch_summary.get('epoch_validation_current_score')} "
                f"current_success_rate={epoch_summary.get('epoch_validation_current_success_rate')} "
                f"candidate={epoch_summary.get('epoch_validation_candidate_score')} "
                f"candidate_success_rate={epoch_summary.get('epoch_validation_candidate_success_rate')}",
                flush=True,
            )
            if memory_info.get("enabled"):
                print(
                    f"[epoch {epoch_idx + 1}/{epochs} longitudinal memory] "
                    f"action={memory_info.get('action')} "
                    f"samples={memory_info.get('sample_count', 0)} "
                    f"pairs={memory_info.get('pair_count', 0)}",
                    flush=True,
                )

        _save_epoch_artifacts(
            out,
            epoch_idx,
            l1_manager_rows=l1_manager_rows,
            l1_judge_rows=l1_judge_rows,
            l2_manager_rows=l2_manager_rows,
            l2_candidate_rows=l2_candidate_rows,
            l2_judge_rows=l2_judge_rows,
            evidence_rows=evidence_rows,
            replay_source_rows=replay_source_rows,
            replay_rows=replay_rows,
            selection_feedback_rows=selection_feedback_rows,
            logs=logs,
            epoch_final_skill_md=evolving_skill_path.read_text(encoding="utf-8"),
            epoch_working_skill_md=(
                rollout_skill_path.read_text(encoding="utf-8")
                if working_branch_state is not None
                else None
            ),
        )
        for previous_epoch_idx in range(epoch_idx):
            _save_epoch_artifacts(
                out,
                previous_epoch_idx,
                l1_manager_rows=l1_manager_rows,
                l1_judge_rows=l1_judge_rows,
                l2_manager_rows=l2_manager_rows,
                l2_candidate_rows=l2_candidate_rows,
                l2_judge_rows=l2_judge_rows,
                evidence_rows=evidence_rows,
                replay_source_rows=replay_source_rows,
                replay_rows=replay_rows,
                selection_feedback_rows=selection_feedback_rows,
                logs=logs,
            )
        save_json(str(logs_path), logs)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(
                {
                    "epoch": epoch_idx + 1,
                    "l2_rows": len(l2_manager_rows),
                }
            )

    epoch_artifact_dirs = _epoch_artifact_dirs(out)
    epoch_artifacts = {
        epoch_dir.name: {
            **{
                key: str(epoch_dir / filename)
                for key, filename in EPOCH_JSONL_FILENAMES.items()
                if (epoch_dir / filename).exists()
            },
            **(
                {"evolving_skill_md": str(epoch_dir / "evolving_skill.md")}
                if (epoch_dir / "evolving_skill.md").exists()
                else {}
            ),
            **(
                {"working_skill_md": str(epoch_dir / "working_skill.md")}
                if (epoch_dir / "working_skill.md").exists()
                else {}
            ),
        }
        for epoch_dir in epoch_artifact_dirs
    }
    artifact_lists = {
        key: [
            str(epoch_dir / filename)
            for epoch_dir in epoch_artifact_dirs
            if (epoch_dir / filename).exists()
        ]
        for key, filename in EPOCH_JSONL_FILENAMES.items()
    }
    close_runner = getattr(runner, "close", None)
    if callable(close_runner):
        close_runner()
    return {
        "mode": "dual_layer_md",
        "evolving_skill_md": str(evolving_skill_path),
        **(
            {
                "working_skill_md": str(working_skill_path),
                "working_branch_state": str(working_branch_state_path),
            }
            if working_branch_state is not None
            else {}
        ),
        "epoch_evolving_skill_md": [
            str(epoch_dir / "evolving_skill.md")
            for epoch_dir in epoch_artifact_dirs
            if (epoch_dir / "evolving_skill.md").exists()
        ],
        **(
            {
                "epoch_working_skill_md": [
                    str(epoch_dir / "working_skill.md")
                    for epoch_dir in epoch_artifact_dirs
                    if (epoch_dir / "working_skill.md").exists()
                ]
            }
            if working_branch_state is not None
            else {}
        ),
        "epoch_artifact_dirs": [str(path) for path in epoch_artifact_dirs],
        "epoch_artifacts": epoch_artifacts,
        "evidence_cards": artifact_lists["evidence_cards"],
        "evidence_replay_sources": artifact_lists["evidence_replay_sources"],
        "l1_skill_manager": artifact_lists["l1_skill_manager"],
        "l1_skill_judge": artifact_lists["l1_skill_judge"],
        "l2_skill_manager": artifact_lists["l2_skill_manager"],
        "l2_skill_judge": artifact_lists["l2_skill_judge"],
        "l2_skill_candidates": artifact_lists["l2_skill_candidates"],
        "replay_label_traces": artifact_lists["replay_label_traces"],
        "selection_task_ids": str(out / "selection_task_ids.json"),
        "selection_feedback_cards": artifact_lists["selection_feedback_cards"],
        "logs": str(logs_path),
        "num_l1_skill_manager_rows": len(l1_manager_rows),
        "num_l1_skill_judge_rows": len(l1_judge_rows),
        "num_l2_skill_manager_rows": len(l2_manager_rows),
        "num_l2_skill_judge_rows": len(l2_judge_rows),
    }


def run_eviskill(args) -> Dict[str, Any]:
    return run_eviskill_evolution(args)
