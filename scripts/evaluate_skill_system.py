#!/usr/bin/env python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from wmsm.action_runtime import (
    ALFWORLD_ACTION_HISTORY_LENGTH,
    APPWORLD_ACTION_HISTORY_LENGTH,
    SCIENCEWORLD_ACTION_HISTORY_LENGTH,
    action_runtime_config,
    action_runtime_differences,
)
from wmsm.json_utils import load_json, load_jsonl, save_json, save_jsonl
from wmsm.llm_clients import build_llm_client
from wmsm.progress import progress_iter
from wmsm.records import normalize_task_results, task_records_payload
from wmsm.runtime import FixedJsonActionLLM, build_alfworld_runner, build_appworld_runner, build_scienceworld_runner
from wmsm.skill_markdown import MarkdownSkillBank
from wmsm.prompts import prompts_for_dataset
from wmsm.evidence_pipeline import (
    _apply_markdown_edits,
    _attach_l1_task_outcomes,
    _archive_evidence_rows,
    _commit_markdown_edits_sequential,
    _drafts_from_l1_response,
    _evidence_id_list,
    _evidence_pool_counts,
    _linked_evidence,
    _mark_evidence_attempts,
    _mark_l2_window_attempts,
    _mark_uncited_selected_evidence,
    _markdown_patch_items,
    _materialize_evidence_candidate,
    _normalize_evidence_runtime_state,
    _normalize_evidence_runtime_states,
    _normalize_markdown_edit,
    _public_evidence_cards,
    _recent_selection_feedback,
    _raw_result_by_task_id,
    _select_l2_evidence_window,
)


BASELINE_MODE = "baseline"
ACTION_ONLY_MODE = "action_agent"
ACTION_SKILL_MD_MODE = "action_agent_skill_md"
DUAL_LAYER_MD_MODE = "action_agent_dual_layer_md"
NO_SKILL_MODES = {BASELINE_MODE, ACTION_ONLY_MODE}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _setup_watch_log(args, out: Path):
    log_path = Path(args.watch_log_file) if args.watch_log_file else out / "watch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return fh


def _build_client(
    *,
    kind: str,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    max_tokens: int,
    load_in_4bit: bool,
    timeout: int,
    max_retries: int,
    retry_initial_sleep: float,
    retry_max_sleep: float,
    debug_dir: Optional[str],
    thinking: Optional[str],
    reasoning_effort: Optional[str],
    extra_body_json: Optional[str],
    context_length_retry: bool = False,
    credential_env_prefix: Optional[str] = None,
    proxy_url: Optional[str] = None,
):
    if not model:
        return None
    extra_body = json.loads(extra_body_json) if extra_body_json else None
    reasoning_effort = None if str(reasoning_effort or "").lower() == "none" else reasoning_effort
    return build_llm_client(
        kind or "openai",
        model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        load_in_4bit=load_in_4bit,
        timeout=timeout,
        max_retries=max_retries,
        retry_initial_sleep=retry_initial_sleep,
        retry_max_sleep=retry_max_sleep,
        debug_dir=debug_dir,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
        context_length_retry=context_length_retry,
        credential_env_prefix=credential_env_prefix,
        proxy_url=proxy_url,
    )


def _build_action_llm(args):
    if getattr(args, "dry_run_action", False):
        return FixedJsonActionLLM()
    return _build_client(
        kind=args.action_llm_kind,
        model=args.action_model,
        api_key=args.action_api_key,
        base_url=args.action_base_url,
        temperature=args.action_temperature,
        max_tokens=args.action_max_new_tokens,
        load_in_4bit=args.action_load_in_4bit,
        timeout=args.action_timeout,
        max_retries=args.action_max_retries,
        retry_initial_sleep=getattr(args, "action_retry_initial_sleep", 5.0),
        retry_max_sleep=getattr(args, "action_retry_max_sleep", 60.0),
        debug_dir=args.llm_debug_dir,
        thinking=args.action_thinking,
        reasoning_effort=args.action_reasoning_effort,
        extra_body_json=args.action_extra_body_json,
        context_length_retry=str(getattr(args, "dataset", "") or "").lower() == "appworld",
        credential_env_prefix="ACTION",
        proxy_url=getattr(args, "action_proxy_url", None),
    )


def _runner(args, action_llm):
    dataset = str(args.dataset or "alfworld").lower()
    parallel_workers = max(1, int(args.bundle_size or 1))
    if dataset == "alfworld":
        return build_alfworld_runner(
            alfworld_path=args.alfworld_path,
            alfworld_config=args.alfworld_config,
            alfworld_split=args.alfworld_split,
            action_llm=action_llm,
            max_steps=args.max_steps,
            top_k_general_skills=args.top_k_general_skills,
            top_k_task_specific_skills=args.top_k_task_specific_skills,
            general_skill_retrieval_threshold=args.general_skill_retrieval_threshold,
            task_specific_skill_retrieval_threshold=args.task_specific_skill_retrieval_threshold,
            task_specific_retrieval_scope=args.task_specific_retrieval_scope,
            max_resets_to_find_task=args.alfworld_max_resets_to_find_task,
            action_history_length=args.alfworld_action_history_length,
            parallel_workers=parallel_workers,
        )
    if dataset == "appworld":
        return build_appworld_runner(
            action_llm=action_llm,
            appworld_data_dir=args.appworld_data_dir,
            output_dir=args.output_dir,
            appworld_env_url=args.appworld_env_url,
            appworld_auto_server=args.appworld_auto_server,
            appworld_server_python=args.appworld_server_python,
            appworld_server_host=args.appworld_server_host,
            appworld_server_port_start=args.appworld_server_port_start,
            appworld_env_timeout=args.appworld_env_timeout,
            action_history_length=args.appworld_action_history_length,
            random_seed=args.appworld_random_seed,
            max_steps=args.max_steps,
            parallel_workers=parallel_workers,
            top_k_general_skills=args.top_k_general_skills,
            top_k_task_specific_skills=args.top_k_task_specific_skills,
            general_skill_retrieval_threshold=args.general_skill_retrieval_threshold,
            task_specific_skill_retrieval_threshold=args.task_specific_skill_retrieval_threshold,
            task_specific_retrieval_scope=args.task_specific_retrieval_scope,
        )
    if dataset in {"scienceworld", "science_world"}:
        return build_scienceworld_runner(
            action_llm=action_llm,
            max_steps=args.max_steps,
            scienceworld_env_url=args.scienceworld_env_url,
            scienceworld_env_timeout=args.scienceworld_env_timeout,
            scienceworld_parallel_workers=parallel_workers,
            scienceworld_auto_server=args.scienceworld_auto_server,
            scienceworld_server_host=args.scienceworld_server_host,
            scienceworld_server_port_start=args.scienceworld_server_port_start,
            scienceworld_path=args.scienceworld_path,
            scienceworld_jar_path=args.scienceworld_jar_path,
            scienceworld_env_step_limit=args.scienceworld_env_step_limit,
            scienceworld_server_python=args.scienceworld_server_python,
            action_history_length=args.scienceworld_action_history_length,
            use_admissible_actions=args.scienceworld_use_admissible_actions,
            top_k_general_skills=args.top_k_general_skills,
            top_k_task_specific_skills=args.top_k_task_specific_skills,
            general_skill_retrieval_threshold=args.general_skill_retrieval_threshold,
            task_specific_skill_retrieval_threshold=args.task_specific_skill_retrieval_threshold,
            task_specific_retrieval_scope=args.task_specific_retrieval_scope,
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


class EmptySkillProvider:
    metadata = {"bank_version": 0, "skill_source": "none"}
    bank_version = 0

    def retrieve(self, query: str, **_: Any) -> Dict[str, Any]:
        return {"task_type": None, "general_skills": [], "task_specific_skills": []}

    def format_for_prompt(self, retrieved_memory: Dict[str, Any]) -> str:
        return ""

    def retrieved_skill_ids(self, retrieved_memory: Dict[str, Any]) -> List[str]:
        return []


def _skill_source_for_mode(mode: str) -> str:
    if mode in NO_SKILL_MODES:
        return "none"
    if mode == ACTION_SKILL_MD_MODE:
        return "static_markdown"
    if mode == DUAL_LAYER_MD_MODE:
        return "evolving_markdown"
    return "unknown"


def _chunked(rows: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(rows), max(1, int(size))):
        yield rows[i:i + max(1, int(size))]


def _task_id(row: Any) -> Any:
    if isinstance(row, dict):
        return row.get("gamefile") or row.get("task_id") or row.get("id")
    return row


def _select_task_refs(args) -> List[Any]:
    if not args.manifest:
        raise ValueError("--manifest is required")
    manifest = load_json(args.manifest)
    if not isinstance(manifest, list):
        raise ValueError("--manifest must be a JSON list")
    if args.shuffle_tasks:
        import random
        rng = random.Random(args.seed)
        manifest = list(manifest)
        rng.shuffle(manifest)
    if args.limit_tasks is not None:
        manifest = manifest[: max(0, int(args.limit_tasks))]
    dataset = str(args.dataset or "").lower()
    if dataset in {"alfworld", "scienceworld", "science_world", "appworld"}:
        return [dict(row) if isinstance(row, dict) else row for row in manifest]
    return [_task_id(row) for row in manifest]


def _normalize_records(results: List[Dict[str, Any]], args):
    return normalize_task_results(
        results,
        max_steps=getattr(args, "record_max_steps", 128),
        max_observation_chars=getattr(args, "record_max_observation_chars", 1500),
        trajectory_mode=getattr(args, "record_trajectory_mode", "full"),
        trajectory_head_steps=getattr(args, "record_trajectory_head_steps", 1),
        trajectory_tail_steps=getattr(args, "record_trajectory_tail_steps", 8),
        action_only=getattr(args, "record_action_only", False),
    )


def _records_payload(results: List[Dict[str, Any]], records, args) -> Dict[str, Any]:
    return task_records_payload(records)


def _summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"tasks": 0, "avg_score": 0.0, "success_rate": 0.0}
    scores = [float(x.get("score", x.get("final_reward", 0.0)) or 0.0) for x in rows]
    successes = [1.0 if bool(x.get("success")) or float(x.get("score", 0.0) or 0.0) >= 1.0 else 0.0 for x in rows]
    summary = {
        "tasks": len(rows),
        "avg_score": sum(scores) / len(scores),
        "success_rate": sum(successes) / len(successes),
    }
    if all(str(row.get("dataset") or "").lower() == "appworld" for row in rows):
        scenario_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            scenario = str(row.get("task_family") or str(row.get("task_id") or "").rsplit("_", 1)[0])
            scenario_rows.setdefault(scenario, []).append(row)
        scenario_success = sum(
            1
            for variants in scenario_rows.values()
            if len(variants) == 3 and all(bool(row.get("success")) for row in variants)
        )
        summary.update(
            {
                "task_goal_completion": 100.0 * sum(successes) / len(successes),
                "scenario_goal_completion": (
                    100.0 * scenario_success / len(scenario_rows) if scenario_rows else 0.0
                ),
                "scenarios": len(scenario_rows),
                "successful_scenarios": scenario_success,
            }
        )
    return summary


def _print_eval_running_summary(task_rows: List[Dict[str, Any]], *, mode: str, bundle_idx: int) -> None:
    rows = [row for row in task_rows or [] if str(row.get("mode") or "") == str(mode)]
    summary = _summarize_results(rows)
    print(
        f"[eval running {mode}] after_bundle={bundle_idx} "
        f"tasks={summary['tasks']} avg_score={summary['avg_score']:.4f} "
        f"success_rate={summary['success_rate']:.4f}",
        flush=True,
    )


def _run_task_set_eval(runner, *, args, out: Path, bundle_idx: int, mode: str, task_ids, skill_bank, desc: str):
    results = runner.run_task_set(task_ids, skill_bank, {}, desc=desc)
    for i, row in enumerate(results):
        row.setdefault("mode", mode)
        row.setdefault("skill_source", _skill_source_for_mode(mode))
        row.setdefault("bundle", bundle_idx)
        row.setdefault("task_index", i)
    return results


def _markdown_edit_summary(edits: List[Dict[str, Any]]) -> str:
    parts = []
    for edit in edits or []:
        op = str(edit.get("op") or "").lower()
        content = str(edit.get("content") or edit.get("target") or "").replace("\n", " ")
        parts.append(f"{op}:{content[:80]}")
    return "; ".join(parts) if parts else "no edits"


def _chat_json_or_error(llm, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if llm is None:
        return {"error": "llm unavailable"}
    try:
        return llm.chat_json(system_prompt, payload)
    except Exception as exc:
        return {"error": str(exc)}


def _dual_layer_md_paths(args, out: Path) -> Dict[str, Path]:
    root = Path(args.dual_layer_md_dir) if args.dual_layer_md_dir else out / "dual_layer_md"
    skill_path = Path(args.save_dual_layer_md_path) if args.save_dual_layer_md_path else root / "evolving_skill.md"
    return {
        "root": root,
        "skill_md": skill_path,
        "evidence": root / "evidence_cards.jsonl",
        "feedback": root / "selection_feedback_cards.jsonl",
        "candidates": root / "l2_candidate_logs.jsonl",
        "logs": root / "bundle_logs.json",
    }


def _init_dual_layer_md_state(args, out: Path, completed_bundles: set) -> Dict[str, Any]:
    paths = _dual_layer_md_paths(args, out)
    paths["root"].mkdir(parents=True, exist_ok=True)
    skill_path = paths["skill_md"]
    resume = (not args.no_resume) and bool(completed_bundles) and skill_path.exists()
    if not resume:
        source = Path(args.dual_layer_initial_skill_md or args.action_skill_md or "")
        if not source.exists():
            raise FileNotFoundError("--dual-layer-initial-skill-md or --action-skill-md is required")
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        evidence_rows: List[Dict[str, Any]] = []
        source_feedback = source.parent / "selection_feedback_cards.jsonl"
        feedback_rows = load_jsonl(str(source_feedback)) if source_feedback.exists() else []
        candidate_rows: List[Dict[str, Any]] = []
        logs: List[Dict[str, Any]] = []
    else:
        evidence_rows = load_jsonl(str(paths["evidence"])) if paths["evidence"].exists() else []
        feedback_rows = load_jsonl(str(paths["feedback"])) if paths["feedback"].exists() else []
        candidate_rows = load_jsonl(str(paths["candidates"])) if paths["candidates"].exists() else []
        logs = load_json(str(paths["logs"])) if paths["logs"].exists() else []
        if not isinstance(logs, list):
            logs = []
    _normalize_evidence_runtime_states(evidence_rows)
    evidence_pool = [row for row in evidence_rows if not bool(row.get("archived"))]
    return {
        "paths": paths,
        "evidence_rows": evidence_rows,
        "feedback_rows": feedback_rows,
        "candidate_rows": candidate_rows,
        "logs": logs,
        "evidence_pool": evidence_pool,
        "evidence_counter": len(evidence_rows) + 1,
    }


def _save_dual_layer_md_state(state: Dict[str, Any]) -> None:
    paths = state["paths"]
    save_jsonl(str(paths["evidence"]), state.get("evidence_rows") or [])
    save_jsonl(str(paths["feedback"]), state.get("feedback_rows") or [])
    save_jsonl(str(paths["candidates"]), state.get("candidate_rows") or [])
    save_json(str(paths["logs"]), state.get("logs") or [])


def _dual_layer_md_models(args) -> Dict[str, Any]:
    def client(prefix: str):
        return _build_client(
            kind=getattr(args, f"{prefix}_llm_kind"),
            model=getattr(args, f"{prefix}_model"),
            api_key=getattr(args, f"{prefix}_api_key"),
            base_url=getattr(args, f"{prefix}_base_url"),
            temperature=getattr(args, f"{prefix}_temperature"),
            max_tokens=getattr(args, f"{prefix}_max_new_tokens"),
            load_in_4bit=getattr(args, f"{prefix}_load_in_4bit"),
            timeout=getattr(args, f"{prefix}_timeout"),
            max_retries=getattr(args, f"{prefix}_max_retries"),
            retry_initial_sleep=getattr(args, f"{prefix}_retry_initial_sleep", 5.0),
            retry_max_sleep=getattr(args, f"{prefix}_retry_max_sleep", 60.0),
            debug_dir=args.llm_debug_dir,
            thinking=getattr(args, f"{prefix}_thinking"),
            reasoning_effort=getattr(args, f"{prefix}_reasoning_effort"),
            extra_body_json=getattr(args, f"{prefix}_extra_body_json"),
        )

    models = {
        "l1_skill_manager": client("l1_skill_manager"),
        "l1_skill_judge": client("l1_skill_judge"),
        "l2_skill_manager": client("l2_skill_manager"),
        "l2_skill_judge": client("l2_skill_judge"),
    }
    missing = [name for name, value in models.items() if value is None]
    if missing:
        raise ValueError(
            "action_agent_dual_layer_md needs model args for: "
            + ", ".join("--" + name.replace("_", "-") + "-model" for name in missing)
        )
    return models


def _run_dual_layer_md_bundle(
    *,
    args,
    out: Path,
    runner,
    bundle_idx: int,
    task_ids,
    state: Dict[str, Any],
    models: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    skill_path = state["paths"]["skill_md"]
    skill_md = skill_path.read_text(encoding="utf-8")
    current_results = _run_task_set_eval(
        runner,
        args=args,
        out=out,
        bundle_idx=bundle_idx,
        mode=DUAL_LAYER_MD_MODE,
        task_ids=task_ids,
        skill_bank=MarkdownSkillBank(str(skill_path)),
        desc=f"eval bundle {bundle_idx} {DUAL_LAYER_MD_MODE}",
    )
    records = _normalize_records(current_results, args)
    payload = _records_payload(current_results, records, args)
    raw_by_id = _raw_result_by_task_id(current_results)
    l1_task_records = _attach_l1_task_outcomes(
        payload.get("task_records") or [],
        raw_by_id,
        dataset=args.dataset,
    )
    record_payload_by_id = {str(x.get("task_id") or ""): x for x in l1_task_records}

    l1_input = {
        "skill_md": skill_md,
        "bundle_trajectories": l1_task_records,
    }
    l1_raw = _chat_json_or_error(models["l1_skill_manager"], prompts_for_dataset(args.dataset)["skill_manager_l1"], l1_input)
    drafts = _drafts_from_l1_response(l1_raw)
    kept_cards: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for draft in drafts:
        eid = f"E{int(state.get('evidence_counter', 1)):06d}"
        state["evidence_counter"] = int(state.get("evidence_counter", 1)) + 1
        card, error = _materialize_evidence_candidate(
            draft=draft,
            evidence_id=eid,
            records_by_id=record_payload_by_id,
        )
        if card is None:
            dropped.append({"draft": draft, "reason": error})
            continue
        l1_label = _chat_json_or_error(
            models["l1_skill_judge"],
            prompts_for_dataset(args.dataset)["skill_judge_l1"],
            {"skill_md": skill_md, "evidence": card},
        )
        if str(l1_label.get("decision") or "").lower() != "keep":
            dropped.append({"evidence": card, "reason": l1_label.get("reason") or l1_label.get("error")})
            continue
        card["l1_skilljudge_reason"] = str(l1_label.get("reason") or "")
        card["bundle"] = bundle_idx
        card["archived"] = False
        _normalize_evidence_runtime_state(card)
        kept_cards.append(card)
        state["evidence_pool"].append(card)
        state["evidence_rows"].append(card)

    l2_info = {"triggered": False, "candidate_count": 0, "accepted_count": 0}
    pool_counts = _evidence_pool_counts(state["evidence_pool"])
    if pool_counts["active"] >= int(args.dual_layer_evidence_window_size):
        skill_md = skill_path.read_text(encoding="utf-8")
        max_l2_window = int(
            getattr(
                args,
                "dual_layer_l2_max_evidence_per_window",
                getattr(args, "dual_layer_evidence_window_size", 20),
            )
            or getattr(args, "dual_layer_evidence_window_size", 20)
        )
        max_l2_window = max(1, max_l2_window)
        selected_pool, evidence_selection_info = _select_l2_evidence_window(
            state["evidence_pool"],
            max_window=max_l2_window,
        )
        public_pool = _public_evidence_cards(selected_pool)
        l2_input = {
            "skill_md": skill_md,
            "evidence_cards": public_pool,
            "max_candidate_edits": int(args.dual_layer_l2_candidate_budget),
            "selection_feedback": []
            if bool(getattr(args, "no_l2_selection_feedback", False))
            else _recent_selection_feedback(state.get("feedback_rows") or []),
        }
        l2_raw = _chat_json_or_error(models["l2_skill_manager"], prompts_for_dataset(args.dataset)["skill_manager_l2"], l2_input)
        l2_manager_error = str(l2_raw.get("error") or "")
        evidence_by_id = {str(row.get("evidence_id")): row for row in public_pool}
        candidate_edits: List[Dict[str, Any]] = []
        for raw_edit in _markdown_patch_items(l2_raw):
            edit = _normalize_markdown_edit(raw_edit)
            if edit is None or not _linked_evidence(edit, evidence_by_id):
                continue
            candidate_edits.append(edit)
            if len(candidate_edits) >= int(args.dual_layer_l2_candidate_budget):
                break

        accepted_edits: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        attempted_evidence_ids = set()
        accepted_evidence_ids = set()
        rejected_evidence_ids = set()
        staled_evidence_ids = set()
        selected_evidence_ids = evidence_selection_info.get("selected_evidence_ids") or []
        cited_evidence_ids = set()
        max_rejects = max(1, int(getattr(args, "dual_layer_l2_max_rejects_per_evidence", 2) or 2))
        if not l2_manager_error:
            attempted_evidence_ids.update(
                _mark_l2_window_attempts(
                    state["evidence_rows"],
                    selected_evidence_ids,
                    round_idx=bundle_idx,
                )
            )
        for edit_idx, edit in enumerate(candidate_edits):
            cited_evidence_ids.update(_evidence_id_list(edit.get("evidence_ids")))
            linked = _linked_evidence(edit, evidence_by_id)
            _, apply_reports = _apply_markdown_edits(skill_md, [edit])
            apply_ok = bool(apply_reports) and all(bool(row.get("applied")) for row in apply_reports)
            if apply_ok:
                l2_label = _chat_json_or_error(
                    models["l2_skill_judge"],
                    prompts_for_dataset(args.dataset)["skill_judge_l2"],
                    {"skill_md": skill_md, "candidate_edit": edit, "linked_evidence": linked},
                )
                decision = str(l2_label.get("decision") or "").lower()
                reason = str(l2_label.get("reason") or l2_label.get("error") or "")
            else:
                failed = next((row for row in apply_reports if not row.get("applied")), {})
                decision = "reject"
                reason = "markdown patch did not apply: " + str(failed.get("reason") or "unknown")
            if decision == "accept":
                accepted_edits.append(edit)
            else:
                updates = _mark_evidence_attempts(
                    state["evidence_rows"],
                    _evidence_id_list(edit.get("evidence_ids")),
                    round_idx=bundle_idx,
                    outcome="reject",
                    max_rejects=max_rejects,
                    count_attempt=False,
                )
                rejected_evidence_ids.update(updates["rejected"])
                staled_evidence_ids.update(updates["staled"])
            decisions.append({"edit_index": edit_idx, "edit": edit, "decision": decision, "reason": reason, "apply_reports": apply_reports})

        if not l2_manager_error:
            updates = _mark_uncited_selected_evidence(
                state["evidence_rows"],
                selected_evidence_ids=selected_evidence_ids,
                cited_evidence_ids=sorted(cited_evidence_ids),
                round_idx=bundle_idx,
                max_rejects=max_rejects,
            )
            rejected_evidence_ids.update(updates["rejected"])
            staled_evidence_ids.update(updates["staled"])

        committed_edits: List[Dict[str, Any]] = []
        if accepted_edits:
            updated_skill_md, commit_reports, committed_edits, failed_commit_edits = (
                _commit_markdown_edits_sequential(skill_md, accepted_edits)
            )
            if committed_edits:
                skill_path.write_text(updated_skill_md, encoding="utf-8")
                for edit in committed_edits:
                    updates = _mark_evidence_attempts(
                        state["evidence_rows"],
                        _evidence_id_list(edit.get("evidence_ids")),
                        round_idx=bundle_idx,
                        outcome="accept",
                        max_rejects=max_rejects,
                        count_attempt=False,
                    )
                    accepted_evidence_ids.update(updates["accepted"])
            for failed in failed_commit_edits:
                edit = failed.get("edit") or {}
                updates = _mark_evidence_attempts(
                    state["evidence_rows"],
                    _evidence_id_list(edit.get("evidence_ids")),
                    round_idx=bundle_idx,
                    outcome="reject",
                    max_rejects=max_rejects,
                    count_attempt=False,
                )
                rejected_evidence_ids.update(updates["rejected"])
                staled_evidence_ids.update(updates["staled"])
        else:
            commit_reports = []
        archived_ids = set()
        for edit in committed_edits:
            archived_ids.update(_evidence_id_list(edit.get("evidence_ids")))
        if archived_ids:
            _archive_evidence_rows(state["evidence_rows"], sorted(archived_ids))
            state["evidence_pool"] = [
                row for row in state["evidence_pool"] if str(row.get("evidence_id")) not in archived_ids
            ]
        _normalize_evidence_runtime_states(state["evidence_pool"])
        pool_counts_after = _evidence_pool_counts(state["evidence_pool"])
        state["candidate_rows"].append(
            {
                "bundle": bundle_idx,
                "input": l2_input,
                "output": {"reasoning": str(l2_raw.get("reasoning") or ""), "edits": candidate_edits},
                "decisions": decisions,
                "accepted_edits": accepted_edits,
                "committed_edits": committed_edits,
                "commit_reports": commit_reports,
                "archived_evidence_ids": sorted(archived_ids),
                "evidence_pool": pool_counts_after,
                "evidence_selection": evidence_selection_info,
                "l2_manager_error": l2_manager_error,
                "attempted_evidence_ids": sorted(attempted_evidence_ids),
                "accepted_evidence_ids": sorted(accepted_evidence_ids),
                "rejected_evidence_ids": sorted(rejected_evidence_ids),
                "staled_evidence_ids": sorted(staled_evidence_ids),
            }
        )
        l2_info = {
            "triggered": True,
            "candidate_count": len(candidate_edits),
            "accepted_count": len(accepted_edits),
            "committed_count": len(committed_edits),
            "archived_evidence_ids": sorted(archived_ids),
            "evidence_pool_size_after": len(state["evidence_pool"]),
            "evidence_pool": pool_counts_after,
            "evidence_selection": evidence_selection_info,
            "l2_manager_error": l2_manager_error,
            "attempted_evidence_ids": sorted(attempted_evidence_ids),
            "accepted_evidence_ids": sorted(accepted_evidence_ids),
            "rejected_evidence_ids": sorted(rejected_evidence_ids),
            "staled_evidence_ids": sorted(staled_evidence_ids),
            "accepted_summary": _markdown_edit_summary(accepted_edits),
            "committed_summary": _markdown_edit_summary(committed_edits),
        }

    log = {
        "bundle": bundle_idx,
        "task_ids": task_ids,
        "draft_count": len(drafts),
        "kept_count": len(kept_cards),
        "dropped": dropped,
        "evidence_pool_size": len(state["evidence_pool"]),
        "evidence_pool": _evidence_pool_counts(state["evidence_pool"]),
        "l2": l2_info,
        "skill_md_path": str(skill_path),
    }
    state["logs"].append(log)
    _save_dual_layer_md_state(state)
    print(
        f"[eval bundle {bundle_idx} {DUAL_LAYER_MD_MODE}] "
        f"score={_summarize_results(current_results)['avg_score']:.3f} "
        f"kept={len(kept_cards)} pool={_evidence_pool_counts(state['evidence_pool'])} l2={l2_info}",
        flush=True,
    )
    return current_results, log


def _load_resume_state(out: Path, config: Dict[str, Any], *, enabled: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], set]:
    task_path = out / "eval_task_results.jsonl"
    bundle_path = out / "eval_bundle_logs.json"
    config_path = out / "eval_run_config.json"
    if not enabled:
        return [], [], [], set()
    if not task_path.exists() or not bundle_path.exists() or not config_path.exists():
        return [], [], [], set()
    old_config = load_json(str(config_path))
    if old_config != config:
        print("[resume] disabled because eval_run_config.json differs", flush=True)
        return [], [], [], set()
    task_rows = load_jsonl(str(task_path))
    bundle_logs = load_json(str(bundle_path))
    completed = {int(row.get("bundle")) for row in bundle_logs if isinstance(row, dict) and row.get("bundle") is not None}
    return task_rows, [], bundle_logs, completed


def _build_summary(args, task_rows: List[Dict[str, Any]], bundle_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in task_rows:
        by_mode.setdefault(str(row.get("mode") or ""), []).append(row)
    return {
        "dataset": args.dataset,
        "modes": args.modes,
        "skill_sources": {
            mode: _skill_source_for_mode(mode)
            for mode in args.modes
        },
        "bundle_size": args.bundle_size,
        "num_task_rows": len(task_rows),
        "by_mode": {mode: _summarize_results(rows) for mode, rows in by_mode.items()},
        "dual_layer": {
            "skill_md": str(_dual_layer_md_paths(args, Path(args.output_dir))["skill_md"])
            if DUAL_LAYER_MD_MODE in args.modes
            else None,
            "accepted_l2_edits": sum(
                int(((log.get("l2") or {}).get("accepted_count") or 0))
                for log in bundle_logs
                if isinstance(log, dict)
            ),
            "committed_l2_edits": sum(
                int(((log.get("l2") or {}).get("committed_count") or 0))
                for log in bundle_logs
                if isinstance(log, dict)
            ),
            "selection_feedback_cards": str(_dual_layer_md_paths(args, Path(args.output_dir))["feedback"])
            if DUAL_LAYER_MD_MODE in args.modes
            else None,
        },
    }


def _eval_run_config(args, runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": args.dataset,
        "manifest": args.manifest,
        "modes": args.modes,
        "bundle_size": args.bundle_size,
        "limit_tasks": args.limit_tasks,
        "shuffle_tasks": args.shuffle_tasks,
        "seed": args.seed,
        "dual_layer_initial_skill_md": args.dual_layer_initial_skill_md,
        "action_skill_md": args.action_skill_md,
        "dual_layer_evidence_window_size": args.dual_layer_evidence_window_size,
        "dual_layer_l2_candidate_budget": args.dual_layer_l2_candidate_budget,
        "dual_layer_l2_max_evidence_per_window": args.dual_layer_l2_max_evidence_per_window,
        "dual_layer_l2_max_rejects_per_evidence": args.dual_layer_l2_max_rejects_per_evidence,
        "no_l2_selection_feedback": args.no_l2_selection_feedback,
        "record_trajectory_mode": args.record_trajectory_mode,
        "record_max_steps": args.record_max_steps,
        "record_trajectory_head_steps": args.record_trajectory_head_steps,
        "record_trajectory_tail_steps": args.record_trajectory_tail_steps,
        "record_max_observation_chars": args.record_max_observation_chars,
        "record_action_only": args.record_action_only,
        "action_runtime": runtime_config,
    }


def _validate_expected_action_runtime(args, current: Dict[str, Any]) -> None:
    expected_path = getattr(args, "expected_action_runtime_config", None)
    if not expected_path:
        return
    expected = load_json(str(expected_path))
    differences = action_runtime_differences(expected, current)
    if differences:
        details = "\n  - ".join(differences)
        raise RuntimeError(
            f"Evaluation action runtime does not match training:\n  - {details}\n"
            "Test and training must use identical ActionModel rollout settings."
        )
    print(f"[runtime alignment] train=test config={expected_path}", flush=True)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _setup_watch_log(args, out)
    runtime_config = action_runtime_config(args)
    _validate_expected_action_runtime(args, runtime_config)
    action_llm = _build_action_llm(args)
    if action_llm is None:
        raise ValueError("--action-model is required unless --dry-run-action is set")
    runner = _runner(args, action_llm)
    task_refs = _select_task_refs(args)
    bundles = list(_chunked(task_refs, args.bundle_size))
    config = _eval_run_config(args, runtime_config)
    task_rows, _, bundle_logs, completed = _load_resume_state(out, config, enabled=not args.no_resume)
    dual_state = None
    dual_models = None
    if DUAL_LAYER_MD_MODE in args.modes:
        dual_state = _init_dual_layer_md_state(args, out, completed)
        dual_models = _dual_layer_md_models(args)

    for bundle_idx, task_ids in enumerate(progress_iter(bundles, desc="eval bundles", disable=args.no_progress)):
        if bundle_idx in completed:
            continue
        for mode in args.modes:
            if mode in NO_SKILL_MODES:
                results = _run_task_set_eval(
                    runner,
                    args=args,
                    out=out,
                    bundle_idx=bundle_idx,
                    mode=mode,
                    task_ids=task_ids,
                    skill_bank=EmptySkillProvider(),
                    desc=f"eval bundle {bundle_idx} {mode}",
                )
                task_rows.extend(results)
                bundle_logs.append({"bundle": bundle_idx, "mode": mode, "summary": _summarize_results(results)})
                _print_eval_running_summary(task_rows, mode=mode, bundle_idx=bundle_idx)
            elif mode == ACTION_SKILL_MD_MODE:
                if not args.action_skill_md:
                    raise ValueError("--action-skill-md is required for action_agent_skill_md")
                results = _run_task_set_eval(
                    runner,
                    args=args,
                    out=out,
                    bundle_idx=bundle_idx,
                    mode=mode,
                    task_ids=task_ids,
                    skill_bank=MarkdownSkillBank(args.action_skill_md),
                    desc=f"eval bundle {bundle_idx} {mode}",
                )
                task_rows.extend(results)
                bundle_logs.append({"bundle": bundle_idx, "mode": mode, "summary": _summarize_results(results)})
                _print_eval_running_summary(task_rows, mode=mode, bundle_idx=bundle_idx)
            elif mode == DUAL_LAYER_MD_MODE:
                results, log = _run_dual_layer_md_bundle(
                    args=args,
                    out=out,
                    runner=runner,
                    bundle_idx=bundle_idx,
                    task_ids=task_ids,
                    state=dual_state,
                    models=dual_models,
                )
                task_rows.extend(results)
                bundle_logs.append({"bundle": bundle_idx, "mode": mode, **log})
                _print_eval_running_summary(task_rows, mode=mode, bundle_idx=bundle_idx)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        save_jsonl(str(out / "eval_task_results.jsonl"), task_rows)
        save_json(str(out / "eval_bundle_logs.json"), bundle_logs)
        save_json(str(out / "eval_run_config.json"), config)
        save_json(str(out / "eval_summary.json"), _build_summary(args, task_rows, bundle_logs))

    summary = _build_summary(args, task_rows, bundle_logs)
    save_json(str(out / "eval_summary.json"), summary)
    close_runner = getattr(runner, "close", None)
    if callable(close_runner):
        close_runner()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def add_common_model_args(
    p,
    prefix: str,
    *,
    default_kind: str = "openai",
    default_temperature=None,
    default_tokens: int = 1024,
    default_max_retries: int = 5,
):
    dest = prefix.replace("-", "_")
    p.add_argument(f"--{prefix}-model", dest=f"{dest}_model", default=None)
    p.add_argument(f"--{prefix}-llm-kind", dest=f"{dest}_llm_kind", choices=["openai", "hf"], default=default_kind)
    p.add_argument(f"--{prefix}-api-key", dest=f"{dest}_api_key", default=None)
    p.add_argument(f"--{prefix}-base-url", dest=f"{dest}_base_url", default=None)
    p.add_argument(f"--{prefix}-proxy-url", dest=f"{dest}_proxy_url", default=None)
    p.add_argument(f"--{prefix}-temperature", dest=f"{dest}_temperature", type=float, default=default_temperature)
    p.add_argument(f"--{prefix}-max-new-tokens", dest=f"{dest}_max_new_tokens", type=int, default=default_tokens)
    p.add_argument(f"--{prefix}-load-in-4bit", dest=f"{dest}_load_in_4bit", action="store_true")
    p.add_argument(f"--{prefix}-timeout", dest=f"{dest}_timeout", type=int, default=180)
    p.add_argument(
        f"--{prefix}-max-retries",
        dest=f"{dest}_max_retries",
        type=int,
        default=default_max_retries,
    )
    p.add_argument(
        f"--{prefix}-retry-initial-sleep",
        dest=f"{dest}_retry_initial_sleep",
        type=float,
        default=5.0,
    )
    p.add_argument(
        f"--{prefix}-retry-max-sleep",
        dest=f"{dest}_retry_max_sleep",
        type=float,
        default=60.0,
    )
    p.add_argument(f"--{prefix}-thinking", dest=f"{dest}_thinking", choices=["enabled", "disabled"], default=None)
    p.add_argument(f"--{prefix}-reasoning-effort", dest=f"{dest}_reasoning_effort", choices=["none", "low", "medium", "high"], default="medium")
    p.add_argument(f"--{prefix}-extra-body-json", dest=f"{dest}_extra_body_json", default=None)


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["alfworld", "scienceworld", "appworld"], default="alfworld")
    p.add_argument("--manifest", required=True)
    p.add_argument("--action-skill-md", default=None, help="Static Markdown skill document for action_agent_dual_layer_md default init.")
    p.add_argument("--dual-layer-initial-skill-md", default=None)
    p.add_argument("--dual-layer-md-dir", default=None)
    p.add_argument("--save-dual-layer-md-path", default=None)
    p.add_argument("--output-dir", default="outputs/skill_system_eval")
    p.add_argument("--watch-log-file", default=None)
    p.add_argument("--llm-debug-dir", default=None)
    p.add_argument("--expected-action-runtime-config", default=None)

    p.add_argument("--alfworld-path", default=None, help="Path to the external ALFWorld installation.")
    p.add_argument("--alfworld-config", default=None, help="Path to the external ALFWorld config.")
    p.add_argument("--alfworld-split", default="train")
    p.add_argument("--alfworld-max-resets-to-find-task", type=int, default=512)
    p.add_argument(
        "--alfworld-action-history-length",
        type=int,
        default=ALFWORLD_ACTION_HISTORY_LENGTH,
    )
    p.add_argument("--scienceworld-env-url", default="http://127.0.0.1:8811")
    p.add_argument("--scienceworld-env-timeout", type=int, default=180)
    p.add_argument(
        "--scienceworld-parallel-workers",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--scienceworld-auto-server", action="store_true")
    p.add_argument("--scienceworld-server-host", default="127.0.0.1")
    p.add_argument("--scienceworld-server-port-start", type=int, default=8811)
    p.add_argument("--scienceworld-path", default=None, help="Path to the external ScienceWorld installation.")
    p.add_argument("--scienceworld-jar-path", default=None, help="Path to the external ScienceWorld JAR.")
    p.add_argument("--scienceworld-env-step-limit", type=int, default=None)
    p.add_argument(
        "--scienceworld-server-python",
        default=None,
        help="Python executable used to launch auto ScienceWorld servers. Defaults to the current Python.",
    )
    p.add_argument(
        "--scienceworld-action-history-length",
        type=int,
        default=SCIENCEWORLD_ACTION_HISTORY_LENGTH,
    )
    p.add_argument(
        "--scienceworld-use-admissible-actions",
        action="store_true",
        help="Pass the full ScienceWorld admissible action list to the action model. Default is SkillNet-style free-form action generation.",
    )
    p.add_argument("--appworld-data-dir", default=None, help="Path to the external AppWorld data directory.")
    p.add_argument("--appworld-env-url", default=None)
    p.add_argument("--appworld-auto-server", action="store_true")
    p.add_argument(
        "--appworld-server-python",
        default=None,
        help="Python executable for the external AppWorld server; defaults to the current interpreter.",
    )
    p.add_argument("--appworld-server-host", default="127.0.0.1")
    p.add_argument("--appworld-server-port-start", type=int, default=8841)
    p.add_argument("--appworld-env-timeout", type=int, default=100)
    p.add_argument(
        "--appworld-action-history-length",
        type=int,
        default=APPWORLD_ACTION_HISTORY_LENGTH,
    )
    p.add_argument("--appworld-random-seed", type=int, default=100)

    add_common_model_args(
        p,
        "action",
        default_kind="openai",
        default_temperature=None,
        default_tokens=512,
        default_max_retries=-1,
    )
    add_common_model_args(p, "l1-skill-manager", default_kind="hf", default_temperature=0.0, default_tokens=2048)
    add_common_model_args(p, "l1-skill-judge", default_kind="hf", default_temperature=0.0, default_tokens=1024)
    add_common_model_args(p, "l2-skill-manager", default_kind="hf", default_temperature=0.0, default_tokens=2048)
    add_common_model_args(p, "l2-skill-judge", default_kind="hf", default_temperature=0.0, default_tokens=1024)
    p.add_argument("--dry-run-action", action="store_true")

    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument(
        "--bundle-size",
        type=int,
        default=2,
        help="Tasks per evaluation bundle and requested rollout worker count for every dataset.",
    )
    p.add_argument("--limit-tasks", type=int, default=None)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=[BASELINE_MODE, ACTION_ONLY_MODE, ACTION_SKILL_MD_MODE, DUAL_LAYER_MD_MODE],
        default=[ACTION_ONLY_MODE, DUAL_LAYER_MD_MODE],
        help=(
            "Evaluation modes. 'baseline' runs only the ActionModel with no skill text; "
            "'action_agent' is its legacy alias."
        ),
    )
    p.add_argument("--shuffle-tasks", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dual-layer-evidence-window-size", type=int, default=20)
    p.add_argument("--dual-layer-l2-candidate-budget", type=int, default=5)
    p.add_argument("--dual-layer-l2-max-evidence-per-window", type=int, default=20)
    p.add_argument("--dual-layer-l2-max-rejects-per-evidence", type=int, default=2)
    p.add_argument("--no-l2-selection-feedback", action="store_true")

    p.add_argument("--record-trajectory-mode", choices=["prefix", "tail", "head_tail", "full"], default="full")
    p.add_argument("--record-max-steps", type=int, default=128)
    p.add_argument("--record-trajectory-head-steps", type=int, default=1)
    p.add_argument("--record-trajectory-tail-steps", type=int, default=8)
    p.add_argument("--record-max-observation-chars", type=int, default=1500)
    p.add_argument("--record-action-only", action="store_true")

    p.add_argument("--top-k-general-skills", type=int, default=5)
    p.add_argument("--top-k-task-specific-skills", type=int, default=2)
    p.add_argument("--general-skill-retrieval-threshold", type=float, default=0.0)
    p.add_argument("--task-specific-skill-retrieval-threshold", type=float, default=0.0)
    p.add_argument("--task-specific-retrieval-scope", choices=["all", "detected"], default="detected")
    p.add_argument("--num-products", type=int, default=1000)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
