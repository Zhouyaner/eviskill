#!/usr/bin/env python
import fcntl
import os
import sys
from pathlib import Path
import subprocess
import shlex
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from wmsm.experiment_naming import default_dual_layer_output_dir
from wmsm.action_runtime import (
    ALFWORLD_ACTION_HISTORY_LENGTH,
    APPWORLD_ACTION_HISTORY_LENGTH,
    SCIENCEWORLD_ACTION_HISTORY_LENGTH,
    action_runtime_config,
    ensure_action_runtime_config,
)
from wmsm.json_utils import load_json, save_json
from wmsm.evidence_pipeline import run_eviskill


def _acquire_output_dir_lock(output_dir: str):
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".generate.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        owner = lock_file.read().strip() or "unknown owner"
        lock_file.close()
        raise RuntimeError(
            f"Output directory is already in use by another generator: {output}. "
            f"Lock owner: {owner}"
        ) from exc
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()}\n")
    lock_file.flush()
    return lock_file


def _ensure_llm_context_config(output_dir: Path, args) -> None:
    """Prevent resuming a compacted run with the full-context semantics (or vice versa)."""
    config_path = output_dir / "llm_context_config.json"
    expected = {
        "schema_version": 1,
        "llm_full_context": bool(getattr(args, "llm_full_context", False)),
        "mode": "full_interactions_and_evidence"
        if bool(getattr(args, "llm_full_context", False))
        else "compacted_legacy",
    }
    if config_path.exists():
        try:
            actual = load_json(str(config_path))
        except Exception as exc:
            raise RuntimeError(f"cannot read {config_path}: {exc}") from exc
        if not isinstance(actual, dict) or bool(actual.get("llm_full_context", False)) != expected["llm_full_context"]:
            raise RuntimeError(
                "LLM context mode differs from the existing run: "
                f"training={actual!r} evaluation={expected!r}. Use a new output directory."
            )
        return

    # The watcher creates its log before the generator reaches this guard. Logs
    # are not evidence/checkpoints and must not make a fresh output look like a
    # compacted run. Any substantive artifact still triggers the guard below.
    existing_artifacts = [
        path.name
        for path in output_dir.iterdir()
        if path.name
        not in {".generate.lock", "llm_context_config.json", "watch.log", "launcher.log"}
    ]
    if expected["llm_full_context"] and existing_artifacts:
        raise RuntimeError(
            "full-context generation cannot resume an output without llm_context_config.json; "
            "use a new output directory"
        )
    save_json(str(config_path), expected)


def _add_value(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _rollout_worker_count(args) -> int:
    return max(1, int(getattr(args, "bundle_size", 1) or 1))


def _replay_range_budget(value: str) -> int:
    budget = int(value)
    if budget < -1:
        raise argparse.ArgumentTypeError("replay range budget must be -1, 0, or a positive integer")
    return budget


def _policy_defer_epoch_limit(value: str) -> int:
    limit = int(value)
    if limit == 0 or limit < -1:
        raise argparse.ArgumentTypeError(
            "policy defer epoch limit must be -1 or a positive integer"
        )
    return limit


class _StoreDeferRuleIgnored(argparse.Action):
    """Set the canonical flag and its deprecated compatibility attribute."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "defer_rule_ignored", True)
        setattr(namespace, "working_branch_defer_rule_ignored", True)


def _manifest_count(path) -> str:
    if not path:
        return "?"
    try:
        rows = load_json(str(path))
    except Exception:
        return "?"
    if isinstance(rows, list):
        return str(len(rows))
    return "?"


def _manifest_count_int(path) -> Optional[int]:
    if not path:
        return None
    try:
        rows = load_json(str(path))
    except Exception:
        return None
    return len(rows) if isinstance(rows, list) else None


def _scienceworld_sibling_test_candidates(path) -> List[Path]:
    if not path:
        return []
    p = Path(path)
    candidates: List[Path] = []
    if p.name == "items.json" and p.parent.name in {"train", "val", "dev", "valid", "selection"}:
        candidates.append(p.parent.parent / "test" / "items.json")
    if p.name in {"train.json", "dev.json", "val.json", "validation.json", "selection.json"}:
        candidates.append(p.with_name("test.json"))
    if p.is_dir():
        candidates.append(p / "test" / "items.json")
        candidates.append(p / "test.json")
    return candidates


def _scienceworld_sibling_selection_candidates(path) -> List[Path]:
    if not path:
        return []
    p = Path(path)
    candidates: List[Path] = []
    if p.name == "items.json" and p.parent.name == "train":
        candidates.append(p.parent.parent / "val" / "items.json")
    if p.name == "train.json":
        candidates.extend([p.with_name("dev.json"), p.with_name("val.json")])
    candidates.append(p.with_name("dev.json"))
    return candidates


def _resolve_selection_manifest_for_display(args) -> str:
    explicit = str(getattr(args, "selection_manifest", "") or "").strip()
    if explicit:
        return explicit
    dataset = str(getattr(args, "dataset", "") or "").lower()
    if dataset in {"scienceworld", "science_world"}:
        for candidate in _scienceworld_sibling_selection_candidates(getattr(args, "manifest", None)):
            if candidate.exists():
                return str(candidate)
    return "(sampled/default)"


def _selection_target_count(args, source_count: Optional[int]) -> str:
    try:
        size = int(getattr(args, "l2_selection_size", 24))
    except (TypeError, ValueError):
        size = 24
    if size <= 0:
        return str(source_count) if source_count is not None else "full"
    if source_count is None:
        return str(size)
    return str(min(size, source_count))


def _alfworld_eval_split_from_manifest(path: Path) -> Optional[str]:
    try:
        rows = load_json(str(path))
    except (FileNotFoundError, ValueError, TypeError):
        return None
    source_splits = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        gamefile = str(row.get("gamefile") or "").replace("\\", "/")
        if "/valid_unseen/" in f"/{gamefile.lstrip('/')}":
            source_splits.add("eval_out_of_distribution")
        elif "/valid_seen/" in f"/{gamefile.lstrip('/')}":
            source_splits.add("eval_in_distribution")
        elif "/train/" in f"/{gamefile.lstrip('/')}":
            source_splits.add("train")
    return next(iter(source_splits)) if len(source_splits) == 1 else None


def _resolve_auto_eval_manifest(args) -> Tuple[Path, Optional[str]]:
    explicit = str(getattr(args, "test_manifest", "") or "").strip()
    if explicit:
        path = Path(explicit)
        if str(getattr(args, "dataset", "") or "").lower() == "alfworld":
            return path, _alfworld_eval_split_from_manifest(path) or getattr(args, "alfworld_split", None)
        return path, getattr(args, "alfworld_split", None)

    dataset = str(getattr(args, "dataset", "") or "").lower()
    if dataset == "alfworld":
        return ROOT / "data" / "splits" / "alfworld_eval_out_of_distribution" / "test.json", "eval_out_of_distribution"
    if dataset in {"scienceworld", "science_world"}:
        for source in [getattr(args, "selection_manifest", None), getattr(args, "manifest", None)]:
            for candidate in _scienceworld_sibling_test_candidates(source):
                if candidate.exists():
                    return candidate, None
        return ROOT / "data" / "splits" / "scienceworld" / "test.json", None
    if dataset == "appworld":
        return ROOT / "data" / "splits" / "appworld_official" / "test_normal" / "items.json", None
    return Path(args.manifest), getattr(args, "alfworld_split", None)


def _print_generation_banner(args) -> None:
    dataset = str(getattr(args, "dataset", "") or "").lower()
    if dataset not in {"alfworld", "scienceworld", "science_world", "appworld"}:
        return

    test_manifest, _ = _resolve_auto_eval_manifest(args)
    selection_manifest = _resolve_selection_manifest_for_display(args)
    auto_eval_status = "disabled" if getattr(args, "no_auto_eval", False) else "enabled"
    workers = _rollout_worker_count(args)
    train_count = _manifest_count_int(args.manifest)
    bundle_size = max(1, int(getattr(args, "bundle_size", 1) or 1))
    bundles_per_epoch = "?"
    total_rollout_bundles = "?"
    if train_count is not None:
        bundles = max(1, (train_count + bundle_size - 1) // bundle_size)
        bundles_per_epoch = str(bundles)
        total_rollout_bundles = str(bundles * max(1, int(getattr(args, "epochs", 1) or 1)))
    print("=" * 60, flush=True)
    dataset_title = {
        "alfworld": "ALFWorld",
        "appworld": "AppWorld",
    }.get(dataset, "ScienceWorld")
    print(f"  WMSM — {dataset_title}", flush=True)
    print("=" * 60, flush=True)
    print(f"  Action model:   {getattr(args, 'action_model', None) or '(none)'}", flush=True)
    print(f"  Teacher model:  {getattr(args, 'teacher_model', None) or '(none)'}", flush=True)
    print(
        f"  Temperature:    action={getattr(args, 'action_temperature', None) if getattr(args, 'action_temperature', None) is not None else 'omitted'} "
        f"teacher={getattr(args, 'teacher_temperature', None) if getattr(args, 'teacher_temperature', None) is not None else 'omitted'}",
        flush=True,
    )
    print(
        f"  Completion:     action={getattr(args, 'action_max_new_tokens', None)} "
        f"teacher={getattr(args, 'teacher_max_tokens', None)}",
        flush=True,
    )
    print(f"  Train:          {args.manifest} ({_manifest_count(args.manifest)} items)", flush=True)
    selection_source_count = _manifest_count_int(selection_manifest)
    print(
        f"  Validation:     {selection_manifest} "
        f"({selection_source_count if selection_source_count is not None else '?'} source items, "
        f"selected {_selection_target_count(args, selection_source_count)})",
        flush=True,
    )
    print(f"  Test:           {test_manifest} ({_manifest_count(test_manifest)} items)", flush=True)
    print(f"  Epochs:         {getattr(args, 'epochs', 1)}", flush=True)
    print(f"  Bundle size:    {getattr(args, 'bundle_size', 1)}", flush=True)
    print(f"  L1 minibatch:   {getattr(args, 'l1_trajectory_minibatch_size', 4)}", flush=True)
    print(f"  Bundles/epoch:  {bundles_per_epoch}", flush=True)
    print(f"  Total bundles:  {total_rollout_bundles}", flush=True)
    epoch_l2_windows = int(getattr(args, "epoch_l2_windows", 4) or 0)
    skill_update_mode = str(getattr(args, "epoch_skill_update_mode", "cluster_step") or "cluster_step").lower()
    print(f"  Skill update:   {skill_update_mode}", flush=True)
    if skill_update_mode == "cluster_step":
        print("  L2 windows/ep:  all semantic windows (coverage mode)", flush=True)
    else:
        print(f"  L2 windows/ep:  {'all' if epoch_l2_windows <= 0 else epoch_l2_windows}", flush=True)
    print(f"  Evidence groups:{getattr(args, 'l2_evidence_window_grouping', 'auto')}", flush=True)
    print(f"  Reuse evidence: {getattr(args, 'reuse_evidence_dir', None) or 'disabled'}", flush=True)
    print(
        "  LLM context:    FULL (no trajectory/evidence trimming; context overflow is explicit)"
        if getattr(args, "llm_full_context", False)
        else (
            f"  LLM compact:    feedback={getattr(args, 'llm_max_observation_chars', 500)} chars/step soft cap, "
            f"input={getattr(args, 'llm_max_input_chars', 30000)} chars target"
        ),
        flush=True,
    )
    if str(getattr(args, "l2_evidence_window_grouping", "auto")).lower() == "semantic":
        print(f"  Embedding model:{getattr(args, 'evidence_embedding_model', 'Qwen/Qwen3-Embedding-0.6B')}", flush=True)
        print(f"  Clustering:     {getattr(args, 'semantic_clustering_method', 'hdbscan')}", flush=True)
        print(
            f"  Semantic window:{getattr(args, 'semantic_cluster_min_window_size', 4)}-"
            f"{getattr(args, 'l2_max_evidence_per_window', 20)} evidence/window",
            flush=True,
        )
    if skill_update_mode == "cluster_step":
        print(f"  Window edits:   coverage merge (hint={getattr(args, 'l2_candidate_budget', 5)})", flush=True)
        print(
            f"  Edit budget/ep: {getattr(args, 'epoch_edit_budget', 4)} (working branch enforced)"
            if getattr(args, "use_evidence_working_branch", False)
            else "  Edit budget/ep: ignored in base cluster_step",
            flush=True,
        )
        print(f"  Global merge:   {'disabled' if getattr(args, 'no_epoch_global_merge', False) else 'enabled'} (cluster_step epoch aggregate)", flush=True)
        print("  Validation:     once per epoch after coverage merge", flush=True)
    else:
        print(f"  Edit budget/ep: {getattr(args, 'epoch_edit_budget', 4)}", flush=True)
        print(f"  Global merge:   {'disabled' if getattr(args, 'no_epoch_global_merge', False) else 'enabled'}", flush=True)
    print(
        "  Long memory:    "
        + (
            f"enabled samples={getattr(args, 'longitudinal_memory_samples', 20)} "
            f"policy={getattr(args, 'longitudinal_memory_pair_policy', 'mixed')}"
            if getattr(args, "use_longitudinal_memory", False)
            else "disabled"
        ),
        flush=True,
    )
    print(f"  Bundle sampler: {getattr(args, 'bundle_sampling_strategy', 'random')}", flush=True)
    validation_policy = (
        "hard success non-regression (ties commit)"
        if bool(getattr(args, "selection_accept_tie", False))
        else "strict hard success improvement (ties reject)"
    )
    if dataset == "alfworld":
        print(
            f"  Action history: {getattr(args, 'alfworld_action_history_length', ALFWORLD_ACTION_HISTORY_LENGTH)}",
            flush=True,
        )
        print(f"  Validation:     {validation_policy}", flush=True)
    elif dataset == "appworld":
        print(
            f"  Action history: {getattr(args, 'appworld_action_history_length', APPWORLD_ACTION_HISTORY_LENGTH)}",
            flush=True,
        )
        print(f"  Environment:    seed={getattr(args, 'appworld_random_seed', 100)} isolated HTTP servers", flush=True)
        print(f"  Validation:     {validation_policy}", flush=True)
    print(f"  Workers:        {workers}", flush=True)
    print(f"  Auto test:      {auto_eval_status}", flush=True)
    print(f"  Output:         {args.output_dir}", flush=True)
    print("=" * 60, flush=True)


def _run_auto_eval(args, info):
    skill_md = info.get("evolving_skill_md") if isinstance(info, dict) else None
    if not skill_md:
        return {"enabled": True, "skipped": True, "reason": "no evolving_skill_md in generation info"}
    skill_path = Path(skill_md)
    if not skill_path.exists():
        return {"enabled": True, "skipped": True, "reason": f"missing evolving skill file: {skill_path}"}

    dataset = str(getattr(args, "dataset", "") or "").lower()
    eval_manifest, eval_split = _resolve_auto_eval_manifest(args)
    eval_out = Path(args.output_dir) / "auto_eval_static_skill"
    eval_script = ROOT / "scripts" / "evaluate_skill_system.py"
    auto_eval_bundle_size = _rollout_worker_count(args)
    training_runtime_path = Path(args.output_dir) / "action_runtime_config.json"
    ensure_action_runtime_config(
        training_runtime_path,
        action_runtime_config(args),
        resume=not getattr(args, "no_resume", False),
        context="generation",
    )
    cmd = [
        sys.executable,
        str(eval_script),
        "--dataset",
        str(args.dataset),
        "--manifest",
        str(eval_manifest),
        "--modes",
        "action_agent_skill_md",
        "--action-skill-md",
        str(skill_path),
        "--output-dir",
        str(eval_out),
        "--bundle-size",
        str(auto_eval_bundle_size),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--expected-action-runtime-config",
        str(training_runtime_path),
    ]

    if getattr(args, "no_progress", False):
        cmd.append("--no-progress")

    _add_value(cmd, "--action-model", getattr(args, "action_model", None))
    _add_value(cmd, "--action-llm-kind", getattr(args, "action_llm_kind", None))
    _add_value(cmd, "--action-api-key", getattr(args, "action_api_key", None))
    _add_value(cmd, "--action-base-url", getattr(args, "action_base_url", None))
    _add_value(cmd, "--action-proxy-url", getattr(args, "action_proxy_url", None))
    _add_value(cmd, "--action-temperature", getattr(args, "action_temperature", None))
    _add_value(cmd, "--action-max-new-tokens", getattr(args, "action_max_new_tokens", None))
    _add_value(cmd, "--action-timeout", getattr(args, "action_timeout", None))
    _add_value(cmd, "--action-max-retries", getattr(args, "action_max_retries", None))
    _add_value(cmd, "--action-retry-initial-sleep", getattr(args, "action_retry_initial_sleep", None))
    _add_value(cmd, "--action-retry-max-sleep", getattr(args, "action_retry_max_sleep", None))
    _add_value(cmd, "--action-thinking", getattr(args, "action_thinking", None))
    _add_value(cmd, "--action-reasoning-effort", getattr(args, "action_reasoning_effort", None))
    _add_value(cmd, "--action-extra-body-json", getattr(args, "action_extra_body_json", None))
    if getattr(args, "action_load_in_4bit", False):
        cmd.append("--action-load-in-4bit")
    if getattr(args, "dry_run_action", False):
        cmd.append("--dry-run-action")

    if dataset == "alfworld":
        _add_value(cmd, "--alfworld-path", getattr(args, "alfworld_path", None))
        _add_value(cmd, "--alfworld-config", getattr(args, "alfworld_config", None))
        _add_value(cmd, "--alfworld-split", eval_split)
        _add_value(cmd, "--alfworld-max-resets-to-find-task", getattr(args, "alfworld_max_resets_to_find_task", None))
        _add_value(cmd, "--alfworld-action-history-length", getattr(args, "alfworld_action_history_length", None))
    elif dataset in {"scienceworld", "science_world"}:
        _add_value(cmd, "--scienceworld-env-url", getattr(args, "scienceworld_env_url", None))
        _add_value(cmd, "--scienceworld-env-timeout", getattr(args, "scienceworld_env_timeout", None))
        if getattr(args, "scienceworld_auto_server", False):
            cmd.append("--scienceworld-auto-server")
        _add_value(cmd, "--scienceworld-server-host", getattr(args, "scienceworld_server_host", None))
        _add_value(cmd, "--scienceworld-server-port-start", getattr(args, "scienceworld_server_port_start", None))
        _add_value(cmd, "--scienceworld-path", getattr(args, "scienceworld_path", None))
        _add_value(cmd, "--scienceworld-jar-path", getattr(args, "scienceworld_jar_path", None))
        _add_value(cmd, "--scienceworld-env-step-limit", getattr(args, "scienceworld_env_step_limit", None))
        _add_value(cmd, "--scienceworld-server-python", getattr(args, "scienceworld_server_python", None))
        _add_value(cmd, "--scienceworld-action-history-length", getattr(args, "scienceworld_action_history_length", None))
        if getattr(args, "scienceworld_use_admissible_actions", False):
            cmd.append("--scienceworld-use-admissible-actions")
    elif dataset == "appworld":
        _add_value(cmd, "--appworld-data-dir", getattr(args, "appworld_data_dir", None))
        _add_value(cmd, "--appworld-env-url", getattr(args, "appworld_env_url", None))
        if getattr(args, "appworld_auto_server", False):
            cmd.append("--appworld-auto-server")
        _add_value(cmd, "--appworld-server-python", getattr(args, "appworld_server_python", None))
        _add_value(cmd, "--appworld-server-host", getattr(args, "appworld_server_host", None))
        _add_value(cmd, "--appworld-server-port-start", getattr(args, "appworld_server_port_start", None))
        _add_value(cmd, "--appworld-env-timeout", getattr(args, "appworld_env_timeout", None))
        _add_value(cmd, "--appworld-action-history-length", getattr(args, "appworld_action_history_length", None))
        _add_value(cmd, "--appworld-random-seed", getattr(args, "appworld_random_seed", None))

    _add_value(cmd, "--record-trajectory-mode", getattr(args, "record_trajectory_mode", None))
    _add_value(cmd, "--record-max-steps", getattr(args, "record_max_steps", None))
    _add_value(cmd, "--record-trajectory-head-steps", getattr(args, "record_trajectory_head_steps", None))
    _add_value(cmd, "--record-trajectory-tail-steps", getattr(args, "record_trajectory_tail_steps", None))
    _add_value(cmd, "--record-max-observation-chars", getattr(args, "record_max_observation_chars", None))
    if getattr(args, "record_action_only", False):
        cmd.append("--record-action-only")

    _add_value(cmd, "--top-k-general-skills", getattr(args, "top_k_general_skills", None))
    _add_value(cmd, "--top-k-task-specific-skills", getattr(args, "top_k_task_specific_skills", None))
    _add_value(cmd, "--general-skill-retrieval-threshold", getattr(args, "general_skill_retrieval_threshold", None))
    _add_value(cmd, "--task-specific-skill-retrieval-threshold", getattr(args, "task_specific_skill_retrieval_threshold", None))
    _add_value(cmd, "--task-specific-retrieval-scope", getattr(args, "task_specific_retrieval_scope", None))

    eval_out.mkdir(parents=True, exist_ok=True)
    command_path = eval_out / "auto_eval_command.txt"
    command_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    print("=" * 60, flush=True)
    print("  FINAL TEST — evaluate final evolving_skill.md", flush=True)
    print("=" * 60, flush=True)
    print(f"  Test manifest: {eval_manifest} ({_manifest_count(eval_manifest)} items)", flush=True)
    print(f"  Skill:         {skill_path}", flush=True)
    print(f"  Output:        {eval_out}", flush=True)
    print(f"[auto-eval] command={shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    summary_path = eval_out / "eval_summary.json"
    result = {
        "enabled": True,
        "skipped": False,
        "returncode": proc.returncode,
        "output_dir": str(eval_out),
        "skill_md": str(skill_path),
        "manifest": str(eval_manifest),
        "command": str(command_path),
        "summary_path": str(summary_path),
    }
    if summary_path.exists():
        try:
            result["summary"] = load_json(str(summary_path))
        except Exception as exc:
            result["summary_error"] = str(exc)
    if proc.returncode != 0:
        result["error"] = f"auto eval exited with code {proc.returncode}"
    return result


def main(args):
    if not args.output_dir:
        args.output_dir = str(default_dual_layer_output_dir(args))
        print(f"[generate] output_dir={args.output_dir}", flush=True)
    output_lock = _acquire_output_dir_lock(args.output_dir)
    try:
        runtime_path = Path(args.output_dir) / "action_runtime_config.json"
        _ensure_llm_context_config(Path(args.output_dir), args)
        ensure_action_runtime_config(
            runtime_path,
            action_runtime_config(args),
            resume=not getattr(args, "no_resume", False),
            context="generation",
        )
        _print_generation_banner(args)
        info = run_eviskill(args)
        save_json(str(Path(args.output_dir) / "eviskill_run_info.json"), info)
        if not getattr(args, "no_auto_eval", False):
            auto_eval = _run_auto_eval(args, info)
            info["auto_eval"] = auto_eval
            save_json(str(Path(args.output_dir) / "eviskill_run_info.json"), info)
            if int(auto_eval.get("returncode", 0) or 0) != 0:
                raise RuntimeError(
                    f"Final auto-eval failed with code {auto_eval.get('returncode')}; "
                    "the watcher can restart and resume the unfinished test bundles."
                )
        print(info, flush=True)
    finally:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_UN)
        output_lock.close()


def add_generation_args(p):
    p.add_argument("--dataset", choices=["alfworld", "scienceworld", "appworld"], default="alfworld")
    p.add_argument("--manifest", required=True)
    p.add_argument("--skill-bank", required=True, help="Initial Markdown skill.md. Kept as an alias for --initial-skill-md.")
    p.add_argument("--initial-skill-md", default=None, help="Initial Markdown skill.md. Defaults to --skill-bank.")
    p.add_argument("--experiment-tag", default=None, help="Optional tag appended to the automatic output directory, e.g. large-modify.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to outputs/{dataset}_{action_model}[_experiment_tag], e.g. outputs/alfworld_gpt_5.5.",
    )
    p.add_argument("--llm-debug-dir", default=None)

    p.add_argument("--alfworld-path", default=None, help="Optional ALFWorld source checkout added to PYTHONPATH.")
    p.add_argument("--alfworld-config", default=None, help="ALFWorld runtime YAML configuration.")
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

    p.add_argument("--action-model", default=None)
    p.add_argument("--action-llm-kind", choices=["openai", "hf"], default=None)
    p.add_argument(
        "--action-api-key",
        default=None,
        help="ActionModel API key. Defaults to ACTION_OPENAI_API_KEY, then OPENAI_API_KEY.",
    )
    p.add_argument(
        "--action-base-url",
        default=None,
        help="ActionModel endpoint. Defaults to ACTION_OPENAI_BASE_URL, then OPENAI_BASE_URL.",
    )
    p.add_argument(
        "--action-proxy-url",
        default=None,
        help="Optional ActionModel HTTP proxy. Defaults to ACTION_OPENAI_PROXY_URL.",
    )
    p.add_argument("--action-temperature", type=float, default=None, help="Optional action-model sampling temperature. OpenAI requests omit temperature by default, matching SkillOPT.")
    p.add_argument("--action-max-new-tokens", type=int, default=512)
    p.add_argument("--action-load-in-4bit", action="store_true")
    p.add_argument("--action-timeout", type=int, default=180)
    p.add_argument("--action-max-retries", type=int, default=-1, help="Maximum LLM retries after failures. Use -1 to retry indefinitely.")
    p.add_argument("--action-retry-initial-sleep", type=float, default=5.0)
    p.add_argument("--action-retry-max-sleep", type=float, default=60.0)
    p.add_argument("--action-thinking", choices=["enabled", "disabled"], default=None)
    p.add_argument("--action-reasoning-effort", choices=["none", "low", "medium", "high"], default="medium")
    p.add_argument("--action-extra-body-json", default=None)
    p.add_argument("--dry-run-action", action="store_true")

    p.add_argument("--teacher-model", required=True)
    p.add_argument("--llm-kind", choices=["openai", "hf"], default="openai")
    p.add_argument(
        "--teacher-api-key",
        "--api-key",
        dest="teacher_api_key",
        default=None,
        help=(
            "Teacher OpenAI-compatible API key. --api-key is retained as a legacy alias. "
            "Defaults to TEACHER_OPENAI_API_KEY, then OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--teacher-base-url",
        "--base-url",
        dest="teacher_base_url",
        default=None,
        help=(
            "Teacher OpenAI-compatible endpoint. --base-url is retained as a legacy alias. "
            "Defaults to TEACHER_OPENAI_BASE_URL, then OPENAI_BASE_URL."
        ),
    )
    p.add_argument(
        "--teacher-proxy-url",
        default=None,
        help="Optional teacher HTTP proxy. Defaults to TEACHER_OPENAI_PROXY_URL.",
    )
    p.add_argument("--teacher-temperature", type=float, default=None, help="Optional teacher sampling temperature. OpenAI requests omit temperature by default, matching SkillOPT.")
    p.add_argument("--teacher-max-tokens", type=int, default=16384)
    p.add_argument("--teacher-load-in-4bit", action="store_true")
    p.add_argument("--teacher-timeout", type=int, default=180)
    p.add_argument("--teacher-max-retries", type=int, default=-1, help="Maximum LLM retries after failures. Use -1 to retry indefinitely.")
    p.add_argument("--teacher-retry-initial-sleep", type=float, default=5.0)
    p.add_argument("--teacher-retry-max-sleep", type=float, default=60.0)
    p.add_argument("--teacher-thinking", choices=["enabled", "disabled"], default=None)
    p.add_argument("--teacher-reasoning-effort", choices=["none", "low", "medium", "high"], default="medium")
    p.add_argument("--teacher-extra-body-json", default=None)

    p.add_argument("--epochs", type=int, default=1, help="Number of outer evolution epochs/rounds. Each epoch rolls out the full evolution manifest once, then runs L2/validation.")
    p.add_argument(
        "--bundle-size",
        type=int,
        default=2,
        help="Tasks per rollout/checkpoint bundle and requested rollout worker count for every dataset.",
    )
    p.add_argument(
        "--l1-trajectory-minibatch-size",
        type=int,
        default=4,
        help="Failure/success trajectories per L1 evidence-extraction call after the epoch rollout.",
    )
    p.add_argument(
        "--bundle-sampling-strategy",
        choices=["random", "shuffle_epoch", "family_epoch", "mixed_family_epoch"],
        default="shuffle_epoch",
        help=(
            "How evolution bundles are sampled. shuffle_epoch is the default full-manifest pass; "
            "random preserves the old with-replacement per-round ablation; "
            "shuffle_epoch shuffles the whole manifest each epoch and slices contiguous bundles; "
            "family_epoch shuffles and round-robin interleaves task_family groups each epoch; "
            "mixed_family_epoch alternates family_epoch bundles with same-family bundles."
        ),
    )
    p.add_argument(
        "--family-group-frequency",
        type=int,
        default=2,
        help="For mixed_family_epoch, every Nth round is a same-family bundle; 1 means always, <=0 falls back to family_epoch.",
    )
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-tasks", type=int, default=None)
    p.add_argument("--num-products", type=int, default=1000)

    p.add_argument("--record-trajectory-mode", choices=["prefix", "tail", "head_tail", "full"], default="full")
    p.add_argument("--record-max-steps", type=int, default=128)
    p.add_argument("--record-trajectory-head-steps", type=int, default=1)
    p.add_argument("--record-trajectory-tail-steps", type=int, default=8)
    p.add_argument("--record-max-observation-chars", type=int, default=1500)
    p.add_argument("--record-action-only", action="store_true")
    p.add_argument(
        "--llm-max-observation-chars",
        type=int,
        default=3000,
        help="Soft per-step cap for ordinary LLM-visible environment feedback. Critical feedback is preserved by the evidence-aware formatter; <0 disables this soft cap.",
    )
    p.add_argument(
        "--llm-max-input-chars",
        type=int,
        default=60000,
        help="Target maximum system-prompt + payload characters. Evidence-aware formatting preserves actions and critical feedback; <=0 disables the target.",
    )
    p.add_argument(
        "--llm-full-context",
        action="store_true",
        help=(
            "Disable L1/L2 LLM-visible trajectory and evidence compaction, including "
            "working-branch trajectory truncation; fail explicitly on model context overflow."
        ),
    )

    p.add_argument("--top-k-general-skills", type=int, default=5)
    p.add_argument("--top-k-task-specific-skills", type=int, default=2)
    p.add_argument("--general-skill-retrieval-threshold", type=float, default=0.0)
    p.add_argument("--task-specific-skill-retrieval-threshold", type=float, default=0.0)
    p.add_argument("--task-specific-retrieval-scope", choices=["all", "detected"], default="detected")

    p.add_argument("--evidence-window-size", type=int, default=20)
    p.add_argument(
        "--reuse-evidence-dir",
        default=None,
        help="Optional existing epoch artifact directory or output root. If present, reuse train EvidenceCards/replay sources and start from skill editing.",
    )
    p.add_argument(
        "--epoch-skill-update-mode",
        choices=["merge_rank", "cluster_step"],
        default="cluster_step",
        help="merge_rank keeps the hard epoch rank/budget ablation; cluster_step runs SkillOPT-style one-step-per-epoch coverage merge plus one validation gate.",
    )
    p.add_argument("--epoch-l2-windows", type=int, default=4, help="Maximum grouped evidence windows for merge_rank. In semantic cluster_step this is ignored and all natural windows are processed.")
    p.add_argument("--epoch-edit-budget", type=int, default=4, help="Maximum globally ranked epoch edits allowed to enter validation in merge_rank and in working-branch cluster_step. Base cluster_step keeps its coverage behavior.")
    p.add_argument("--epoch-merge-max-candidates", type=int, default=24, help="merge_rank candidate cap; in cluster_step this is only the recursive merge chunk size, not a truncation limit.")
    p.add_argument("--epoch-merge-max-content-chars", type=int, default=1200, help="Legacy merge_rank-only cap for candidate edit content; cluster_step keeps complete accepted edits.")
    p.add_argument("--epoch-merge-skill-max-chars", type=int, default=12000, help="Legacy merge_rank-only skill.md cap; cluster_step always includes the complete skill.")
    p.add_argument("--no-epoch-global-merge", action="store_true", help="Disable the epoch merge LLM call and use deterministic accepted-edit fallback.")
    p.add_argument(
        "--working-branch-llm-rank",
        action="store_true",
        help=(
            "Use one epoch-level LLM rank over the complete cluster_step merged pool "
            "before applying the working-branch edit budget."
        ),
    )
    p.add_argument("--l2-candidate-budget", type=int, default=5, help="Window edit compactness hint. It is a hard cap only outside cluster_step coverage mode.")
    p.add_argument(
        "--l2-replay-ranges-per-edit",
        type=_replay_range_budget,
        default=4,
        help=(
            "Replay ranges per edit: controls window-local L2 replay and, when the "
            "evidence working branch is enabled, the per-edit post-reject final-context "
            "replay limit. -1 replays all eligible linked ranges, 0 disables replay, "
            "and N>0 replays at most N ranges per edit in each applicable stage."
        ),
    )
    p.add_argument(
        "--post-reject-replay-cap",
        type=_replay_range_budget,
        default=8,
        help=(
            "Working-branch post-reject Action replay budget per epoch: -1 removes the "
            "epoch cap, 0 prevents rejected candidates from entering the working branch, "
            "and N>0 permits at most N final-context replays. The per-edit limit still "
            "comes from --l2-replay-ranges-per-edit."
        ),
    )
    p.add_argument(
        "--working-branch-provisional-cap",
        type=int,
        default=4,
        help="Maximum number of live provisional edits retained by the evidence working branch.",
    )
    p.add_argument(
        "--working-branch-defer-uncovered-evidence",
        action="store_true",
        help=(
            "In AppWorld working-branch cluster_step, defer Evidence not covered by "
            "the proposer, global merge, or coverage repair instead of creating "
            "fallback edits."
        ),
    )
    p.add_argument(
        "--defer-rule-ignored",
        action=_StoreDeferRuleIgnored,
        dest="defer_rule_ignored",
        nargs=0,
        default=False,
        help=(
            "Globally defer L2 edits whose replay failure attribution is "
            "rule_ignored. This applies with or without the evidence working branch."
        ),
    )
    p.add_argument(
        "--working-branch-defer-rule-ignored",
        action=_StoreDeferRuleIgnored,
        dest="defer_rule_ignored",
        nargs=0,
        default=False,
        help="Deprecated compatibility alias for --defer-rule-ignored.",
    )
    p.set_defaults(working_branch_defer_rule_ignored=False)
    p.add_argument(
        "--working-branch-max-policy-defer-epochs",
        type=_policy_defer_epoch_limit,
        default=-1,
        help=(
            "Working-branch limit on distinct epochs an EvidenceCard may be deferred "
            "by its policy before becoming stale. -1 disables the limit; 0 is invalid."
        ),
    )
    p.add_argument(
        "--epoch-trajectory-comparison-tasks",
        type=int,
        default=0,
        help=(
            "Working-branch train tasks compared with the preceding epoch to create "
            "standalone contrastive failure EvidenceCards. 0 disables the feature."
        ),
    )
    p.add_argument(
        "--epoch-final-replay-ranges-per-edit",
        type=_replay_range_budget,
        default=2,
        help="Epoch-final replay ranges per merged edit: -1 replays all linked ranges, 0 skips epoch-final replay/Judge and proceeds directly to validation, and N>0 replays at most N ranges.",
    )
    p.add_argument(
        "--no-replay",
        action="store_true",
        help=(
            "Disable every L2 ActionAgent replay. Local L2 Judges use only the candidate "
            "edit and linked EvidenceCards, replay attribution/reflection is disabled, "
            "and working-branch reject transitions mechanically retain eligible edits."
        ),
    )
    p.add_argument(
        "--semantic-window-max-evidence",
        "--l2-max-evidence-per-window",
        dest="l2_max_evidence_per_window",
        type=int,
        default=20,
        help="Maximum active EvidenceCards included in each evidence window. The old --l2-max-evidence-per-window name is kept as a compatibility alias.",
    )
    p.add_argument(
        "--l2-evidence-window-grouping",
        choices=["auto", "card", "round", "semantic"],
        default="auto",
        help="How to pack EvidenceCards. auto uses round grouping for ScienceWorld and card grouping for other datasets; semantic uses embedding clusters.",
    )
    p.add_argument("--evidence-embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="SentenceTransformer model used by semantic EvidenceCard clustering.")
    p.add_argument("--evidence-embedding-device", default="auto", help="Device for semantic EvidenceCard embeddings; auto lets sentence-transformers decide.")
    p.add_argument("--evidence-embedding-batch-size", type=int, default=32, help="Batch size for semantic EvidenceCard embedding.")
    p.add_argument(
        "--semantic-clustering-method",
        choices=["hdbscan", "graph"],
        default="hdbscan",
        help="Semantic EvidenceCard clustering method. hdbscan adapts to local embedding density; graph uses fixed mutual-kNN cosine threshold.",
    )
    p.add_argument("--semantic-hdbscan-min-cluster-size", default="auto", help="Minimum HDBSCAN cluster size for semantic EvidenceCards, or auto.")
    p.add_argument("--semantic-hdbscan-min-samples", default="auto", help="HDBSCAN min_samples for semantic EvidenceCards, or auto.")
    p.add_argument("--semantic-cluster-threshold", type=float, default=0.68, help="Graph-mode cosine threshold for mutual-kNN semantic EvidenceCard edges.")
    p.add_argument("--semantic-cluster-top-k", type=int, default=8, help="Graph-mode mutual nearest-neighbor count used by semantic EvidenceCard clustering.")
    p.add_argument(
        "--semantic-window-min-evidence",
        "--semantic-cluster-min-window-size",
        dest="semantic_cluster_min_window_size",
        type=int,
        default=4,
        help="Target minimum final semantic evidence window size. Smaller clusters are merged/rebalanced when possible; evidence is not dropped. The old --semantic-cluster-min-window-size name is kept as a compatibility alias.",
    )
    p.add_argument("--semantic-cluster-merge-threshold", type=float, default=0.58, help="Centroid cosine threshold for merging small semantic clusters into nearby windows.")
    p.add_argument("--semantic-cluster-max-reason-chars", type=int, default=300, help="Maximum pattern characters included in semantic embedding text.")
    p.add_argument("--semantic-cluster-max-actions", type=int, default=12, help="Maximum trigger actions included in semantic embedding text.")
    p.add_argument("--l2-max-rejects-per-evidence", type=int, default=2, help="Mark evidence stale after repeated explicit L2 judge rejects. cluster_step defers uncovered evidence instead of consuming it.")
    p.add_argument("--no-l2-selection-feedback", action="store_true", help="Disable lightweight selection feedback extraction for later Skill Edit Proposer calls. ScienceWorld epoch-style evolution never feeds validation/dev cases back into skill edit prompts.")
    p.add_argument("--no-epoch-validation", action="store_true", help="Skip the epoch selection validation gate and commit globally merged accepted edits directly.")
    p.add_argument("--selection-manifest", default=None, help="Held-out validation/selection manifest used to gate epoch candidate skills. ScienceWorld defaults to sibling dev.json when available.")
    p.add_argument("--test-manifest", default=None, help="Held-out test manifest used by the automatic final static evaluation. ScienceWorld can infer sibling test/items.json from split-dir manifests.")
    p.add_argument("--l2-selection-size", type=int, default=24, help="Number of fixed manifest tasks held out from evolution bundles for the L2 selection gate.")
    p.add_argument("--l2-selection-seed", type=int, default=42, help="Seed for the fixed L2 selection task subset.")
    p.add_argument("--l2-selection-feedback-max-positive", type=int, default=3, help="Legacy feedback setting for non-ScienceWorld selection comparison cases.")
    p.add_argument("--l2-selection-feedback-max-negative", type=int, default=3, help="Legacy feedback setting for non-ScienceWorld selection comparison cases.")
    p.add_argument("--l2-selection-feedback-history", type=int, default=5, help="Legacy feedback history size for Skill Edit Proposer inputs when feedback is enabled.")
    p.add_argument("--l2-selection-feedback-step-delta", type=int, default=5, help="Legacy same-score step delta used when choosing selection feedback cases.")
    p.add_argument("--selection-success-margin", type=float, default=0.0, help="Required success-rate improvement margin for the selection gate primary metric.")
    p.add_argument(
        "--selection-accept-tie",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Commit a candidate when validation hard success rate exactly ties the "
            "current skill. Disabled by default; use --selection-accept-tie to restore "
            "tie commits."
        ),
    )
    p.add_argument("--selection-score-tiebreak-margin", type=float, default=0.01, help="Required score improvement when selection success-rate is tied.")
    p.add_argument("--selection-step-tiebreak-delta", type=float, default=1.0, help="Required average-step reduction when success-rate and score are tied.")
    p.add_argument("--use-longitudinal-memory", action="store_true", help="Enable epoch-level Longitudinal Skill Memory updates after evidence-based skill edits.")
    p.add_argument(
        "--use-evidence-working-branch",
        action="store_true",
        help=(
            "Enable the active Evidence working branch: rollout from working_skill.md, "
            "retain evolving_skill.md as the validated best, and use capped post-reject "
            "replay plus adjacent-epoch contrastive train evidence."
        ),
    )
    p.add_argument("--longitudinal-memory-samples", type=int, default=20, help="Number of evolution-train tasks sampled for previous/current skill comparison.")
    p.add_argument(
        "--longitudinal-memory-pair-policy",
        choices=["mixed", "changed", "unchanged"],
        default="mixed",
        help="Which longitudinal comparison pairs are shown to the memory writer.",
    )
    p.add_argument("--longitudinal-memory-max-trajectory-chars", type=int, default=3000, help="Maximum compact trajectory characters per rollout in longitudinal memory prompts.")
    p.add_argument("--longitudinal-memory-gate-with-selection", action="store_true", help="Validate the Longitudinal Skill Memory candidate on the selection set before writing it.")
    p.add_argument("--no-auto-eval", action="store_true", help="Disable automatic static evaluation of the final evolving_skill.md.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-progress", action="store_true")


def parse_args():
    p = argparse.ArgumentParser()
    add_generation_args(p)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
