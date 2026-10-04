"""Normalize the effective Action Agent runtime and persist run snapshots.

The dataset YAML files are the source of truth for user-editable experiment
parameters. This module deliberately does not load a second configuration
file: it normalizes parsed arguments, records derived runtime values, and
checks that resumed evaluation uses the same effective settings.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ACTION_RUNTIME_SCHEMA_VERSION = 1
OPTIONAL_RUNTIME_DEFAULTS = {
    "common.action_proxy_url": None,
}
ALFWORLD_ACTION_HISTORY_LENGTH = 8
SCIENCEWORLD_ACTION_HISTORY_LENGTH = 8
APPWORLD_ACTION_HISTORY_LENGTH = 8


def _value(args: Any, name: str, default: Any = None) -> Any:
    value = getattr(args, name, default)
    return default if value is None and default is not None else value


def _effective_action_kind(args: Any) -> str:
    return str(getattr(args, "action_llm_kind", None) or "openai")


def _normalized_extra_body(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return str(value)


def action_runtime_config(args: Any) -> Dict[str, Any]:
    dataset = str(getattr(args, "dataset", "") or "").lower()
    if dataset == "science_world":
        dataset = "scienceworld"

    action_base_url = (
        getattr(args, "action_base_url", None)
        or os.environ.get("ACTION_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None
    )
    if action_base_url:
        action_base_url = str(action_base_url).rstrip("/")

    common = {
        "action_model": getattr(args, "action_model", None),
        "action_llm_kind": _effective_action_kind(args),
        "action_base_url": action_base_url,
        "action_proxy_url": (
            getattr(args, "action_proxy_url", None)
            or os.environ.get("ACTION_OPENAI_PROXY_URL")
            or os.environ.get("OPENAI_PROXY_URL")
            or None
        ),
        "action_temperature": getattr(args, "action_temperature", None),
        "action_max_new_tokens": _value(args, "action_max_new_tokens", 512),
        "action_load_in_4bit": bool(getattr(args, "action_load_in_4bit", False)),
        "action_timeout": _value(args, "action_timeout", 180),
        "action_max_retries": _value(args, "action_max_retries", -1),
        "action_retry_initial_sleep": _value(args, "action_retry_initial_sleep", 5.0),
        "action_retry_max_sleep": _value(args, "action_retry_max_sleep", 60.0),
        "action_thinking": getattr(args, "action_thinking", None),
        "action_reasoning_effort": getattr(args, "action_reasoning_effort", "medium"),
        "action_extra_body": _normalized_extra_body(
            getattr(args, "action_extra_body_json", None)
        ),
        "dry_run_action": bool(getattr(args, "dry_run_action", False)),
        "max_steps": _value(args, "max_steps", 50),
        "parallel_workers": max(1, int(_value(args, "bundle_size", 1) or 1)),
        "seed": _value(args, "seed", 42),
        "top_k_general_skills": _value(args, "top_k_general_skills", 5),
        "top_k_task_specific_skills": _value(args, "top_k_task_specific_skills", 2),
        "general_skill_retrieval_threshold": _value(
            args, "general_skill_retrieval_threshold", 0.0
        ),
        "task_specific_skill_retrieval_threshold": _value(
            args, "task_specific_skill_retrieval_threshold", 0.0
        ),
        "task_specific_retrieval_scope": getattr(
            args, "task_specific_retrieval_scope", "detected"
        ),
    }

    if dataset == "alfworld":
        dataset_runtime = {
            "alfworld_path": getattr(args, "alfworld_path", None),
            "alfworld_config": getattr(args, "alfworld_config", None),
            "alfworld_max_resets_to_find_task": _value(
                args, "alfworld_max_resets_to_find_task", 512
            ),
            "action_history_length": _value(
                args,
                "alfworld_action_history_length",
                ALFWORLD_ACTION_HISTORY_LENGTH,
            ),
        }
    elif dataset == "scienceworld":
        dataset_runtime = {
            "scienceworld_env_url": getattr(args, "scienceworld_env_url", None),
            "scienceworld_env_timeout": _value(args, "scienceworld_env_timeout", 180),
            "scienceworld_auto_server": bool(
                getattr(args, "scienceworld_auto_server", False)
            ),
            "scienceworld_server_host": getattr(
                args, "scienceworld_server_host", "127.0.0.1"
            ),
            "scienceworld_server_port_start": _value(
                args, "scienceworld_server_port_start", 8811
            ),
            "scienceworld_path": getattr(args, "scienceworld_path", None),
            "scienceworld_jar_path": getattr(args, "scienceworld_jar_path", None),
            "scienceworld_env_step_limit": getattr(
                args, "scienceworld_env_step_limit", None
            ),
            "scienceworld_server_python": getattr(
                args, "scienceworld_server_python", None
            ),
            "action_history_length": _value(
                args,
                "scienceworld_action_history_length",
                SCIENCEWORLD_ACTION_HISTORY_LENGTH,
            ),
            "use_admissible_actions": bool(
                getattr(args, "scienceworld_use_admissible_actions", False)
            ),
        }
    elif dataset == "appworld":
        # The AppWorld runner uses the active interpreter by default. Keep the
        # runtime snapshot aligned with that behavior instead of recording a
        # machine-specific `python` path.
        server_python = str(
            getattr(args, "appworld_server_python", None)
            or os.environ.get("APPWORLD_SERVER_PYTHON")
            or sys.executable
        )
        data_value = getattr(args, "appworld_data_dir", None) or os.environ.get("APPWORLD_DATA_DIR")
        data_path = Path(data_value).expanduser().resolve() if data_value else Path.cwd()
        if (data_path / "data" / "tasks").is_dir():
            data_path = data_path / "data"
        try:
            version_result = subprocess.run(
                [server_python, "-c", "import appworld; print(appworld.__version__)"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            package_version = version_result.stdout.strip() or "unknown"
        except Exception:
            package_version = "unknown"
        dataset_runtime = {
            "appworld_data_dir": str(data_path),
            "appworld_package_version": package_version,
            "appworld_env_url": getattr(args, "appworld_env_url", None),
            "appworld_auto_server": bool(getattr(args, "appworld_auto_server", False)),
            "appworld_server_python": server_python,
            "appworld_server_host": getattr(args, "appworld_server_host", "127.0.0.1"),
            "appworld_server_port_start": _value(args, "appworld_server_port_start", 8841),
            "appworld_env_timeout": _value(args, "appworld_env_timeout", 100),
            "action_history_length": _value(
                args,
                "appworld_action_history_length",
                APPWORLD_ACTION_HISTORY_LENGTH,
            ),
            "action_context_mode": "full_interactions",
            "random_seed": _value(args, "appworld_random_seed", 100),
            "max_api_calls_per_interaction": 1000,
            "raise_on_extra_parameters": True,
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    return {
        "schema_version": ACTION_RUNTIME_SCHEMA_VERSION,
        "dataset": dataset,
        "common": common,
        "dataset_runtime": dataset_runtime,
    }


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    rows: Dict[str, Any] = {}
    for key in sorted(value):
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        rows.update(_flatten(value[key], child_prefix))
    return rows


def action_runtime_differences(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> List[str]:
    expected_flat = _flatten(expected)
    actual_flat = _flatten(actual)
    differences = []
    for key in sorted(set(expected_flat) | set(actual_flat)):
        default = OPTIONAL_RUNTIME_DEFAULTS.get(key, "<missing>")
        expected_value = expected_flat.get(key, default)
        actual_value = actual_flat.get(key, default)
        if expected_value != actual_value:
            differences.append(
                f"{key}: training={expected_value!r} evaluation={actual_value!r}"
            )
    return differences


def ensure_action_runtime_config(
    path: Path,
    current: Dict[str, Any],
    *,
    resume: bool,
    context: str,
) -> Dict[str, Any]:
    path = Path(path)
    if resume and path.exists():
        expected = json.loads(path.read_text(encoding="utf-8"))
        differences = action_runtime_differences(expected, current)
        if differences:
            details = "\n  - ".join(differences)
            raise RuntimeError(
                f"{context} action runtime differs from the existing run:\n  - {details}\n"
                "Use the original rollout parameters or a new output directory."
            )
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return current
