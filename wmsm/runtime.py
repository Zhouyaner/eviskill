import json
import re
from typing import Any, Optional

from .action_runtime import (
    ALFWORLD_ACTION_HISTORY_LENGTH,
    APPWORLD_ACTION_HISTORY_LENGTH,
    SCIENCEWORLD_ACTION_HISTORY_LENGTH,
)


def build_alfworld_runner(
    *,
    alfworld_path: str,
    alfworld_config: str,
    alfworld_split: str,
    action_llm: Any,
    max_steps: int = 50,
    top_k_general_skills: int = 6,
    top_k_task_specific_skills: int = 6,
    general_skill_retrieval_threshold: float = 0.0,
    task_specific_skill_retrieval_threshold: float = 0.0,
    task_specific_retrieval_scope: str = "detected",
    max_resets_to_find_task: int = 512,
    action_history_length: int = ALFWORLD_ACTION_HISTORY_LENGTH,
    initialize_full_env: bool = True,
    parallel_workers: int = 1,
):
    from .agents import AlfWorldActionAgent
    from .alfworld_runner import AlfWorldRunner

    return AlfWorldRunner(
        AlfWorldActionAgent(action_llm),
        alfworld_path=alfworld_path,
        alfworld_config=alfworld_config,
        alfworld_split=alfworld_split,
        max_steps=max_steps,
        top_k_general_skills=top_k_general_skills,
        top_k_task_specific_skills=top_k_task_specific_skills,
        general_skill_retrieval_threshold=general_skill_retrieval_threshold,
        task_specific_skill_retrieval_threshold=task_specific_skill_retrieval_threshold,
        task_specific_retrieval_scope=task_specific_retrieval_scope,
        max_resets_to_find_task=max_resets_to_find_task,
        action_history_length=action_history_length,
        initialize_full_env=initialize_full_env,
        parallel_workers=parallel_workers,
    )


def build_scienceworld_runner(
    *,
    action_llm: Any,
    scienceworld_env_url: str = "http://127.0.0.1:8811",
    scienceworld_env_timeout: int = 180,
    scienceworld_parallel_workers: int = 1,
    scienceworld_auto_server: bool = False,
    scienceworld_server_host: str = "127.0.0.1",
    scienceworld_server_port_start: int = 8811,
    scienceworld_path: Optional[str] = None,
    scienceworld_jar_path: Optional[str] = None,
    scienceworld_env_step_limit: Optional[int] = None,
    scienceworld_server_python: Optional[str] = None,
    top_k_general_skills: int = 6,
    top_k_task_specific_skills: int = 6,
    general_skill_retrieval_threshold: float = 0.0,
    task_specific_skill_retrieval_threshold: float = 0.0,
    task_specific_retrieval_scope: str = "detected",
    max_steps: int = 100,
    action_history_length: int = SCIENCEWORLD_ACTION_HISTORY_LENGTH,
    use_admissible_actions: bool = False,
):
    from .agents import ScienceWorldActionAgent
    from .scienceworld_runner import ScienceWorldRunner

    return ScienceWorldRunner(
        ScienceWorldActionAgent(action_llm),
        remote_environment_url=scienceworld_env_url,
        remote_timeout=scienceworld_env_timeout,
        parallel_workers=scienceworld_parallel_workers,
        auto_server=scienceworld_auto_server,
        server_host=scienceworld_server_host,
        server_port_start=scienceworld_server_port_start,
        scienceworld_path=scienceworld_path,
        scienceworld_jar_path=scienceworld_jar_path,
        scienceworld_env_step_limit=scienceworld_env_step_limit,
        scienceworld_server_python=scienceworld_server_python,
        max_steps=max_steps,
        top_k_general_skills=top_k_general_skills,
        top_k_task_specific_skills=top_k_task_specific_skills,
        general_skill_retrieval_threshold=general_skill_retrieval_threshold,
        task_specific_skill_retrieval_threshold=task_specific_skill_retrieval_threshold,
        task_specific_retrieval_scope=task_specific_retrieval_scope,
        action_history_length=action_history_length,
        use_admissible_actions=use_admissible_actions,
    )


def build_appworld_runner(
    *,
    action_llm: Any,
    appworld_data_dir: Optional[str] = None,
    output_dir: str,
    appworld_env_url: Optional[str] = None,
    appworld_auto_server: bool = False,
    appworld_server_python: Optional[str] = None,
    appworld_server_host: str = "127.0.0.1",
    appworld_server_port_start: int = 8841,
    appworld_env_timeout: int = 100,
    action_history_length: int = APPWORLD_ACTION_HISTORY_LENGTH,
    random_seed: int = 100,
    max_steps: int = 50,
    parallel_workers: int = 1,
    top_k_general_skills: int = 5,
    top_k_task_specific_skills: int = 2,
    general_skill_retrieval_threshold: float = 0.0,
    task_specific_skill_retrieval_threshold: float = 0.0,
    task_specific_retrieval_scope: str = "detected",
):
    from .agents import AppWorldActionAgent
    from .appworld_runner import AppWorldRunner

    return AppWorldRunner(
        AppWorldActionAgent(action_llm),
        data_dir=appworld_data_dir,
        output_dir=output_dir,
        environment_url=appworld_env_url,
        auto_server=appworld_auto_server,
        server_python=appworld_server_python,
        server_host=appworld_server_host,
        server_port_start=appworld_server_port_start,
        environment_timeout=appworld_env_timeout,
        max_steps=max_steps,
        action_history_length=action_history_length,
        random_seed=random_seed,
        parallel_workers=parallel_workers,
        top_k_general_skills=top_k_general_skills,
        top_k_task_specific_skills=top_k_task_specific_skills,
        general_skill_retrieval_threshold=general_skill_retrieval_threshold,
        task_specific_skill_retrieval_threshold=task_specific_skill_retrieval_threshold,
        task_specific_retrieval_scope=task_specific_retrieval_scope,
    )


class FixedJsonActionLLM:
    """Tiny action LLM for script dry-runs; real experiments should pass an actual model."""

    def chat_json(self, system_prompt, payload):
        payload = payload or {}
        admissible = payload.get("admissible_actions") or (payload.get("context") or {}).get("admissible_actions") or []
        action = admissible[0] if admissible else "search[shirt]"
        return {"thought": "fixed dry-run action", "action": action}

    def chat(self, messages, temperature=None, max_tokens=None):
        prompt = "\n".join(str((m or {}).get("content", "")) for m in messages or [] if isinstance(m, dict))
        match = re.search(r'Admissible actions of the current situation:\s*(\[[\s\S]*?\])', prompt)
        if match:
            try:
                actions = json.loads(match.group(1))
            except Exception:
                actions = []
        else:
            actions = []
        action = str(actions[0]) if actions else "search[shirt]"
        return f"<think>Choose the first admissible dry-run action.</think>\n<action>{action}</action>"
