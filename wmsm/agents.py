import json
import re
from typing import Any, Dict, List

from .json_utils import extract_json_obj
from .prompts import (
    ALFWORLD_ACTION_AGENT_PROMPT,
    APPWORLD_ACTION_AGENT_PROMPT,
    SCIENCEWORLD_ACTION_AGENT_PROMPT,
)
from .prompts.appworld import APPWORLD_ACTION_FEW_SHOT_MESSAGES
from .prompts.alfworld import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS


APPWORLD_FALLBACK_CODE = "print(apis.api_docs.show_app_descriptions())"


def _format_recent_history(rows: List[Dict[str, Any]]) -> str:
    parts = []
    for row in rows or []:
        step = row.get("step", len(parts)) if isinstance(row, dict) else len(parts)
        observation = str((row or {}).get("observation", "")) if isinstance(row, dict) else ""
        action = str((row or {}).get("action", "")) if isinstance(row, dict) else str(row)
        if observation:
            parts.append(f"Step {step}: observation={observation} action={action}")
        else:
            parts.append(f"Step {step}: action={action}")
    return "\n".join(parts) if parts else "None"


def _format_alfworld_recent_history(rows: List[Dict[str, Any]]) -> str:
    parts = []
    for index, row in enumerate(rows or []):
        item = row if isinstance(row, dict) else {}
        step_num = int(item.get("step", index) or 0) + 1
        observation = str(item.get("observation") or "")
        action = str(item.get("action") or "")
        parts.append(
            f"[Observation {step_num}: '{observation}', Action {step_num}: '{action}']"
        )
    return "\n".join(parts)


def _format_score(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".") or "0"
    except (TypeError, ValueError):
        return "unavailable"


def _format_scienceworld_recent_history(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No previous steps."
    parts: List[str] = []
    for index, raw in enumerate(rows):
        row = raw if isinstance(raw, dict) else {}
        before = row.get("score_before")
        after = row.get("score_after")
        delta = row.get("score_delta")
        if before is None or after is None or delta is None:
            score_line = "Score: unavailable"
        else:
            try:
                delta_text = f"{float(delta):+.4f}".rstrip("0").rstrip(".")
            except (TypeError, ValueError):
                delta_text = "unavailable"
            score_line = (
                f"Score: {_format_score(before)} -> {_format_score(after)} "
                f"(delta {delta_text})"
            )
        parts.append(
            "\n".join(
                [
                    f"Step {row.get('step', index)}",
                    f"Observation: {str(row.get('observation') or '')}",
                    f"Action: {str(row.get('action') or '')}",
                    f"Result: {str(row.get('next_observation') or row.get('env_feedback') or '')}",
                    score_line,
                ]
            )
        )
    return "\n\n".join(parts)


def _extract_tagged_action(text: str) -> str:
    match = re.search(r"<action>\s*(.*?)\s*</action>", str(text or ""), flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def _extract_tagged_thought(text: str) -> str:
    match = re.search(r"<think>\s*(.*?)\s*</think>", str(text or ""), flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def _match_admissible_action(candidate: str, admissible_actions: List[str]) -> str:
    candidate = str(candidate or "").strip()
    actions = [str(x).strip() for x in admissible_actions or [] if str(x).strip()]
    if not actions:
        return candidate
    if candidate in actions:
        return candidate
    normalized = re.sub(r"\s+", " ", candidate).strip().lower()
    for action in actions:
        if re.sub(r"\s+", " ", action).strip().lower() == normalized:
            return action
    lower_text = str(candidate or "").lower()
    matches = [action for action in actions if action.lower() in lower_text]
    if len(matches) == 1:
        return matches[0]
    return candidate


def _fallback_action(text: str, admissible_actions: List[str]) -> str:
    tagged = _extract_tagged_action(text)
    if tagged:
        return _match_admissible_action(tagged, admissible_actions)
    actions = [str(x).strip() for x in admissible_actions or [] if str(x).strip()]
    lower_text = str(text or "").lower()
    matches = [action for action in actions if action.lower() in lower_text]
    if len(matches) == 1:
        return matches[0]
    return actions[0] if actions else ""


def extract_appworld_code(text: str) -> str:
    raw = str(text or "").strip()
    action_match = re.search(r"<action>\s*(.*?)\s*</action>", raw, flags=re.I | re.S)
    action_text = action_match.group(1).strip() if action_match else raw
    fenced = re.search(r"```(?:python|py)\s*\n?(.*?)```", action_text, flags=re.I | re.S)
    if fenced:
        code = fenced.group(1).strip()
        if code:
            return code
    partial = re.search(r"```(?:python|py)\s*\n?(.*)$", action_text, flags=re.I | re.S)
    if partial:
        code = partial.group(1).strip()
        if code:
            return code
    obj = extract_json_obj(action_text) or {}
    code = str(obj.get("code") or "").strip() if isinstance(obj, dict) else ""
    if code:
        return code
    try:
        compile(action_text, "<appworld-action>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return APPWORLD_FALLBACK_CODE
    return action_text or APPWORLD_FALLBACK_CODE


def _appworld_reasoning_without_code(text: str) -> str:
    raw = str(text or "").strip()
    tagged_thought = _extract_tagged_thought(raw)
    if tagged_thought:
        return tagged_thought
    raw = re.sub(r"```(?:python|py)\s*\n?.*?```", "", raw, flags=re.I | re.S)
    raw = re.sub(r"```(?:python|py)\s*\n?.*$", "", raw, flags=re.I | re.S)
    raw = re.sub(r"</?action>", "", raw, flags=re.I)
    obj = extract_json_obj(str(text or "")) or {}
    if isinstance(obj, dict) and obj.get("thought"):
        return str(obj.get("thought") or "").strip()
    return raw.strip()


class ActionAgent:
    def __init__(self, llm: Any, system_prompt: str = ""):
        self.llm = llm
        self.system_prompt = system_prompt

    def act(self, task, context, world_knowledge=None, retrieved_memory_text="", recent_history=None, step_id=0):
        payload = {
            "task": task,
            "observation": context.get("observation", ""),
            "admissible_actions": context.get("admissible_actions") or [],
            "skills": retrieved_memory_text or "",
            "recent_history": recent_history or [],
            "step": step_id,
        }
        if world_knowledge:
            payload["world_knowledge"] = world_knowledge
        text = self.llm.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        obj = extract_json_obj(text) or {}
        action = str(obj.get("action") or "").strip()
        if not action:
            action = _fallback_action(text, payload["admissible_actions"])
        return {
            "thought": str(obj.get("thought") or _extract_tagged_thought(text) or ""),
            "action": action,
        }


class AlfWorldActionAgent(ActionAgent):
    def __init__(self, llm: Any):
        super().__init__(llm, system_prompt=ALFWORLD_ACTION_AGENT_PROMPT)

    def act(self, task, context, world_knowledge=None, retrieved_memory_text="", recent_history=None, step_id=0):
        admissible_actions = [str(x) for x in context.get("admissible_actions") or []]
        step_count = max(0, int(step_id))
        current_observation = str(context.get("observation") or "")
        admissible_text = "\n ".join(f"'{action}'" for action in admissible_actions)
        skill_prefix = ""
        if str(retrieved_memory_text or "").strip():
            skill_prefix = (
                "\n\n## Skill Knowledge\n"
                "Below is a skill document with learned strategies. "
                "Use these guidelines to inform your decisions:\n\n"
                f"{str(retrieved_memory_text).strip()}\n"
            )
        if step_count <= 0:
            environment_prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=current_observation,
                admissible_actions=admissible_text,
            )
        else:
            history = _format_alfworld_recent_history(recent_history or [])
            environment_prompt = ALFWORLD_TEMPLATE.format(
                task_description=str(task or ""),
                step_count=step_count,
                history_length=len(recent_history or []),
                action_history=history,
                current_step=step_count + 1,
                current_observation=current_observation,
                admissible_actions=admissible_text,
            )
        user_prompt = skill_prefix + environment_prompt
        text = self.llm.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        thought = _extract_tagged_thought(text)
        action = _match_admissible_action(_extract_tagged_action(text), admissible_actions)
        if not action and admissible_actions:
            action = _fallback_action(text, admissible_actions)
            thought = thought or "No parsable action tag; using the first admissible action."
        return {"thought": thought, "action": action.strip()}


class ScienceWorldActionAgent(ActionAgent):
    def __init__(self, llm: Any):
        super().__init__(llm, system_prompt=SCIENCEWORLD_ACTION_AGENT_PROMPT)

    def act(self, task, context, world_knowledge=None, retrieved_memory_text="", recent_history=None, step_id=0):
        max_steps = context.get("max_steps")
        progress = (
            f"Step {int(step_id) + 1} of at most {int(max_steps)}."
            if max_steps not in (None, "")
            else f"Step {int(step_id) + 1}."
        )
        user_prompt = (
            f"## Task\n{str(task or '')}\n\n"
            f"## Task Family\n{str(context.get('task_family') or 'scienceworld')}\n\n"
            "## Skill Knowledge\n"
            f"{retrieved_memory_text or 'None'}\n\n"
            "## Progress\n"
            f"{progress}\n"
            f"Current score: {_format_score(context.get('current_score'))}\n\n"
            "## Recent History\n"
            f"{_format_scienceworld_recent_history(recent_history or [])}\n\n"
            "## Current Observation\n"
            f"{str(context.get('observation', ''))}\n\n"
        )
        admissible_actions = [str(x) for x in context.get("admissible_actions") or []]
        if admissible_actions:
            user_prompt += (
                "Candidate actions available in this state:\n"
                f"{json.dumps(admissible_actions, ensure_ascii=False)}\n\n"
            )
        user_prompt += "Now choose the next action."
        text = self.llm.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        thought = _extract_tagged_thought(text)
        action = _extract_tagged_action(text)
        if admissible_actions:
            action = _match_admissible_action(action, admissible_actions)
            if not action:
                action = _fallback_action(text, admissible_actions)
        elif not action:
            match = re.search(r"Action:\s*(.+)", str(text or ""), flags=re.I)
            action = match.group(1).strip().strip("\"'`*") if match else ""
        return {"thought": thought, "action": action.strip()}


class AppWorldActionAgent(ActionAgent):
    def __init__(self, llm: Any):
        super().__init__(llm, system_prompt=APPWORLD_ACTION_AGENT_PROMPT)

    def act(self, task, context, world_knowledge=None, retrieved_memory_text="", recent_history=None, step_id=0):
        supervisor = context.get("supervisor") if isinstance(context.get("supervisor"), dict) else {}
        app_descriptions = context.get("app_descriptions") or []
        max_steps = context.get("max_steps")
        task_context = {
            "task": str(task or ""),
            "supervisor": supervisor,
            "datetime": str(context.get("datetime") or ""),
            "app_descriptions": app_descriptions,
            "interaction": int(step_id) + 1,
            "max_interactions": max_steps,
            "skill_md": str(retrieved_memory_text or ""),
        }
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            *[dict(message) for message in APPWORLD_ACTION_FEW_SHOT_MESSAGES],
            {
                "role": "user",
                "content": (
                    "Actual AppWorld task context:\n"
                    + json.dumps(task_context, ensure_ascii=False, indent=2)
                    + "\n\nGenerate the first Python interaction."
                ),
            },
        ]
        history = [row for row in (recent_history or []) if isinstance(row, dict)]
        for row in history:
            reasoning = str(row.get("thought") or "")
            code = str(row.get("action") or "")
            assistant_content = (reasoning + "\n\n" if reasoning else "") + f"```python\n{code}\n```"
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Output:\n```\n{str(row.get('next_observation') or '')}\n```\n\n"
                        "Continue the actual task with exactly one next Python code block."
                    ),
                }
            )
        text = self.llm.chat(messages)
        return {
            "thought": _appworld_reasoning_without_code(text),
            "action": extract_appworld_code(text),
        }
