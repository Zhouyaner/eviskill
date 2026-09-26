import re
from pathlib import Path
from typing import Any


def _compact_slug(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    if "/" in text or "\\" in text:
        text = Path(text).name
    text = text.lower()
    text = text.replace(".", "")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    text = re.sub(r"^gpt_", "gpt", text)
    return text or fallback


def _model_slug(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    if "/" in text or "\\" in text:
        text = Path(text).name
    text = text.lower()
    text = re.sub(r"[^a-z0-9.]+", "_", text).strip("_.")
    text = re.sub(r"\.{2,}", ".", text)
    return text or fallback


def skill_source_slug(path: Any) -> str:
    text = str(path or "").strip()
    lower = text.lower()
    stem = Path(text).stem if text else "skill"
    if "skillopt" in lower and stem.lower() == "initial":
        return "skillopt_initial"
    return _compact_slug(stem, fallback="skill")


def default_dual_layer_output_dir(args: Any) -> Path:
    dataset = _compact_slug(getattr(args, "dataset", None), fallback="dataset")
    action_model = getattr(args, "action_model", None)
    if not action_model and bool(getattr(args, "dry_run_action", False)):
        action_model = "dry_run_action"
    action = _model_slug(action_model, fallback="action")
    name = f"{dataset}_{action}"
    tag = str(getattr(args, "experiment_tag", "") or "").strip()
    if tag:
        name += f"_{_compact_slug(tag, fallback='tag')}"
    return Path("outputs") / name
