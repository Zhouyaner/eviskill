import re
from typing import Any, Optional


TASK_TYPE_ALIASES = {
    "pick_and_place": "pick_and_place",
    "pick_and_place_simple": "pick_and_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
    "pick_two": "pick_two_obj_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "look_at": "look_at_obj_in_light",
    "clean": "clean",
    "clean_and_place": "clean",
    "pick_clean_then_place_in_recep": "clean",
    "heat": "heat",
    "heat_and_place": "heat",
    "pick_heat_then_place_in_recep": "heat",
    "cool": "cool",
    "cool_and_place": "cool",
    "pick_cool_then_place_in_recep": "cool",
    "examine": "examine",
}


def normalize_task_type(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return TASK_TYPE_ALIASES.get(text)


def normalize_scienceworld_task_family(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"^task[_ -]*\d+[_ -]*", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or None


def normalize_task_type_for_dataset(value: Any, dataset: Any = None) -> Optional[str]:
    key = str(dataset or "").lower()
    if key in {"scienceworld", "science_world"}:
        return normalize_scienceworld_task_family(value)
    if key == "alfworld":
        return normalize_alfworld_task_family(value)
    return normalize_task_type(value) or normalize_scienceworld_task_family(value)


def normalize_alfworld_task_family(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return TASK_TYPE_ALIASES.get(text)


def detect_alfworld_task_family(task_description: str) -> str:
    text = str(task_description or "").lower()
    if "desklamp" in text or "under the lamp" in text or "look at" in text:
        return "look_at_obj_in_light"
    if "clean" in text or "wash" in text:
        return "clean"
    if "heat" in text or "warm" in text or "microwave" in text:
        return "heat"
    if "cool" in text or "chill" in text or "fridge" in text or "refrigerator" in text:
        return "cool"
    if "examine" in text or "inspect" in text:
        return "examine"
    return "pick_and_place"
