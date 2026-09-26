from typing import Any, Dict, Iterable, List

from .schemas import TaskRecord, TrajectoryStep


def compact_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if max_chars is None or int(max_chars) < 0:
        return text
    return text[:max_chars]


def select_trajectory_steps(
    trajectory: Iterable[Dict[str, Any]],
    *,
    mode: str = "prefix",
    max_steps: int = 24,
    head_steps: int = 1,
    tail_steps: int = 8,
) -> List[Dict[str, Any]]:
    rows = list(trajectory or [])
    mode = str(mode or "prefix")
    if mode == "full":
        return rows
    if mode == "prefix":
        return rows[: max(0, int(max_steps))]
    if mode == "tail":
        tail = max(0, int(tail_steps))
        return rows[-tail:] if tail else []
    if mode == "head_tail":
        head = max(0, int(head_steps))
        tail = max(0, int(tail_steps))
        selected: List[Dict[str, Any]] = []
        seen = set()
        for raw in rows[:head] + (rows[-tail:] if tail else []):
            step_key = raw.get("step", len(selected)) if isinstance(raw, dict) else len(selected)
            key = str(step_key)
            if key in seen:
                continue
            seen.add(key)
            selected.append(raw)
        return selected
    raise ValueError(f"Unknown trajectory selection mode: {mode}")


def normalize_task_result(
    result: Dict[str, Any],
    *,
    max_steps: int = 24,
    max_observation_chars: int = 1500,
    trajectory_mode: str = "prefix",
    trajectory_head_steps: int = 1,
    trajectory_tail_steps: int = 8,
    action_only: bool = False,
) -> TaskRecord:
    steps: List[TrajectoryStep] = []
    for raw in select_trajectory_steps(
        result.get("trajectory") or [],
        mode=trajectory_mode,
        max_steps=max_steps,
        head_steps=trajectory_head_steps,
        tail_steps=trajectory_tail_steps,
    ):
        steps.append(
            TrajectoryStep(
                step=int(raw.get("step", len(steps)) or 0),
                observation="" if action_only else compact_text(raw.get("observation", ""), max_observation_chars),
                action=str(raw.get("action", "")),
                reward=float(raw.get("score_delta", raw.get("reward", raw.get("score", 0.0))) or 0.0),
            )
        )

    return TaskRecord(
        task_id=str(result.get("task_id", "")),
        instruction=str(result.get("instruction", "")),
        trajectory=steps,
        final_reward=float(result.get("final_reward", result.get("score", 0.0)) or 0.0),
        task_family=str(result.get("task_family", "")),
    )


def normalize_task_results(results: Iterable[Dict[str, Any]], **kwargs: Any) -> List[TaskRecord]:
    return [normalize_task_result(result, **kwargs) for result in results]


def task_records_payload(records: Iterable[TaskRecord], *, include_rewards: bool = False) -> Dict[str, Any]:
    rows = []
    for record in records:
        data = record.to_dict()
        if not include_rewards:
            data.pop("final_reward", None)
            for step in data.get("trajectory", []):
                step.pop("reward", None)
        rows.append(data)
    return {"task_records": rows}


def task_records_payload_from_results(results: Iterable[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    return task_records_payload(normalize_task_results(results or [], **kwargs))


def old_task_record_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    record = normalize_task_result(result, max_steps=24, max_observation_chars=1500)
    data = record.to_dict()
    data["observed_reward"] = data.pop("final_reward", record.final_reward)
    return data
