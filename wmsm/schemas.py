from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class TrajectoryStep:
    step: int
    observation: str
    action: str
    reward: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            step=int(data.get("step", 0) or 0),
            observation=str(data.get("observation", "")),
            action=str(data.get("action", "")),
            reward=float(data.get("score_delta", data.get("reward", data.get("score", 0.0))) or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    task_id: str
    instruction: str
    trajectory: List[TrajectoryStep]
    final_reward: float
    task_family: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRecord":
        return cls(
            task_id=str(data.get("task_id", "")),
            instruction=str(data.get("instruction", "")),
            trajectory=[TrajectoryStep.from_dict(x) for x in data.get("trajectory", [])],
            final_reward=float(
                data.get("final_reward", data.get("observed_reward", data.get("reward_before", 0.0))) or 0.0
            ),
            task_family=str(data.get("task_family", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["trajectory"] = [step.to_dict() for step in self.trajectory]
        return data
