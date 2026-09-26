from pathlib import Path
from typing import Any, Dict, List, Optional


class MarkdownSkillBank:
    """Adapter that injects one complete Markdown skill document into ActionAgent."""

    def __init__(self, markdown_path: str, *, skill_id: str = "skill_md"):
        self.markdown_path = str(markdown_path)
        self.skill_id = str(skill_id or "skill_md")
        self.markdown_text = Path(markdown_path).read_text(encoding="utf-8")
        self.metadata: Dict[str, Any] = {"bank_version": 0, "skill_md_path": self.markdown_path}

    @property
    def bank_version(self) -> int:
        return 0

    def retrieve(self, query: str, **_: Any) -> Dict[str, Any]:
        return {
            "task_type": None,
            "general_skills": [
                {
                    "skill_id": self.skill_id,
                    "title": "Markdown Skill Document",
                    "principle": "Use the full Markdown skill document.",
                    "when_to_apply": "Every task.",
                    "section": "general_skills",
                    "task_type": None,
                }
            ],
            "task_specific_skills": [],
            "markdown_path": self.markdown_path,
        }

    def format_for_prompt(self, retrieved_memory: Dict[str, Any]) -> str:
        return self.markdown_text

    def retrieved_skill_ids(self, retrieved_memory: Dict[str, Any]) -> List[str]:
        return [self.skill_id]

    def save(self, path: Optional[str] = None) -> None:
        return None
