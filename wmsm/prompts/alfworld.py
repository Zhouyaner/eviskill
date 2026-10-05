from .common import (
    SKILL_JUDGE_L1_EVIDENCE_PROMPT,
    SKILL_JUDGE_L2_MARKDOWN_PROMPT,
    SKILL_MANAGER_L1_EVIDENCE_PROMPT,
    SKILL_MANAGER_L1_FAILURE_EVIDENCE_PROMPT,
    SKILL_MANAGER_L1_SUCCESS_EVIDENCE_PROMPT,
    SKILL_MANAGER_L2_MARKDOWN_PROMPT,
    SKILL_REFLECT_L2_MARKDOWN_PROMPT,
)


ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_ACTION_AGENT_PROMPT = "You are an expert agent operating in the ALFRED Embodied Environment."


ALFWORLD_SKILL_MANAGER_L1_EVIDENCE_PROMPT = SKILL_MANAGER_L1_EVIDENCE_PROMPT
ALFWORLD_SKILL_MANAGER_L1_FAILURE_EVIDENCE_PROMPT = SKILL_MANAGER_L1_FAILURE_EVIDENCE_PROMPT
ALFWORLD_SKILL_MANAGER_L1_SUCCESS_EVIDENCE_PROMPT = SKILL_MANAGER_L1_SUCCESS_EVIDENCE_PROMPT
ALFWORLD_SKILL_MANAGER_L2_MARKDOWN_PROMPT = SKILL_MANAGER_L2_MARKDOWN_PROMPT
ALFWORLD_SKILL_JUDGE_L1_EVIDENCE_PROMPT = SKILL_JUDGE_L1_EVIDENCE_PROMPT
ALFWORLD_SKILL_JUDGE_L2_MARKDOWN_PROMPT = SKILL_JUDGE_L2_MARKDOWN_PROMPT
ALFWORLD_SKILL_REFLECT_L2_MARKDOWN_PROMPT = SKILL_REFLECT_L2_MARKDOWN_PROMPT


__all__ = [
    "ALFWORLD_ACTION_AGENT_PROMPT",
    "ALFWORLD_TEMPLATE_NO_HIS",
    "ALFWORLD_TEMPLATE",
    "ALFWORLD_SKILL_MANAGER_L1_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_MANAGER_L1_FAILURE_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_MANAGER_L1_SUCCESS_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_MANAGER_L2_MARKDOWN_PROMPT",
    "ALFWORLD_SKILL_JUDGE_L1_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_JUDGE_L2_MARKDOWN_PROMPT",
    "ALFWORLD_SKILL_REFLECT_L2_MARKDOWN_PROMPT",
]
