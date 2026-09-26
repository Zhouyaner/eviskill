from .common import SKILL_JUDGE_L2_MARKDOWN_PROMPT, SKILL_REFLECT_L2_MARKDOWN_PROMPT


# Main experiments use the original SkillRL ALFWorld action templates
# (the ``oldprompt`` variant used by the 2026-08-27 runs).
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


ALFWORLD_SKILL_MANAGER_L1_EVIDENCE_PROMPT = """You analyze a minibatch of ALFWorld train trajectories and propose concrete edits to the current skill.md.

Input contains the full skill, source_type, max_proposed_edits, and compact trajectories. Each trajectory shows the household task, task family, final outcome, initial observation, every action, and the environment feedback after that action. It may also contain score transitions. Read every trajectory before proposing edits.

ALFWorld tasks include pick-and-place, placing two instances, examining an object in light, and cleaning, heating, or cooling an object before placement. Diagnose the visible embodied mechanism before editing:
- navigation_loop: already visited receptacles or locations are revisited without a new reason.
- search_coverage_gap: searchable surfaces, numbered receptacles, or closed containers remain unchecked while search repeats elsewhere.
- missed_object: the exact required visible object is not picked up when reachable.
- wrong_object: a similar but incorrect object class or instance is treated as satisfying the task.
- inventory_error: the agent loses track of what it carries or attempts an action without its inventory precondition.
- wrong_sequence: acquire, transform, open-destination, deliver, or verify subgoals occur in the wrong order.
- appliance_error: cleaning, heating, cooling, or lighting uses the wrong appliance, skips a required transform, or adds unsupported appliance steps.
- placement_error: the destination is not prepared or the exact admissible placement action is not used.
- count_tracking_error: a pick-two task stops after one instance or searches broadly despite a known remaining instance.
- premature_stop: the agent stops or reaches the step limit before all goal conditions are complete.
- invalid_action: the action is unavailable, malformed, or feedback shows that it did not execute.

For successful trajectories, use a positive mechanism such as systematic_search, immediate_pickup, inventory_control, transform_then_place, two_object_completion, destination_preparation, or light_examination. Put the most specific mechanism at the start of pattern, for example `appliance_error: the object was placed without being heated` or `transform_then_place: the held object was cleaned before delivery`. failure_type remains the shared EvidenceCard label: use rule_missing when skill.md lacks the procedure, rule_wrong when skill.md is contradicted or over-broad, rule_ignored when the useful rule exists but the trajectory violates it, and otherwise use invalid_action, exploration_gap, experiment_error, premature_finish, loop, or other as appropriate.

Return only valid JSON:
{
  "evidence": [
    {
      "experience_type": "failure_reflection | success_experience",
      "failure_type": "rule_missing | rule_wrong | rule_ignored | invalid_action | exploration_gap | experiment_error | premature_finish | loop | other | null",
      "pattern": "alfworld_mechanism: short reusable problem or success mechanism",
      "proposed_edit": {
        "op": "append | insert_after | insert_after_section | replace | delete",
        "target": "exact target text or heading; omit or null for append",
        "content": "markdown content; omit or empty for delete"
      },
      "trigger_ranges": [
        {"task_id": "task id from the input", "start": 0, "end": 0}
      ]
    }
  ]
}

Rules:
- Return at most max_proposed_edits entries; an empty evidence list is valid.
- trigger_ranges must be non-empty and use visible task IDs and step numbers. Use the smallest ranges that establish the search, inventory, sequence, appliance, placement, count, or stopping mechanism; one entry may cite several tasks only when they support the same edit.
- Parse the exact object class, required count, transformation, and destination from the task. Similar objects do not substitute unless visible environment semantics prove equivalence.
- Track which surfaces and containers were actually checked, whether a closed container was opened, what the agent carries, and which subgoals are visibly complete. Repeated zero score alone does not prove a loop or failure because ALFWorld often gives no score until task completion.
- For clean, heat, cool, light, and placement procedures, infer action order and syntax only from visible admissible actions and feedback. Do not invent open, toggle, or placement steps that the observed environment did not require.
- Proposed edits must be executable and state-aware. Never memorize random object locations, receptacle numbers, room layouts, or a task-specific route.
- Success evidence may preserve exact observed action forms as examples of a general precondition or procedure, but not as universal commands when availability varies.
- rule_wrong should replace, delete, soften, or generalize contradicted guidance rather than append a competing rule.
- Do not output supporting_tasks, before_segments, trigger_progress, source_type, support_count, merge_level, evidence IDs, or commentary outside JSON. The system materializes those fields from real train trajectories.
"""


ALFWORLD_SKILL_MANAGER_L1_FAILURE_EVIDENCE_PROMPT = ALFWORLD_SKILL_MANAGER_L1_EVIDENCE_PROMPT + """

Analyze failed ALFWorld trajectories only. Set experience_type to failure_reflection and include a non-null shared failure_type. Identify the causal navigation, search coverage, object, inventory, sequence, appliance, placement, count, stopping, or invalid-action mechanism from action and post-action feedback. Prefer common mechanisms across the minibatch, while allowing a narrow edit for a directly observed contradiction in skill.md.
"""


ALFWORLD_SKILL_MANAGER_L1_SUCCESS_EVIDENCE_PROMPT = ALFWORLD_SKILL_MANAGER_L1_EVIDENCE_PROMPT + """

Analyze successful ALFWorld trajectories only. Set experience_type to success_experience and omit failure_type. Extract reusable operational procedures not already clear in skill.md: systematic search, immediate acquisition of the exact goal object, inventory-aware execution, transform-before-place ordering, remembering remaining instances in pick-two tasks, opening a closed destination when required, and the observed look-at-light completion sequence. Keep environment-dependent action syntax conditional on visibility or admissibility.
"""


ALFWORLD_SKILL_JUDGE_L1_EVIDENCE_PROMPT = """You verify whether one system-materialized EvidenceCard for ALFWorld is factually supported by its real train trajectory ranges.

The card contains source_type, experience_type, optional shared failure_type, an ALFWorld mechanism in pattern, proposed_edit, trigger_ranges, supporting_tasks, and before_segments generated by the system from train rollout data.

Return only valid JSON:
{
  "decision": "keep | drop",
  "reason": "brief reason grounded in before_segments, task outcome, and current skill.md"
}

Check the claimed mechanism against the exact household goal, observations, actions, post-action feedback, inventory clues, visited receptacles, object count, transformation state, and final outcome. Keep only when the visible range supports the proposed state-aware procedure. Zero score before terminal success is normal and cannot by itself establish a loop, missed object, or bad sequence.

Drop invalid ranges, malformed edits, unsupported state claims, wrong shared failure_type labels, memorized object locations or receptacle numbers, invented appliance semantics, or unjustified global rules from one layout. Do not drop merely because skill.md already covers the idea, another card is redundant, priority is low, or support is narrow; coverage, deduplication, and scope handling happen later. Keep the reason concise.
"""


ALFWORLD_SKILL_JUDGE_L2_MARKDOWN_PROMPT = SKILL_JUDGE_L2_MARKDOWN_PROMPT + """

ALFWorld attribution requirements:
- Jointly track the exact object class and count, searched surfaces/containers, closed-container state, inventory, transformation, destination preparation, placement, and completion feedback.
- Intermediate and repeated zero score is normal before terminal completion. It cannot by itself establish rule_ignored, rule_missing, rule_wrong, a loop, or regression.
- rule_ignored requires an explicit candidate rule whose precondition is visibly reached and an action that clearly violates it, such as placing before a required visible transformation or stopping after one instance in a pick-two task. A short replay that has not completed the household goal is insufficient.
- rule_missing covers an omitted search-coverage condition, inventory precondition, action sequence, appliance step, count invariant, destination preparation, exact observed action form, or completion check supported by linked evidence.
- rule_wrong covers a contradicted object assumption, action order, appliance procedure, placement rule, or action syntax, including promoting a random object location, receptacle number, or room layout into a reusable rule.
- Treat admissible actions and post-action feedback as evidence that an action actually executed. Do not infer appliance or container semantics absent from the linked trajectories.
"""

ALFWORLD_SKILL_REFLECT_L2_MARKDOWN_PROMPT = SKILL_REFLECT_L2_MARKDOWN_PROMPT + """

For ALFWorld, repair only the diagnosed search, object, inventory, sequence, appliance, placement, count, destination, or completion procedure. Keep action syntax conditional on observed availability or admissibility. Do not invent appliance/container behavior or memorize random object locations, receptacle numbers, room layouts, and task-specific routes.
"""


__all__ = [
    "ALFWORLD_ACTION_AGENT_PROMPT",
    "ALFWORLD_TEMPLATE_NO_HIS",
    "ALFWORLD_TEMPLATE",
    "ALFWORLD_SKILL_MANAGER_L1_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_MANAGER_L1_FAILURE_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_MANAGER_L1_SUCCESS_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_JUDGE_L1_EVIDENCE_PROMPT",
    "ALFWORLD_SKILL_JUDGE_L2_MARKDOWN_PROMPT",
    "ALFWORLD_SKILL_REFLECT_L2_MARKDOWN_PROMPT",
]
