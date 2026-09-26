# ScienceWorld Skill

## Overview
This skill guides agents in ScienceWorld, a text-based science environment.
The agent should solve tasks by exploring, collecting useful objects, using
tools, running simple experiments, and verifying results.

**Output format**: Always output `<think>...</think>` for reasoning, then
`<action>...</action>` for the chosen action.

---

## Core Principles

1. **Parse the goal first**: Identify the target object, property, measurement,
   comparison, or final state requested by the task.
2. **Use admissible actions only**: Always choose an action from the provided
   admissible action list.
3. **Inspect before acting**: Look around, examine relevant objects, and open
   containers before deciding what to take or use.
4. **Collect necessary tools**: Pick up only objects needed for the task, such
   as containers, thermometers, wires, batteries, magnets, scales, seeds, or
   target materials.
5. **Run controlled experiments**: Change one variable at a time, observe the
   result, and compare it with the task goal.
6. **Measure when possible**: Use instruments instead of guessing temperature,
   mass, volume, length, conductivity, or other properties.
7. **Observe after changes**: After heating, cooling, mixing, planting,
   connecting, moving, or waiting, check the new state before continuing.
8. **Avoid loops**: Do not repeat actions that do not change the observation.
9. **Verify before finishing**: Only answer, submit, or perform the final
   placement after confirming the goal condition is satisfied.

---

## Common Task Patterns

- **Classification**: Examine each candidate and select the one whose observed
  features match the requested category.
- **Heating/cooling/state change**: Put the object in a suitable container, use
  the correct heat or cold source, wait if needed, then observe or measure.
- **Mixing/solutions**: Use a clean container, add substances one at a time,
  stir/shake if available, and observe the result.
- **Electricity/conductivity**: Build a closed circuit with battery, wires, and
  device; insert the test material and observe whether the device works.
- **Plants/life cycles**: Use the correct organism or stage, provide required
  resources, and keep species or conditions consistent.
- **Forces/motion/comparison**: Use the same object across trials and change
  only the requested surface, slope, force, or condition.

---

## Mistakes to Avoid

- Guessing when the environment provides a way to test.
- Mixing materials unintentionally or using the wrong container.
- Changing several experimental conditions at once.
- Forgetting to wait or observe after an intervention.
- Giving a final answer before checking the evidence.
