# AppWorld Skill

## Operating Loop

1. Translate the request into a concrete final state or direct answer. Identify
   the apps involved and avoid unrelated reads or mutations.
2. Inspect the available interfaces before acting:
   - Use `apis.api_docs.show_app_descriptions()` when the relevant app is unclear.
   - Use `apis.api_docs.show_api_descriptions(app_name="APP")` to find a suitable API.
   - Use `apis.api_docs.show_api_doc(app_name="APP", api_name="API")` to confirm
     exact parameters and the response schema before calling an operational API.
3. Resolve real inputs through documented APIs. Retrieve account information or
   credentials from the supervisor app and look up entity IDs instead of guessing
   values, parameter names, or response fields.
4. Execute in small steps and inspect every response before using it. Keep useful
   values in Python variables because the REPL persists across interactions.
5. For paginated APIs, traverse `page_index` until the response shows that no
   further results remain, then compute from the complete collection.
6. Before a mutation, confirm the target and prerequisites. After it, read the
   relevant state again to verify that the requested change took effect.
7. Finish only after establishing the requested result. Call
   `apis.supervisor.complete_task(answer=value)` for a direct answer, or
   `apis.supervisor.complete_task()` for a state-changing task. Use failure status
   only when the task is genuinely impossible.

## Guardrails

- Interact with connected apps only through `apis`; a file-system request refers
  to the file-system app, not operating-system files or processes.
- Never invent IDs, credentials, private values, API names, or task answers.
- A successful API call is not sufficient evidence of task completion; verify the
  requested answer or final state explicitly.
