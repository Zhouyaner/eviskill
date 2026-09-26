import contextlib
import hashlib
import json
import os
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .action_runtime import ALFWORLD_ACTION_HISTORY_LENGTH
from .agents import AlfWorldActionAgent
from .task_family import detect_alfworld_task_family, normalize_alfworld_task_family


_TEXTWORLD_GYM_REGISTRY_LOCK = threading.Lock()
_TEXTWORLD_REPLAY_ENV_LOCK = threading.RLock()


def _safe_checkpoint_part(value: Any, *, limit: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    text = text.strip("._")
    return (text or "item")[:limit]


def _atomic_save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_checkpoint(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        corrupt = path.with_name(path.name + ".corrupt")
        try:
            os.replace(path, corrupt)
        except Exception:
            pass
        print(f"[ALFWorld checkpoint] ignored corrupt checkpoint path={path} error={exc}", flush=True)
        return None


def _checkpoint_matches(
    checkpoint: Optional[Dict[str, Any]],
    *,
    task_id: str,
    gamefile: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> bool:
    if not checkpoint:
        return False
    if str(checkpoint.get("task_id") or "") != str(task_id):
        return False
    stored_gamefile = checkpoint.get("gamefile")
    if gamefile and stored_gamefile and str(stored_gamefile) != str(gamefile):
        return False
    stored_meta = checkpoint.get("metadata") or {}
    for key in ("namespace", "mode", "bundle", "task_index"):
        if metadata and key in metadata and key in stored_meta and stored_meta[key] != metadata[key]:
            return False
    return True


@contextlib.contextmanager
def _quiet_textworld_replay_output():
    if os.environ.get("WMSM_ALFWORLD_REPLAY_VERBOSE"):
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def add_alfworld_path(path: Optional[str]) -> None:
    if not path:
        return
    root = Path(path).expanduser().resolve()
    candidates = [root]
    nested_env = root / "agent_system" / "environments" / "env_package" / "alfworld"
    if nested_env.exists():
        candidates.append(nested_env)
    parts = root.parts
    if "agent_system" in parts:
        idx = parts.index("agent_system")
        candidates.append(Path(*parts[:idx]))
    if root.name == "alfworld" and root.parent.name == "env_package":
        candidates.append(root)
        candidates.append(root.parents[2])
    for cand in candidates:
        value = str(cand)
        if value and value not in sys.path:
            sys.path.insert(0, value)


def resolve_alfworld_config(config_path: str) -> str:
    path = Path(config_path).expanduser()
    if path.exists():
        return str(path.resolve())
    raise FileNotFoundError(f"ALFWorld config file does not exist: {config_path}")


def canonical_alfworld_gamefile(gamefile: str) -> str:
    """Return a data-root-independent identity for an ALFWorld gamefile."""
    path = Path(str(gamefile or "")).expanduser()
    parts = path.parts
    if "json_2.1.1" in parts:
        index = parts.index("json_2.1.1")
        return Path(*parts[index:]).as_posix()
    return path.as_posix()


def resolve_alfworld_gamefile(gamefile: str) -> str:
    """Resolve absolute or SkillOPT-style ALFWORLD_DATA-relative paths."""
    raw = str(gamefile or "").strip()
    if not raw:
        raise FileNotFoundError("ALFWorld gamefile path is empty")
    path = Path(raw).expanduser()
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        data_root = str(os.environ.get("ALFWORLD_DATA") or "").strip()
        if data_root:
            candidates.append(Path(data_root).expanduser() / path)
        candidates.append(Path.home() / ".cache" / "alfworld" / path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    searched = ", ".join(str(candidate) for candidate in candidates) or raw
    raise FileNotFoundError(
        f"ALFWorld gamefile does not exist: {raw}. Searched: {searched}. "
        "Set ALFWORLD_DATA to the directory containing json_2.1.1."
    )


def load_alfworld_config(config_path: str) -> Dict[str, Any]:
    import alfworld.agents.modules.generic as generic

    resolved = resolve_alfworld_config(config_path)
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], resolved]
        return generic.load_config()
    finally:
        sys.argv = old_argv


def make_alfworld_env(config_path: str, split: str):
    config = load_alfworld_config(config_path)
    env_type = config["env"]["type"]
    if env_type == "AlfredTWEnv":
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        env_cls = AlfredTWEnv
    elif env_type == "AlfredThorEnv":
        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
        env_cls = AlfredThorEnv
    else:
        from alfworld.agents.environment import get_environment
        env_cls = get_environment(env_type)
    env = env_cls(config, train_eval=split)
    return env.init_env(batch_size=1)


def make_alfworld_env_for_gamefile(config_path: str, gamefile: str):
    config = load_alfworld_config(config_path)
    if config["env"]["type"] != "AlfredTWEnv":
        raise ValueError("Exact ALFWorld gamefile replay is currently implemented for AlfredTWEnv only.")

    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    game_path = str(Path(gamefile).expanduser().resolve())
    if not Path(game_path).exists():
        raise FileNotFoundError(f"ALFWorld gamefile does not exist: {gamefile}")

    domain_randomization = bool(config["env"].get("domain_randomization", False))
    wrappers = [AlfredDemangler(shuffle=domain_randomization), AlfredInfos]
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    training_method = config["general"]["training_method"]
    if training_method == "dqn":
        max_episode_steps = config["rl"]["training"]["max_nb_steps_per_episode"]
    elif training_method == "dagger":
        max_episode_steps = config["dagger"]["training"]["max_nb_steps_per_episode"]
    else:
        raise NotImplementedError(f"Unsupported ALFWorld training_method={training_method}")

    with _TEXTWORLD_GYM_REGISTRY_LOCK:
        with _quiet_textworld_replay_output():
            env_id = textworld.gym.register_games(
                [game_path],
                request_infos,
                batch_size=1,
                asynchronous=True,
                max_episode_steps=max_episode_steps,
                wrappers=wrappers,
                name=f"alfworld-replay-{uuid.uuid4().hex}",
            )
            return textworld.gym.make(env_id)


def parse_task_instruction(observation: str) -> str:
    match = re.search(r"Your task is to:\s*(.*)", str(observation or ""), flags=re.I | re.S)
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return str(observation or "").strip().split("\n")[0].strip()


def normalize_task(text: str) -> str:
    value = " ".join(str(text or "").strip().lower().split())
    return value[:-1] if value.endswith(".") else value


def episode_success(done: bool, score: float) -> bool:
    return bool(done and float(score) > 0.0)


def detect_bad_step(prev_obs: str, next_obs: str, action: str) -> bool:
    nxt = str(next_obs or "").lower()
    if any(x in nxt for x in ["nothing happens", "i don't understand", "you can't", "not possible", "invalid"]):
        return True
    return str(prev_obs or "").strip() == str(next_obs or "").strip() and str(action) not in {"look", "inventory"}


class AlfWorldRunner:
    def __init__(
        self,
        action_agent: AlfWorldActionAgent,
        *,
        alfworld_path: str,
        alfworld_config: str,
        alfworld_split: str = "train",
        max_steps: int = 50,
        top_k_general_skills: int = 6,
        top_k_task_specific_skills: int = 6,
        general_skill_retrieval_threshold: float = 0.0,
        task_specific_skill_retrieval_threshold: float = 0.0,
        task_specific_retrieval_scope: str = "detected",
        max_resets_to_find_task: int = 512,
        action_history_length: int = ALFWORLD_ACTION_HISTORY_LENGTH,
        initialize_full_env: bool = True,
        parallel_workers: int = 1,
    ):
        add_alfworld_path(alfworld_path)
        self.alfworld_path = alfworld_path
        self.alfworld_config = alfworld_config
        self.initialize_full_env = bool(initialize_full_env)
        self._full_env = make_alfworld_env(alfworld_config, alfworld_split) if initialize_full_env else None
        self._single_env = None
        self._single_gamefile = None
        self.env = self._full_env
        self.action_agent = action_agent
        self.alfworld_split = alfworld_split
        self.max_steps = int(max_steps)
        self.top_k_general_skills = int(top_k_general_skills)
        self.top_k_task_specific_skills = int(top_k_task_specific_skills)
        self.general_skill_retrieval_threshold = float(general_skill_retrieval_threshold)
        self.task_specific_skill_retrieval_threshold = float(task_specific_skill_retrieval_threshold)
        self.task_specific_retrieval_scope = task_specific_retrieval_scope
        self.max_resets_to_find_task = int(max_resets_to_find_task)
        self.action_history_length = max(0, int(action_history_length))
        self.parallel_workers = max(1, int(parallel_workers or 1))

    def _new_parallel_worker(self):
        return AlfWorldRunner(
            self.action_agent,
            alfworld_path=self.alfworld_path,
            alfworld_config=self.alfworld_config,
            alfworld_split=self.alfworld_split,
            max_steps=self.max_steps,
            top_k_general_skills=self.top_k_general_skills,
            top_k_task_specific_skills=self.top_k_task_specific_skills,
            general_skill_retrieval_threshold=self.general_skill_retrieval_threshold,
            task_specific_skill_retrieval_threshold=self.task_specific_skill_retrieval_threshold,
            task_specific_retrieval_scope=self.task_specific_retrieval_scope,
            max_resets_to_find_task=self.max_resets_to_find_task,
            action_history_length=self.action_history_length,
            initialize_full_env=False,
            parallel_workers=1,
        )

    def _reset(self):
        obs, info = self.env.reset()
        return obs[0], info

    def _step(self, action: str):
        obs, scores, dones, infos = self.env.step([action])
        return obs[0], float(scores[0]), bool(dones[0]), infos

    def _admissible(self, info: Dict[str, Any]) -> List[str]:
        actions = (info or {}).get("admissible_commands")
        if isinstance(actions, list) and actions and isinstance(actions[0], list):
            return [str(x) for x in actions[0]]
        if isinstance(actions, list):
            return [str(x) for x in actions]
        return []

    def _task_from_id(self, task_id: Any) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        if isinstance(task_id, dict):
            task = task_id.get("task") or task_id.get("instruction")
            gamefile = task_id.get("gamefile")
            task_family = normalize_alfworld_task_family(
                task_id.get("task_family") or task_id.get("family") or task_id.get("task_type")
            )
            return (
                str(task_id.get("task_id") or task_id.get("id") or task or ""),
                str(task) if task else None,
                str(gamefile) if gamefile else None,
                task_family,
            )
        raw = str(task_id)
        target = raw if any(x in raw.lower() for x in ["put", "clean", "heat", "cool", "look", "examine", "find"]) else None
        return raw, target, None, None

    def _reset_to_gamefile(self, gamefile: str):
        game_path = resolve_alfworld_gamefile(gamefile)
        if self._single_gamefile != game_path:
            if self._single_env is not None:
                try:
                    self._single_env.close()
                except Exception:
                    pass
            self._single_env = make_alfworld_env_for_gamefile(self.alfworld_config, game_path)
            self._single_gamefile = game_path
        self.env = self._single_env
        with _TEXTWORLD_GYM_REGISTRY_LOCK:
            with _quiet_textworld_replay_output():
                obs, info = self._reset()
        return obs, info, parse_task_instruction(obs)

    def _reset_to_task(self, target_task: Optional[str], gamefile: Optional[str] = None):
        if gamefile:
            return self._reset_to_gamefile(gamefile)
        if self._full_env is None:
            raise RuntimeError(
                "Replay-only AlfWorldRunner requires an exact gamefile. "
                "Pass initialize_full_env=True for reset-based task search."
            )
        self.env = self._full_env
        if not target_task:
            obs, info = self._reset()
            return obs, info, parse_task_instruction(obs)
        target_norm = normalize_task(target_task)
        for trial in range(1, self.max_resets_to_find_task + 1):
            obs, info = self._reset()
            task = parse_task_instruction(obs)
            if normalize_task(task) == target_norm:
                return obs, info, task
            if trial == 1 or trial % 100 == 0:
                print(
                    f"[ALFWorld replay search] trial={trial}/{self.max_resets_to_find_task} "
                    f"target={target_task} current={task}",
                    flush=True,
                )
        raise RuntimeError(f"Cannot replay ALFWorld task after resets: {target_task}")

    def run_task(
        self,
        task_id,
        skill_bank,
        world_knowledge=None,
        force_skill_ids=None,
        checkpoint_path: Optional[str] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
    ):
        tid, target_task, gamefile, task_family = self._task_from_id(task_id)
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
        checkpoint_metadata = dict(checkpoint_metadata or {})
        checkpoint = _load_checkpoint(checkpoint_file)
        if _checkpoint_matches(
            checkpoint,
            task_id=str(tid),
            gamefile=gamefile,
            metadata=checkpoint_metadata,
        ) and checkpoint.get("status") == "complete" and isinstance(checkpoint.get("result"), dict):
            result = checkpoint["result"]
            print(
                f"[ALFWorld checkpoint] reuse complete task_id={tid} "
                f"steps={result.get('steps')} path={checkpoint_file}",
                flush=True,
            )
            return result

        with _TEXTWORLD_REPLAY_ENV_LOCK:
            obs, info, instruction = self._reset_to_task(target_task, gamefile=gamefile)
        task_family = task_family or detect_alfworld_task_family(instruction)
        cur = obs
        traj = []
        final_score, done = 0.0, False
        retrieved_memory = skill_bank.retrieve(
            instruction,
            task_type_override=task_family,
            top_k_general=self.top_k_general_skills,
            top_k_task_specific=self.top_k_task_specific_skills,
            min_score_general=self.general_skill_retrieval_threshold,
            min_score_task_specific=self.task_specific_skill_retrieval_threshold,
            task_specific_scope=self.task_specific_retrieval_scope,
            force_skill_ids=force_skill_ids,
        )
        retrieved_memory_text = skill_bank.format_for_prompt(retrieved_memory)
        retrieved_skill_ids = skill_bank.retrieved_skill_ids(retrieved_memory)

        def save_checkpoint(status: str, result: Optional[Dict[str, Any]] = None) -> None:
            if checkpoint_file is None:
                return
            payload = {
                "schema_version": 1,
                "dataset": "alfworld",
                "status": status,
                "metadata": checkpoint_metadata,
                "task_id": str(tid),
                "gamefile": gamefile,
                "instruction": instruction,
                "task_family": task_family,
                "max_steps": self.max_steps,
                "next_step": len(traj),
                "final_score": final_score,
                "done": done,
                "retrieved_skill_ids": retrieved_skill_ids,
                "trajectory": traj,
            }
            if result is not None:
                payload["result"] = result
            _atomic_save_json(checkpoint_file, payload)

        if _checkpoint_matches(
            checkpoint,
            task_id=str(tid),
            gamefile=gamefile,
            metadata=checkpoint_metadata,
        ) and checkpoint.get("status") == "in_progress":
            prefix = [
                row for row in checkpoint.get("trajectory", [])
                if isinstance(row, dict)
            ][: self.max_steps]
            if prefix:
                print(
                    f"[ALFWorld checkpoint] resume task_id={tid} "
                    f"replay_steps={len(prefix)} path={checkpoint_file}",
                    flush=True,
                )
            for raw in prefix:
                if done:
                    break
                action = str(raw.get("action", ""))
                with _TEXTWORLD_REPLAY_ENV_LOCK:
                    next_obs, score, done, info = self._step(action)
                final_score = float(score)
                row = {
                    "step": len(traj),
                    "task": instruction,
                    "observation": cur,
                    "thought": str(raw.get("thought", "")),
                    "action": action,
                    "next_observation": next_obs,
                    "score": final_score,
                    "reward": final_score,
                    "done": done,
                    "bad_step": detect_bad_step(cur, next_obs, action),
                }
                traj.append(row)
                cur = next_obs

        for step in range(len(traj), self.max_steps):
            if done:
                break
            admissible = self._admissible(info)
            context = {"observation": cur, "admissible_actions": admissible}
            decision = self.action_agent.act(
                instruction,
                context,
                world_knowledge or {},
                retrieved_memory_text,
                traj[-self.action_history_length:] if self.action_history_length else [],
                step,
            )
            action = decision.get("action") or (admissible[0] if admissible else "look")
            if admissible and action not in admissible:
                action = admissible[0]
            # TextWorld's Tatsu parser is process-global and is not thread-safe.
            # Keep LLM calls concurrent, but serialize environment transitions.
            with _TEXTWORLD_REPLAY_ENV_LOCK:
                next_obs, score, done, info = self._step(action)
            final_score = float(score)
            traj.append(
                {
                    "step": step,
                    "task": instruction,
                    "observation": cur,
                    "thought": decision.get("thought", ""),
                    "action": action,
                    "next_observation": next_obs,
                    "score": final_score,
                    "reward": final_score,
                    "done": done,
                    "bad_step": detect_bad_step(cur, next_obs, action),
                }
            )
            cur = next_obs
            save_checkpoint("in_progress")
            if done:
                break

        result = {
            "dataset": "alfworld",
            "task_id": str(tid),
            "gamefile": gamefile,
            "instruction": instruction,
            "task_family": task_family,
            "success": episode_success(done, final_score),
            "score": final_score,
            "steps": len(traj),
            "trajectory": traj,
            "retrieved_memory": retrieved_memory,
            "retrieved_memory_text": retrieved_memory_text,
            "retrieved_skill_ids": retrieved_skill_ids,
        }
        save_checkpoint("complete", result=result)
        return result

    def run_task_set(
        self,
        task_ids,
        skill_bank,
        world_knowledge=None,
        desc=None,
        force_skill_ids=None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_namespace: Optional[str] = None,
    ):
        task_ids = list(task_ids or [])
        total = len(task_ids)

        def run_one(worker, idx: int, tid: Any):
            checkpoint_path = None
            checkpoint_metadata = None
            if checkpoint_dir:
                parsed_tid, _, _, _ = self._task_from_id(tid)
                namespace = _safe_checkpoint_part(checkpoint_namespace or desc or "run", limit=120)
                digest = hashlib.sha1(str(tid).encode("utf-8")).hexdigest()[:10]
                name = (
                    f"{namespace}__task_{idx - 1:03d}__"
                    f"{_safe_checkpoint_part(parsed_tid, limit=60)}__{digest}.json"
                )
                checkpoint_path = str(Path(checkpoint_dir) / name)
                checkpoint_metadata = {
                    "namespace": checkpoint_namespace or desc or "run",
                    "task_index": idx - 1,
                    "desc": desc,
                }
            result = worker.run_task(
                tid,
                skill_bank,
                world_knowledge,
                force_skill_ids=force_skill_ids,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata=checkpoint_metadata,
            )
            if desc:
                print(
                    f"[{desc}] task {idx}/{total} id={result.get('task_id')} "
                    f"family={result.get('task_family')} score={result.get('score')} "
                    f"steps={result.get('steps')} skills={result.get('retrieved_skill_ids') or []}",
                    flush=True,
                )
            return result

        can_parallelize = all(self._task_from_id(tid)[2] for tid in task_ids)
        workers = min(self.parallel_workers, total) if total else 1
        if workers <= 1 or not can_parallelize:
            if workers > 1 and not can_parallelize:
                print(
                    "[ALFWorld runner] exact gamefile is missing for at least one task; "
                    "falling back to serial rollout.",
                    flush=True,
                )
            results = [run_one(self, idx, tid) for idx, tid in enumerate(task_ids, start=1)]
        else:
            print(f"[ALFWorld runner] parallel workers={workers} tasks={total}", flush=True)
            ordered_results: List[Optional[Dict[str, Any]]] = [None] * total
            partitions: List[List[Tuple[int, Any]]] = [[] for _ in range(workers)]
            for idx, tid in enumerate(task_ids, start=1):
                partitions[(idx - 1) % workers].append((idx, tid))

            def run_partition(assigned: List[Tuple[int, Any]]) -> None:
                worker = self._new_parallel_worker()
                try:
                    for idx, tid in assigned:
                        ordered_results[idx - 1] = run_one(worker, idx, tid)
                finally:
                    worker.close()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_partition, partition) for partition in partitions]
                for future in as_completed(futures):
                    future.result()
            results = [row for row in ordered_results if row is not None]
            if len(results) != total:
                raise RuntimeError(
                    f"ALFWorld parallel rollout lost results: got {len(results)} expected {total}"
                )
        if desc:
            score_avg = sum(float(x.get("score", 0.0) or 0.0) for x in results) / total if total else 0.0
            steps_avg = sum(int(x.get("steps", 0) or 0) for x in results) / total if total else 0.0
            success = sum(1 for x in results if x.get("success"))
            success_rate = success / total if total else 0.0
            print(
                f"[{desc}] summary score_avg={score_avg:.4f} "
                f"success_rate={success_rate:.4f} success={success}/{total} steps_avg={steps_avg:.2f}",
                flush=True,
            )
        return results

    def close(self) -> None:
        environments = []
        if self._single_env is not None:
            environments.append(self._single_env)
        if self._full_env is not None and self._full_env is not self._single_env:
            environments.append(self._full_env)
        for env in environments:
            try:
                env.close()
            except Exception:
                pass
        self._single_env = None
        self._full_env = None
        self.env = None

    def run_from_trace_range(
        self,
        original_result,
        *,
        start_step,
        continue_steps,
        skill_bank,
        world_knowledge=None,
        force_skill_ids=None,
    ):
        """Replay original actions up to start_step, then continue with skill_bank."""
        task_ref = {
            "task_id": original_result.get("task_id"),
            "task": original_result.get("instruction"),
            "gamefile": original_result.get("gamefile"),
            "task_family": original_result.get("task_family"),
        }
        tid, target_task, gamefile, task_family = self._task_from_id(task_ref)
        obs, info, instruction = self._reset_to_task(target_task, gamefile=gamefile)
        task_family = task_family or detect_alfworld_task_family(instruction)
        prefix = [
            x for x in original_result.get("trajectory", [])
            if int(x.get("step", 0) or 0) < int(start_step)
        ]
        replay_traj = []
        cur = obs
        done = False
        final_score = 0.0
        for raw in prefix:
            action = str(raw.get("action", ""))
            with _TEXTWORLD_REPLAY_ENV_LOCK:
                next_obs, score, done, info = self._step(action)
            final_score = float(score)
            replay_traj.append(
                {
                    "step": int(raw.get("step", len(replay_traj)) or 0),
                    "task": instruction,
                    "observation": cur,
                    "action": action,
                    "next_observation": next_obs,
                    "score": final_score,
                    "reward": final_score,
                    "done": done,
                    "bad_step": detect_bad_step(cur, next_obs, action),
                }
            )
            cur = next_obs
            if done:
                break

        retrieved_memory = skill_bank.retrieve(
            instruction,
            task_type_override=task_family,
            top_k_general=self.top_k_general_skills,
            top_k_task_specific=self.top_k_task_specific_skills,
            min_score_general=self.general_skill_retrieval_threshold,
            min_score_task_specific=self.task_specific_skill_retrieval_threshold,
            task_specific_scope=self.task_specific_retrieval_scope,
            force_skill_ids=force_skill_ids,
        )
        retrieved_memory_text = skill_bank.format_for_prompt(retrieved_memory)
        retrieved_skill_ids = skill_bank.retrieved_skill_ids(retrieved_memory)
        after_segments = []
        max_continue = max(0, int(continue_steps))
        for offset in range(max_continue):
            if done:
                break
            step = int(start_step) + offset
            admissible = self._admissible(info)
            context = {"observation": cur, "admissible_actions": admissible}
            decision = self.action_agent.act(
                instruction,
                context,
                world_knowledge or {},
                retrieved_memory_text,
                replay_traj[-self.action_history_length:] if self.action_history_length else [],
                step,
            )
            action = decision.get("action") or (admissible[0] if admissible else "look")
            if admissible and action not in admissible:
                action = admissible[0]
            with _TEXTWORLD_REPLAY_ENV_LOCK:
                next_obs, score, done, info = self._step(action)
            final_score = float(score)
            row = {
                "step": step,
                "task": instruction,
                "observation": cur,
                "thought": decision.get("thought", ""),
                "action": action,
                "next_observation": next_obs,
                "score": final_score,
                "reward": final_score,
                "done": done,
                "bad_step": detect_bad_step(cur, next_obs, action),
            }
            replay_traj.append(row)
            after_segments.append(row)
            cur = next_obs
        return {
            "dataset": "alfworld",
            "task_id": str(tid),
            "instruction": instruction,
            "task_family": task_family,
            "prefix_actions": [x.get("action", "") for x in prefix],
            "after_segments": [after_segments],
            "retrieved_skill_ids": retrieved_skill_ids,
            "retrieved_memory": retrieved_memory,
            "score": final_score,
            "done": done,
        }
