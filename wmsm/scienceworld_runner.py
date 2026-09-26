import atexit
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .action_runtime import SCIENCEWORLD_ACTION_HISTORY_LENGTH
from .agents import ScienceWorldActionAgent
from .task_family import normalize_scienceworld_task_family


def _clean_task_family(value: Any, fallback: Any = None) -> str:
    raw = str(value or "").strip()
    if raw:
        return raw
    return normalize_scienceworld_task_family(fallback) or "scienceworld"


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _task_ref(task_ref: Any) -> Dict[str, Any]:
    if isinstance(task_ref, dict):
        task = task_ref.get("task") or task_ref.get("task_name") or task_ref.get("instruction")
        task_id = task_ref.get("task_id") or task_ref.get("id") or task
        return {
            "task_id": str(task_id or ""),
            "task": str(task or ""),
            "task_family": _clean_task_family(task_ref.get("task_family") or task_ref.get("family"), task),
            "variation_idx": int(task_ref.get("variation_idx", task_ref.get("variation", 0)) or 0),
            "simplification": str(task_ref.get("simplification", task_ref.get("simplification_str", "")) or ""),
            "recommended_max_steps": _optional_int(task_ref.get("recommended_max_steps")),
        }
    text = str(task_ref)
    return {
        "task_id": text,
        "task": text,
        "task_family": _clean_task_family(text),
        "variation_idx": 0,
        "simplification": "",
        "recommended_max_steps": None,
    }


def _is_bad_step(prev_obs: str, next_obs: str, action: str) -> bool:
    nxt = str(next_obs or "").lower()
    bad_markers = [
        "i don't understand",
        "invalid",
        "nothing happens",
        "you can't",
        "not possible",
        "unknown action",
    ]
    if any(marker in nxt for marker in bad_markers):
        return True
    return str(prev_obs or "").strip() == str(next_obs or "").strip() and str(action or "") not in {"look", "look around", "inventory"}


def _normalize_score(value: Any) -> float:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _score_transition_fields(
    score_before: float,
    score_after: float,
    *,
    before_available: bool,
    after_available: bool,
) -> Dict[str, Optional[float]]:
    if not before_available or not after_available:
        return {
            "score_before": None,
            "score_after": None,
            "score_delta": None,
        }
    before = float(score_before)
    after = float(score_after)
    return {
        "score_before": before,
        "score_after": after,
        "score_delta": round(after - before, 10),
    }


class RemoteScienceWorldEnv:
    def __init__(self, remote_environment_url: str, timeout: int = 180):
        import requests

        self.session = requests.Session()
        self.session.trust_env = False
        self.base_url = str(remote_environment_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("ScienceWorld requires --scienceworld-env-url, for example http://127.0.0.1:8811")
        self.timeout = int(timeout)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"Remote ScienceWorld {path} failed:\n{resp.text}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Remote ScienceWorld {path} error:\n{data}")
        return data

    def reset(self, ref: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
        data = self._post(
            "/reset",
            {
                "task": ref.get("task"),
                "task_id": ref.get("task_id"),
                "variation_idx": ref.get("variation_idx", 0),
                "simplification": ref.get("simplification", ""),
            },
        )
        info = dict(data.get("info") or {})
        raw_score = data.get("score", info.get("score"))
        info["_score_available"] = raw_score is not None
        if raw_score is not None:
            info["_normalized_score"] = _normalize_score(raw_score)
        return str(data.get("observation") or ""), info, str(data.get("instruction") or ref.get("task") or "")

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        data = self._post("/step", {"action": str(action or "")})
        info = dict(data.get("info") or {})
        raw_score = data.get("score", info.get("score"))
        score_available = raw_score is not None
        normalized_score = _normalize_score(raw_score) if score_available else 0.0
        info["_score_available"] = score_available
        if score_available:
            info["_normalized_score"] = normalized_score
        return str(data.get("observation") or ""), normalized_score, bool(data.get("done", False)), info


def _split_remote_urls(remote_environment_url: str) -> List[str]:
    urls = [str(x or "").strip().rstrip("/") for x in str(remote_environment_url or "").split(",")]
    return [x for x in urls if x]


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((str(host or "127.0.0.1"), int(port)))
        except OSError:
            return False
    return True


def _find_free_ports(host: str, start_port: int, count: int) -> List[int]:
    ports: List[int] = []
    port = max(1, int(start_port or 8811))
    while len(ports) < max(1, int(count)):
        if _is_port_available(host, port):
            ports.append(port)
        port += 1
    return ports


def _default_scienceworld_server_python(value: Optional[str] = None) -> str:
    if value:
        return str(value)
    env_value = os.environ.get("SCIENCEWORLD_SERVER_PYTHON")
    if env_value:
        return env_value
    return sys.executable


class ManagedScienceWorldServerPool:
    def __init__(
        self,
        *,
        workers: int,
        host: str = "127.0.0.1",
        port_start: int = 8811,
        scienceworld_path: Optional[str] = None,
        scienceworld_jar_path: Optional[str] = None,
        env_step_limit: int = 120,
        server_python: Optional[str] = None,
        startup_timeout: int = 120,
    ):
        self.workers = max(1, int(workers or 1))
        self.host = str(host or "127.0.0.1")
        self.port_start = int(port_start or 8811)
        self.scienceworld_path = scienceworld_path
        self.scienceworld_jar_path = scienceworld_jar_path
        self.env_step_limit = int(env_step_limit or 120)
        self.server_python = _default_scienceworld_server_python(server_python)
        self.startup_timeout = int(startup_timeout or 120)
        self.processes: List[subprocess.Popen] = []
        self.urls: List[str] = []
        self._closed = False

    def start(self) -> List[str]:
        if self.urls:
            return list(self.urls)
        root = Path(__file__).resolve().parents[1]
        server_script = root / "scripts" / "serve_scienceworld_env.py"
        ports = _find_free_ports(self.host, self.port_start, self.workers)
        for port in ports:
            cmd = [
                self.server_python,
                str(server_script),
                "--host",
                self.host,
                "--port",
                str(port),
                "--scienceworld-env-step-limit",
                str(self.env_step_limit),
            ]
            if self.scienceworld_path:
                cmd.extend(["--scienceworld-path", str(self.scienceworld_path)])
            if self.scienceworld_jar_path:
                cmd.extend(["--scienceworld-jar-path", str(self.scienceworld_jar_path)])
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            self.processes.append(proc)
            self.urls.append(f"http://{self.host}:{port}")
        try:
            for proc, url in zip(self.processes, self.urls):
                self._wait_until_healthy(url, proc)
        except Exception:
            self.close()
            raise
        atexit.register(self.close)
        print(
            "[ScienceWorld auto-server] "
            f"started {len(self.urls)} server(s): {', '.join(self.urls)}",
            flush=True,
        )
        return list(self.urls)

    def _wait_until_healthy(self, url: str, proc: subprocess.Popen) -> None:
        import requests

        session = requests.Session()
        session.trust_env = False
        deadline = time.time() + self.startup_timeout
        last_error: Any = None
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"ScienceWorld auto-server exited early for {url} with code {proc.returncode}")
            try:
                resp = session.get(f"{url}/health", timeout=2)
                if resp.status_code < 400 and (resp.json() or {}).get("ok"):
                    return
                last_error = f"status={resp.status_code} body={resp.text[:300]}"
            except Exception as exc:
                last_error = exc
            time.sleep(0.5)
        raise RuntimeError(f"Timed out waiting for ScienceWorld auto-server {url}: {last_error}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5
        for proc in self.processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.processes.clear()


class ScienceWorldRunner:
    def __init__(
        self,
        action_agent: ScienceWorldActionAgent,
        *,
        remote_environment_url: str,
        remote_timeout: int = 180,
        max_steps: int = 100,
        top_k_general_skills: int = 6,
        top_k_task_specific_skills: int = 6,
        general_skill_retrieval_threshold: float = 0.0,
        task_specific_skill_retrieval_threshold: float = 0.0,
        task_specific_retrieval_scope: str = "detected",
        action_history_length: int = SCIENCEWORLD_ACTION_HISTORY_LENGTH,
        use_admissible_actions: bool = False,
        parallel_workers: int = 1,
        auto_server: bool = False,
        server_host: str = "127.0.0.1",
        server_port_start: int = 8811,
        scienceworld_path: Optional[str] = None,
        scienceworld_jar_path: Optional[str] = None,
        scienceworld_env_step_limit: Optional[int] = None,
        scienceworld_server_python: Optional[str] = None,
    ):
        self.action_agent = action_agent
        self.max_steps = int(max_steps)
        self.remote_timeout = int(remote_timeout)
        self.top_k_general_skills = int(top_k_general_skills)
        self.top_k_task_specific_skills = int(top_k_task_specific_skills)
        self.general_skill_retrieval_threshold = float(general_skill_retrieval_threshold)
        self.task_specific_skill_retrieval_threshold = float(task_specific_skill_retrieval_threshold)
        self.task_specific_retrieval_scope = task_specific_retrieval_scope
        self.action_history_length = max(0, int(action_history_length))
        self.use_admissible_actions = bool(use_admissible_actions)
        self.server_pool: Optional[ManagedScienceWorldServerPool] = None

        requested_workers = max(1, int(parallel_workers or 1))
        if auto_server:
            self.server_pool = ManagedScienceWorldServerPool(
                workers=requested_workers,
                host=server_host,
                port_start=server_port_start,
                scienceworld_path=scienceworld_path,
                scienceworld_jar_path=scienceworld_jar_path,
                env_step_limit=int(scienceworld_env_step_limit or max_steps),
                server_python=scienceworld_server_python,
                startup_timeout=remote_timeout,
            )
            self.env_urls = self.server_pool.start()
        else:
            self.env_urls = _split_remote_urls(remote_environment_url)
            if not self.env_urls:
                raise ValueError("ScienceWorld requires --scienceworld-env-url or --scienceworld-auto-server")
        self.parallel_workers = min(requested_workers, len(self.env_urls))
        if requested_workers > 1 and self.parallel_workers <= 1 and not auto_server:
            print(
                "[ScienceWorld runner] parallel_workers requested but only one env URL is available; "
                "use --scienceworld-auto-server or pass comma-separated --scienceworld-env-url values.",
                flush=True,
            )
        self.env = RemoteScienceWorldEnv(self.env_urls[0], timeout=self.remote_timeout)

    def _admissible(self, info: Dict[str, Any]) -> List[str]:
        for key in ("valid", "admissible_actions", "valid_actions", "available_actions"):
            actions = (info or {}).get(key)
            if isinstance(actions, list):
                return [str(x) for x in actions]
        return []

    def run_task(self, task_id, skill_bank, world_knowledge=None, force_skill_ids=None):
        return self._run_task_with_env(self.env, task_id, skill_bank, world_knowledge, force_skill_ids=force_skill_ids)

    def _run_task_with_env(self, env, task_id, skill_bank, world_knowledge=None, force_skill_ids=None):
        ref = _task_ref(task_id)
        obs, info, instruction = env.reset(ref)
        task_family = _clean_task_family(ref.get("task_family"), ref.get("task"))
        task_max_steps = _optional_int(ref.get("recommended_max_steps")) or self.max_steps
        cur = obs
        traj: List[Dict[str, Any]] = []
        score_available = bool(info.get("_score_available", info.get("score") is not None))
        final_score = float(info.get("_normalized_score", _normalize_score(info.get("score", 0.0))))
        done = False
        retrieval_query = f"{instruction}\nTask family: {task_family}\nObservation: {cur}"
        retrieved_memory = skill_bank.retrieve(
            retrieval_query,
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

        for step in range(task_max_steps):
            if done:
                break
            admissible = self._admissible(info) if self.use_admissible_actions else []
            context = {
                "observation": cur,
                "admissible_actions": admissible,
                "task_family": task_family,
                "current_score": final_score if score_available else None,
                "max_steps": task_max_steps,
            }
            recent_history = traj[-self.action_history_length:] if self.action_history_length else []
            decision = self.action_agent.act(instruction, context, world_knowledge or {}, retrieved_memory_text, recent_history, step)
            action = decision.get("action") or "look around"
            if admissible and action not in admissible:
                action = admissible[0]
            score_before = final_score
            before_available = score_available
            next_obs, next_score, done, info = env.step(action)
            after_available = bool(info.get("_score_available", next_score is not None))
            score_fields = _score_transition_fields(
                score_before,
                float(next_score or 0.0),
                before_available=before_available,
                after_available=after_available,
            )
            if after_available:
                final_score = float(next_score)
            score_available = after_available
            traj.append(
                {
                    "step": step,
                    "task": instruction,
                    "observation": cur,
                    "thought": decision.get("thought", ""),
                    "action": action,
                    "next_observation": next_obs,
                    **score_fields,
                    "done": done,
                    "bad_step": _is_bad_step(cur, next_obs, action),
                }
            )
            cur = next_obs

        return {
            "dataset": "scienceworld",
            "task_id": str(ref.get("task_id") or ""),
            "task": str(ref.get("task") or ""),
            "instruction": instruction,
            "task_family": task_family,
            "variation_idx": int(ref.get("variation_idx", 0) or 0),
            "simplification": str(ref.get("simplification") or ""),
            "recommended_max_steps": _optional_int(ref.get("recommended_max_steps")),
            "max_steps": task_max_steps,
            "success": bool(done and final_score >= 1.0),
            "score": final_score,
            "steps": len(traj),
            "trajectory": traj,
            "retrieved_memory": retrieved_memory,
            "retrieved_memory_text": retrieved_memory_text,
            "retrieved_skill_ids": retrieved_skill_ids,
        }

    def run_task_set(self, task_ids, skill_bank, world_knowledge=None, desc=None, force_skill_ids=None, **_: Any):
        task_ids = list(task_ids or [])
        total = len(task_ids)
        if total <= 1 or self.parallel_workers <= 1:
            results = []
            for idx, tid in enumerate(task_ids, start=1):
                result = self.run_task(tid, skill_bank, world_knowledge, force_skill_ids=force_skill_ids)
                results.append(result)
                if desc:
                    print(
                        f"[{desc}] task {idx}/{total} id={result.get('task_id')} "
                        f"family={result.get('task_family')} score={result.get('score')} "
                        f"steps={result.get('steps')} skills={result.get('retrieved_skill_ids') or []}",
                        flush=True,
                    )
            if desc:
                self._print_task_set_summary(desc, results, total)
            return results

        workers = min(self.parallel_workers, total, len(self.env_urls))
        results: List[Optional[Dict[str, Any]]] = [None] * total
        partitions: List[List[Tuple[int, Any]]] = [[] for _ in range(workers)]
        for idx, tid in enumerate(task_ids, start=1):
            partitions[(idx - 1) % workers].append((idx, tid))

        def _worker(url: str, assigned: List[Tuple[int, Any]]) -> None:
            env = RemoteScienceWorldEnv(url, timeout=self.remote_timeout)
            for idx, tid in assigned:
                result = self._run_task_with_env(
                    env,
                    tid,
                    skill_bank,
                    world_knowledge,
                    force_skill_ids=force_skill_ids,
                )
                result["scienceworld_env_url"] = url
                results[idx - 1] = result
                if desc:
                    print(
                        f"[{desc}] task {idx}/{total} id={result.get('task_id')} "
                        f"family={result.get('task_family')} score={result.get('score')} "
                        f"steps={result.get('steps')} skills={result.get('retrieved_skill_ids') or []}",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_worker, self.env_urls[i], partitions[i])
                for i in range(workers)
            ]
            for future in as_completed(futures):
                future.result()

        final_results = [x for x in results if x is not None]
        if len(final_results) != total:
            raise RuntimeError(f"ScienceWorld parallel rollout lost results: got {len(final_results)} expected {total}")
        if desc:
            self._print_task_set_summary(desc, final_results, total)
        return final_results

    def _print_task_set_summary(self, desc: str, results: List[Dict[str, Any]], total: int) -> None:
        score_avg = sum(float(x.get("score", 0.0) or 0.0) for x in results) / total if total else 0.0
        steps_avg = sum(int(x.get("steps", 0) or 0) for x in results) / total if total else 0.0
        success = sum(1 for x in results if x.get("success"))
        success_rate = success / total if total else 0.0
        print(
            f"[{desc}] summary score_avg={score_avg:.4f} "
            f"success_rate={success_rate:.4f} success={success}/{total} steps_avg={steps_avg:.2f}",
            flush=True,
        )

    def close(self) -> None:
        if self.server_pool is not None:
            self.server_pool.close()

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
        return self._run_from_trace_range_with_env(
            self.env,
            original_result,
            start_step=start_step,
            continue_steps=continue_steps,
            skill_bank=skill_bank,
            world_knowledge=world_knowledge,
            force_skill_ids=force_skill_ids,
        )

    def _run_from_trace_range_with_env(
        self,
        env,
        original_result,
        *,
        start_step,
        continue_steps,
        skill_bank,
        world_knowledge=None,
        force_skill_ids=None,
    ):
        ref = _task_ref(
            {
                "task_id": original_result.get("task_id"),
                "task": original_result.get("task") or original_result.get("instruction"),
                "task_family": original_result.get("task_family"),
                "variation_idx": original_result.get("variation_idx", 0),
                "simplification": original_result.get("simplification", ""),
                "recommended_max_steps": original_result.get("recommended_max_steps")
                or original_result.get("max_steps"),
            }
        )
        obs, info, instruction = env.reset(ref)
        task_family = _clean_task_family(ref.get("task_family"), ref.get("task"))
        task_max_steps = _optional_int(ref.get("recommended_max_steps")) or self.max_steps
        prefix = [
            x for x in original_result.get("trajectory", [])
            if int(x.get("step", 0) or 0) < int(start_step)
        ]
        replay_traj: List[Dict[str, Any]] = []
        cur = obs
        done = False
        score_available = bool(info.get("_score_available", info.get("score") is not None))
        final_score = float(info.get("_normalized_score", _normalize_score(info.get("score", 0.0))))
        for raw in prefix:
            action = str(raw.get("action", ""))
            score_before = final_score
            before_available = score_available
            next_obs, next_score, done, info = env.step(action)
            after_available = bool(info.get("_score_available", next_score is not None))
            score_fields = _score_transition_fields(
                score_before,
                float(next_score or 0.0),
                before_available=before_available,
                after_available=after_available,
            )
            if after_available:
                final_score = float(next_score)
            score_available = after_available
            replay_traj.append(
                {
                    "step": int(raw.get("step", len(replay_traj)) or 0),
                    "task": instruction,
                    "observation": cur,
                    "action": action,
                    "next_observation": next_obs,
                    **score_fields,
                    "done": done,
                    "bad_step": _is_bad_step(cur, next_obs, action),
                }
            )
            cur = next_obs
            if done:
                break

        retrieval_query = f"{instruction}\nTask family: {task_family}\nObservation: {cur}"
        retrieved_memory = skill_bank.retrieve(
            retrieval_query,
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
        remaining_steps = max(0, task_max_steps - int(start_step))
        for offset in range(min(max(0, int(continue_steps)), remaining_steps)):
            if done:
                break
            step = int(start_step) + offset
            admissible = self._admissible(info) if self.use_admissible_actions else []
            context = {
                "observation": cur,
                "admissible_actions": admissible,
                "task_family": task_family,
                "current_score": final_score if score_available else None,
                "max_steps": task_max_steps,
            }
            recent_history = replay_traj[-self.action_history_length:] if self.action_history_length else []
            decision = self.action_agent.act(instruction, context, world_knowledge or {}, retrieved_memory_text, recent_history, step)
            action = decision.get("action") or "look around"
            if admissible and action not in admissible:
                action = admissible[0]
            score_before = final_score
            before_available = score_available
            next_obs, next_score, done, info = env.step(action)
            after_available = bool(info.get("_score_available", next_score is not None))
            score_fields = _score_transition_fields(
                score_before,
                float(next_score or 0.0),
                before_available=before_available,
                after_available=after_available,
            )
            if after_available:
                final_score = float(next_score)
            score_available = after_available
            row = {
                "step": step,
                "task": instruction,
                "observation": cur,
                "thought": decision.get("thought", ""),
                "action": action,
                "next_observation": next_obs,
                **score_fields,
                "done": done,
                "bad_step": _is_bad_step(cur, next_obs, action),
            }
            replay_traj.append(row)
            after_segments.append(row)
            cur = next_obs

        return {
            "dataset": "scienceworld",
            "task_id": str(ref.get("task_id") or ""),
            "instruction": instruction,
            "task_family": task_family,
            "variation_idx": int(ref.get("variation_idx", 0) or 0),
            "simplification": str(ref.get("simplification") or ""),
            "recommended_max_steps": _optional_int(ref.get("recommended_max_steps")),
            "max_steps": task_max_steps,
            "prefix_actions": [x.get("action", "") for x in prefix],
            "after_segments": [after_segments],
            "retrieved_skill_ids": retrieved_skill_ids,
            "retrieved_memory": retrieved_memory,
            "score": final_score,
            "done": done,
        }

    def run_from_trace_range_set(
        self,
        jobs,
        *,
        skill_bank,
        world_knowledge=None,
        force_skill_ids=None,
        desc: Optional[str] = None,
    ):
        jobs = list(jobs or [])
        total = len(jobs)
        if total <= 1 or self.parallel_workers <= 1:
            results = []
            for idx, job in enumerate(jobs, start=1):
                try:
                    replay = self.run_from_trace_range(
                        job.get("original_result") or {},
                        start_step=job.get("start_step", 0),
                        continue_steps=job.get("continue_steps", 1),
                        skill_bank=skill_bank,
                        world_knowledge=world_knowledge,
                        force_skill_ids=force_skill_ids,
                    )
                    results.append({"ok": True, "result": replay})
                    if desc:
                        print(
                            f"[{desc}] replay {idx}/{total} done "
                            f"evidence={job.get('evidence_id')} task={job.get('task_id')} "
                            f"range={job.get('range_index')}",
                            flush=True,
                        )
                except Exception as exc:
                    results.append({"ok": False, "error": str(exc)})
                    if desc:
                        print(
                            f"[{desc}] replay {idx}/{total} error "
                            f"evidence={job.get('evidence_id')} task={job.get('task_id')} "
                            f"range={job.get('range_index')} error={exc}",
                            flush=True,
                        )
            return results

        workers = min(self.parallel_workers, total, len(self.env_urls))
        results: List[Optional[Dict[str, Any]]] = [None] * total
        partitions: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(workers)]
        for idx, job in enumerate(jobs):
            partitions[idx % workers].append((idx, job))

        def _worker(url: str, assigned: List[Tuple[int, Dict[str, Any]]]) -> None:
            env = RemoteScienceWorldEnv(url, timeout=self.remote_timeout)
            for idx, job in assigned:
                try:
                    replay = self._run_from_trace_range_with_env(
                        env,
                        job.get("original_result") or {},
                        start_step=job.get("start_step", 0),
                        continue_steps=job.get("continue_steps", 1),
                        skill_bank=skill_bank,
                        world_knowledge=world_knowledge,
                        force_skill_ids=force_skill_ids,
                    )
                    results[idx] = {"ok": True, "result": replay}
                    if desc:
                        print(
                            f"[{desc}] replay {idx + 1}/{total} done "
                            f"evidence={job.get('evidence_id')} task={job.get('task_id')} "
                            f"range={job.get('range_index')} env={url}",
                            flush=True,
                        )
                except Exception as exc:
                    results[idx] = {"ok": False, "error": str(exc)}
                    if desc:
                        print(
                            f"[{desc}] replay {idx + 1}/{total} error "
                            f"evidence={job.get('evidence_id')} task={job.get('task_id')} "
                            f"range={job.get('range_index')} env={url} error={exc}",
                            flush=True,
                        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_worker, self.env_urls[i], partitions[i])
                for i in range(workers)
            ]
            for future in as_completed(futures):
                future.result()

        final_results = [result if result is not None else {"ok": False, "error": "missing replay result"} for result in results]
        return final_results
