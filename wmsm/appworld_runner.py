import atexit
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .action_runtime import APPWORLD_ACTION_HISTORY_LENGTH
from .agents import ActionAgent
from .appworld_manifests import resolve_appworld_data_dir
from .bundles import get_task_id


APPWORLD_FALLBACK_CODE = "print(apis.api_docs.show_app_descriptions())"
APPWORLD_TRANSPORT_ATTEMPTS = 4
APPWORLD_TRANSPORT_RETRY_BASE_SECONDS = 2.0


def _is_transient_appworld_transport_error(exc: BaseException) -> bool:
    """Identify transport failures that are safe to recover by rerunning a task."""
    text = str(exc or "").lower()
    return any(
        marker in text
        for marker in (
            "connection reset by peer",
            "connection aborted",
            "connection refused",
            "remoteappworld call",
            "remote appworld call",
            "read timed out",
            "max retries exceeded",
            "network is unreachable",
            "status=429",
            "http 429",
            "upstream load",
            "上游负载已饱和",
            "ssl eof",
            "eof occurred in violation of protocol",
            "empty completion content",
            "http 502",
            "http 503",
            "http 504",
        )
    )


def _split_urls(value: Optional[str]) -> List[str]:
    return [part.strip().rstrip("/") for part in str(value or "").split(",") if part.strip()]


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
    port = max(1, int(start_port or 8841))
    while len(ports) < max(1, int(count)):
        if _is_port_available(host, port):
            ports.append(port)
        port += 1
    return ports


def _default_server_python(value: Optional[str] = None) -> str:
    if value:
        return str(Path(value).expanduser())
    env_value = os.environ.get("APPWORLD_SERVER_PYTHON")
    if env_value:
        return env_value
    return sys.executable


def prepare_appworld_runtime_root(data_dir: str, runtime_root: str) -> Tuple[Path, Path]:
    data_path = resolve_appworld_data_dir(data_dir)
    root = Path(runtime_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_link = root / "data"
    if data_link.is_symlink():
        if data_link.resolve() != data_path:
            raise RuntimeError(
                f"AppWorld runtime data symlink points to {data_link.resolve()}, expected {data_path}"
            )
    elif data_link.exists():
        if data_link.resolve() != data_path:
            raise RuntimeError(
                f"AppWorld runtime root already contains a different data path: {data_link}"
            )
    else:
        data_link.symlink_to(data_path, target_is_directory=True)
    (root / "experiments" / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "server_logs").mkdir(parents=True, exist_ok=True)
    return root, data_path


def appworld_package_version(server_python: Optional[str] = None) -> str:
    command = [
        _default_server_python(server_python),
        "-c",
        "import appworld; print(appworld.__version__)",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load_app_descriptions(server_python: str, runtime_root: Path) -> List[Dict[str, str]]:
    script = (
        "import json,sys; import appworld; appworld.update_root(sys.argv[1]); "
        "from appworld.apps import APP_TO_DESCRIPTION; "
        "print(json.dumps([{'name':k,'description':v} for k,v in APP_TO_DESCRIPTION.items() "
        "if k != 'admin']))"
    )
    try:
        result = subprocess.run(
            [server_python, "-c", script, str(runtime_root)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        rows = json.loads(result.stdout)
        return [
            {"name": str(row.get("name") or ""), "description": str(row.get("description") or "")}
            for row in rows
            if isinstance(row, dict) and row.get("name")
        ]
    except Exception:
        return []


_SECRET_KEY_RE = re.compile(
    r"(?:password|passcode|access[_-]?token|refresh[_-]?token|verification[_-]?code|"
    r"one[_-]?time[_-]?code|otp|api[_-]?key|secret|cvv|card[_-]?number)",
    re.IGNORECASE,
)
_QUOTED_SECRET_RE = re.compile(
    r"(?i)(\b(?:[a-z0-9_]*(?:password|passcode|token|secret|otp|verification_code|api_key|cvv|card_number)[a-z0-9_]*)"
    r"\b\s*(?::|=)\s*)(['\"])(.*?)(\2)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]+")


def redact_appworld_optimizer_value(value: Any) -> Any:
    """Redact credential values while retaining API/code structure."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _SECRET_KEY_RE.search(str(key)) and item not in (None, "")
                else redact_appworld_optimizer_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_appworld_optimizer_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_appworld_optimizer_value(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
            redacted = redact_appworld_optimizer_value(parsed)
            return json.dumps(redacted, ensure_ascii=False, indent=1)
        except (TypeError, ValueError):
            pass
    text = _QUOTED_SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(4)}", text)
    return _BEARER_RE.sub("Bearer [REDACTED]", text)


def sanitize_appworld_evaluation(
    payload: Any,
    *,
    include_requirement_labels: bool = False,
) -> Tuple[bool, Dict[str, int], List[Dict[str, Any]]]:
    data = payload if isinstance(payload, dict) else {}
    passes = data.get("passes") if isinstance(data.get("passes"), list) else []
    failures = data.get("failures") if isinstance(data.get("failures"), list) else []
    try:
        num_tests = int(data.get("num_tests", len(passes) + len(failures)) or 0)
    except (TypeError, ValueError):
        num_tests = len(passes) + len(failures)
    passed = len(passes)
    failed = max(len(failures), num_tests - passed)
    success = bool(data.get("success")) and num_tests > 0 and passed == num_tests
    summary = {
        "num_tests": num_tests,
        "passed_tests": passed,
        "failed_tests": failed,
    }
    labels: List[Dict[str, Any]] = []
    if include_requirement_labels:
        statuses = [True] * passed + [False] * failed
        labels = [
            {"label": f"requirement_{index + 1}", "passed": status}
            for index, status in enumerate(statuses)
        ]
    return success, summary, labels


def _execution_failed(code: str, output: str) -> bool:
    if not str(code or "").strip():
        return True
    text = str(output or "").lower()
    return any(
        marker in text
        for marker in (
            "execution failed. traceback:",
            "execution timed out",
            "syntax error in line",
            "no code available to execute",
            "maximum number of executions",
        )
    )


def _task_ref(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        row = dict(value)
        task_id = str(row.get("task_id") or row.get("id") or "")
    else:
        task_id = str(value)
        row = {"task_id": task_id}
    try:
        family, raw_variant = task_id.rsplit("_", 1)
        variant = int(raw_variant)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid AppWorld task ID: {task_id!r}") from None
    row.update(
        {
            "dataset": "appworld",
            "task_id": task_id,
            "task_family": str(row.get("task_family") or family),
            "variant": int(row.get("variant", variant)),
        }
    )
    return row


def _unique_experiment_name(task_id: str) -> str:
    safe_task = re.sub(r"[^a-z0-9_-]+", "_", str(task_id).lower()).strip("_")
    return f"wmsm_{os.getpid()}_{safe_task}_{uuid.uuid4().hex[:12]}"


class ManagedAppWorldServerPool:
    def __init__(
        self,
        *,
        workers: int,
        data_dir: str,
        runtime_root: str,
        host: str = "127.0.0.1",
        port_start: int = 8841,
        server_python: Optional[str] = None,
        startup_timeout: int = 180,
    ) -> None:
        self.workers = max(1, int(workers or 1))
        self.host = str(host or "127.0.0.1")
        self.port_start = int(port_start or 8841)
        self.server_python = _default_server_python(server_python)
        self.startup_timeout = max(1, int(startup_timeout or 180))
        self.runtime_root, self.data_dir = prepare_appworld_runtime_root(data_dir, runtime_root)
        self.processes: List[subprocess.Popen] = []
        self.log_handles: List[Any] = []
        self.urls: List[str] = []
        self.package_version = appworld_package_version(self.server_python)
        self.app_descriptions = _load_app_descriptions(self.server_python, self.runtime_root)
        self._closed = False

    def _launch_worker(self, worker_index: int, port: int) -> subprocess.Popen:
        log_path = self.runtime_root / "server_logs" / f"environment_{port}.log"
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        command = [
            self.server_python,
            "-m",
            "appworld.cli",
            "serve",
            "environment",
            "--port",
            str(port),
            "--root",
            str(self.runtime_root),
            "--no-show-usage",
        ]
        process = subprocess.Popen(
            command,
            cwd=str(self.runtime_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        if worker_index == len(self.processes):
            self.processes.append(process)
            self.log_handles.append(log_handle)
            self.urls.append(f"http://{self.host}:{port}")
        else:
            old_handle = self.log_handles[worker_index]
            try:
                old_handle.close()
            except Exception:
                pass
            self.processes[worker_index] = process
            self.log_handles[worker_index] = log_handle
        return process

    def start(self) -> List[str]:
        if self.urls:
            return list(self.urls)
        ports = _find_free_ports(self.host, self.port_start, self.workers)
        for worker_index, port in enumerate(ports):
            self._launch_worker(worker_index, port)
        try:
            for process, url in zip(self.processes, self.urls):
                self._wait_until_healthy(url, process)
        except Exception:
            self.close()
            raise
        atexit.register(self.close)
        print(
            f"[AppWorld auto-server] started {len(self.urls)} isolated server(s): "
            f"{', '.join(self.urls)} version={self.package_version} data={self.data_dir}",
            flush=True,
        )
        return list(self.urls)

    def recover_worker(self, worker_index: int) -> None:
        """Restart only a dead worker; live workers keep their assigned port."""
        if self._closed or worker_index < 0 or worker_index >= len(self.processes):
            return
        process = self.processes[worker_index]
        if process.poll() is None:
            return
        port = int(self.urls[worker_index].rsplit(":", 1)[-1])
        print(
            f"[AppWorld auto-server] recovering worker={worker_index} "
            f"port={port} previous_exit={process.returncode}",
            flush=True,
        )
        replacement = self._launch_worker(worker_index, port)
        self._wait_until_healthy(self.urls[worker_index], replacement)

    def _wait_until_healthy(self, url: str, process: subprocess.Popen) -> None:
        import requests

        session = requests.Session()
        session.trust_env = False
        deadline = time.time() + self.startup_timeout
        last_error: Any = None
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"AppWorld server exited early for {url} with code {process.returncode}; "
                    f"see {self.runtime_root / 'server_logs'}"
                )
            try:
                response = session.get(f"{url}/", timeout=2)
                if response.status_code < 400:
                    return
                last_error = f"status={response.status_code}"
            except Exception as exc:
                last_error = exc
            time.sleep(0.5)
        raise RuntimeError(f"Timed out waiting for AppWorld server {url}: {last_error}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.processes.clear()
        for handle in self.log_handles:
            try:
                handle.close()
            except Exception:
                pass
        self.log_handles.clear()


class RemoteAppWorldEnv:
    def __init__(
        self,
        environment_url: str,
        *,
        timeout: int = 100,
        max_interactions: int = 50,
        random_seed: int = 100,
    ) -> None:
        import requests

        self.base_url = str(environment_url).rstrip("/")
        self.timeout = max(1, int(timeout or 100))
        self.max_interactions = max(1, int(max_interactions or 50))
        self.random_seed = int(random_seed if random_seed is not None else 100)
        self.session = requests.Session()
        self.session.trust_env = False
        self.active_task_id: Optional[str] = None

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        response = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout + 30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Remote AppWorld call {path} failed with HTTP {response.status_code}"
            )
        data = response.json()
        return data.get("output")

    def initialize(self, task_id: str, experiment_name: str) -> Dict[str, Any]:
        if self.active_task_id:
            self.close()
        output = self._post(
            "/initialize",
            {
                "task_id": str(task_id),
                "experiment_name": str(experiment_name),
                "remote_docker": False,
                "max_interactions": self.max_interactions,
                "max_api_calls_per_interaction": 1000,
                "raise_on_unsafe_syntax": True,
                "null_patch_unsafe_execution": True,
                "load_ground_truth": True,
                "raise_on_failure": True,
                "random_seed": self.random_seed,
                "timeout_seconds": self.timeout,
                "show_api_response_schemas": True,
                "gc_threshold": 500000,
                "raise_on_extra_parameters": True,
                "import_utils": False,
                "parse_datetimes": False,
                "allow_datetime_change": False,
                "add_login_shortcut": False,
                "munchify_response": False,
            },
        )
        self.active_task_id = str(task_id)
        return output if isinstance(output, dict) else {}

    def execute(self, code: str) -> str:
        if not self.active_task_id:
            raise RuntimeError("AppWorld environment is not initialized")
        output = self._post(
            "/execute", {"task_id": self.active_task_id, "code": str(code or "")}
        )
        return str(output or "")

    def task_completed(self) -> bool:
        if not self.active_task_id:
            return False
        return bool(
            self._post("/task_completed", {"task_id": self.active_task_id})
        )

    def evaluate(self, *, include_requirement_labels: bool = False):
        if not self.active_task_id:
            return False, {"num_tests": 0, "passed_tests": 0, "failed_tests": 0}, []
        output = self._post(
            "/evaluate",
            {"task_id": self.active_task_id, "suppress_errors": True, "report": False},
        )
        return sanitize_appworld_evaluation(
            output, include_requirement_labels=include_requirement_labels
        )

    def close(self) -> None:
        if not self.active_task_id:
            return
        task_id = self.active_task_id
        self.active_task_id = None
        try:
            self._post("/close", {"task_id": task_id})
        except Exception:
            pass


class AppWorldRunner:
    def __init__(
        self,
        action_agent: ActionAgent,
        *,
        data_dir: str,
        output_dir: str,
        environment_url: Optional[str] = None,
        auto_server: bool = False,
        server_python: Optional[str] = None,
        server_host: str = "127.0.0.1",
        server_port_start: int = 8841,
        environment_timeout: int = 100,
        max_steps: int = 50,
        action_history_length: int = APPWORLD_ACTION_HISTORY_LENGTH,
        random_seed: int = 100,
        parallel_workers: int = 1,
        top_k_general_skills: int = 5,
        top_k_task_specific_skills: int = 2,
        general_skill_retrieval_threshold: float = 0.0,
        task_specific_skill_retrieval_threshold: float = 0.0,
        task_specific_retrieval_scope: str = "detected",
    ) -> None:
        self.action_agent = action_agent
        self.data_dir = str(resolve_appworld_data_dir(data_dir))
        self.output_dir = str(Path(output_dir).expanduser().resolve())
        self.max_steps = max(1, int(max_steps or 50))
        self.action_history_length = max(0, int(action_history_length or 0))
        self.random_seed = int(random_seed if random_seed is not None else 100)
        self.environment_timeout = max(1, int(environment_timeout or 100))
        self.requested_parallel_workers = max(1, int(parallel_workers or 1))
        self.server_pool: Optional[ManagedAppWorldServerPool] = None
        server_python = _default_server_python(server_python)
        runtime_root = str(Path(self.output_dir) / "appworld_runtime")
        if auto_server:
            self.server_pool = ManagedAppWorldServerPool(
                workers=self.requested_parallel_workers,
                data_dir=self.data_dir,
                runtime_root=runtime_root,
                host=server_host,
                port_start=server_port_start,
                server_python=server_python,
            )
            self.env_urls = self.server_pool.start()
            self.app_descriptions = list(self.server_pool.app_descriptions)
        else:
            prepare_appworld_runtime_root(self.data_dir, runtime_root)
            self.env_urls = _split_urls(environment_url)
            self.app_descriptions = _load_app_descriptions(server_python, Path(runtime_root))
        if not self.env_urls:
            raise ValueError(
                "AppWorld needs --appworld-auto-server or at least one --appworld-env-url"
            )
        self.parallel_workers = min(self.requested_parallel_workers, len(self.env_urls))
        if self.parallel_workers < self.requested_parallel_workers:
            print(
                f"[AppWorld runner] requested workers={self.requested_parallel_workers}, "
                f"available isolated servers={len(self.env_urls)}; using {self.parallel_workers}",
                flush=True,
            )
        self.env = self._make_env(0)
        self.top_k_general_skills = int(top_k_general_skills)
        self.top_k_task_specific_skills = int(top_k_task_specific_skills)
        self.general_skill_retrieval_threshold = float(general_skill_retrieval_threshold)
        self.task_specific_skill_retrieval_threshold = float(task_specific_skill_retrieval_threshold)
        self.task_specific_retrieval_scope = str(task_specific_retrieval_scope)

    def _make_env(self, worker_index: int) -> RemoteAppWorldEnv:
        return RemoteAppWorldEnv(
            self.env_urls[worker_index],
            timeout=self.environment_timeout,
            max_interactions=self.max_steps,
            random_seed=self.random_seed,
        )

    def _retrieve(self, skill_bank, instruction: str, family: str, observation: str, force_skill_ids=None):
        query = f"{instruction}\nTask family: {family}\nLatest execution output: {observation}"
        kwargs = {
            "top_k_general": self.top_k_general_skills,
            "top_k_task_specific": self.top_k_task_specific_skills,
            "min_score_general": self.general_skill_retrieval_threshold,
            "min_score_task_specific": self.task_specific_skill_retrieval_threshold,
            "task_specific_scope": self.task_specific_retrieval_scope,
            "force_skill_ids": force_skill_ids,
        }
        try:
            memory = skill_bank.retrieve(query, task_type_override=family, **kwargs)
        except TypeError:
            memory = skill_bank.retrieve(query, **kwargs)
        return memory, skill_bank.format_for_prompt(memory), skill_bank.retrieved_skill_ids(memory)

    def run_task(self, task_id, skill_bank, world_knowledge=None, force_skill_ids=None):
        return self._run_task_with_env(
            self.env,
            task_id,
            skill_bank,
            world_knowledge=world_knowledge,
            force_skill_ids=force_skill_ids,
        )

    def _run_task_with_env(self, env, task_id, skill_bank, world_knowledge=None, force_skill_ids=None):
        ref = _task_ref(task_id)
        task_key = ref["task_id"]
        metadata = env.initialize(task_key, _unique_experiment_name(task_key))
        instruction = str(metadata.get("instruction") or ref.get("instruction") or "")
        supervisor = metadata.get("supervisor") if isinstance(metadata.get("supervisor"), dict) else {}
        datetime_value = str(metadata.get("datetime") or "")
        family = str(ref.get("task_family") or task_key.rsplit("_", 1)[0])
        initial_observation = "No Python code has been executed yet."
        memory, memory_text, skill_ids = self._retrieve(
            skill_bank, instruction, family, initial_observation, force_skill_ids
        )
        trajectory: List[Dict[str, Any]] = []
        current_output = initial_observation
        task_completed = False
        try:
            for step in range(self.max_steps):
                recent_history = trajectory[-self.action_history_length :] if self.action_history_length else []
                context = {
                    "observation": current_output,
                    "task_family": family,
                    "supervisor": supervisor,
                    "datetime": datetime_value,
                    "app_descriptions": self.app_descriptions,
                    "max_steps": self.max_steps,
                }
                decision = self.action_agent.act(
                    instruction,
                    context,
                    world_knowledge or {},
                    memory_text,
                    recent_history,
                    step,
                )
                code = str(decision.get("action") or APPWORLD_FALLBACK_CODE).strip()
                output = env.execute(code)
                task_completed = env.task_completed()
                bad_step = _execution_failed(code, output)
                trajectory.append(
                    {
                        "step": step,
                        "task": instruction,
                        "observation": current_output,
                        "thought": str(decision.get("thought") or ""),
                        "action": code,
                        "next_observation": output,
                        "done": task_completed,
                        "bad_step": bad_step,
                        "execution_ok": not bad_step,
                        "task_completed": task_completed,
                    }
                )
                current_output = output
                if task_completed:
                    break
            success, evaluation_summary, requirement_labels = env.evaluate(
                include_requirement_labels=str(ref.get("split") or "") == "train"
            )
        finally:
            env.close()
        result = {
            "dataset": "appworld",
            "task_id": task_key,
            "instruction": instruction,
            "task_family": family,
            "variant": int(ref.get("variant", 0) or 0),
            "split": str(ref.get("split") or ""),
            "success": bool(success),
            "score": 1.0 if success else 0.0,
            "steps": len(trajectory),
            "max_steps": self.max_steps,
            "task_completed": bool(task_completed),
            "evaluation_summary": evaluation_summary,
            "trajectory": trajectory,
            "retrieved_memory": memory,
            "retrieved_memory_text": memory_text,
            "retrieved_skill_ids": skill_ids,
        }
        if requirement_labels:
            result["train_evaluation_requirements"] = requirement_labels
        return result

    def run_task_set(self, task_ids, skill_bank, world_knowledge=None, desc=None, force_skill_ids=None, **_: Any):
        task_ids = list(task_ids or [])
        total = len(task_ids)
        if total == 0:
            return []
        workers = min(self.parallel_workers, total)
        ordered: List[Optional[Dict[str, Any]]] = [None] * total
        partitions: List[List[Tuple[int, Any]]] = [[] for _ in range(workers)]
        for index, task_id in enumerate(task_ids):
            partitions[index % workers].append((index, task_id))

        def run_partition(worker_index: int, assigned: List[Tuple[int, Any]]) -> None:
            env = self._make_env(worker_index)
            for index, task_id in assigned:
                result = None
                for attempt in range(APPWORLD_TRANSPORT_ATTEMPTS):
                    try:
                        result = self._run_task_with_env(
                            env,
                            task_id,
                            skill_bank,
                            world_knowledge=world_knowledge,
                            force_skill_ids=force_skill_ids,
                        )
                        break
                    except Exception as exc:
                        if (
                            attempt >= APPWORLD_TRANSPORT_ATTEMPTS - 1
                            or not _is_transient_appworld_transport_error(exc)
                        ):
                            raise
                        delay = min(
                            30.0,
                            APPWORLD_TRANSPORT_RETRY_BASE_SECONDS * (2**attempt),
                        )
                        print(
                            f"[{desc or 'AppWorld'}] task {index + 1}/{total} "
                            f"id={get_task_id(task_id)} transport failure; "
                            f"retrying from fresh world attempt={attempt + 2}/"
                            f"{APPWORLD_TRANSPORT_ATTEMPTS} sleep={delay:.1f}s error={exc}",
                            flush=True,
                        )
                        try:
                            env.close()
                        except Exception:
                            pass
                        if self.server_pool is not None:
                            self.server_pool.recover_worker(worker_index)
                        time.sleep(delay)
                        env = self._make_env(worker_index)
                if result is None:
                    raise RuntimeError(f"AppWorld task produced no result: {task_id}")
                ordered[index] = result
                if desc:
                    print(
                        f"[{desc}] task {index + 1}/{total} id={result.get('task_id')} "
                        f"family={result.get('task_family')} score={result.get('score')} "
                        f"steps={result.get('steps')} skills={result.get('retrieved_skill_ids')}",
                        flush=True,
                    )

        if workers == 1:
            run_partition(0, partitions[0])
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(run_partition, worker_index, partitions[worker_index])
                    for worker_index in range(workers)
                ]
                for future in as_completed(futures):
                    future.result()
        results = [row for row in ordered if row is not None]
        if len(results) != total:
            raise RuntimeError(f"AppWorld rollout lost results: got {len(results)} expected {total}")
        if desc:
            successes = sum(1 for row in results if row.get("success"))
            steps_avg = sum(int(row.get("steps", 0) or 0) for row in results) / total
            print(
                f"[{desc}] summary score_avg={successes / total:.4f} "
                f"success_rate={successes / total:.4f} success={successes}/{total} "
                f"steps_avg={steps_avg:.2f}",
                flush=True,
            )
        return results

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
        ref = _task_ref(original_result)
        task_key = ref["task_id"]
        metadata = env.initialize(task_key, _unique_experiment_name(task_key))
        instruction = str(metadata.get("instruction") or original_result.get("instruction") or "")
        supervisor = metadata.get("supervisor") if isinstance(metadata.get("supervisor"), dict) else {}
        datetime_value = str(metadata.get("datetime") or "")
        family = str(ref.get("task_family") or task_key.rsplit("_", 1)[0])
        prefix = [
            row
            for row in (original_result.get("trajectory") or [])
            if isinstance(row, dict) and int(row.get("step", 0) or 0) < int(start_step)
        ]
        replay_trajectory: List[Dict[str, Any]] = []
        current_output = "No Python code has been executed yet."
        task_completed = False
        try:
            for raw in prefix:
                code = str(raw.get("action") or "")
                output = env.execute(code)
                task_completed = env.task_completed()
                bad_step = _execution_failed(code, output)
                replay_trajectory.append(
                    {
                        "step": int(raw.get("step", len(replay_trajectory)) or 0),
                        "task": instruction,
                        "observation": current_output,
                        "action": code,
                        "next_observation": output,
                        "done": task_completed,
                        "bad_step": bad_step,
                        "execution_ok": not bad_step,
                        "task_completed": task_completed,
                    }
                )
                current_output = output
                if task_completed:
                    break
            memory, memory_text, skill_ids = self._retrieve(
                skill_bank, instruction, family, current_output, force_skill_ids
            )
            after_segments: List[Dict[str, Any]] = []
            for offset in range(max(0, int(continue_steps))):
                if task_completed or len(replay_trajectory) >= self.max_steps:
                    break
                step = int(start_step) + offset
                recent_history = replay_trajectory[-self.action_history_length :] if self.action_history_length else []
                context = {
                    "observation": current_output,
                    "task_family": family,
                    "supervisor": supervisor,
                    "datetime": datetime_value,
                    "app_descriptions": self.app_descriptions,
                    "max_steps": self.max_steps,
                }
                decision = self.action_agent.act(
                    instruction,
                    context,
                    world_knowledge or {},
                    memory_text,
                    recent_history,
                    step,
                )
                code = str(decision.get("action") or APPWORLD_FALLBACK_CODE).strip()
                output = env.execute(code)
                task_completed = env.task_completed()
                bad_step = _execution_failed(code, output)
                row = {
                    "step": step,
                    "task": instruction,
                    "observation": current_output,
                    "thought": str(decision.get("thought") or ""),
                    "action": code,
                    "next_observation": output,
                    "done": task_completed,
                    "bad_step": bad_step,
                    "execution_ok": not bad_step,
                    "task_completed": task_completed,
                }
                replay_trajectory.append(row)
                after_segments.append(row)
                current_output = output
            success, evaluation_summary, _ = env.evaluate(include_requirement_labels=False)
        finally:
            env.close()
        return {
            "dataset": "appworld",
            "task_id": task_key,
            "instruction": instruction,
            "task_family": family,
            "variant": int(ref.get("variant", original_result.get("variant", 0)) or 0),
            "prefix_actions": [str(row.get("action") or "") for row in prefix],
            "after_segments": [after_segments],
            "retrieved_skill_ids": skill_ids,
            "retrieved_memory": memory,
            "score": 1.0 if success else 0.0,
            "success": bool(success),
            "done": bool(task_completed),
            "task_completed": bool(task_completed),
            "evaluation_summary": evaluation_summary,
        }

    def run_from_trace_range_set(
        self,
        jobs,
        *,
        skill_bank,
        world_knowledge=None,
        force_skill_ids=None,
        desc=None,
    ):
        jobs = list(jobs or [])
        total = len(jobs)
        if total == 0:
            return []
        workers = min(self.parallel_workers, total)
        ordered: List[Optional[Dict[str, Any]]] = [None] * total
        partitions: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(workers)]
        for index, job in enumerate(jobs):
            partitions[index % workers].append((index, job))

        def run_partition(worker_index: int, assigned: List[Tuple[int, Dict[str, Any]]]) -> None:
            env = self._make_env(worker_index)
            for index, job in assigned:
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
                    ordered[index] = {"ok": True, "result": replay}
                    if desc:
                        print(
                            f"[{desc}] replay {index + 1}/{total} done "
                            f"evidence={job.get('evidence_id')} task={job.get('task_id')}",
                            flush=True,
                        )
                except Exception as exc:
                    ordered[index] = {"ok": False, "error": str(exc)}

        if workers == 1:
            run_partition(0, partitions[0])
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(run_partition, worker_index, partitions[worker_index])
                    for worker_index in range(workers)
                ]
                for future in as_completed(futures):
                    future.result()
        return [
            row if row is not None else {"ok": False, "error": "missing replay result"}
            for row in ordered
        ]

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        if self.server_pool is not None:
            self.server_pool.close()


__all__ = [
    "APPWORLD_FALLBACK_CODE",
    "AppWorldRunner",
    "ManagedAppWorldServerPool",
    "RemoteAppWorldEnv",
    "appworld_package_version",
    "prepare_appworld_runtime_root",
    "redact_appworld_optimizer_value",
    "sanitize_appworld_evaluation",
]
