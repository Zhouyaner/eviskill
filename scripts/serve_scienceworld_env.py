#!/usr/bin/env python
import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple


def to_jsonable(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return str(x)


def add_scienceworld_path(path: str = None) -> None:
    if not path:
        return
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def make_env(scienceworld_path: str = None, jar_path: str = None, env_step_limit: int = 120):
    add_scienceworld_path(scienceworld_path)
    from scienceworld import ScienceWorldEnv

    return ScienceWorldEnv("", serverPath=jar_path, envStepLimit=int(env_step_limit))


def score_from_info(info: dict, fallback: float = 0.0) -> float:
    try:
        return float((info or {}).get("score", fallback) or 0.0)
    except (TypeError, ValueError):
        return float(fallback or 0.0)


def task_description(env, fallback: str = "") -> str:
    for name in ("get_task_description", "getTaskDescription", "taskdescription"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                text = str(fn() or "").strip()
                if text:
                    return text
            except Exception:
                pass
    return str(fallback or "")


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def _response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(to_jsonable(payload), ensure_ascii=False).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        # The client timed out or was interrupted while ScienceWorld was
        # preparing the response. The environment process can keep serving.
        return


def make_handler(env, args):
    current = {"task": None, "variation_idx": 0, "simplification": ""}

    class ScienceWorldHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._handle_health()
            elif self.path == "/valid_actions":
                self._handle_valid_actions()
            else:
                _response(self, {"error": f"unknown route: {self.path}"}, status=404)

        def do_POST(self):
            if self.path == "/reset":
                self._handle_reset()
            elif self.path == "/step":
                self._handle_step()
            elif self.path == "/valid_actions":
                self._handle_valid_actions()
            elif self.path == "/close":
                self._handle_close()
            else:
                _response(self, {"error": f"unknown route: {self.path}"}, status=404)

        def log_message(self, fmt, *values):
            return

        def _handle_health(self):
            try:
                tasks = env.get_task_names()
            except Exception:
                tasks = []
            _response(
                self,
                {
                    "ok": True,
                    "tasks": len(tasks),
                    "current_task": current.get("task"),
                    "env_step_limit": args.scienceworld_env_step_limit,
                },
            )

        def _handle_reset(self):
            try:
                data = _read_json(self)
                task = str(data.get("task") or data.get("task_name") or data.get("task_id") or "")
                variation_idx = int(data.get("variation_idx", data.get("variation", 0)) or 0)
                simplification = str(data.get("simplification", data.get("simplification_str", "")) or "")
                env.load(task, variation_idx, simplification)
                obs, info = env.reset()
                info = dict(info or {})
                info.setdefault("valid", env.get_valid_action_object_combinations())
                instruction = task_description(env, fallback=task)
                current.update({"task": task, "variation_idx": variation_idx, "simplification": simplification})
                _response(
                    self,
                    {
                        "observation": str(obs),
                        "instruction": instruction,
                        "score": score_from_info(info),
                        "done": False,
                        "info": info,
                        "task": task,
                        "variation_idx": variation_idx,
                        "simplification": simplification,
                    },
                )
            except Exception as exc:
                _response(self, {"error": repr(exc), "traceback": traceback.format_exc()}, status=500)

        def _handle_step(self):
            try:
                data = _read_json(self)
                action = str(data.get("action") or "")
                obs, reward, done, info = env.step(action)
                info = dict(info or {})
                _response(
                    self,
                    {
                        "observation": str(obs),
                        "reward": float(reward or 0.0),
                        "score": score_from_info(info, fallback=reward),
                        "done": bool(done),
                        "info": info,
                    },
                )
            except Exception as exc:
                _response(self, {"error": repr(exc), "traceback": traceback.format_exc()}, status=500)

        def _handle_valid_actions(self):
            try:
                _response(self, {"valid": env.get_valid_action_object_combinations()})
            except Exception as exc:
                _response(self, {"error": repr(exc), "traceback": traceback.format_exc()}, status=500)

        def _handle_close(self):
            try:
                env.close()
            except Exception:
                pass
            _response(self, {"ok": True})

    return ScienceWorldHandler


def parse_args():
    p = argparse.ArgumentParser(description="Serve ScienceWorld as an isolated HTTP environment.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8811)
    p.add_argument("--scienceworld-path", default=None, help="Path to a local ScienceWorld checkout; added to PYTHONPATH.")
    p.add_argument("--scienceworld-jar-path", default=None, help="Path to scienceworld.jar. Defaults to the package jar.")
    # The manifests include tasks with recommended_max_steps=120. This server
    # limit is only a hard ceiling; task-specific rollout budgets remain set by
    # each manifest row.
    p.add_argument("--scienceworld-env-step-limit", type=int, default=120)
    return p.parse_args()


def main():
    args = parse_args()
    env = make_env(
        scienceworld_path=args.scienceworld_path,
        jar_path=args.scienceworld_jar_path,
        env_step_limit=args.scienceworld_env_step_limit,
    )
    server = HTTPServer((args.host, int(args.port)), make_handler(env, args))
    print(f"[ScienceWorld Server] http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        try:
            env.close()
        except Exception:
            pass
        server.server_close()


if __name__ == "__main__":
    main()
