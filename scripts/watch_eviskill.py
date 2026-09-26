#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eviskill import add_generation_args
from wmsm.experiment_naming import default_dual_layer_output_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the EviSkill evolution pipeline with automatic restart. Pass pipeline args after --."
    )
    p.add_argument("--python", default=sys.executable, help="Python executable used to launch the generator.")
    p.add_argument("--script", default="scripts/run_eviskill.py")
    p.add_argument("--max-restarts", type=int, default=-1, help="-1 means unlimited restarts.")
    p.add_argument("--sleep-seconds", type=float, default=120.0)
    p.add_argument("--log-file", default=None)
    p.add_argument("generator_args", nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args = parse_args()
    generator_args = list(args.generator_args)
    if generator_args and generator_args[0] == "--":
        generator_args = generator_args[1:]
    if not generator_args:
        raise SystemExit("Pass EviSkill pipeline arguments after --")

    script = Path(args.script)
    if not script.is_absolute():
        script = ROOT / script
    generator_parser = argparse.ArgumentParser(add_help=False)
    add_generation_args(generator_parser)
    known_generator_args, _ = generator_parser.parse_known_args(generator_args)
    output_dir = known_generator_args.output_dir or str(default_dual_layer_output_dir(known_generator_args))
    if not known_generator_args.output_dir:
        generator_args = ["--output-dir", output_dir] + generator_args
    log_fh = None
    log_path = Path(args.log_file) if args.log_file else Path(output_dir) / "watch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", encoding="utf-8")
    print(f"[watch] output_dir={output_dir}", flush=True)
    print(f"[watch] log_file={log_path}", flush=True)

    restarts = 0
    while True:
        cmd = [args.python, str(script)] + generator_args
        msg = f"[{datetime.now().isoformat(timespec='seconds')}] launch restart={restarts}: {' '.join(cmd)}"
        print(msg, flush=True)
        if log_fh:
            print(msg, file=log_fh, flush=True)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        if log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                env=env,
            )
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(1)
                if chunk == "" and proc.poll() is not None:
                    break
                if not chunk:
                    continue
                print(chunk, end="", flush=True)
                print(chunk, end="", file=log_fh, flush=True)
            code = proc.wait()
        else:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
            code = proc.wait()
        done_msg = f"[{datetime.now().isoformat(timespec='seconds')}] exit code={code}"
        print(done_msg, flush=True)
        if log_fh:
            print(done_msg, file=log_fh, flush=True)
        if code == 0:
            return 0
        restarts += 1
        if args.max_restarts >= 0 and restarts > args.max_restarts:
            return code
        sleep_msg = f"sleep {args.sleep_seconds:.1f}s before resume"
        print(sleep_msg, flush=True)
        if log_fh:
            print(sleep_msg, file=log_fh, flush=True)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
