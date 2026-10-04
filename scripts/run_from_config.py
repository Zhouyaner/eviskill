#!/usr/bin/env python3
"""Run EviSkill from one reproducible YAML experiment configuration.

The YAML files contain the benchmark protocol and machine-specific paths. API
credentials are intentionally resolved from the process environment by the
existing LLM client and are never written to the resolved configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")
_PLACEHOLDER = re.compile(r"(?:^|/)(?:path|your|replace[-_ ]?me)(?:/|$)", re.I)
_SECRET = re.compile(r"(?:key|token|secret|password)", re.I)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, fallback = match.group(1), match.group(2)
            if name in variables and str(variables[name]) != "":
                return str(variables[name])
            if fallback is not None:
                return fallback
            raise ValueError(
                f"configuration references ${name}, but {name} is not set"
            )

        previous = value
        for _ in range(8):
            expanded = _VAR.sub(replace, previous)
            if expanded == previous:
                return expanded
            previous = expanded
        return previous
    if isinstance(value, list):
        return [_expand(item, variables) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand(item, variables) for key, item in value.items()}
    return value


def _as_cli_name(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def _parser_actions() -> Dict[str, argparse.Action]:
    # Import lazily so --help for this launcher remains useful even when the
    # package is being inspected before optional runtime dependencies are built.
    from scripts.run_eviskill import add_generation_args

    parser = argparse.ArgumentParser(add_help=False)
    add_generation_args(parser)
    actions: Dict[str, argparse.Action] = {}
    for action in parser._actions:
        actions[action.dest] = action
        for option in action.option_strings:
            if option.startswith("--"):
                actions[option[2:].replace("-", "_")] = action
    return actions


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _arguments_to_cli(arguments: Mapping[str, Any]) -> list[str]:
    actions = _parser_actions()
    command: list[str] = []
    for dest, value in arguments.items():
        if dest not in actions:
            raise ValueError(
                f"unknown pipeline argument {dest!r}; use a destination from scripts/run_eviskill.py"
            )
        action = actions[dest]
        requested_option = _as_cli_name(dest)
        option = (
            requested_option
            if requested_option in action.option_strings
            else next(
                (flag for flag in action.option_strings if flag.startswith("--")),
                requested_option,
            )
        )
        if value is None or value is False:
            # BooleanOptionalAction needs an explicit negative flag when the
            # config intentionally differs from its parser default.
            if value is False and isinstance(action, argparse.BooleanOptionalAction):
                negative = next(
                    (flag for flag in action.option_strings if flag.startswith("--no-")),
                    None,
                )
                if negative:
                    command.append(negative)
            continue
        if value is True:
            command.append(option)
            continue
        if isinstance(value, (dict, list)):
            value = _json_value(value)
        command.extend([option, str(value)])
    return command


def _redact(value: Any, key: str = "") -> Any:
    if _SECRET.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {name: _redact(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    return value


def _check_placeholders(arguments: Mapping[str, Any], allow: bool) -> None:
    if allow:
        return
    bad = []
    for key, value in arguments.items():
        if isinstance(value, str) and _PLACEHOLDER.search(value):
            bad.append(f"{key}={value}")
    if bad:
        raise ValueError(
            "configuration still contains placeholder paths; edit the YAML or use "
            "environment overrides before running: " + ", ".join(bad)
        )


def _load_config(path: Path, profile: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"top-level YAML value must be a mapping: {path}")
    profiles = raw.get("profiles") or {}
    if profile not in profiles:
        names = ", ".join(sorted(profiles))
        raise ValueError(f"unknown profile {profile!r}; available profiles: {names}")

    configured_env = raw.get("environment") or {}
    if not isinstance(configured_env, Mapping):
        raise ValueError("environment must be a mapping")
    variables = dict(os.environ)
    # Resolve config-provided environment values in declaration order. This
    # permits ALFWORLD_DATA or a server path to be referenced by a path field.
    for key, value in configured_env.items():
        if _SECRET.search(str(key)):
            raise ValueError(
                f"secret-like environment key {key!r} must be supplied by the shell, not YAML"
            )
        resolved = _expand(value, variables)
        if not isinstance(resolved, (str, int, float, bool)):
            raise ValueError(f"environment value {key!r} must be scalar")
        variables[str(key)] = str(resolved)

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("defaults must be a mapping")
    profile_data = profiles[profile]
    if not isinstance(profile_data, Mapping):
        raise ValueError(f"profile {profile!r} must be a mapping")

    arguments = _deep_merge(
        _deep_merge(defaults.get("arguments") or {}, raw.get("paths") or {}),
        profile_data.get("arguments") or {},
    )
    # Useful interpolation values are explicit and do not alter the process
    # environment seen by the benchmark.
    variables.update({"CONFIG_DIR": str(path.parent), "PROFILE": profile})
    arguments = _expand(arguments, variables)
    for key, value in arguments.items():
        if _SECRET.search(str(key)) and str(key).endswith("_api_key"):
            if value not in (None, "local"):
                raise ValueError(
                    f"{key} must be omitted (or set to 'local'); provide credentials through the process environment"
                )
    watcher = _expand(defaults.get("watcher") or {}, variables)
    watcher = _deep_merge(watcher, _expand(profile_data.get("watcher") or {}, variables))
    resolved = {
        "schema_version": raw.get("schema_version", 1),
        "source_config": str(path),
        "profile": profile,
        "provenance": raw.get("provenance") or {},
        "environment": {str(k): variables[str(k)] for k in configured_env},
        "arguments": arguments,
        "watcher": watcher,
    }
    return resolved, {str(k): variables[str(k)] for k in configured_env}


def _write_snapshot(output_dir: Path, resolved: Mapping[str, Any], command: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(_redact(resolved), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_command.txt").write_text(
        shlex.join(list(command)) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--profile", required=True, help="Model profile in the YAML")
    parser.add_argument("--python", default=sys.executable, help="Python used by the watcher")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    parser.add_argument("--allow-placeholders", action="store_true", help="Allow /path/to placeholders for command inspection")
    parser.add_argument("--no-watch", action="store_true", help="Run run_eviskill.py directly without the restart watcher")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra pipeline flags after --")
    args = parser.parse_args()
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    if not config_path.exists():
        raise SystemExit(f"configuration does not exist: {config_path}")

    try:
        resolved, configured_env = _load_config(config_path, args.profile)
        arguments = resolved["arguments"]
        _check_placeholders(arguments, args.allow_placeholders)
        pipeline = _arguments_to_cli(arguments) + extra
        # Parse the final command once to catch missing required arguments and
        # misspelled flags before a long-running benchmark starts.
        from scripts.run_eviskill import add_generation_args

        check = argparse.ArgumentParser(add_help=False)
        add_generation_args(check)
        check.parse_args(pipeline)
    except (ValueError, yaml.YAMLError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        raise SystemExit(f"invalid EviSkill configuration: {exc}") from exc

    watcher = resolved.get("watcher") or {}
    if args.no_watch:
        command = [args.python, str(ROOT / "scripts" / "run_eviskill.py"), *pipeline]
    else:
        command = [args.python, str(ROOT / "scripts" / "watch_eviskill.py")]
        for key, flag in (("max_restarts", "--max-restarts"), ("sleep_seconds", "--sleep-seconds"), ("log_file", "--log-file")):
            if watcher.get(key) is not None:
                command.extend([flag, str(watcher[key])])
        command.extend(["--python", args.python, "--", *pipeline])

    output_dir = Path(str(arguments["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    _write_snapshot(output_dir, resolved, command)
    for key, value in configured_env.items():
        os.environ[key] = value

    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(ROOT), env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
