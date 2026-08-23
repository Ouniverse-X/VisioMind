"""User-facing Voltron session TUI for closed-loop task execution."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator, TextIO

from voltron.agents.brain.body.planner_backend import OpenAICompatiblePlanner, OpenAIPlannerConfig
from voltron.agents.brain.body.rule_based_planner import RuleBasedPlanner
from voltron.config_loader import parse_args_with_config
from voltron.entrypoints.examples.closed_loop import main as closed_loop_main
from voltron.runtime.session.voltron_session import (
    VoltronEvent,
    build_configured_voltron_session,
    build_mock_voltron_session,
)


_RESET = "\033[0m"
_COLORS = {
    "USER": "\033[36m",
    "BRAIN": "\033[35m",
    "PLAN": "\033[1m",
    "DISPATCH": "\033[33m",
    "RESULT": "\033[32m",
    "MEMORY": "\033[34m",
    "FINAL": "\033[1m",
    "ERROR": "\033[31m",
    "SYSTEM": "\033[2m",
}


@dataclass
class VoltronTuiRenderer:
    use_color: bool = True
    stream: TextIO | None = None
    event_detail: str = "summary"
    _seen_agent_results: set[tuple[str | None, str | None, str | None]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.stream is None:
            self.stream = sys.stdout

    def write(self, label: str, text: str) -> None:
        assert self.stream is not None
        self.stream.write(self._format(label, text) + "\n")
        self.stream.flush()

    def write_event(self, event: VoltronEvent) -> None:
        assert self.stream is not None
        line = self.render_event(event)
        if line is None:
            return
        self.stream.write(line + "\n")
        self.stream.flush()

    def render_event(self, event: VoltronEvent) -> str | None:
        if event.event_type == "user_command":
            return self._format("USER", event.message)
        if event.event_type.startswith("brain_") or event.event_type == "brain_plan":
            return self._format("BRAIN", _render_brain_event(event))
        if event.event_type == "subtask_dispatch":
            return self._format("DISPATCH", event.message)
        if event.event_type == "agent_result":
            if not self._should_render_agent_result(event):
                return None
            return self._format("RESULT", _render_agent_result(event))
        if event.event_type in {"context_patch", "brain_context_patch"}:
            return self._format("MEMORY", event.message)
        if event.event_type == "task_final":
            return self._format("FINAL", event.message)
        if event.event_type == "environment_reset":
            return self._format("SYSTEM", "environment reset")
        return self._format("SYSTEM", event.message)

    def _format(self, label: str, text: str) -> str:
        prefix = f"[{label}]"
        if not self.use_color:
            return f"{prefix} {text}"
        return f"{_COLORS.get(label, '')}{prefix}{_RESET} {text}"

    def _should_render_agent_result(self, event: VoltronEvent) -> bool:
        if self.event_detail == "full":
            return True
        status = str(event.payload.get("status") or "").lower()
        if status and status != "success":
            return True
        key = (
            event.task_id,
            str(event.payload.get("subtask_id") or ""),
            str(event.payload.get("agent") or event.source or ""),
        )
        if key in self._seen_agent_results:
            return False
        self._seen_agent_results.add(key)
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Voltron closed-loop session console")
    parser.add_argument("--runtime", choices=("mock", "behavior"), default="mock")
    parser.add_argument(
        "--config",
        default=None,
        help="Closed-loop JSON config path. Used by --runtime behavior.",
    )
    parser.add_argument("--planner", choices=("rule", "openai"), default="rule")
    parser.add_argument("--brain-base-url", default=os.getenv("VOLTRON_OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--brain-model", default=os.getenv("VOLTRON_OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--brain-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--brain-timeout-s", type=float, default=30.0)
    parser.add_argument("--brain-temperature", type=float, default=0.1)
    parser.add_argument("--brain-max-retries", type=int, default=0)
    parser.add_argument("--task-type", choices=("manipulation", "navigation", "interaction", "observation"), default="interaction")
    parser.add_argument("--radio-demo", action="store_true", default=True)
    parser.add_argument("--no-radio-demo", action="store_false", dest="radio_demo")
    _add_runtime_log_args(parser)
    _add_event_detail_args(parser)
    parser.add_argument("--no-color", action="store_true")
    return parser


def parse_tui_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace | None]:
    bootstrap = _build_runtime_bootstrap_parser()
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    if bootstrap_args.runtime != "behavior":
        return build_parser().parse_args(argv), None

    tui_args, runtime_argv = _build_behavior_tui_parser().parse_known_args(argv)
    if tui_args.config:
        runtime_argv.extend(["--config", tui_args.config])
    runtime_args = parse_args_with_config(closed_loop_main.build_parser(), runtime_argv)
    return tui_args, runtime_args


def main(argv: list[str] | None = None) -> int:
    args, runtime_args = parse_tui_args(argv)
    if args.runtime == "behavior" and runtime_args is not None:
        _apply_behavior_tui_defaults(args, runtime_args)
    session = None
    renderer = VoltronTuiRenderer(
        use_color=not args.no_color,
        stream=_build_tui_stream(args),
        event_detail=getattr(args, "event_detail", "summary"),
    )
    try:
        if args.runtime == "behavior":
            if runtime_args is None:
                raise RuntimeError("behavior runtime args were not parsed")
            with _route_runtime_output(args):
                session = build_configured_voltron_session(runtime_args, event_sink=renderer.write_event)
            default_task_type = runtime_args.task_type
        else:
            session = build_mock_voltron_session(
                event_sink=renderer.write_event,
                planner=build_planner(args),
                radio_demo=args.radio_demo,
            )
            default_task_type = args.task_type

        renderer.write("SYSTEM", "Voltron session TUI. Type /help for commands, /quit to exit.")
        while True:
            try:
                user_text = input("voltron> ").strip()
            except (EOFError, KeyboardInterrupt):
                renderer.write("SYSTEM", "exit")
                return 0
            if not user_text:
                continue
            if user_text in {"/quit", "/exit"}:
                return 0
            if user_text == "/help":
                renderer.write("SYSTEM", "Enter a task request. Commands: /help, /quit")
                continue
            try:
                with _route_runtime_output(args):
                    session.run_user_command(user_text, task_type=default_task_type)
            except Exception as exc:
                renderer.write("ERROR", f"{type(exc).__name__}: {exc}")
        return 0
    finally:
        if session is not None and hasattr(session, "close"):
            with _route_runtime_output(args):
                session.close()


def build_planner(args: argparse.Namespace) -> Any:
    if args.planner == "openai":
        return OpenAICompatiblePlanner(
            OpenAIPlannerConfig(
                base_url=args.brain_base_url,
                model=args.brain_model,
                api_key_env=args.brain_api_key_env,
                timeout_s=float(args.brain_timeout_s),
                temperature=float(args.brain_temperature),
                max_retries=int(args.brain_max_retries),
            )
        )
    return RuleBasedPlanner()


def _build_runtime_bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime", choices=("mock", "behavior"), default="mock")
    return parser


def _build_behavior_tui_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime", choices=("mock", "behavior"), default="mock")
    parser.add_argument("--config", default=None)
    _add_runtime_log_args(parser)
    _add_event_detail_args(parser)
    parser.add_argument("--no-color", action="store_true")
    return parser


def _add_runtime_log_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-log-mode",
        choices=("quiet", "file", "tui"),
        default="quiet",
        help="Where to send raw runtime stdout/stderr. TUI events are always shown.",
    )
    parser.add_argument(
        "--runtime-log-file",
        default=None,
        help="Runtime log file used when --runtime-log-mode=file.",
    )
    parser.add_argument(
        "--show-runtime-logs",
        action="store_const",
        const="tui",
        dest="runtime_log_mode",
        help="Alias for --runtime-log-mode=tui.",
    )


def _add_event_detail_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--event-detail",
        choices=("summary", "full"),
        default="summary",
        help="summary hides repeated per-control-step agent_result events; full shows every event.",
    )
    parser.add_argument(
        "--show-step-results",
        action="store_const",
        const="full",
        dest="event_detail",
        help="Alias for --event-detail=full.",
    )


def _apply_behavior_tui_defaults(args: argparse.Namespace, runtime_args: argparse.Namespace) -> None:
    if getattr(args, "runtime_log_mode", "quiet") == "quiet":
        runtime_args.progress_log_every = 0


def _build_tui_stream(args: argparse.Namespace) -> TextIO:
    if getattr(args, "runtime_log_mode", "quiet") == "tui":
        return sys.stdout
    try:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        return os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding=encoding, errors="replace")
    except (AttributeError, OSError):
        return sys.stdout


@contextmanager
def _route_runtime_output(args: argparse.Namespace) -> Iterator[None]:
    mode = getattr(args, "runtime_log_mode", "quiet")
    if mode == "tui":
        yield
        return

    target = _open_runtime_log_target(args)
    try:
        if _can_redirect_standard_fds(target):
            with _redirect_standard_fds(target):
                yield
        else:
            with redirect_stdout(target), redirect_stderr(target):
                yield
    finally:
        target.close()


def _open_runtime_log_target(args: argparse.Namespace) -> TextIO:
    if getattr(args, "runtime_log_mode", "quiet") == "file":
        path = Path(getattr(args, "runtime_log_file", None) or "logs/voltron_tui_runtime.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8", buffering=1)
    return open(os.devnull, "w", encoding="utf-8")


def _can_redirect_standard_fds(target: TextIO) -> bool:
    try:
        sys.stdout.fileno()
        sys.stderr.fileno()
        target.fileno()
    except (AttributeError, OSError):
        return False
    return True


@contextmanager
def _redirect_standard_fds(target: TextIO) -> Iterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(target.fileno(), 1)
        os.dup2(target.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _render_brain_event(event: VoltronEvent) -> str:
    if event.event_type == "brain_plan":
        subtasks = event.payload.get("subtasks")
        if isinstance(subtasks, list) and subtasks:
            return f"{event.message}: " + "; ".join(
                f"{item.get('subtask_id')}:{item.get('agent')}:{item.get('action')}"
                for item in subtasks
                if isinstance(item, dict)
            )
    thinking = event.payload.get("thinking_summary")
    if thinking:
        return str(thinking)
    return event.message


def _render_agent_result(event: VoltronEvent) -> str:
    status = event.payload.get("status")
    subtask_id = event.payload.get("subtask_id")
    agent = event.payload.get("agent") or event.source
    result = event.payload.get("result", {})
    compact = _compact_json(result)
    control_step = event.payload.get("control_step")
    step_text = f" step={control_step}" if control_step is not None else ""
    return f"{agent} {subtask_id}{step_text} status={status} {compact}"


def _compact_json(value: Any, *, max_len: int = 260) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
