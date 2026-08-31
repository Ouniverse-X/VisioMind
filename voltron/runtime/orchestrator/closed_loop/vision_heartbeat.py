from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Callable

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.enums import AgentName, AgentStatus
from voltron.shared.results import AgentResult
from voltron.shared.telemetry.payload_sanitizer import strip_image_payloads

LOGGER = logging.getLogger(__name__)


class _DaemonHeartbeatExecutor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._shutdown = False
        self._threads: list[threading.Thread] = []

    def submit(
        self, fn: Callable[..., AgentResult | None], *args: Any
    ) -> concurrent.futures.Future[AgentResult | None]:
        future: concurrent.futures.Future[AgentResult | None] = concurrent.futures.Future()

        with self._lock:
            if self._shutdown:
                future.set_exception(RuntimeError("Vision heartbeat executor is shut down"))
                return future

            thread = threading.Thread(
                target=self._run,
                args=(future, fn, args),
                name="voltron-vision-heartbeat",
                daemon=True,
            )
            self._threads.append(thread)

        thread.start()
        return future

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        del cancel_futures
        with self._lock:
            self._shutdown = True
            threads = list(self._threads)

        if wait:
            for thread in threads:
                thread.join()

    @staticmethod
    def _run(
        future: concurrent.futures.Future[AgentResult | None],
        fn: Callable[..., AgentResult | None],
        args: tuple[Any, ...],
    ) -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)


class VisionHeartbeatRunner:
    def __init__(
        self,
        *,
        vision_agent: Any,
        interval_steps: int,
        emit_event: Callable[..., None] | None = None,
        max_in_flight: int = 1,
    ) -> None:
        self.vision_agent = vision_agent
        self.interval_steps = max(0, int(interval_steps))
        self.emit_event = emit_event
        del max_in_flight
        self._executor = _DaemonHeartbeatExecutor()
        self._future: concurrent.futures.Future[AgentResult | None] | None = None
        self._last_trigger_step = 0
        self._missed_count = 0
        self._active = False

    def start(self, *, context: ExecutionContext, environment: Any) -> None:
        del context, environment
        self._active = self.interval_steps > 0 and callable(
            getattr(self.vision_agent, "execute", None)
        )
        self._last_trigger_step = 0
        self._missed_count = 0

    def stop(self, *, flush: bool = True, timeout_s: float = 10.0) -> None:
        self._active = False
        future = self._future
        self._future = None
        wait_for_executor = bool(flush)
        cancel_futures = not flush
        if flush and future is not None:
            try:
                future.result(timeout=max(0.0, float(timeout_s)))
            except (TimeoutError, concurrent.futures.TimeoutError):
                future.cancel()
                wait_for_executor = False
                cancel_futures = True
                LOGGER.warning("Vision heartbeat did not finish before shutdown timeout")
            except Exception:
                LOGGER.exception("Vision heartbeat did not finish cleanly")
        self._executor.shutdown(wait=wait_for_executor, cancel_futures=cancel_futures)

    def on_environment_step(
        self,
        *,
        context: ExecutionContext,
        environment: Any,
        env_step: int,
        source_subtask: Subtask,
        feedback: dict[str, Any] | None = None,
    ) -> concurrent.futures.Future[AgentResult | None] | None:
        if not self._active:
            return None
        env_step = int(env_step)
        if env_step <= 0 or env_step - self._last_trigger_step < self.interval_steps:
            return None
        if self._future is not None and not self._future.done():
            self._missed_count += 1
            return None

        subtask = self._build_heartbeat_subtask(
            context=context,
            environment=environment,
            env_step=env_step,
            source_subtask=source_subtask,
            feedback=feedback or {},
        )
        self._last_trigger_step = env_step
        self._future = self._executor.submit(self._execute_heartbeat, subtask, context, env_step)
        return self._future

    def _build_heartbeat_subtask(
        self,
        *,
        context: ExecutionContext,
        environment: Any,
        env_step: int,
        source_subtask: Subtask,
        feedback: dict[str, Any],
    ) -> Subtask:
        parameters = {
            "env_step": env_step,
            "monitored_subtask_id": source_subtask.subtask_id,
            "monitored_agent": source_subtask.agent.value,
            "monitored_action": source_subtask.action,
            "target": dict(source_subtask.target),
            "environment_feedback": dict(feedback),
            "missed_heartbeats": self._missed_count,
            "allow_task_complete": False,
            "instruction": self._heartbeat_instruction(source_subtask=source_subtask),
        }
        self._missed_count = 0
        subtask = Subtask(
            subtask_id=f"{context.task_request.task_id}_vision_heartbeat_{env_step}",
            agent=AgentName.VISION,
            action="environment_heartbeat",
            target=dict(source_subtask.target),
            parameters=parameters,
        )
        runtime_inputs = environment.build_runtime_inputs(subtask, context)
        if isinstance(runtime_inputs, dict):
            subtask.parameters = {**parameters, **runtime_inputs}
        return subtask

    def _execute_heartbeat(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        env_step: int,
    ) -> AgentResult | None:
        execute = getattr(self.vision_agent, "execute", None)
        if not callable(execute):
            return None
        result = execute(subtask, context)
        if getattr(result, "status", None) == AgentStatus.SUCCESS:
            self._emit_success(subtask=subtask, context=context, result=result, env_step=env_step)
        return result

    def _emit_success(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        env_step: int,
    ) -> None:
        if self.emit_event is None:
            return
        try:
            self.emit_event(
                event_type="vision_heartbeat",
                source=AgentName.VISION.value,
                message=f"vision heartbeat at environment step {env_step}",
                payload={
                    "subtask_id": subtask.subtask_id,
                    "monitored_subtask_id": subtask.parameters.get("monitored_subtask_id"),
                    "monitored_agent": subtask.parameters.get("monitored_agent"),
                    "env_step": env_step,
                    "result": strip_image_payloads(dict(getattr(result, "result", {}))),
                },
                task_id=context.task_request.task_id,
            )
        except Exception:
            LOGGER.exception("Failed to emit Vision heartbeat event")

    @staticmethod
    def _heartbeat_instruction(*, source_subtask: Subtask) -> str:
        target = (
            source_subtask.target.get("object") or source_subtask.target.get("room") or "the task"
        )
        return (
            "Summarize the current environment state for task memory. "
            f"Focus on progress toward {target}, visible target state, and risks for the active "
            f"{source_subtask.agent.value} subtask. Do not decide whether the task or subtask is complete."
        )
