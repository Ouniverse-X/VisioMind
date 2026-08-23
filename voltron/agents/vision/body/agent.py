"""Vision agent for scene understanding and semantic memory updates."""

from __future__ import annotations

import time
from typing import Any

from voltron.agents.vision.tools import VisionPhotoCaptureTool, scene_report
from voltron.shared.enums import AgentName, AgentStatus
from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.results import AgentResult
from voltron.shared.contracts import AgentCapability, MemoryAdapter, ToolInvocation, VisionAdapter
from voltron.shared.registries import ToolCatalog
from voltron.shared.results import ToolResult


class VisionAgent:
    """Execute visual perception subtasks and write structured observations."""

    def __init__(self, memory: MemoryAdapter, vision: VisionAdapter, tools: ToolCatalog | None = None):
        self.memory = memory
        self.vision = vision
        self.tools = tools or self._build_default_tools()

    @staticmethod
    def _build_default_tools() -> ToolCatalog:
        catalog = ToolCatalog()
        photo_capture_tool = VisionPhotoCaptureTool()
        for tool_name in photo_capture_tool.tool_names:
            catalog.register(tool_name, photo_capture_tool)
        return catalog

    def capability_manifest(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                capability_id="vision.photo.capture",
                agent=AgentName.VISION,
                kind="tool",
                action_names=("take_photo", "capture_photo"),
                description="Capture robot camera photos from requested views without VLM analysis.",
                intent_examples=(
                    "take a photo from the head camera",
                    "capture photos from the wrist cameras",
                ),
                input_schema={"views": ["head", "left_wrist", "right_wrist"]},
                output_schema={
                    "photos": [
                        {
                            "view": "string",
                            "path": "string",
                            "mime_type": "image/png",
                        }
                    ]
                },
            )
        ]

    def execute(self, subtask: Subtask, context: ExecutionContext) -> AgentResult:
        start = time.time()
        normalized_action = str(subtask.action or "").strip().lower()

        if normalized_action in {"take_photo", "capture_photo"}:
            return self._execute_tool_subtask(subtask=subtask, context=context, start=start)

        if self._is_environment_heartbeat_subtask(subtask):
            return self._execute_environment_heartbeat(subtask=subtask, context=context, start=start)

        images = subtask.parameters.get("images")
        if not isinstance(images, list) or not images:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="VLM_INPUT_MISSING",
                result={"message": "subtask.parameters['images'] is required"},
                latency_ms=self._latency_ms(start),
            )

        instruction = str(subtask.parameters.get("instruction", subtask.action))

        allow_task_complete = scene_report.allow_task_complete(subtask)

        try:
            report = self.vision.analyze(
                images_b64=images,
                instruction=instruction,
                task_name=context.task_request.task_id,
                image_view_order=subtask.parameters.get("image_view_order"),
            )
            memory_stats = self.memory.record_perception(report)
            task_complete = bool(report.task_complete) if allow_task_complete else False
            structured_scene_report = scene_report.build_scene_report(
                report=report,
                subtask=subtask,
                task_complete=task_complete,
            )
        except Exception as exc:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code=scene_report.classify_error_code(exc),
                result={"message": str(exc)},
                latency_ms=self._latency_ms(start),
            )

        result_payload = {
            "task_complete": task_complete,
            "objects": [obj.name for obj in report.objects],
            "relations": [rel.relation for rel in report.relations],
            "scene_report": structured_scene_report,
            "memory_update": memory_stats,
            "raw_text": report.raw_text,
        }
        environment_feedback = subtask.parameters.get("environment_feedback")
        if isinstance(environment_feedback, dict):
            result_payload["environment_feedback"] = dict(environment_feedback)
        environment_vlm_heartbeat = subtask.parameters.get("environment_vlm_heartbeat")
        if isinstance(environment_vlm_heartbeat, dict):
            result_payload["environment_vlm_heartbeat"] = dict(environment_vlm_heartbeat)

        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS,
            result=result_payload,
            latency_ms=self._latency_ms(start),
        )

    def run_episode(self, *, subtask: Subtask, context: ExecutionContext, runtime: Any) -> AgentResult:
        """Run a complete perception episode using runtime-provided observations."""

        static_parameters = dict(subtask.parameters)
        last_result: AgentResult | None = None

        for control_step in range(1, int(runtime.max_control_steps) + 1):
            runtime.prepare_control_step(
                subtask=subtask,
                context=context,
                static_parameters=static_parameters,
                control_step=control_step,
            )
            result = self.execute(subtask, context)
            last_result = runtime.publish_agent_result(
                subtask=subtask,
                context=context,
                result=result,
                control_step=control_step,
            )

            if result.status == AgentStatus.FAILURE:
                if hasattr(runtime, "record_agent_failure"):
                    runtime.record_agent_failure(
                        subtask=subtask,
                        context=context,
                        result=result,
                        failure_reason=result.error_code or "AGENT_FAILURE",
                    )
                return result

            step_outcome = runtime.apply_agent_result(subtask=subtask, result=result, context=context)
            if getattr(step_outcome, "feedback", None):
                runtime.update_feedback(
                    subtask=subtask,
                    context=context,
                    result=result,
                    control_step=control_step,
                    feedback=step_outcome.feedback,
                )
            if not bool(getattr(step_outcome, "done", False)):
                continue

            if getattr(step_outcome, "success", None) is False:
                return runtime.environment_failure_result(
                    subtask=subtask,
                    context=context,
                    result=result,
                    control_step=control_step,
                    feedback=getattr(step_outcome, "feedback", {}),
                    failure_reason=getattr(step_outcome, "failure_reason", None),
                )

            if hasattr(runtime, "record_agent_success"):
                runtime.record_agent_success(subtask=subtask, context=context, result=result)
            return result

        if last_result is not None and hasattr(runtime, "record_agent_failure"):
            runtime.record_agent_failure(
                subtask=subtask,
                context=context,
                result=last_result,
                failure_reason="SUBTASK_TIMEOUT",
            )
        return runtime.timeout_result(subtask=subtask)

    @staticmethod
    def _latency_ms(start: float) -> int:
        return int((time.time() - start) * 1000)

    def _execute_tool_subtask(self, *, subtask: Subtask, context: ExecutionContext, start: float) -> AgentResult:
        tool_name = "vision.photo.capture"
        invocation = ToolInvocation(
            tool_name=tool_name,
            payload={
                "views": subtask.parameters.get("views"),
                "camera_capture": subtask.parameters.get("camera_capture"),
                "run_dir": subtask.parameters.get("run_dir") or context.runtime_state.get("run_dir"),
                "trace_id": context.trace_id,
                "subtask_id": subtask.subtask_id,
            },
        )

        try:
            tool = self.tools.get(tool_name)
        except KeyError:
            tool_result = ToolResult(
                tool_name=tool_name,
                ok=False,
                payload={"message": f"Unknown Vision tool {tool_name!r}"},
                error_code="unknown_tool",
            )
        else:
            if hasattr(tool, "invoke"):
                tool_result = tool.invoke(invocation)
            elif callable(tool):
                tool_result = tool(invocation)
            else:
                tool_result = ToolResult(
                    tool_name=tool_name,
                    ok=False,
                    payload={"message": f"Registered Vision tool {tool_name!r} is not invokable"},
                    error_code="invalid_tool",
                )

        result_payload = {
            "tool_result": {
                "tool_name": tool_result.tool_name,
                "ok": tool_result.ok,
                "error_code": tool_result.error_code,
            },
            **tool_result.payload,
        }
        if tool_result.ok:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.SUCCESS,
                result=result_payload,
                latency_ms=self._latency_ms(start),
            )
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.FAILURE,
            error_code=self._tool_error_code(tool_result.error_code),
            result=result_payload,
            latency_ms=self._latency_ms(start),
        )

    @staticmethod
    def _tool_error_code(error_code: str | None) -> str:
        return {
            "camera_adapter_missing": "VISION_TOOL_CAMERA_ADAPTER_MISSING",
            "camera_view_unavailable": "VISION_TOOL_CAMERA_UNAVAILABLE",
            "camera_capture_failed": "VISION_TOOL_CAMERA_CAPTURE_FAILED",
            "invalid_camera_frame": "VISION_TOOL_INVALID_CAMERA_FRAME",
            "photo_write_failed": "VISION_TOOL_WRITE_FAILED",
        }.get(error_code or "", "VISION_TOOL_FAILED")

    @staticmethod
    def _is_environment_heartbeat_subtask(subtask: Subtask) -> bool:
        action = str(subtask.action or "").strip().lower()
        return action in {"environment_heartbeat", "monitor_environment"}

    def _execute_environment_heartbeat(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        start: float,
    ) -> AgentResult:
        images = subtask.parameters.get("images")
        if not isinstance(images, list) or not images:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="VLM_INPUT_MISSING",
                result={"message": "subtask.parameters['images'] is required"},
                latency_ms=self._latency_ms(start),
            )

        instruction = str(subtask.parameters.get("instruction") or "Summarize current environment state.")
        try:
            report = self.vision.analyze(
                images_b64=images,
                instruction=instruction,
                task_name=context.task_request.task_id,
                image_view_order=subtask.parameters.get("image_view_order"),
            )
        except Exception as exc:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code=scene_report.classify_error_code(exc),
                result={"message": str(exc)},
                latency_ms=self._latency_ms(start),
            )

        task_complete = bool(report.task_complete)
        result_payload = {
            "task_complete": task_complete,
            "objects": [obj.name for obj in report.objects],
            "relations": [rel.relation for rel in report.relations],
            "scene_report": {"task_complete": task_complete},
            "memory_update": {},
            "raw_text": report.raw_text,
        }
        environment_feedback = subtask.parameters.get("environment_feedback")
        if isinstance(environment_feedback, dict):
            result_payload["environment_feedback"] = dict(environment_feedback)

        monitor_summary = self._record_environment_heartbeat_summary(
            subtask=subtask,
            context=context,
            report=report,
            task_complete=task_complete,
        )
        if monitor_summary is not None:
            result_payload["monitor_summary"] = monitor_summary

        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS,
            result=result_payload,
            latency_ms=self._latency_ms(start),
        )

    def _record_environment_heartbeat_summary(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        report: Any,
        task_complete: bool,
    ) -> dict[str, Any] | None:
        recorder = getattr(self.memory, "record_monitor_summary", None)
        if not callable(recorder):
            return None

        payload = self._build_environment_heartbeat_summary_payload(
            subtask=subtask,
            context=context,
            report=report,
            task_complete=task_complete,
        )
        recorded = recorder(payload)
        if isinstance(recorded, dict):
            return {"recorded": bool(recorded.get("recorded", False)), **recorded, "payload": payload}
        return {"recorded": False, "payload": payload}

    @staticmethod
    def _build_environment_heartbeat_summary_payload(
        *,
        subtask: Subtask,
        context: ExecutionContext,
        report: Any,
        task_complete: bool,
    ) -> dict[str, Any]:
        monitored_subtask_id = str(
            subtask.parameters.get("monitored_subtask_id") or subtask.subtask_id
        ).strip()
        monitored_agent = str(subtask.parameters.get("monitored_agent") or "").strip() or None
        target = VisionAgent._monitor_target(subtask)
        env_step = VisionAgent._positive_int(subtask.parameters.get("env_step"))
        environment_feedback = subtask.parameters.get("environment_feedback")
        result = "success" if task_complete else "running"
        payload = {
            "subtask_id": monitored_subtask_id,
            "agent": AgentName.VISION.value,
            "monitored_agent": monitored_agent,
            "env_step": env_step,
            "summary": str(getattr(report, "raw_text", "") or "").strip(),
            "target": target,
            "result": result,
            "observed_objects": [obj.name for obj in getattr(report, "objects", [])],
        }
        if isinstance(environment_feedback, dict):
            payload["environment_feedback"] = dict(environment_feedback)
        return payload

    @staticmethod
    def _monitor_target(subtask: Subtask) -> str | None:
        for source in (subtask.target, subtask.parameters.get("target")):
            if isinstance(source, dict):
                for key in ("object", "object_name", "target", "name"):
                    value = source.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            elif isinstance(source, str) and source.strip():
                return source.strip()
        return None

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed
