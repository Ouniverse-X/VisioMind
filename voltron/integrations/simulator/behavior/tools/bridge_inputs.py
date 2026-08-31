from __future__ import annotations

from typing import Any

from voltron.integrations.simulator.behavior.tools import (
    bridge_environment as behavior_bridge_environment,
)
from voltron.integrations.simulator.behavior.tools import (
    bridge_localization as behavior_bridge_localization,
)
from voltron.integrations.simulator.behavior.tools import runtime_inputs as behavior_runtime_inputs
from voltron.integrations.simulator.behavior.tools import (
    scene_state_sampling as behavior_scene_state_sampling,
)
from voltron.integrations.simulator.behavior.tools.camera_capture import (
    BehaviorCameraCaptureAdapter,
)


def _sample_scene_state(runtime: Any, *, scene_id: str | None) -> dict[str, Any] | None:
    try:
        return behavior_scene_state_sampling.sample_scene_runtime_state(runtime, scene_id=scene_id)
    except Exception:
        return None


def build_runtime_inputs(runtime: Any, *, subtask: Any) -> dict[str, Any]:
    runtime_subtask = runtime._sync_runtime_subtask(subtask)
    environment_vlm_heartbeat = behavior_bridge_environment.read_environment_vlm_heartbeat(runtime)
    camera_capture = BehaviorCameraCaptureAdapter(lambda: runtime._last_obs)
    return behavior_runtime_inputs.build_runtime_inputs_for_subtask(
        subtask=subtask,
        runtime_subtask=runtime_subtask,
        last_obs=runtime._last_obs,
        last_info=runtime._last_info,
        environment_vlm_heartbeat=environment_vlm_heartbeat,
        camera_capture=camera_capture,
        run_dir=getattr(runtime, "_record_dir", None),
        build_vision_inputs=behavior_runtime_inputs.build_vision_runtime_inputs,
        build_action_inputs=behavior_runtime_inputs.build_action_runtime_inputs,
        build_navigation_inputs=lambda subtask,
        observation: behavior_bridge_localization.build_vln_runtime_inputs(
            subtask=subtask,
            observation=observation,
            env_kwargs=runtime.env_kwargs,
            last_info=runtime._last_info,
            last_obs=runtime._last_obs,
            navigation_runtime_state=runtime._navigation_runtime_state,
            scene_id=behavior_bridge_localization.extract_scene_id(
                last_info=runtime._last_info,
                last_obs=runtime._last_obs,
                scene_id=runtime._scene_id,
            ),
            scene_state=_sample_scene_state(
                runtime,
                scene_id=behavior_bridge_localization.extract_scene_id(
                    last_info=runtime._last_info,
                    last_obs=runtime._last_obs,
                    scene_id=runtime._scene_id,
                ),
            ),
        ),
        extract_images_b64=behavior_bridge_localization.extract_images_b64,
        to_policy_observation=behavior_bridge_localization.to_policy_observation,
        policy_observation_source=lambda language_instruction=None: behavior_bridge_localization.build_policy_observation_source(
            last_obs=runtime._last_obs,
            root_task_instruction=runtime._root_task_instruction,
            language_instruction=language_instruction,
        ),
        instruction_for_subtask=runtime._instruction_for_subtask,
        policy_backend=getattr(runtime, "_policy_backend", None),
        env_id=runtime.env_id,
    )
