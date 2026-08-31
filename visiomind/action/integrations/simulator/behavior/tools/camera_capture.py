from __future__ import annotations

from typing import Any, Callable

from visiomind.action.shared.contracts import CameraFrame

VIEW_TO_OBS_KEY = {
    "head": "video.observation.images.rgb.head_256_256",
    "left_wrist": "video.observation.images.rgb.left_wrist_256_256",
    "right_wrist": "video.observation.images.rgb.right_wrist_256_256",
}


class BehaviorCameraCaptureAdapter:
    def __init__(self, observation_provider: Callable[[], dict[str, Any]]) -> None:
        self._observation_provider = observation_provider

    def capture(self, views: list[str]) -> dict[str, CameraFrame]:
        obs = self._observation_provider()
        frames: dict[str, CameraFrame] = {}
        for view in views:
            key = VIEW_TO_OBS_KEY.get(view)
            if key is None or key not in obs:
                continue
            frames[view] = CameraFrame(view=view, data=obs[key])
        return frames
