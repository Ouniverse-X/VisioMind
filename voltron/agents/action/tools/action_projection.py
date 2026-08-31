from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbodimentActionSpec:
    navigation_keys: tuple[str, ...]
    manipulation_keys: tuple[str, ...]


_DEFAULT_SPECS: dict[str, EmbodimentActionSpec] = {
    "behavior_r1_pro": EmbodimentActionSpec(
        navigation_keys=("base",),
        manipulation_keys=(
            "torso",
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ),
    ),
    "unitree_g1": EmbodimentActionSpec(
        navigation_keys=("navigate_command", "base_height_command", "waist"),
        manipulation_keys=("left_arm", "right_arm", "left_hand", "right_hand"),
    ),
}


class ActionProjection:
    def __init__(self, spec: EmbodimentActionSpec):
        self.spec = spec
        self._last_safe_action: dict[str, Any] = {}

    @classmethod
    def from_embodiment(cls, embodiment_tag: str) -> "ActionProjection":
        spec = _DEFAULT_SPECS.get(embodiment_tag)
        if spec is None:
            raise ValueError(
                f"No action projection spec for embodiment '{embodiment_tag}'. "
                "Please register one in _DEFAULT_SPECS."
            )
        return cls(spec)

    def update_last_safe_action(self, action: dict[str, Any]) -> None:
        self._last_safe_action = {k: self._clone_value(v) for k, v in action.items()}

    def project_navigation(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._project(
            action, allowed_keys=set(self.spec.navigation_keys), freeze_non_owned=True
        )

    def project_manipulation(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._project(
            action, allowed_keys=set(self.spec.manipulation_keys), freeze_non_owned=False
        )

    def _project(
        self,
        action: dict[str, Any],
        allowed_keys: set[str],
        *,
        freeze_non_owned: bool,
    ) -> dict[str, Any]:
        projected: dict[str, Any] = {}
        normalized_allowed_keys = set(allowed_keys)
        normalized_allowed_keys.update(f"action.{key}" for key in allowed_keys)

        owned_present = [k for k in action if k in normalized_allowed_keys]
        if not owned_present:
            raise ValueError(
                f"No owned action keys found. allowed={sorted(allowed_keys)}, "
                f"actual={sorted(action.keys())}"
            )

        for key, value in action.items():
            if key in normalized_allowed_keys:
                projected[key] = self._clone_value(value)
                continue

            if freeze_non_owned and key in self._last_safe_action:
                projected[key] = self._clone_value(self._last_safe_action[key])
            else:
                projected[key] = self._zero_like(value)

        return projected

    @staticmethod
    def _clone_value(value: Any) -> Any:
        if hasattr(value, "clone"):
            return value.clone()

        if hasattr(value, "copy"):
            return value.copy()
        return value

    @staticmethod
    def _zero_like(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return np.zeros_like(value)
        if hasattr(value, "new_zeros") and hasattr(value, "shape"):
            return value.new_zeros(value.shape)
        if isinstance(value, list):
            return [ActionProjection._zero_like(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ActionProjection._zero_like(item) for item in value)
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return type(value)(0)
        return ActionProjection._clone_value(value)
