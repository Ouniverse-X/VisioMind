from __future__ import annotations

from typing import Any, Literal

import numpy as np

from visiomind.action.shared.errors import AdapterError

OpenPICometActionMode = Literal["dict", "raw"]


class OpenPICometActionAdapter:
    EXPECTED_DIM = 23
    RAW_ACTION_KEY = "robot_r1"
    SLICES: tuple[tuple[str, int, int], ...] = (
        ("action.base", 0, 3),
        ("action.torso", 3, 7),
        ("action.left_arm", 7, 14),
        ("action.left_gripper", 14, 15),
        ("action.right_arm", 15, 22),
        ("action.right_gripper", 22, 23),
    )

    @staticmethod
    def convert(action: Any, *, mode: OpenPICometActionMode = "raw") -> dict[str, np.ndarray]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape != (OpenPICometActionAdapter.EXPECTED_DIM,):
            raise AdapterError(
                f"OpenPI Comet action expected shape ({OpenPICometActionAdapter.EXPECTED_DIM},), got {arr.shape}"
            )
        if mode == "raw":
            return {OpenPICometActionAdapter.RAW_ACTION_KEY: arr.astype(np.float32, copy=True)}
        if mode == "dict":
            return {
                key: arr[start:end].astype(np.float32, copy=True).reshape(1, 1, -1)
                for key, start, end in OpenPICometActionAdapter.SLICES
            }
        raise AdapterError(f"Unsupported OpenPI Comet action mode: {mode!r}")


__all__ = ["OpenPICometActionAdapter", "OpenPICometActionMode"]
