"""DOMINO simulator-GT geometry helpers without model dependencies."""

from __future__ import annotations

from typing import Any
import numpy as np


def is_grasped(task_env: Any) -> bool:
    contact = getattr(task_env, "_first_target_gripper_contact_valid", None)
    if contact is not None and bool(np.asarray(contact, dtype=np.bool_).any()):
        return True
    for name in ("is_grasped", "target_grasped", "grasped"):
        value = getattr(task_env, name, False)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
    return False


def ee_geometry(task_env: Any, target: np.ndarray) -> dict[str, Any]:
    left = np.asarray(task_env.robot.get_left_ee_pose()[:3], dtype=np.float64)
    right = np.asarray(task_env.robot.get_right_ee_pose()[:3], dtype=np.float64)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("DOMINO robot end-effector positions must have shape [3]")
    left_distance = float(np.linalg.norm(left - target))
    right_distance = float(np.linalg.norm(right - target))
    return {
        "left_ee_xyz": left.tolist(),
        "right_ee_xyz": right.tolist(),
        "left_target_dist": left_distance,
        "right_target_dist": right_distance,
        "ee_target_dist": min(left_distance, right_distance),
    }

