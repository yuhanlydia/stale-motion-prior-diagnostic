"""Drop-in DOMINO policy adapter for the inference-only diagnostic.

Configuration is passed via SMP_* environment variables so paired conditions
can use the exact same DynamicWAM deploy YAML and checkpoint.
"""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dynamicwam.runtime.ciwam.adapters.domino.composite import (
    composite_robotwin_frame,
    extract_state,
    head_camera_frame,
)
from dynamicwam.runtime.ciwam.execution import NativeStepper
from dynamicwam.runtime.ciwam.flow import HeadFlowBuffer
from dynamicwam.runtime.ciwam.wam.policy import DynamicWAMPolicy

from .change import ChangeDetector, ChangeDetectorConfig
from .intervention import HistoryIntervention, InterventionConfig, Mode
from .logging import JsonlLogger
from .geometry import ee_geometry, is_grasped


def _target_xyz(task_env: Any) -> np.ndarray:
    config = task_env.get_dynamic_motion_config()
    actor = config["target_actor"]
    return np.asarray(actor.get_pose().p, dtype=np.float64)


def _workspace_ok(task_env: Any, xyz: np.ndarray) -> bool:
    checker = getattr(task_env, "is_out_of_bounds", None)
    if callable(checker):
        try:
            return not bool(checker(xyz))
        except TypeError:
            pass
    return bool(np.isfinite(xyz).all())


class DiagnosticDeployModel:
    def __init__(self, usr_args: dict[str, Any]) -> None:
        self.policy = DynamicWAMPolicy(usr_args["dynamicwam_deploy_config"], project_root=usr_args["dynamicwam_root"])
        self.policy.setup()
        mode = Mode(os.environ.get("SMP_MODE", "full"))
        random_queries = tuple(int(x) for x in os.environ.get("SMP_RANDOM_RESETS", "").split(",") if x)
        self.history = HistoryIntervention(
            lambda: HeadFlowBuffer(self.policy.runtime.head_flow_config),
            InterventionConfig(mode=mode, stale_queries=int(os.environ.get("SMP_STALE_QUERIES", "2")), random_reset_queries=random_queries),
        )
        self.detector = ChangeDetector(ChangeDetectorConfig())
        self.logger = JsonlLogger(Path(os.environ.get("SMP_LOG", "runs/queries.jsonl")))
        self.stepper: NativeStepper | None = None
        self.pending: deque[np.ndarray] = deque()
        self.query = 0
        self.pending_change_event = None

    def bind_env(self, env: Any) -> None:
        if self.stepper is None or self.stepper.env is not env:
            self.stepper = NativeStepper(env, action_interval_seconds=self.policy.runtime.action_interval_seconds)
            self.policy.set_instruction(env.get_instruction())

    def finish_episode(self) -> None:
        if self.stepper is not None:
            summary = self.stepper.telemetry.summary()
            print(f"SYNC-EPISODE-SUMMARY {summary}", flush=True)
        self.stepper = None
        self.pending.clear()
        self.history.reset_episode()
        self.detector.reset()
        self.policy.reset()
        self.query = 0
        self.pending_change_event = None


def get_model(usr_args: dict[str, Any]) -> DiagnosticDeployModel:
    return DiagnosticDeployModel(usr_args)


def eval(TASK_ENV: Any, model: DiagnosticDeployModel, observation: dict[str, Any]) -> None:
    model.bind_env(TASK_ENV)
    clock = getattr(TASK_ENV, "_scene_step_clock", None)
    if clock is None:
        raise RuntimeError("diagnostic requires TASK_ENV._scene_step_clock")
    now = float(clock.snapshot().time_seconds)
    target = _target_xyz(TASK_ENV)
    event = model.detector.update(
        query=model.query,
        position_xyz=target,
        time_seconds=now,
        pre_grasp=not is_grasped(TASK_ENV),
        in_workspace=_workspace_ok(TASK_ENV, target),
    )
    if event.changed:
        model.pending_change_event = event
    model.history.push(head_camera_frame(observation), simulator_time_seconds=now, change_point=event.changed)
    if not model.pending:
        motion, intervention = model.history.observation()
        packet = {
            "frame": composite_robotwin_frame(observation, model.policy.runtime.observation_config),
            "state": extract_state(observation),
            "flow_frames": motion.flow_rgb,
            "motion_features": motion.motion_features,
            "motion_interval_valid_mask": motion.interval_valid_mask,
            "motion_acceleration_valid_mask": motion.acceleration_valid_mask,
        }
        actions = model.policy.sample(packet)
        model.pending.extend(actions)
        logged_event = model.pending_change_event or event
        geometry = ee_geometry(TASK_ENV, target)
        model.logger.write({
            "task": os.environ.get("SMP_TASK", "unknown"),
            "level": int(os.environ.get("SMP_LEVEL", "-1")),
            "requested_start_seed": int(os.environ.get("SMP_REQUESTED_SEED", "-1")),
            "seed": int(getattr(TASK_ENV, "_smp_episode_seed")),
            "query": model.query,
            "simulator_time": now,
            "target_xyz": target.tolist(),
            "change_point": bool(model.pending_change_event is not None),
            "change_angle_deg": logged_event.direction_deg,
            "speed_ratio": logged_event.speed_ratio,
            "target_speed": logged_event.speed,
            "pre_grasp": not is_grasped(TASK_ENV),
            **geometry,
            **intervention,
        })
        model.pending_change_event = None
        model.query += 1
    if model.stepper is None:
        raise RuntimeError("environment not bound")
    model.stepper.execute(model.pending.popleft())


def reset_model(model: DiagnosticDeployModel) -> None:
    model.finish_episode()
