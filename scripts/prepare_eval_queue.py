#!/usr/bin/env python3
"""Create exact paired DOMINO configs and an argv-only evaluation queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


CONDITIONS = ("full", "reset_change", "random_reset", "stale_hold")


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamicwam-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument(
        "--policy-name",
        default="ciwam.adapters.domino.deploy_policy_sync_flow",
        help="Policy adapter import path to place in generated configs.",
    )
    parser.add_argument(
        "--trajectory-root", type=Path,
        help="Optional root containing frozen trajectory_snapshots to reuse.",
    )
    args = parser.parse_args()
    root = args.dynamicwam_root.resolve()
    domino = root / "external" / "DOMINO"
    python = root / "external" / "robotwin" / "bin" / "python"
    source_task_config = domino / "task_config" / "demo_clean_dynamic.yml"
    if not all(path.exists() for path in (domino, python, source_task_config)):
        raise FileNotFoundError("DynamicWAM/DOMINO runtime is incomplete")
    base_task = yaml.safe_load(source_task_config.read_text(encoding="utf-8"))
    task_config_names = {}
    for level in args.levels:
        if level not in (1, 2, 3):
            raise ValueError(f"invalid DOMINO level: {level}")
        config = dict(base_task)
        config["dynamic_level"] = level
        config["eval_video_log"] = True
        name = f"smp_clean_dynamic_l{level}"
        write_yaml(domino / "task_config" / f"{name}.yml", config)
        task_config_names[level] = name
    output = args.output_root.resolve()
    trajectory_root = (args.trajectory_root or output).resolve()
    configs = output / "configs"
    rows = []
    pythonpath = ":".join((
        str(root / "src" / "dynamicwam" / "runtime"),
        str(root / "src"),
        str(root / "external" / "curobo" / "src"),
        str(domino / "script"),
    ))
    for task in args.tasks:
        if not (domino / "envs" / f"{task}.py").exists():
            raise ValueError(f"unknown DOMINO task: {task}")
        for level in args.levels:
            for seed in args.seeds:
                for condition in args.conditions:
                    label = f"{task}_l{level}_s{seed}_{condition}"
                    trajectory_snapshot = trajectory_root / "trajectory_snapshots" / f"{task}_l{level}_requested_s{seed}.pkl"
                    run_config = {
                        "policy_name": args.policy_name,
                        "task_name": task,
                        "task_config": task_config_names[level],
                        "ckpt_setting": f"dynamicwam-full-{condition}",
                        "instruction_type": "unseen",
                        "seed": seed,
                        "start_seed": seed,
                        "episode_num": 1,
                        "eval_output_root": str(output / "official_results" / label),
                        "dynamicwam_root": str(root),
                        "dynamicwam_deploy_config": str(root / "configs" / "absolute_motion_v2.yaml"),
                    }
                    config_path = configs / f"{label}.yml"
                    write_yaml(config_path, run_config)
                    rows.append({
                        "task": task,
                        "level": level,
                        "seed": seed,
                        "condition": condition,
                        "cwd": str(domino),
                        "env": {
                            "PYTHONPATH": pythonpath,
                            "CUDA_VISIBLE_DEVICES": "0",
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "PYTHONUNBUFFERED": "1",
                            "SMP_TRAJECTORY_SNAPSHOT": str(trajectory_snapshot),
                        },
                        "command": [str(python), str(domino / "script" / "eval_policy.py"), "--config", str(config_path)],
                    })
    queue = output / "queue.jsonl"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    provenance = {
        "dynamicwam_commit": "feb7921",
        "checkpoint_sha256": "7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7",
        "source_task_config": str(source_task_config),
        "tasks": args.tasks,
        "levels": args.levels,
        "seeds": args.seeds,
        "conditions": args.conditions,
        "rows": len(rows),
    }
    (output / "queue_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} runs to {queue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
