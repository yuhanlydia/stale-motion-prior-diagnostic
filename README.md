# Stale Motion Prior Diagnostic

Inference-only causal diagnostics for the hypothesis:

> Truthful motion history helps under stationary dynamics but becomes a stale,
> causally harmful prior immediately after an abrupt regime change.

The first gate uses the released DynamicWAM checkpoint on DOMINO. It does not
train, fine-tune, alter KV state, or change current RGB observations. Only the
explicit history-flow/kinematic buffer is intervened on.

## Gate-0 design

The four decisive paired conditions are:

| Condition | Intervention |
|---|---|
| `full` | Released DynamicWAM behavior |
| `reset_change` | Clear pre-change motion history at a simulator-GT change; retain current and all subsequent observations |
| `random_reset` | Same reset count and normalized-time distribution, but away from true changes |
| `stale_hold` | Update current RGB, but freeze the pre-change motion packet for two policy queries |

The preregistered detector uses only simulator target motion: direction change
greater than 45 degrees or relative speed change greater than 0.5, with minimum
speed, pre-grasp, workspace, and two-query cooldown eligibility checks.

Primary outcomes remain DOMINO SR and MS. Mechanism outcomes are three-query
post-change gripper-target distance AUC and recovery lag. The main interaction
is `(RESET-FULL)_L3 - (RESET-FULL)_L1`; random reset must not reproduce the L3
gain. The strongest local ordering is `RESET > FULL > STALE`.

## Reproducibility pins

- DynamicWAM code: record with `git rev-parse HEAD` (initial checkout used
  `db91c4fe856b42d5500150e11e4f86f407f21dbc`).
- DynamicWAM checkpoint revision:
  `925cbb7aef5033c924f809ae87479d39fe9f76ff`.
- Full checkpoint SHA-256:
  `7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7`.
- DOMINO evaluation commit is read from the released config/manifest, not from
  an unpinned standalone clone.

At 2026-09-01 the checkpoint YAML has one additional benchmark integrity field
not accepted by DynamicWAM `main`. Do not delete it. Pin a compatible upstream
commit or wait for the upstream schema fix; mixing latest code and released
config silently is not acceptable.

## Install

```bash
git clone https://github.com/Autumn1337/DynamicWAM.git
git clone https://github.com/OWNER/stale-motion-prior-diagnostic.git
cd stale-motion-prior-diagnostic
python -m venv .venv
.venv/bin/pip install -e '.[test,analysis]'
.venv/bin/pytest
```

Install this package in both the DynamicWAM inference environment and the
DOMINO evaluation environment. The only integration patch replaces the
official adapter module with a three-line import shim while preserving a
recoverable backup:

```bash
python scripts/install_adapter.py /path/to/DynamicWAM
# restore later:
python scripts/install_adapter.py /path/to/DynamicWAM --restore
```

The adapter reads `SMP_MODE`, `SMP_LOG`, `SMP_STALE_QUERIES`, and
`SMP_RANDOM_RESETS`. Thus checkpoint, observation, task, instruction, seed, and
official eval configuration stay identical across conditions.

## Generate the paired smoke matrix

```bash
smp-matrix \
  --tasks grab_roller move_stapler_pad place_mouse_pad \
  --levels 1 3 \
  --seeds 137 138 139 140 141 142 143 144 145 146 \
  --output runs/smoke.jsonl
```

Before execution, add each row's exact argv-only `command` for the pinned
DynamicWAM/DOMINO evaluator. Do not use shell strings. Run under a hard ten-hour
budget with crash-safe per-run logs and a resumable journal:

```bash
python scripts/run_budget.py runs/smoke_commands.jsonl --hours 10
```

Queue rows should be ordered by `(task, level, seed)` and then all four
conditions, ensuring paired seeds complete together as often as possible.

## Random-reset control

Random reset timing is a two-pass control. First run FULL to collect valid
episode lengths and true-change normalized times. Freeze that distribution and
draw random reset queries with the same count, stratified by task/level, while
excluding true-change cooldown windows. Commit the generated schedule before
opening condition outcomes. Never tune it using policy loss or final MS.

## Analysis

Episode result JSONL must contain `task`, `level`, `seed`, `condition`, `sr`,
and `ms`. Then:

```bash
smp-analyze runs/episode_results.jsonl --draws 10000
```

Pilot inference uses paired episode bootstrap. The 35-task confirmatory run
must use task-balanced hierarchical bootstrap (resample tasks, then paired
seeds within task). Report delta SR, delta MS, delta distance AUC, and delta
recovery lag with 95% confidence intervals.

## Gates and stopping

1. **A — reproduction:** one official L1 task produces nonzero SR/MS plus
   replay and complete telemetry.
2. **B — events:** at least 70% of selected L3 episodes contain an eligible
   pre-grasp abrupt change.
3. **C — phenomenon:** L3 reset improves paired MS, random reset does not, and
   the L3 reset gain exceeds L1.
4. **D — causal ordering:** preferably RESET > FULL > STALE in local distance
   AUC or recovery lag.

Stop if there is no L1-vs-L3 interaction. Do not proceed to flow-only reset,
kinematic-only reset, history-length sweeps, severity sweeps, or any learned
method until Gate-0 survives.

## Known limitations before the first rollout

- The official DynamicWAM environment currently requires FlashAttention
  2.8.3.post1. On hosts without `nvcc`, install a Torch 2.7/CUDA 12.6/CPython
  3.10/SM86-compatible wheel or add the matching CUDA toolkit.
- A 16GB RTX A4000 may be below the practical memory requirement of the
  released video-action model. Checkpoint verification is CPU-safe; rollout
  OOM is a Gate-A infrastructure failure, not evidence against the hypothesis.
- Generic target pose is available through DOMINO's dynamic-motion config.
  Gripper pose extraction is embodiment-specific and must be validated before
  distance AUC/recovery lag are treated as complete.

### RoboTwin embodiments archive bug

The pinned `embodiments.zip` SHA is correct, but the archive contains macOS
resource forks while DynamicWAM's manifest records a cleaned extraction tree.
The upstream pre-extraction count therefore fails before it can clean anything.
`scripts/extract_robotwin_archive.py` verifies the pinned archive SHA, rejects
unsafe members, excludes only `__MACOSX`, refuses overwrite, and writes a JSON
provenance report. This workaround must remain visible in every run manifest.
