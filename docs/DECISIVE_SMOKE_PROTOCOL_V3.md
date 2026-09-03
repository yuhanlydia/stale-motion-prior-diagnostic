# Stale Motion Prior — v3 Decisive Smoke

## Why v3 exists

The first GPU rollout exposed a protocol flaw rather than a scientific result: Level 1 has no abrupt change, therefore `reset_change` is identical to `full`. The old statistic `(RESET-FULL)_L3 - (RESET-FULL)_L1` consequently degenerates because its Level-1 term is structurally zero.

v3 creates a real stationary control. For each Level-1 FULL episode, it borrows the **number and normalized phase of true Level-3 changes from the same task and requested start seed**, then performs a pre-grasp history reset at the nearest valid Level-1 query. No outcome, policy loss, SR, or MS is used to choose this time.

The second fix is statistical: DOMINO may advance an infeasible requested seed to different actual simulator seeds at different levels. Cross-level effects therefore pair on `requested_start_seed`, while within-level conditions remain required to resolve to the same actual simulator seed.

## Scientific question

> Does a history reset help specifically after a dynamics regime shift, while the same reset is neutral or harmful under stationary motion?

This is stronger than asking whether reset helps Level 3 in isolation.

## Conditions actually required

### Level 1 — stationary control

Run only:

- `full`
- `random_reset`

`random_reset` is *not random in count or phase*: it is the outcome-blind stationary reset matched to the same task/requested-seed Level-3 change schedule.

Do not waste GPU time on L1 `reset_change` or `stale_hold`; without a true change they are no-op controls.

### Level 3 — abrupt dynamics

Run:

- `full`
- `reset_change`
- `random_reset`
- `stale_hold`

The L3 random control keeps the existing design: same reset count and empirical task/level phase distribution, pre-grasp only, outside all true-change cooldown windows.

## Frozen task matrix

Tasks:

- `grab_roller`
- `move_stapler_pad`
- `place_mouse_pad`

Requested seeds: `137..146`.

If all FULL episodes must be rerun, v3 contains 180 condition runs:

- Level 1: `3 tasks x 10 seeds x 2 conditions = 60`
- Level 3: `3 tasks x 10 seeds x 4 conditions = 120`

If valid FULL logs, official metrics and frozen trajectory snapshots already exist from the same pinned checkpoint/runtime, reuse them. Then only 120 new intervention runs are needed:

- L1 stationary reset: 30
- L3 reset/random/stale: 90

Do not rerun FULL merely to consume GPU time.

## Building the v3 schedule

The builder must see FULL logs from both levels:

```bash
PYTHONPATH=src python scripts/build_random_reset_schedule_phase_matched.py \
  runs/full_logs_l1 runs/full_logs_l3 \
  --output runs/random_reset_schedule_v3.jsonl \
  --cooldown 2 \
  --seed 20260903 \
  --stationary-reference-level 3
```

A Level-1 episode is marked unavailable if its matching task/requested-seed Level-3 FULL episode has no usable pre-grasp change. Do not silently replace it with a zero-reset control.

Inject the frozen schedule using `scripts/apply_random_reset_schedule.py` before opening intervention outcomes.

## Primary statistic

Define

- `G_change = (MS_reset_change - MS_full)_L3`
- `G_stationary = (MS_random_reset - MS_full)_L1`

The v3 headline statistic is

`R = G_change - G_stationary`.

The stale-prior hypothesis predicts `R > 0`.

Required supporting gates:

1. `G_change > 0`.
2. `(RESET_CHANGE - RANDOM_RESET)_L3 > 0` — change specificity.
3. `G_stationary <= 0` is preferred. If stationary resets also improve performance, the claim weakens toward generic history regularization.
4. `(STALE_HOLD - FULL)_L3 < 0` is the expected causal sign.

The analyzer still emits the old reset-vs-reset L3-minus-L1 statistic for audit only; it is no longer the primary v3 gate.

## Statistics

Use paired 10,000-draw bootstrap for the three-task smoke. Pair cross-level cells by `requested_start_seed`; actual simulator seed is still checked within each task/level/requested-seed set of conditions.

For a later 35-task confirmation, use task-balanced hierarchical bootstrap.

## Result publication

Raw `runs/` stays gitignored. After harvesting authoritative DOMINO metrics, publish only a small audit summary:

```bash
python scripts/harvest_official_metrics.py runs/v3_queue.jsonl \
  --output runs/episode_results_v3.jsonl

python scripts/publish_results.py \
  --results runs/episode_results_v3.jsonl \
  --full-logs runs/full_logs_l1 runs/full_logs_l3 \
  --label decisive_smoke_v3

git add published_results/
git commit -m "Publish stale-motion v3 smoke results"
git push
```

This exists so the scientific result can be audited from GitHub without committing multi-GB simulator artifacts.

## Stop / continue

**STOP** the stale-motion-prior direction if `R` is absent/reversed, or if L3 `RESET_CHANGE` fails to beat L3 `RANDOM_RESET`. Do not rescue the hypothesis with post-hoc detector thresholds, longer stale holds, selected tasks, or learned gates.

**CONTINUE** only after the v3 causal reversal survives. The next experiment is then decomposition of the two DynamicWAM history paths: flow reset only, kinematic reset only, both, neither. Only after that should a learned history-validity mechanism be designed.
