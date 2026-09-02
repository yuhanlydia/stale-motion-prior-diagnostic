# Stale Motion Prior — Validity-Fixed Decisive Smoke

## Scientific question

Test one phenomenon before introducing any learned method:

> Does truthful recent motion history become causally harmful immediately after an abrupt target-dynamics regime change?

The benchmark remains DOMINO and the policy remains the released DynamicWAM checkpoint. No weights, KV state, current RGB observation, language input, or action head are trained or altered. Only the explicit history-flow / kinematic history interface is intervened on.

## Conditions

- `full`: paired deterministic DynamicWAM history.
- `reset_change`: at a simulator-GT abrupt change, discard pre-change motion history, keep the current RGB frame, and rebuild history only from post-change observations.
- `random_reset`: same reset count and matched normalized episode phase as true changes, but only at `pre_grasp=True` query locations and outside the true-change cooldown.
- `stale_hold`: current RGB continues to update while the pre-change motion packet is deliberately held for two policy queries.

## Code-validity changes

1. Non-segmented velocity/trajectory tasks no longer return a fake commanded regime index of zero. The kinematic detector remains active unless DOMINO explicitly exposes `current_segment_idx`.
2. Low-speed direction cosine uses `max(||v1||||v2||, eps)` rather than `||v1||||v2|| + eps`.
3. The phase-matched random-reset builder samples only from `pre_grasp=True` FULL queries.
4. The strict budget runner marks a row complete only when the child exits successfully, a non-empty query log exists, `SYNC-EPISODE-SUMMARY` exists, and exactly one new official `*_metrics.json` is produced.
5. Query logs record `actions_remaining_at_change`, `steps_until_next_policy_query`, and `chunk_phase` so action-chunk latency can be separated from stale-history effects.

## Stage A — FULL equivalence audit

Use `grab_roller`, Level 1, seeds `137..141`.

Run each seed once with the unmodified official adapter and once with diagnostic `full` under the same DOMINO evaluator/checkpoint/config. Compare SR/MS and, where both sides expose it, the first action/action chunk. The diagnostic adapter intentionally fixes policy sampling RNG for paired causal runs. If bit-exact equivalence with the official stochastic path is not possible, call the experimental baseline **paired_full**, not “exact released behavior”, and document the output discrepancy before proceeding.

## Stage B — L3 regime-source/event audit

Run FULL only:

- `grab_roller`, L3, seeds `137..146`
- `move_stapler_pad`, L3, seeds `137..146`
- `place_mouse_pad`, L3, seeds `137..146`

For every episode record whether an explicit `current_segment_idx` source exists, commanded and fallback switches, chosen `change_source`, `pre_grasp`, target speed / direction / speed-ratio, and action-chunk phase at the change.

**Gate B:** at least 70% of selected L3 episodes must contain an eligible pre-grasp change before first grasp/contact. If not, do not loosen thresholds after seeing policy outcomes; change task selection or stop.

## Stage C — decisive paired smoke

Use exactly:

- tasks: `grab_roller`, `move_stapler_pad`, `place_mouse_pad`
- levels: `1, 3`
- seeds: `137..146`
- conditions: `full`, `reset_change`, `random_reset`, `stale_hold`

This is 240 task-level condition runs.

Run order per task/level/seed:

1. FULL first, freezing simulator trajectory and telemetry.
2. Build and freeze `random_reset` schedule before opening intervention outcomes.
3. Run RESET_CHANGE, RANDOM_RESET, STALE_HOLD on the same actual simulator seed/trajectory.

Build the random control with:

```bash
PYTHONPATH=src python scripts/build_random_reset_schedule_phase_matched.py \
  runs/full_logs \
  --output runs/random_reset_schedule_v2.jsonl \
  --cooldown 2 \
  --seed 20260902
```

Use `scripts/apply_random_reset_schedule.py` to inject the frozen queries into the RANDOM queue. Use the strict runner for all expensive rows:

```bash
PYTHONPATH=src python scripts/run_budget_strict.py runs/smoke_commands.jsonl \
  --hours 10 \
  --journal runs/strict_journal.jsonl \
  --logs runs/logs
```

Harvest official DOMINO metrics with `harvest_official_metrics.py` and analyze with `smp-analyze --draws 10000`.

## Primary endpoints

Headline metrics remain official DOMINO SR and Manipulation Score (MS).

Primary phenomenon statistic:

`I_MS = (MS_reset - MS_full)_L3 - (MS_reset - MS_full)_L1`.

The hypothesis predicts `I_MS > 0`.

Additional preregistered checks:

- `(RESET-FULL)_L3 > 0` on MS.
- RANDOM_RESET does not reproduce the L3 RESET gain.
- For lower-is-better post-change distance AUC and recovery lag, expected ordering is `RESET_CHANGE < FULL < STALE_HOLD`.
- Stratify local recovery by `actions_remaining_at_change`; do not mistake action-chunk commitment latency for stale-history persistence.

Use paired episode bootstrap (10,000 draws) for smoke. If expanded to all tasks, use task-balanced hierarchical bootstrap (resample task, then paired seed within task).

## Stop / continue

**STOP** the stale-motion-prior hypothesis if the L1-vs-L3 RESET interaction is absent or reversed. Do not rescue it by tuning detector thresholds, stale duration, history length, or task subset using outcome data.

**CONTINUE** only if L3 shows a change-specific RESET advantage that is not reproduced by RANDOM_RESET. Then run causal decomposition only: flow-history reset, kinematic-history reset, both, neither. Only after that should a learned history-validity gate be considered.
