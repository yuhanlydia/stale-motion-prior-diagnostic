# Stale Motion Prior — v4 Control-Fairness Protocol

## Why v4 exists

The first v3 intervention result is informative but not yet decisive:

- `grab_roller`, L3, seeds 137–146
- `RESET_CHANGE` completed 10/10 without runtime failure
- RESET SR: 0.0%
- RESET MS: 28.31
- paired mean `RESET - FULL`: **-1.78 MS**

This is a real negative early result for hard reset on `grab_roller`; it is not discarded.

However, before any v3 RANDOM/Level-1 stationary controls were run, code review found a control-semantic confound. `RESET_CHANGE` clears the DynamicWAM history at the exact simulator step of the true regime change and then accumulates every post-change frame until the next policy query. The old `RANDOM_RESET` cleared the buffer only inside `observation()`, immediately before policy inference. With DynamicWAM's history `policy_stride=4`, the two conditions could therefore expose very different amounts of fresh post-reset history.

v4 fixes only this control path. Existing FULL results and the ten completed `grab_roller RESET_CHANGE` runs remain reusable.

## v4 matched reset event

Every true FULL change log already records:

- policy query index
- normalized episode phase
- `actions_remaining_at_change`
- `chunk_phase`

For each control reset, v4 chooses a pre-grasp non-change query at the nearest normalized phase and fires the reset when the pending action count equals the source change's `actions_remaining_at_change`.

Thus both true-change reset and control reset follow:

`reset buffer -> push current event-step frame -> accumulate remaining action-step frames -> next policy query`.

The variable of interest is now *whether the reset coincides with a dynamics regime change*, rather than how much time the buffer had to recover.

## Stationary Level-1 control

For the same task and requested seed, Level 1 borrows both:

1. normalized change phase from Level-3 FULL;
2. `actions_remaining_at_change` from Level-3 FULL.

Native Level-1 detector events are ignored for control scheduling. This produces a stationary reset with the same timing and chunk-rebuild opportunity as the Level-3 change reset.

## Existing artifacts to reuse

Do not rerun valid artifacts from the pinned runtime/checkpoint:

- L1 FULL: 30 episodes, SR 46.67%, MS 60.24
- L2 FULL: 30 episodes, SR 60.00%, MS 72.36
- existing L3 FULL matched baselines
- `grab_roller` L3 RESET_CHANGE seeds 137–146

No v3 RANDOM/stationary control had started when the control bug was found, so no control result needs to be discarded.

## Zero-GPU diagnostic first

Before launching more episodes, use the already completed FULL/RESET artifacts:

```bash
PYTHONPATH=src python scripts/analyze_reset_by_chunk_phase.py \
  --results runs/episode_results_partial_v3.jsonl \
  --full-logs runs/full_logs_l3 \
  --output published_results/reset_chunk_phase_partial.json
```

Interpretation:

- If `RESET-FULL` improves as `actions_remaining_at_change` increases, the hard reset is likely suffering from cold-start when the regime change occurs close to a policy query.
- If the reset remains negative even when many actions remain before the next query, the evidence against hard reset is stronger.

This diagnostic is exploratory; do not treat the correlation as a confirmatory result.

## Rebuild v4 control schedule

Use FULL logs from both levels:

```bash
PYTHONPATH=src python scripts/build_random_reset_schedule_phase_matched.py \
  runs/full_logs_l1 runs/full_logs_l3 \
  --output runs/random_reset_schedule_v4.jsonl \
  --cooldown 2 \
  --seed 20260904 \
  --stationary-reference-level 3
```

Every usable scheduled row must have:

- `unavailable=false`
- `pre_grasp_matched=true`
- `chunk_phase_matched=true`
- nonempty `random_reset_events` whenever its reference episode has a true change.

Inject with:

```bash
python scripts/apply_random_reset_schedule.py \
  runs/v4_queue.jsonl runs/random_reset_schedule_v4.jsonl \
  --output runs/v4_queue_scheduled.jsonl
```

The applied queue carries `SMP_RANDOM_RESET_EVENTS=query:actions_remaining,...`.

## Compute-efficient next GPU order

The current negative result covers only one task, so do not spend all remaining control budget yet.

### Stage 1 — cross-task hard-reset sign screen

Run only 5 seeds each:

- `move_stapler_pad`, L3, RESET_CHANGE, seeds 137–141
- `place_mouse_pad`, L3, RESET_CHANGE, seeds 137–141

This is **10 new runs**. It is a development screen, not the final statistical test.

Decision:

- If both new task means are non-positive and the pooled 20 RESET pairs remain non-positive, hard reset has little prospect. Do not complete 30 RESET runs just for sample size.
- If either task has a clear positive RESET sign, complete the remaining RESET seeds and proceed to matched controls.

### Stage 2 — matched controls only if warranted

Run v4 event-step controls:

- L1 stationary random reset
- L3 random reset
- L3 stale hold

The headline causal comparisons remain:

- `G_change = (RESET_CHANGE - FULL)_L3`
- `G_stationary = (stationary reset - FULL)_L1`
- change specificity: `(RESET_CHANGE - RANDOM_RESET)_L3`
- stale perturbation: `(STALE_HOLD - FULL)_L3`

## 16GB / 24GB execution

DynamicWAM's released synchronous runner is batch-1 by construction. Do not change simulator semantics or force multi-episode batching for this causal study.

For 16GB:

- apply `patches/dynamicwam-cpu-t5.patch`;
- BF16 model inference;
- one DynamicWAM process per GPU;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- `CUDA_MODULE_LOADING=LAZY`;
- reuse loaded model only within the official evaluator lifecycle if supported; do not run two model processes concurrently on a 16GB card.

For 24GB use the same scientific settings. Extra memory is not a reason to alter batch semantics.

## Stop / pivot rule

Three possible outcomes should be distinguished:

1. **RESET improves and RANDOM does not**: strong stale-prior evidence; proceed to flow-vs-kinematic decomposition.
2. **RESET hurts but its harm is strongly concentrated near query-boundary changes**: the gap may still be real, but full reset is too aggressive; investigate selective history attenuation rather than declaring history universally useful.
3. **RESET is non-positive across tasks, chunk position does not explain it, and STALE_HOLD is not consistently worse than FULL**: stop the stale-motion-prior direction.

Do not tune change thresholds, stale duration, or task subsets after reading outcomes and then relabel the same data confirmatory.
