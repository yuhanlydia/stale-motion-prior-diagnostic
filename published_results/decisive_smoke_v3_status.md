# Decisive Smoke v3 — Run Status

Status: **paused by decision on 2026-09-03**.

The active `L3 / reset_change` queue was stopped after 10 of 30 paired
episodes completed. All 10 completed episodes finished without runtime
failure. The remaining 20 reset episodes, plus the planned L3
`random_reset` and `stale_hold` queues, were not started.

## Observed partial result

Compared with the matched L3 FULL baseline from Stage C:

| completed episodes | RESET SR | RESET MS | mean MS (RESET − FULL) |
|---:|---:|---:|---:|
| 10 | 0.0% | 28.31 | −1.78 |

All ten completed episodes were `grab_roller` (seeds 137–146). The partial
sample does not establish a preregistered causal conclusion, but it shows no
early RESET benefit and motivated pausing the expensive queue before running
the remaining intervention conditions.

Existing L1 and L2 baseline results remain unchanged:

- L1 FULL: 30/30 complete, SR 46.67%, MS 60.24.
- L2 FULL: 30/30 complete, SR 60.00%, MS 72.36.

Raw simulator artifacts remain locally under `runs/` (ignored by Git). This
tracked note is the hand-off state; no claim that the stale-motion-prior
hypothesis is confirmed or rejected should be made from this partial sample.
