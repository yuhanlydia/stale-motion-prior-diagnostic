# Decisive Smoke v4 — Control Fix and Next Run

The 10 completed `grab_roller` L3 `RESET_CHANGE` episodes from v3 are retained: SR 0.0%, MS 28.31, paired mean RESET−FULL = −1.78 MS.

A control-semantic bug was found before RANDOM/stationary controls were run. True-change reset happened at the environment step of the change and could rebuild history before the next query; old RANDOM reset happened immediately before policy inference. v4 matches random/stationary resets to the source change's `actions_remaining_at_change` and normalized phase.

No completed RESET result is invalidated by this fix. The next step is first to run the zero-GPU chunk-phase diagnostic on existing data, then only 10 cross-task RESET screening episodes (5 each on `move_stapler_pad` and `place_mouse_pad`) before committing to the full control matrix.

See `docs/DECISIVE_SMOKE_PROTOCOL_V4.md`.
