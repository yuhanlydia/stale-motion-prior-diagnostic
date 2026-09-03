from stale_motion_prior.chunk_analysis import summarize_reset_by_chunk_phase


def test_chunk_phase_summary_uses_first_full_change_and_paired_delta():
    results = [
        {
            "task": "t",
            "level": 3,
            "requested_start_seed": 1,
            "seed": 11,
            "condition": "full",
            "ms": 30,
            "sr": 0,
        },
        {
            "task": "t",
            "level": 3,
            "requested_start_seed": 1,
            "seed": 11,
            "condition": "reset_change",
            "ms": 28,
            "sr": 0,
        },
        {
            "task": "t",
            "level": 3,
            "requested_start_seed": 2,
            "seed": 12,
            "condition": "full",
            "ms": 30,
            "sr": 0,
        },
        {
            "task": "t",
            "level": 3,
            "requested_start_seed": 2,
            "seed": 12,
            "condition": "reset_change",
            "ms": 34,
            "sr": 0,
        },
    ]
    logs = [
        {
            "task": "t",
            "level": 3,
            "requested_start_seed": 1,
            "seed": 11,
            "query": 2,
            "change_point": True,
            "pre_grasp": True,
            "actions_remaining_at_change": 2,
        },
        {
            "task": "t",
            "level": 3,
            "requested_start_seed": 2,
            "seed": 12,
            "query": 2,
            "change_point": True,
            "pre_grasp": True,
            "actions_remaining_at_change": 10,
        },
    ]
    summary = summarize_reset_by_chunk_phase(results, logs)
    assert summary["n"] == 2
    assert summary["mean_delta_ms"] == 1.0
    assert summary["low_remaining_mean_delta_ms"] == -2.0
    assert summary["high_remaining_mean_delta_ms"] == 4.0
