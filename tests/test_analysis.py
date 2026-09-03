from stale_motion_prior.analysis import analyze_rows


def test_direct_paired_interaction_and_controls():
    rows = []
    values = {
        1: {"full": 10, "reset_change": 9, "random_reset": 8, "stale_hold": 9},
        3: {"full": 10, "reset_change": 14, "random_reset": 10, "stale_hold": 7},
    }
    for level, conditions in values.items():
        for condition, ms in conditions.items():
            rows.append({"task": "t", "level": level, "seed": 1, "condition": condition, "ms": ms, "sr": ms})
    result = analyze_rows(rows, draws=100)
    assert result["reset_minus_full_ms_L1"]["estimate"] == -1
    assert result["reset_minus_full_ms_L3"]["estimate"] == 4
    assert result["reset_minus_full_ms_interaction_L3_minus_L1"]["estimate"] == 5
    assert result["random_minus_full_ms_L3"]["estimate"] == 0
    assert result["stale_minus_full_ms_L3"]["estimate"] == -3
    assert result["reset_minus_random_ms_L3"]["estimate"] == 4


def test_cross_level_pairing_uses_requested_seed_and_reports_stationary_reversal():
    rows = []
    values = {
        1: {"full": 10, "reset_change": 10, "random_reset": 8, "stale_hold": 10},
        3: {"full": 10, "reset_change": 14, "random_reset": 9, "stale_hold": 7},
    }
    actual_seed = {1: 101, 3: 202}
    for level, conditions in values.items():
        for condition, ms in conditions.items():
            rows.append({
                "task": "t",
                "level": level,
                "requested_start_seed": 7,
                "seed": actual_seed[level],
                "condition": condition,
                "ms": ms,
                "sr": ms,
            })
    result = analyze_rows(rows, draws=100)
    assert result["reset_minus_full_ms_L3"]["estimate"] == 4
    assert result["random_minus_full_ms_L1"]["estimate"] == -2
    assert result["reset_minus_random_ms_L3"]["estimate"] == 5
    assert result[
        "stale_prior_reversal_ms_L3_change_minus_L1_stationary"
    ]["estimate"] == 6
    assert result["reset_minus_full_ms_interaction_L3_minus_L1"]["estimate"] == 4
