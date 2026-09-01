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
