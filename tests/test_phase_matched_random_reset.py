from stale_motion_prior.random_reset import build_phase_matched_schedule


def _row(
    query,
    *,
    pre_grasp=True,
    change=False,
    task="task",
    level=3,
    req=7,
    seed=101,
    remaining=None,
):
    row = {
        "task": task,
        "level": level,
        "requested_start_seed": req,
        "seed": seed,
        "query": query,
        "pre_grasp": pre_grasp,
        "change_point": change,
    }
    if change:
        row["actions_remaining_at_change"] = 0 if remaining is None else remaining
    return row


def test_random_resets_are_pregrasp_and_outside_change_cooldown():
    records = [
        _row(0),
        _row(1),
        _row(2, change=True),
        _row(3),
        _row(4),
        _row(5, pre_grasp=False),
        _row(6, pre_grasp=False),
    ]
    schedule = build_phase_matched_schedule(records, cooldown=1, rng_seed=3)
    assert len(schedule) == 1
    selected = schedule[0]["random_reset_queries"]
    assert selected == [0] or selected == [4]
    assert not schedule[0]["unavailable"]
    assert schedule[0]["chunk_phase_matched"]


def test_random_reset_schedule_is_deterministic():
    records = []
    for seed in (101, 102):
        for query in range(8):
            records.append(
                _row(query, change=query == 3, seed=seed, req=seed, remaining=5)
            )
    first = build_phase_matched_schedule(records, cooldown=1, rng_seed=11)
    second = build_phase_matched_schedule(records, cooldown=1, rng_seed=11)
    assert first == second


def test_episode_without_safe_pregrasp_query_is_marked_unavailable():
    records = [_row(0), _row(1, change=True), _row(2)]
    schedule = build_phase_matched_schedule(records, cooldown=2, rng_seed=0)
    assert schedule[0]["random_reset_queries"] == []
    assert schedule[0]["random_reset_events"] == []
    assert schedule[0]["unavailable"] is True


def test_episode_without_change_has_zero_reset_control_by_default():
    schedule = build_phase_matched_schedule([_row(0), _row(1)], cooldown=2)
    assert schedule == [
        {
            "task": "task",
            "level": 3,
            "seed": 7,
            "episode_seed": 101,
            "random_reset_queries": [],
            "random_reset_events": [],
            "unavailable": False,
            "pre_grasp_matched": True,
            "chunk_phase_matched": True,
            "borrowed_reference_level": None,
        }
    ]


def test_stationary_level_borrows_same_requested_seed_reference_phase():
    records = []
    for query in range(7):
        records.append(_row(query, level=1, seed=101))
    for query in range(7):
        records.append(
            _row(
                query,
                level=3,
                seed=202,
                change=query == 3,
                remaining=6,
            )
        )
    schedule = build_phase_matched_schedule(
        records,
        cooldown=1,
        rng_seed=0,
        stationary_reference_level=3,
    )
    stationary = next(row for row in schedule if row["level"] == 1)
    assert stationary["random_reset_queries"] == [3]
    assert stationary["random_reset_events"][0]["actions_remaining"] == 6
    assert stationary["unavailable"] is False
    assert stationary["borrowed_reference_level"] == 3


def test_stationary_level_without_reference_change_is_unavailable():
    records = [_row(query, level=1, seed=101) for query in range(5)]
    schedule = build_phase_matched_schedule(
        records,
        stationary_reference_level=3,
    )
    assert schedule[0]["random_reset_queries"] == []
    assert schedule[0]["random_reset_events"] == []
    assert schedule[0]["unavailable"] is True
    assert schedule[0]["borrowed_reference_level"] == 3


def test_stationary_reference_ignores_native_detector_event():
    records = [
        _row(
            query,
            level=1,
            seed=101,
            req=7,
            change=query == 1,
            remaining=2,
        )
        for query in range(7)
    ] + [
        _row(
            query,
            level=3,
            seed=202,
            req=7,
            change=query == 3,
            remaining=5,
        )
        for query in range(7)
    ]
    schedule = build_phase_matched_schedule(
        records,
        cooldown=1,
        rng_seed=0,
        stationary_reference_level=3,
    )
    stationary = next(row for row in schedule if row["level"] == 1)
    assert stationary["borrowed_reference_level"] == 3
    assert stationary["random_reset_queries"] == [3]
    assert stationary["random_reset_events"][0]["actions_remaining"] == 5
    assert stationary["unavailable"] is False


def test_l3_random_control_preserves_change_chunk_phase():
    records = [
        _row(
            query,
            change=query == 3,
            remaining=7,
        )
        for query in range(8)
    ]
    schedule = build_phase_matched_schedule(
        records,
        cooldown=1,
        rng_seed=0,
    )[0]
    assert len(schedule["random_reset_events"]) == 1
    event = schedule["random_reset_events"][0]
    assert event["actions_remaining"] == 7
    assert event["query"] != 3


def test_change_without_chunk_telemetry_is_unavailable():
    records = [
        {
            "task": "task",
            "level": 3,
            "requested_start_seed": 7,
            "seed": 101,
            "query": 2,
            "pre_grasp": True,
            "change_point": True,
        }
    ]
    schedule = build_phase_matched_schedule(records)[0]
    assert schedule["unavailable"] is True
    assert schedule["unavailable_reason"] == "missing_chunk_phase_telemetry"
