from stale_motion_prior.random_reset import build_phase_matched_schedule


def _row(query, *, pre_grasp=True, change=False, task='task', level=3, req=7, seed=101):
    return {
        'task': task,
        'level': level,
        'requested_start_seed': req,
        'seed': seed,
        'query': query,
        'pre_grasp': pre_grasp,
        'change_point': change,
    }


def test_random_resets_are_pregrasp_and_outside_change_cooldown():
    records = [_row(0), _row(1), _row(2, change=True), _row(3), _row(4), _row(5, pre_grasp=False), _row(6, pre_grasp=False)]
    schedule = build_phase_matched_schedule(records, cooldown=1, rng_seed=3)
    assert len(schedule) == 1
    selected = schedule[0]['random_reset_queries']
    assert selected == [0] or selected == [4]
    assert not schedule[0]['unavailable']


def test_random_reset_schedule_is_deterministic():
    records = []
    for seed in (101, 102):
        for q in range(8):
            records.append(_row(q, change=q == 3, seed=seed, req=seed))
    a = build_phase_matched_schedule(records, cooldown=1, rng_seed=11)
    b = build_phase_matched_schedule(records, cooldown=1, rng_seed=11)
    assert a == b


def test_episode_without_safe_pregrasp_query_is_marked_unavailable():
    records = [_row(0), _row(1, change=True), _row(2)]
    schedule = build_phase_matched_schedule(records, cooldown=2, rng_seed=0)
    assert schedule[0]['random_reset_queries'] == []
    assert schedule[0]['unavailable'] is True


def test_episode_without_change_has_zero_reset_control_by_default():
    schedule = build_phase_matched_schedule([_row(0), _row(1)], cooldown=2)
    assert schedule == [{
        'task': 'task', 'level': 3, 'seed': 7, 'episode_seed': 101,
        'random_reset_queries': [], 'unavailable': False,
        'pre_grasp_matched': True, 'borrowed_reference_level': None,
    }]


def test_stationary_level_borrows_same_requested_seed_reference_phase():
    records = []
    for query in range(7):
        records.append(_row(query, level=1, seed=101))
    for query in range(7):
        records.append(_row(query, level=3, seed=202, change=query == 3))
    schedule = build_phase_matched_schedule(
        records,
        cooldown=1,
        rng_seed=0,
        stationary_reference_level=3,
    )
    stationary = next(row for row in schedule if row['level'] == 1)
    assert stationary['random_reset_queries'] == [3]
    assert stationary['unavailable'] is False
    assert stationary['borrowed_reference_level'] == 3


def test_stationary_level_without_reference_change_is_unavailable():
    records = [_row(query, level=1, seed=101) for query in range(5)]
    schedule = build_phase_matched_schedule(
        records,
        stationary_reference_level=3,
    )
    assert schedule[0]['random_reset_queries'] == []
    assert schedule[0]['unavailable'] is True
    assert schedule[0]['borrowed_reference_level'] == 3


def test_stationary_reference_ignores_native_detector_event():
    records = [
        _row(query, level=1, seed=101, req=7, change=query == 1)
        for query in range(7)
    ] + [
        _row(query, level=3, seed=202, req=7, change=query == 3)
        for query in range(7)
    ]
    schedule = build_phase_matched_schedule(
        records, cooldown=1, rng_seed=0, stationary_reference_level=3
    )
    stationary = next(row for row in schedule if row['level'] == 1)
    assert stationary['borrowed_reference_level'] == 3
    assert stationary['random_reset_queries'] == [3]
    assert stationary['unavailable'] is False
