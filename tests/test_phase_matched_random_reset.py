from stale_motion_prior.random_reset import build_phase_matched_schedule


def _row(query, *, pre_grasp=True, change=False, task='task', level=3, req=7, seed=101):
    return {'task': task, 'level': level, 'requested_start_seed': req, 'seed': seed, 'query': query, 'pre_grasp': pre_grasp, 'change_point': change}


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


def test_episode_without_change_has_zero_reset_control():
    schedule = build_phase_matched_schedule([_row(0), _row(1)], cooldown=2)
    assert schedule == [{
        'task': 'task', 'level': 3, 'seed': 7, 'episode_seed': 101,
        'random_reset_queries': [], 'unavailable': False,
        'pre_grasp_matched': True,
    }]
