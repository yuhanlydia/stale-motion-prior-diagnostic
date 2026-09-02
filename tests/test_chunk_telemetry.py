from stale_motion_prior.chunk_telemetry import change_chunk_telemetry


def test_change_inside_chunk_reports_remaining_actions_and_phase():
    telemetry = change_chunk_telemetry(pending_actions=9, last_chunk_size=16)
    assert telemetry['actions_remaining_at_change'] == 9
    assert telemetry['steps_until_next_policy_query'] == 9
    assert telemetry['chunk_phase'] == 7 / 16


def test_change_at_policy_boundary_has_no_active_chunk_phase():
    telemetry = change_chunk_telemetry(pending_actions=0, last_chunk_size=16)
    assert telemetry['actions_remaining_at_change'] == 0
    assert telemetry['steps_until_next_policy_query'] == 0
    assert telemetry['chunk_phase'] is None
