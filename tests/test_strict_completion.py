from pathlib import Path

from stale_motion_prior.strict_completion import completion_status


def test_completion_requires_exactly_one_new_official_metric(tmp_path: Path):
    query_log = tmp_path / 'queries.jsonl'
    query_log.write_text('{}\n')
    ok, reason = completion_status(returncode=0, query_log=query_log, log_text='SYNC-EPISODE-SUMMARY {}', new_metrics=[])
    assert not ok
    assert reason == 'missing_official_metrics'


def test_completion_rejects_ambiguous_official_metrics(tmp_path: Path):
    query_log = tmp_path / 'queries.jsonl'
    query_log.write_text('{}\n')
    ok, reason = completion_status(returncode=0, query_log=query_log, log_text='SYNC-EPISODE-SUMMARY {}', new_metrics=['a_metrics.json', 'b_metrics.json'])
    assert not ok
    assert reason == 'ambiguous_official_metrics'


def test_completion_accepts_one_metric_and_required_telemetry(tmp_path: Path):
    query_log = tmp_path / 'queries.jsonl'
    query_log.write_text('{}\n')
    ok, reason = completion_status(returncode=0, query_log=query_log, log_text='SYNC-EPISODE-SUMMARY {}', new_metrics=['a_metrics.json'])
    assert ok
    assert reason is None


def test_official_baseline_allows_missing_diagnostic_telemetry(tmp_path: Path):
    ok, reason = completion_status(
        returncode=0,
        query_log=tmp_path / 'missing.jsonl',
        log_text='official adapter',
        new_metrics=['a_metrics.json'],
        require_diagnostic_telemetry=False,
    )
    assert ok
    assert reason is None
