import json
from pathlib import Path
import subprocess
import sys


def test_strict_runner_fails_row_without_official_metric(tmp_path: Path):
    repo = Path(__file__).parents[1]
    output = tmp_path / 'official'
    output.mkdir()
    config = tmp_path / 'eval.yml'
    config.write_text(f'eval_output_root: {output}\n')
    queue = tmp_path / 'queue.jsonl'
    row = {
        'task': 'task', 'level': 3, 'seed': 9, 'condition': 'full',
        'command': [
            sys.executable, '-c',
            "import os; from pathlib import Path; Path(os.environ['SMP_LOG']).write_text('{}\\n'); print('SYNC-EPISODE-SUMMARY {}')",
            '--config', str(config),
        ],
    }
    queue.write_text(json.dumps(row) + '\n')
    journal = tmp_path / 'journal.jsonl'
    subprocess.run([
        sys.executable, str(repo / 'scripts' / 'run_budget_strict.py'), str(queue),
        '--hours', '0.01', '--journal', str(journal), '--logs', str(tmp_path / 'logs'),
    ], check=True, env={**__import__('os').environ, 'PYTHONPATH': str(repo / 'src')})
    result = json.loads(journal.read_text().splitlines()[-1])
    assert result['status'] == 'failed'
    assert result['failure_reason'] == 'missing_official_metrics'


def test_strict_runner_accepts_exactly_one_new_metric(tmp_path: Path):
    repo = Path(__file__).parents[1]
    output = tmp_path / 'official'
    output.mkdir()
    config = tmp_path / 'eval.yml'
    config.write_text(f'eval_output_root: {output}\n')
    queue = tmp_path / 'queue.jsonl'
    code = (
        "import os,sys; from pathlib import Path; "
        "Path(os.environ['SMP_LOG']).write_text('{}\\n'); "
        "Path(sys.argv[-1]).joinpath('new_metrics.json').write_text('{}'); "
        "print('SYNC-EPISODE-SUMMARY {}')"
    )
    row = {
        'task': 'task', 'level': 3, 'seed': 9, 'condition': 'full',
        'command': [sys.executable, '-c', code, '--config', str(config), str(output)],
    }
    queue.write_text(json.dumps(row) + '\n')
    journal = tmp_path / 'journal.jsonl'
    subprocess.run([
        sys.executable, str(repo / 'scripts' / 'run_budget_strict.py'), str(queue),
        '--hours', '0.01', '--journal', str(journal), '--logs', str(tmp_path / 'logs'),
    ], check=True, env={**__import__('os').environ, 'PYTHONPATH': str(repo / 'src')})
    result = json.loads(journal.read_text().splitlines()[-1])
    assert result['status'] == 'complete'
    assert len(result['official_metrics']) == 1
