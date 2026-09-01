import json
from pathlib import Path
import subprocess
import sys


def test_query_log_is_absolute_across_child_cwd(tmp_path: Path):
    repo = Path(__file__).parents[1]
    child_cwd = tmp_path / "child"
    child_cwd.mkdir()
    queue = tmp_path / "queue.jsonl"
    row = {
        "task": "task",
        "level": 1,
        "seed": 7,
        "condition": "full",
        "cwd": str(child_cwd),
        "command": [
            sys.executable,
            "-c",
            "import os,pathlib; pathlib.Path(os.environ['SMP_LOG']).write_text('{}\\n'); print('SYNC-EPISODE-SUMMARY {}')",
        ],
    }
    queue.write_text(json.dumps(row) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_budget.py"),
            str(queue),
            "--hours",
            "0.01",
            "--journal",
            "journal.jsonl",
            "--logs",
            "logs",
        ],
        cwd=tmp_path,
        check=True,
    )
    result = json.loads((tmp_path / "journal.jsonl").read_text())
    assert result["status"] == "complete"
    assert (tmp_path / "logs" / "task_1_7_full.jsonl").is_file()
    assert not (child_cwd / "logs").exists()
