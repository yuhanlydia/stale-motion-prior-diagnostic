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


def test_journal_records_only_new_official_metrics(tmp_path: Path):
    repo = Path(__file__).parents[1]
    output = tmp_path / "official"
    output.mkdir()
    (output / "old_metrics.json").write_text("{}")
    config = tmp_path / "eval.yml"
    config.write_text(f"eval_output_root: {output}\n")
    queue = tmp_path / "queue.jsonl"
    row = {
        "task": "task", "level": 3, "seed": 9, "condition": "full",
        "command": [
            sys.executable, "-c",
            "import os,pathlib,sys; pathlib.Path(os.environ['SMP_LOG']).write_text('{}\\n'); pathlib.Path(sys.argv[3]).joinpath('new_metrics.json').write_text('{}'); print('SYNC-EPISODE-SUMMARY {}')",
            "--config", str(config), str(output),
        ],
    }
    queue.write_text(json.dumps(row) + "\n")
    subprocess.run([
        sys.executable, str(repo / "scripts" / "run_budget.py"), str(queue),
        "--hours", "0.01", "--journal", str(tmp_path / "journal.jsonl"),
        "--logs", str(tmp_path / "logs"),
    ], check=True)
    result = json.loads((tmp_path / "journal.jsonl").read_text())
    assert result["official_metrics"] == [str(output / "new_metrics.json")]
