import json
from pathlib import Path
import subprocess
import sys


def test_filter_queue_combines_selectors(tmp_path: Path):
    repo = Path(__file__).parents[1]
    rows = [
        {"task": task, "level": level, "seed": seed, "condition": condition}
        for task in ("a", "b") for level in (1, 3)
        for seed in (7, 8) for condition in ("full", "stale_hold")
    ]
    source = tmp_path / "source.jsonl"
    output = tmp_path / "selected.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    subprocess.run([
        sys.executable, str(repo / "scripts" / "filter_queue.py"), str(source),
        "--conditions", "stale_hold", "--tasks", "b", "--levels", "3",
        "--seeds", "8", "--output", str(output),
    ], check=True)
    assert [json.loads(line) for line in output.read_text().splitlines()] == [{
        "condition": "stale_hold", "level": 3, "seed": 8, "task": "b",
    }]
