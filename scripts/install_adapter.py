#!/usr/bin/env python3
"""Install the diagnostic adapter into a pinned DynamicWAM checkout.

The original file is retained as .smp-original.py and can be restored with
--restore. This is the only source patch required by the experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


RELATIVE = Path("src/dynamicwam/runtime/ciwam/adapters/domino/deploy_policy_sync_flow.py")
SHIM = '''"""Generated shim for stale-motion-prior-diagnostic."""\nfrom stale_motion_prior.domino_adapter import eval, get_model, reset_model\n\n__all__ = ["eval", "get_model", "reset_model"]\n'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dynamicwam_root", type=Path)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    target = args.dynamicwam_root.resolve() / RELATIVE
    backup = target.with_suffix(".smp-original.py")
    if args.restore:
        if not backup.exists():
            raise FileNotFoundError(backup)
        shutil.copy2(backup, target)
        print(f"restored {target}")
        return 0
    if not target.exists():
        raise FileNotFoundError(target)
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(SHIM, encoding="utf-8")
    print(f"installed diagnostic shim at {target}; backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

