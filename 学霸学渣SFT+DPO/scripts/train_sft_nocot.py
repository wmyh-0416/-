#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sft_cot import main as shared_train_main  # noqa: E402


if __name__ == "__main__":
    if "--variant" not in sys.argv:
        sys.argv.extend(["--variant", "nocot"])
    raise SystemExit(shared_train_main())
