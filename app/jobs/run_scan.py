from __future__ import annotations

import json
import os
from app.config import get_settings
from app.jobs.scan import run_scan


def main() -> None:
    settings = get_settings()
    force = os.getenv("FORCE_SCAN", "0").strip() == "1"
    res = run_scan(settings, force=force)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
