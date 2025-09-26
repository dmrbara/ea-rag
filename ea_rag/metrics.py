from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import paths


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_run_id(prefix: str | None = None) -> str:
    base = _utc_now_id()
    if prefix:
        return f"{base}_{prefix}"
    return base


def record_timing(step: str, run_id: str, duration_ms: int, meta: Dict[str, Any] | None = None) -> None:
    timings_dir = paths.cache_dir / "timings"
    timings_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _utc_now_id(),
        "run_id": run_id,
        "step": step,
        "duration_ms": duration_ms,
        "meta": meta or {},
    }
    with (timings_dir / "timings.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


