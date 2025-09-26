from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class PredictRun:
    timestamp: str
    run_id: str
    duration_ms: int
    model: str
    total_items: int


@dataclass(frozen=True)
class IngestRun:
    timestamp: str
    run_id: str
    duration_ms: int
    collection: str
    sentences: int


def _iter_timings(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_predict_runs(timings_path: Path) -> List[PredictRun]:
    runs: List[PredictRun] = []
    for row in _iter_timings(timings_path):
        try:
            if row.get("step") != "predict":
                continue
            meta = row.get("meta") or {}
            model = str(meta.get("model") or "unknown")
            total_items = int(meta.get("total") or 0)
            runs.append(
                PredictRun(
                    timestamp=str(row.get("timestamp") or ""),
                    run_id=str(row.get("run_id") or ""),
                    duration_ms=int(row.get("duration_ms") or 0),
                    model=model,
                    total_items=total_items,
                )
            )
        except Exception:
            # Skip malformed rows
            continue
    return runs


def summarize_by_model(runs: List[PredictRun]) -> List[Dict[str, object]]:
    agg: Dict[str, Dict[str, object]] = {}
    for r in runs:
        a = agg.setdefault(r.model, {"model": r.model, "runs": 0, "items": 0, "total_ms": 0})
        a["runs"] = int(a["runs"]) + 1
        a["items"] = int(a["items"]) + int(r.total_items)
        a["total_ms"] = int(a["total_ms"]) + int(r.duration_ms)
    # Sort by total_ms desc
    rows = list(agg.values())
    rows.sort(key=lambda x: int(x["total_ms"]), reverse=True)
    return rows


def format_ms(ms: int) -> str:
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.3f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.1f}s"


def summarize_per_run(runs: List[PredictRun]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in runs:
        # Convert UTC id like 20250921T193751Z to local human-readable time
        when = r.timestamp
        try:
            dt_utc = datetime.strptime(r.timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            when = dt_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        rows.append(
            {
                "timestamp": r.timestamp,
                "when": when,
                "run_id": r.run_id,
                "model": r.model,
                "items": r.total_items,
                "duration_ms": r.duration_ms,
            }
        )
    # Sort newest first by timestamp (UTC id strings are lexicographically sortable)
    rows.sort(key=lambda x: str(x["timestamp"]), reverse=True)
    return rows


# Ingestion timings utilities

def load_ingest_runs(timings_path: Path) -> List[IngestRun]:
    runs: List[IngestRun] = []
    for row in _iter_timings(timings_path):
        try:
            if row.get("step") != "ingest":
                continue
            meta = row.get("meta") or {}
            collection = str(meta.get("collection") or "unknown")
            sentences = int(meta.get("sentences") or 0)
            runs.append(
                IngestRun(
                    timestamp=str(row.get("timestamp") or ""),
                    run_id=str(row.get("run_id") or ""),
                    duration_ms=int(row.get("duration_ms") or 0),
                    collection=collection,
                    sentences=sentences,
                )
            )
        except Exception:
            continue
    return runs


def summarize_ingest_by_collection(runs: List[IngestRun]) -> List[Dict[str, object]]:
    agg: Dict[str, Dict[str, object]] = {}
    for r in runs:
        a = agg.setdefault(r.collection, {"collection": r.collection, "runs": 0, "sentences": 0, "total_ms": 0})
        a["runs"] = int(a["runs"]) + 1
        a["sentences"] = int(a["sentences"]) + int(r.sentences)
        a["total_ms"] = int(a["total_ms"]) + int(r.duration_ms)
    rows = list(agg.values())
    rows.sort(key=lambda x: int(x["total_ms"]), reverse=True)
    return rows


def summarize_ingest_per_run(runs: List[IngestRun]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in runs:
        when = r.timestamp
        try:
            dt_utc = datetime.strptime(r.timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            when = dt_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        rows.append(
            {
                "timestamp": r.timestamp,
                "when": when,
                "run_id": r.run_id,
                "collection": r.collection,
                "sentences": r.sentences,
                "duration_ms": r.duration_ms,
            }
        )
    rows.sort(key=lambda x: str(x["timestamp"]), reverse=True)
    return rows

