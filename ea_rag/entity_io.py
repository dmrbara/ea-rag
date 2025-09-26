from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Callable
import json


@dataclass(frozen=True)
class EntityPair:
    source_uri: str
    target_name: str
    line_number: int


def _normalize_uri_token(token: str) -> str:
    if token.startswith("@"):
        return token[1:]
    return token


def parse_entity_pairs(path: str | Path, progress_cb: Optional[Callable[[int], None]] = None) -> List[Tuple[str, str]]:
    """
    Parse an entity-pairs file where each non-empty line is formatted as:
        @<URI> <TAB or SPACES> <Target Name>

    Returns list of (source_uri, target_name) preserving order.
    Ignores blank lines and lines starting with '#'.
    """
    p = Path(path)
    pairs: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "\t" in line:
                left, right = line.split("\t", 1)
            else:
                # Fallback: split on first run of whitespace
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    # Skip malformed line gracefully
                    continue
                left, right = parts[0], parts[1]

            source_uri = _normalize_uri_token(left.strip())
            target_name = right.strip()
            if source_uri and target_name:
                pairs.append((source_uri, target_name))
                if progress_cb is not None:
                    progress_cb(1)
    return pairs


def _write_lines(lines: Iterable[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_split_sources_targets(
    pairs: Iterable[Tuple[str, str]],
    sources_out: str | Path,
    targets_out: str | Path,
) -> None:
    sources = [src for src, _ in pairs]
    targets = [tgt for _, tgt in pairs]
    if len(sources) != len(targets):
        raise ValueError("Sources and targets length mismatch after parsing")

    _write_lines(sources, Path(sources_out))
    _write_lines(targets, Path(targets_out))


def write_parsed_jsonl(
    pairs: Iterable[Tuple[str, str]],
    out_path: str | Path,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(json.dumps({"source_uri": src, "target_name": tgt}, ensure_ascii=False) + "\n")


