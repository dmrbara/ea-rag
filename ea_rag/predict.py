from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import concurrent.futures as _fut
import threading
import httpx

import pandas as pd

from .kg_tsv import resolve_name_to_entity_uri
from .retrieval import retrieve_candidates, retrieve_source_sentences
from .llm import predict_alignment_openrouter, prompt_to_text


def _read_sources(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _name_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].replace("_", " ")


def _aggregate_candidates(
    rows: List[Dict[str, Any]],
    max_per_uri: int = 3,
    context_chars: Optional[int] = None,
) -> Dict[str, List[str]]:
    # Group sentences by entity_uri
    agg: Dict[str, List[str]] = {}
    for row in rows:
        payload = row.get("payload", {})
        uri = payload.get("entity_uri")
        sent = payload.get("sentence")
        if not uri or not sent:
            continue
        lst = agg.setdefault(uri, [])
        if len(lst) < max_per_uri:
            if context_chars is not None and len(sent) > context_chars:
                lst.append(sent[:context_chars])
            else:
                lst.append(sent)
    return agg


def _prompt_with_retrieval(
    source_uri: str,
    candidate_texts: Dict[str, List[str]],
    include_rationale: bool,
    source_sentences: Optional[List[str]] = None,
) -> Dict[str, Any]:
    name = _name_from_uri(source_uri)
    candidates = [
        {"target_uri": uri, "context": " ".join(sents)} for uri, sents in list(candidate_texts.items())[:10]
    ]
    prompt: Dict[str, Any] = {
        "source_uri": source_uri,
        "name": name,
        "candidates": candidates,
        "task": (
            "Select the best aligned target. Return JSON {target_uri:str, score:float, rationale:str}."
            if include_rationale else
            "Select the best aligned target. Return JSON {target_uri:str, score:float}."
        ),
    }
    if source_sentences:
        prompt["source_sentences"] = source_sentences
    return prompt


def _prompt_baseline(source_uri: str, candidate_names: List[str], include_rationale: bool, *, source_sentences: Optional[List[str]] = None) -> Dict[str, Any]:
    name = _name_from_uri(source_uri)
    candidates = [
        {"target_name": c} for c in candidate_names[:10]
    ]
    prompt = {
        "source_uri": source_uri,
        "name": name,
        "candidates": candidates,
        "task": (
            "Select the best aligned target. Return JSON {target_uri:str, score:float, rationale:str}."
            if include_rationale else
            "Select the best aligned target. Return JSON {target_uri:str, score:float}."
        ),
    }
    if source_sentences:
        prompt["source_sentences"] = source_sentences
    return prompt


def run_predictions(
    sources_path: str | Path,
    collection: str,
    model: str,
    retrieval: bool,
    out_path: str | Path,
    k: int = 10,
    include_rationale: bool = False,
    max_per_uri: int = 3,
    context_chars: Optional[int] = None,
    max_tokens: Optional[int] = 128,
    temperature: Optional[float] = None,
    json_mode: bool = True,
    save_candidates: bool = True,
    source_collection: Optional[str] = None,
    max_source_sentences: int = 3,
    progress_cb: Optional[Callable[[int], None]] = None,
    *,
    concurrency: int = 8,
    request_timeout: float = 60.0,
    max_retries: int = 5,
    backoff_base: float = 0.75,
) -> None:
    sources = _read_sources(Path(sources_path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    # Create a shared HTTP client for OpenRouter
    client = httpx.Client(timeout=request_timeout, http2=True)

    def _process_one(src: str) -> Tuple[str, Dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            source_sents: Optional[List[str]] = None
            if source_collection:
                src_rows = retrieve_source_sentences(src, source_collection, limit=max_source_sentences)
                src_texts: List[str] = []
                for r in src_rows:
                    payload = r.get("payload", {})
                    sent = payload.get("sentence")
                    if sent:
                        if context_chars is not None and len(sent) > context_chars:
                            src_texts.append(sent[:context_chars])
                        else:
                            src_texts.append(sent)
                source_sents = src_texts[:max_source_sentences] if src_texts else None

            if retrieval:
                raw = retrieve_candidates(_name_from_uri(src), k=k, collection=collection)
                candidates_by_uri = _aggregate_candidates(raw, max_per_uri=max_per_uri, context_chars=context_chars)
                prompt = _prompt_with_retrieval(src, candidates_by_uri, include_rationale, source_sentences=source_sents)
            else:
                name = _name_from_uri(src)
                candidate_names = [name, name.lower(), name.upper(), name.replace(" ", "")]  # placeholder baseline
                prompt = _prompt_baseline(src, candidate_names, include_rationale, source_sentences=source_sents)

            resp = predict_alignment_openrouter(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                client=client,
                max_retries=max_retries,
                backoff_base=backoff_base,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            row = {
                "source_uri": src,
                "model": model,
                "retrieval": retrieval,
                **({"candidates": list(prompt.get("candidates", []))} if save_candidates else {}),
                **({"prompt_text": prompt_to_text(prompt)} if save_candidates else {}),
                "prediction": resp,
                "latency_ms": latency_ms,
            }
            return src, row
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return src, {
                "source_uri": src,
                "model": model,
                "retrieval": retrieval,
                "prediction": {"error": str(e)},
                "latency_ms": latency_ms,
            }

    with out.open("w", encoding="utf-8") as f:
        with _fut.ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as ex:
            future_to_src = {ex.submit(_process_one, src): src for src in sources}
            for fut in _fut.as_completed(future_to_src):
                _, row = fut.result()
                with lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if progress_cb is not None:
                    try:
                        progress_cb(1)
                    except Exception:
                        pass
    # Close the HTTP client
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


