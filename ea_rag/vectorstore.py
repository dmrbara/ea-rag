from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
import time
import random
from pathlib import Path
from typing import Iterable, List, Optional, Callable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import env
from .embeddings import embed_sentences


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_ingestion_id(
    subgraphs_path: Path,
    embedding_model: str,
) -> str:
    payload = {
        "subgraphs_sha256": _file_sha256(subgraphs_path),
        "embedding_model": embedding_model,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def get_client(url: Optional[str] = None, api_key: Optional[str] = None) -> QdrantClient:
    # Prefer explicit arg, then QDRANT_API_KEY from env. For local Docker, api_key may be None.
    key = api_key if api_key is not None else env.qdrant_api_key
    # Prefer gRPC for higher throughput; falls back to REST if gRPC port is unavailable
    return QdrantClient(url=url or env.qdrant_url, api_key=key, prefer_grpc=True, grpc_port=6334)


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    exists = False
    try:
        info = client.get_collection(name)
        exists = info is not None
    except Exception:
        exists = False
    if not exists:
        client.recreate_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )


def collection_has_ingestion(client: QdrantClient, collection: str, ingestion_id: str) -> bool:
    try:
        res = client.scroll(
            collection_name=collection,
            with_payload=True,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="ingestion_id", match=qmodels.MatchValue(value=ingestion_id)),
                    qmodels.FieldCondition(key="sentinel", match=qmodels.MatchValue(value=True)),
                ]
            ),
            limit=1,
        )
        points, _ = res
        return len(points) > 0
    except Exception:
        return False


def ingest_subgraph_jsonl_to_qdrant(
    jsonl_path: Path,
    collection: str,
    ingestion_id: str,
    embed_batch_size: int = 1024,
    upsert_batch_size: int = 1024,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    client = get_client()

    # Streaming pipeline: accumulate texts across rows to build large batches,
    # embed, then upsert immediately. This reduces memory footprint and overlaps
    # network work with Qdrant ingestion more efficiently.
    pending_texts: List[str] = []
    pending_payloads: List[dict] = []
    pending_ids: List[int] = []
    # Derive a stable numeric offset from ingestion_id to avoid ID collisions across shards.
    # Clamp to 36 bits so (offset * 10_000_000) fits safely within 64-bit integer limits.
    shard_offset = (int(hashlib.sha256(ingestion_id.encode("utf-8")).hexdigest(), 16) & ((1 << 36) - 1)) * 10_000_000
    next_id = shard_offset
    first_vector_size: Optional[int] = None

    def _retry_call(fn, *args, attempts: int = 6, base_delay: float = 0.5, max_delay: float = 8.0, **kwargs):
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if i == attempts - 1:
                    raise
                delay = min(max_delay, base_delay * (2 ** i)) * (0.5 + random.random())
                time.sleep(delay)

    def _upsert_vectors(vecs: List[List[float]], payloads: List[dict], ids: List[int]) -> None:
        nonlocal first_vector_size
        if not vecs:
            return
        if first_vector_size is None:
            first_vector_size = len(vecs[0])
            ensure_collection(client, collection, vector_size=first_vector_size)
        for i in range(0, len(vecs), upsert_batch_size):
            _retry_call(
                client.upsert,
                collection_name=collection,
                points=qmodels.Batch(
                    ids=ids[i : i + upsert_batch_size],
                    vectors=vecs[i : i + upsert_batch_size],
                    payloads=payloads[i : i + upsert_batch_size],
                ),
            )

    def _flush() -> None:
        nonlocal pending_texts, pending_payloads, pending_ids
        if not pending_texts:
            return
        vecs = _retry_call(embed_sentences, pending_texts)
        _upsert_vectors(vecs, pending_payloads, pending_ids)
        if progress_cb is not None:
            try:
                progress_cb(len(pending_texts))
            except Exception:
                pass
        pending_texts = []
        pending_payloads = []
        pending_ids = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            entity_uri = row.get("entity_uri")
            entity_name = row.get("entity_name")
            triple_count = row.get("triple_count")
            sentences: List[str] = row.get("sentences", [])
            if not sentences:
                continue
            for sent in sentences:
                pending_texts.append(sent)
                pending_payloads.append(
                    {
                        "entity_uri": entity_uri,
                        "entity_name": entity_name,
                        "sentence": sent,
                        "ingestion_id": ingestion_id,
                        "triple_count": triple_count,
                    }
                )
                pending_ids.append(next_id)
                next_id += 1
                if len(pending_texts) >= embed_batch_size:
                    _flush()

    # Final flush
    _flush()

    # If nothing was ingested, exit early
    if first_vector_size is None:
        return

    # Write a sentinel point to mark completion
    client.upsert(
        collection_name=collection,
        points=qmodels.Batch(
            ids=[shard_offset + 9_999_999],
            vectors=[[0.0] * first_vector_size],
            payloads=[{"sentinel": True, "ingestion_id": ingestion_id}],
        ),
    )


