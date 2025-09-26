from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .embeddings import embed_query
from .config import env


_CLIENT: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QdrantClient(url=env.qdrant_url, api_key=env.qdrant_api_key)
    return _CLIENT


def retrieve_candidates(query_text: str, k: int, collection: str) -> List[Dict[str, Any]]:
    vec = embed_query(query_text)
    client = _get_client()
    res = client.search(
        collection_name=collection,
        query_vector=vec,
        limit=k,
        with_payload=True,
        score_threshold=None,
    )
    out: List[Dict[str, Any]] = []
    for pt in res:
        out.append({
            "score": pt.score,
            "payload": pt.payload,
        })
    return out


def retrieve_source_sentences(entity_uri: str, collection: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch sentences for a specific source entity by exact match on payload.entity_uri
    using Qdrant scroll with a filter. Returns list of dicts with payload and score=None.
    """
    client = _get_client()
    results: List[Dict[str, Any]] = []
    next_offset: Optional[Tuple[int, int]] = None
    fetched = 0
    while True:
        res = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="entity_uri", match=qmodels.MatchValue(value=entity_uri))]
            ),
            limit=min(256, (limit - fetched)) if (limit is not None and limit > fetched) else 256,
            offset=next_offset,
        )
        points, next_offset = res
        for pt in points:
            results.append({"score": None, "payload": pt.payload})
        fetched += len(points)
        if not next_offset:
            break
        if limit is not None and fetched >= limit:
            break
    return results

