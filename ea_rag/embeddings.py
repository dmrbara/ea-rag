from __future__ import annotations

from typing import Iterable, List, Optional, Dict

from langchain_openai import OpenAIEmbeddings

from .config import models


_EMBEDDER_CACHE: Dict[str, OpenAIEmbeddings] = {}


def _get_embedder(model: Optional[str] = None) -> OpenAIEmbeddings:
    embedding_model = model or models.embedding_model
    embedder = _EMBEDDER_CACHE.get(embedding_model)
    if embedder is None:
        embedder = OpenAIEmbeddings(model=embedding_model)
        _EMBEDDER_CACHE[embedding_model] = embedder
    return embedder


def embed_sentences(
    sentences: Iterable[str],
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Embed a list of sentences using OpenAI embeddings via LangChain.

    Returns a list of vectors (list[float]) in the same order as the input.
    """
    texts = list(sentences)
    if not texts:
        return []

    embedder = _get_embedder(model)
    vectors = embedder.embed_documents(texts)
    return vectors


def embed_query(
    text: str,
    model: Optional[str] = None,
) -> List[float]:
    embedder = _get_embedder(model)
    return embedder.embed_query(text)


