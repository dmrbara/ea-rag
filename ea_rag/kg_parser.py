from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDFS, SKOS


Triple = Tuple[str, str, str | Literal]


def load_graph(path: str | Path, fmt: Optional[str] = None) -> Graph:
    """
    Load an RDF graph with basic auto-detection and fallbacks.

    If fmt is provided, it is used directly. Otherwise, try formats based
    on file suffix and common RDF serializations.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"KG file not found: {p}")

    candidates: List[str]
    if fmt:
        candidates = [fmt]
    else:
        candidates = []
        suffix = p.suffix.lower()
        if suffix in {".ttl"}:
            candidates.append("turtle")
        elif suffix in {".nt"}:
            candidates.append("nt")
        elif suffix in {".rdf", ".xml"}:
            candidates.append("xml")
        elif suffix in {".n3"}:
            candidates.append("n3")
        elif suffix in {".trig"}:
            candidates.append("trig")
        elif suffix in {".nq", ".nquads"}:
            candidates.append("nquads")
        # Generic fallbacks (de-dup will keep order)
        candidates.extend(["turtle", "nt", "xml", "n3"])  # common
        seen: set[str] = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    parse_errors: List[str] = []
    for candidate in candidates:
        g = Graph()
        try:
            g.parse(str(p), format=candidate)
            return g
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{candidate}: {exc.__class__.__name__}")
            continue

    raise ValueError(
        "Failed to parse KG. Tried formats: "
        + ", ".join(candidates)
        + ("; errors: " + "; ".join(parse_errors) if parse_errors else "")
    )


def _local_name(uri: str) -> str:
    # Split on common separators for CURIE-like forms
    for sep in ["#", "/", ":"]:
        if sep in uri:
            uri = uri.rsplit(sep, 1)[-1]
    return uri.replace("_", " ")


def get_label(g: Graph, uri: str) -> str:
    ref = URIRef(uri)
    # Prefer rdfs:label
    for o in g.objects(ref, RDFS.label):
        if isinstance(o, Literal) and str(o).strip():
            return str(o)
    # Fallback to skos:prefLabel
    for o in g.objects(ref, SKOS.prefLabel):
        if isinstance(o, Literal) and str(o).strip():
            return str(o)
    # Fallback to local name
    return _local_name(uri)


def extract_entities_and_triples(g: Graph) -> Tuple[Set[str], List[Triple]]:
    entities: Set[str] = set()
    triples: List[Triple] = []
    for s, p, o in g:
        s_is_iri = isinstance(s, URIRef)
        p_is_iri = isinstance(p, URIRef)
        o_is_iri = isinstance(o, URIRef)

        if s_is_iri:
            entities.add(str(s))
        if o_is_iri:
            entities.add(str(o))

        if s_is_iri and p_is_iri:
            triples.append((str(s), str(p), str(o) if o_is_iri else o))
    return entities, triples


def one_hop_triples(g: Graph, entity_uri: str, limit: Optional[int] = None) -> List[Triple]:
    ref = URIRef(entity_uri)
    results: List[Triple] = []
    # Outgoing
    for p, o in g.predicate_objects(ref):
        if isinstance(p, URIRef):
            results.append((entity_uri, str(p), str(o) if isinstance(o, URIRef) else o))
            if limit and len(results) >= limit:
                return results
    # Incoming
    for s, p in g.subject_predicates(ref):
        if isinstance(p, URIRef):
            results.append((str(s), str(p), entity_uri))
            if limit and len(results) >= limit:
                return results
    return results


