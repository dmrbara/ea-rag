from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass(frozen=True)
class TSVLiteral:
    text: str
    unit: Optional[str] = None


TSVObject = Union[str, TSVLiteral]  # entity URI or literal
Triple = Tuple[str, str, TSVObject]


def _token_to_entity_uri(token: str) -> str:
    return f"http://local/entity/{token}"


def _token_to_predicate_uri(token: str) -> str:
    return f"http://local/predicate/{token}"


def _humanize_from_token(token: str) -> str:
    return token.replace("_", " ")


def _is_iri(token: str) -> bool:
    """Heuristic: treat strings with a URI scheme as IRIs (e.g., http://, https://, urn:)."""
    if not token:
        return False
    # Common schemes
    return (
        "://" in token
        or token.startswith("urn:")
    )


def _entity_uri_from_token(token: str) -> str:
    """Return an entity URI, preserving full IRIs and only remapping bare tokens."""
    return token if _is_iri(token) else _token_to_entity_uri(token)


def _predicate_uri_from_token(token: str) -> str:
    """Return a predicate URI, preserving full IRIs and only remapping bare tokens."""
    return token if _is_iri(token) else _token_to_predicate_uri(token)


def _humanize_from_iri(iri: str) -> str:
    """Humanize an IRI by taking the local name after '/', '#', or ':', then replacing underscores."""
    local = iri
    for sep in ["#", "/", ":"]:
        if sep in local:
            local = local.rsplit(sep, 1)[-1]
    return local.replace("_", " ")


@dataclass
class TSVGraph:
    # Entities
    entity_tokens: Set[str] = field(default_factory=set)
    entity_uri_by_token: Dict[str, str] = field(default_factory=dict)
    token_by_entity_uri: Dict[str, str] = field(default_factory=dict)
    label_by_entity_uri: Dict[str, str] = field(default_factory=dict)

    # Adjacency
    outgoing: Dict[str, List[Tuple[str, TSVObject]]] = field(default_factory=dict)  # entity_uri -> [(pred_uri, obj)]
    incoming: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)  # entity_uri -> [(subj_uri, pred_uri)]


def load_tsv_graph(path: str | Path) -> TSVGraph:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"KG file not found: {p}")

    g = TSVGraph()
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # Try split on multiple spaces as a fallback
                parts = line.split()
                if len(parts) != 3:
                    # Skip malformed lines
                    continue
            s_tok, p_tok, o_tok = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not s_tok or not p_tok or not o_tok:
                continue

            s_uri = _entity_uri_from_token(s_tok)
            p_uri = _predicate_uri_from_token(p_tok)

            # Parse object: literal (e.g., "1.75"^^m) or entity token
            obj: TSVObject
            lit = _parse_literal_token(o_tok)
            if lit is not None:
                obj = lit
            else:
                obj = _entity_uri_from_token(o_tok)

            # Register entities and labels
            # Register subject entity
            if s_tok not in g.entity_tokens:
                g.entity_tokens.add(s_tok)
                g.entity_uri_by_token[s_tok] = s_uri
                g.token_by_entity_uri[s_uri] = s_tok
                # Prefer humanizing from IRI when input is an IRI; otherwise from token
                if _is_iri(s_tok):
                    g.label_by_entity_uri[s_uri] = _humanize_from_iri(s_uri)
                else:
                    g.label_by_entity_uri[s_uri] = _humanize_from_token(s_tok)

            # Register object entity if it is an entity URI
            if isinstance(obj, str):
                o_tok_entity = o_tok
                if o_tok_entity not in g.entity_tokens:
                    g.entity_tokens.add(o_tok_entity)
                    g.entity_uri_by_token[o_tok_entity] = obj
                    g.token_by_entity_uri[obj] = o_tok_entity
                    if _is_iri(o_tok_entity):
                        g.label_by_entity_uri[obj] = _humanize_from_iri(obj)
                    else:
                        g.label_by_entity_uri[obj] = _humanize_from_token(o_tok_entity)

            # Outgoing
            g.outgoing.setdefault(s_uri, []).append((p_uri, obj))
            # Incoming
            if isinstance(obj, str):
                g.incoming.setdefault(obj, []).append((s_uri, p_uri))

    return g


def get_label(g: TSVGraph, uri: str) -> str:
    return g.label_by_entity_uri.get(uri, uri.rsplit("/", 1)[-1].replace("_", " "))


def one_hop_triples(g: TSVGraph, entity_uri: str, limit: Optional[int] = None) -> List[Triple]:
    results: List[Triple] = []

    for p_uri, obj in g.outgoing.get(entity_uri, []):
        results.append((entity_uri, p_uri, obj))
        if limit and len(results) >= limit:
            return results

    for s_uri, p_uri in g.incoming.get(entity_uri, []):
        results.append((s_uri, p_uri, entity_uri))
        if limit and len(results) >= limit:
            return results

    return results


def resolve_name_to_entity_uri(g: TSVGraph, name: str) -> Optional[str]:
    """
    Resolve a target name to an entity URI. Tries in order:
    - exact token match
    - token match after replacing spaces <-> underscores
    - case-insensitive token match
    """
    if name in g.entity_tokens:
        return g.entity_uri_by_token[name]

    alt = name.replace(" ", "_")
    if alt in g.entity_tokens:
        return g.entity_uri_by_token[alt]

    # Case-insensitive search on tokens
    lower = name.lower()
    for tok in g.entity_tokens:
        if tok.lower() == lower or tok.lower() == alt.lower():
            return g.entity_uri_by_token[tok]

    return None


def _parse_literal_token(token: str) -> Optional[TSVLiteral]:
    """
    Parse a literal token like "1.75"^^m or "6929"^^xsd:integer or "Crescas".
    Returns TSVLiteral(text, unit) or None if token should be treated as an entity token.
    """
    t = token.strip()
    if not t:
        return None

    # If it looks like a quoted literal optionally with ^^type (type may be prefixed or full IRI)
    if t.startswith('"') and '"' in t[1:]:
        try:
            end_quote = t.rindex('"')
        except ValueError:
            return None
        text = t[1:end_quote]
        rest = t[end_quote + 1 :].strip()
        unit: Optional[str] = None
        if rest.startswith("^^"):
            dtype = rest[2:]
            # Normalize angle-bracketed IRIs: ^^<http://...#double> -> http://...#double
            if dtype.startswith("<") and dtype.endswith(">"):
                dtype = dtype[1:-1]
            # Treat common XSD datatypes as no unit; otherwise keep as unit label
            lowered = dtype.lower()
            if lowered.startswith("xsd:") or "/www.w3.org/2001/xmlschema#" in lowered:
                unit = None
            else:
                unit = dtype
        return TSVLiteral(text=text, unit=unit)

    # Unquoted numeric tokens: treat as literal text
    if t.replace(".", "", 1).isdigit():
        return TSVLiteral(text=t, unit=None)

    # Otherwise treat as entity token (IRI or local token)
    return None


def merge_graphs(base: TSVGraph, other: TSVGraph) -> TSVGraph:
    """Merge 'other' into 'base' and return base."""
    for tok in other.entity_tokens:
        if tok not in base.entity_tokens:
            base.entity_tokens.add(tok)
            uri = other.entity_uri_by_token[tok]
            base.entity_uri_by_token[tok] = uri
            base.token_by_entity_uri[uri] = tok
            base.label_by_entity_uri[uri] = other.label_by_entity_uri.get(uri, _humanize_from_token(tok))
    # Merge outgoing and incoming
    for s_uri, lst in other.outgoing.items():
        base.outgoing.setdefault(s_uri, []).extend(lst)
    for o_uri, lst in other.incoming.items():
        base.incoming.setdefault(o_uri, []).extend(lst)
    return base


