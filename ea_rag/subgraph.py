from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rdflib import Graph, Literal

from .kg_parser import Triple as RDFTriple, get_label as get_label_rdf, one_hop_triples as one_hop_triples_rdf
from .kg_tsv import TSVGraph, TSVLiteral, get_label as get_label_tsv, one_hop_triples as one_hop_triples_tsv


def _humanize_predicate(iri: str) -> str:
    # Simple humanization: take local name and split camelCase / underscores
    local = iri
    for sep in ["#", "/", ":"]:
        if sep in local:
            local = local.rsplit(sep, 1)[-1]
    text = local.replace("_", " ")
    # Insert spaces before capitals: birthPlace -> birth Place
    out: List[str] = []
    for ch in text:
        if out and ch.isupper() and (out[-1].islower()):
            out.append(" ")
        out.append(ch)
    words = "".join(out).lower().strip()
    # If predicate already looks like a verb phrase, don't add 'has'
    verb_prefixes = (
        "is ", "was ", "were ", "has ", "have ",
        "plays ", "play ", "lives ", "lived ",
        "works ", "worked ", "located ", "born ",
        "died ", "married ", "studied ", "teaches ", "taught ",
        "coaches ", "coached ", "manages ", "managed ",
        "represents ", "represented ",
    )
    for pref in verb_prefixes:
        if words.startswith(pref):
            return words
    return f"has {words}"


def triples_to_sentences(g: Graph, triples: List[RDFTriple]) -> List[str]:
    sentences: List[str] = []
    for s, p, o in triples:
        subj = get_label_rdf(g, s)
        pred = _humanize_predicate(p)
        if isinstance(o, Literal):
            obj = str(o)
        else:
            obj = get_label_rdf(g, o)
        sentence = f"{subj} {pred} {obj}."
        sentences.append(sentence)
    return sentences


def build_target_subgraph_sentences_rdf(
    g: Graph,
    target_uri: str,
    max_triples: int | None = 64,
) -> dict:
    triples = one_hop_triples_rdf(g, target_uri, limit=max_triples)
    sentences = triples_to_sentences(g, triples)
    incoming = sum(1 for s, _, o in triples if o == target_uri)
    outgoing = len(triples) - incoming
    return {
        "entity_name": target_uri.rsplit("/", 1)[-1],
        "entity_uri": target_uri,
        "sentences": sentences,
        "triple_count": len(triples),
        "meta": {"incoming": incoming, "outgoing": outgoing},
    }


def build_target_subgraph_sentences_tsv(
    g: TSVGraph,
    target_uri: str,
    max_triples: int | None = 64,
) -> dict:
    triples = one_hop_triples_tsv(g, target_uri, limit=max_triples)
    sentences: List[str] = []
    for s, p, o in triples:
        subj = get_label_tsv(g, s)
        pred = _humanize_predicate(p)
        if isinstance(o, TSVLiteral):
            literal_text = o.text
            if o.unit:
                obj_repr = f"{literal_text} {o.unit}"
            else:
                obj_repr = literal_text
        else:
            obj_repr = get_label_tsv(g, o)
        sentences.append(f"{subj} {pred} {obj_repr}.")
    incoming = sum(1 for s, _, o in triples if o == target_uri)
    outgoing = len(triples) - incoming
    return {
        "entity_name": target_uri.rsplit("/", 1)[-1],
        "entity_uri": target_uri,
        "sentences": sentences,
        "triple_count": len(triples),
        "meta": {"incoming": incoming, "outgoing": outgoing},
    }


