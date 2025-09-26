from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class SyntheticStats:
    num_entities: int
    relation_lines: int
    attribute_lines: int
    pair_lines: int


def _entity_uri(token: str) -> str:
    return f"http://local/entity/{token}"


def generate_synthetic_dataset(
    out_rel: Path,
    out_attr: Path,
    out_pairs: Path,
    num_entities: int = 1000,
    seed: Optional[int] = None,
) -> SyntheticStats:
    """
    Generate a synthetic DBP/YAGO-like dataset with:
      - 3 relation triples per entity (TSV)
      - 3 attribute triples per entity (TSV, quoted literals supported by kg_tsv)
      - 1 entity pair per entity in the format: @URI<TAB>Name

    The TSV subject/predicate/object entries are tokens; downstream TSV loader maps
    them to URIs like http://local/entity/<token> and http://local/predicate/<token>.
    """
    # Determinism hook reserved (not strictly needed as generation is formulaic)
    if seed is not None:
        import random as _random

        _random.seed(seed)

    out_rel.parent.mkdir(parents=True, exist_ok=True)
    out_attr.parent.mkdir(parents=True, exist_ok=True)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)

    rel_count = 0
    attr_count = 0
    pairs_count = 0

    with out_rel.open("w", encoding="utf-8") as f_rel, out_attr.open("w", encoding="utf-8") as f_attr, out_pairs.open("w", encoding="utf-8") as f_pairs:
        for i in range(num_entities):
            e_tok = f"E{i}"

            # Relations (3 per entity)
            club_tok = f"Club_{i % 50}"
            country_tok = f"Country_{i % 20}"
            team_tok = f"Team_{i % 100}"

            f_rel.write(f"{e_tok}\tplaysFor\t{club_tok}\n")
            f_rel.write(f"{e_tok}\tisCitizenOf\t{country_tok}\n")
            f_rel.write(f"{team_tok}\thasPlayer\t{e_tok}\n")
            rel_count += 3

            # Attributes (3 per entity)
            shirt_num = i % 30
            height_m = 1.50 + (i % 60) / 100.0
            f_attr.write(f'{e_tok}\tshirtNumber\t"{shirt_num}"^^xsd:integer\n')
            f_attr.write(f'{e_tok}\theightMeters\t"{height_m:.2f}"^^m\n')
            f_attr.write(f'{e_tok}\thasMotto\t"Motto_{i}"\n')
            attr_count += 3

            # Entity pairs (1 per entity): @URI<TAB>Name
            f_pairs.write(f"@{_entity_uri(e_tok)}\t{e_tok}\n")
            pairs_count += 1

    return SyntheticStats(
        num_entities=num_entities,
        relation_lines=rel_count,
        attribute_lines=attr_count,
        pair_lines=pairs_count,
    )


