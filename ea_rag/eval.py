from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any


def _read_truth_csv(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source_uri")
            tgt = row.get("target_uri")
            if src and tgt:
                mapping[src] = tgt
    return mapping


def _read_predictions(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _rank_of_true(candidates: List[dict], true_uri: str) -> Optional[int]:
    # candidates expected to be ordered (as retrieved). Use 'target_uri' when present
    for idx, c in enumerate(candidates, start=1):
        cand_uri = c.get("target_uri")
        if cand_uri == true_uri:
            return idx
    return None


def evaluate(
    pred_path: str | Path,
    truth_path: str | Path,
    out_json: str | Path,
    cases_csv: str | Path | None = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    pred_p = Path(pred_path)
    truth_p = Path(truth_path)
    out_p = Path(out_json)

    truth = _read_truth_csv(truth_p)
    preds = _read_predictions(pred_p)

    total = 0
    with_truth = 0
    correct_top1 = 0

    # For retrieval-only metrics
    rr_sum = 0.0
    rr_count = 0
    hit_at = {1: 0, 3: 0, 5: 0, 10: 0}
    hit_count = 0

    case_rows: List[Dict[str, Any]] = []

    for row in preds:
        total += 1
        source_uri = row.get("source_uri")
        predicted = row.get("prediction", {}) or {}
        predicted_uri = predicted.get("target_uri")
        retrieval = bool(row.get("retrieval"))
        model = row.get("model")

        true_uri = truth.get(source_uri)
        if not true_uri:
            if progress_cb is not None:
                progress_cb(1)
            continue

        with_truth += 1
        is_correct = predicted_uri == true_uri
        if is_correct:
            correct_top1 += 1

        rr = None
        hits = {1: None, 3: None, 5: None, 10: None}

        if retrieval:
            candidates = row.get("candidates", []) or []
            # Only evaluate when candidate list contains URIs
            if any(isinstance(c, dict) and "target_uri" in c for c in candidates):
                rank = _rank_of_true(candidates, true_uri)
                if rank is not None:
                    rr = 1.0 / rank
                    rr_sum += rr
                    rr_count += 1
                # Hits
                hit_count += 1
                for k in hit_at.keys():
                    hits[k] = rank is not None and rank <= k
                    if hits[k]:
                        hit_at[k] += 1

        case_rows.append(
            {
                "source_uri": source_uri,
                "true_uri": true_uri,
                "predicted_uri": predicted_uri,
                "correct": bool(is_correct),
                "rr": rr,
                "hit@1": hits[1],
                "hit@3": hits[3],
                "hit@5": hits[5],
                "hit@10": hits[10],
                "model": model,
                "retrieval": retrieval,
            }
        )

        if progress_cb is not None:
            progress_cb(1)

    acc1 = (correct_top1 / with_truth) if with_truth else 0.0
    mrr = (rr_sum / rr_count) if rr_count else None
    summary = {
        "total_rows": total,
        "evaluated": with_truth,
        "accuracy@1": acc1,
        "mrr": mrr,
        "hit@1": (hit_at[1] / hit_count) if hit_count else None,
        "hit@3": (hit_at[3] / hit_count) if hit_count else None,
        "hit@5": (hit_at[5] / hit_count) if hit_count else None,
        "hit@10": (hit_at[10] / hit_count) if hit_count else None,
        "counts": {
            "rr_count": rr_count,
            "hit_count": hit_count,
            "correct_top1": correct_top1,
        },
    }

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=2)

    if cases_csv is not None:
        cp = Path(cases_csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_uri",
            "true_uri",
            "predicted_uri",
            "correct",
            "rr",
            "hit@1",
            "hit@3",
            "hit@5",
            "hit@10",
            "model",
            "retrieval",
        ]
        with cp.open("w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in case_rows:
                writer.writerow(r)


# --- Extras: build truth from entity_pairs.txt ---
from .entity_io import parse_entity_pairs
from .kg_tsv import load_tsv_graph, resolve_name_to_entity_uri
from .kg_parser import load_graph as load_rdf_graph, get_label as get_label_rdf


def build_truth_from_pairs(
    pairs_path: str | Path,
    kg_path: str | Path,
    kg_format: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build a truth mapping {source_uri -> target_uri} from an entity_pairs.txt file.

    - Left side: source URI, possibly prefixed with '@' → normalized to plain URI
    - Right side: target may be a name/token (no scheme) OR a full URI

    For TSV KGs, the final target URIs are the synthetic URIs used during ingestion.
    For RDF KGs, we resolve via labels.
    """
    pairs = parse_entity_pairs(pairs_path)
    truth: Dict[str, str] = {}

    chosen = kg_format
    g_rdf = None
    g_tsv = None
    if chosen == "rdf":
        g_rdf = load_rdf_graph(kg_path)
    elif chosen == "tsv":
        g_tsv = load_tsv_graph(kg_path)
    else:
        # Autodetect: try RDF then TSV
        try:
            g_rdf = load_rdf_graph(kg_path)
            chosen = "rdf"
        except Exception:
            g_tsv = load_tsv_graph(kg_path)
            chosen = "tsv"

    if chosen == "tsv" and g_tsv is not None:
        for src_uri, target_name in pairs:
            # If right side is a full IRI, use it directly (TSV graph stores IRIs as-is)
            if "://" in target_name:
                truth[src_uri] = target_name
                continue
            # Otherwise resolve token/name via TSV index
            resolved = (
                resolve_name_to_entity_uri(g_tsv, target_name)
                or resolve_name_to_entity_uri(g_tsv, target_name.replace("_", " "))
            )
            if resolved:
                truth[src_uri] = resolved
        return truth

    # RDF fallback: build label->uri index (simple exact match, underscore-space variant)
    assert g_rdf is not None
    label_to_uri: Dict[str, str] = {}
    for s in g_rdf.subjects():
        s_str = str(s)
        label = get_label_rdf(g_rdf, s_str)
        if label and label not in label_to_uri:
            label_to_uri[label] = s_str
    for src_uri, target_name in pairs:
        if "://" in target_name:
            # already a URI
            truth[src_uri] = target_name
            continue
        label = target_name.replace("_", " ")
        uri = label_to_uri.get(label)
        if uri:
            truth[src_uri] = uri
    return truth


def evaluate_with_pairs(
    pred_path: str | Path,
    pairs_path: str | Path,
    kg_path: str | Path,
    out_json: str | Path,
    cases_csv: str | Path | None = None,
    kg_format: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    preds = _read_predictions(Path(pred_path))
    truth = build_truth_from_pairs(pairs_path, kg_path, kg_format=kg_format)
    # Reuse core evaluation logic
    # Construct temporary files not needed; compute metrics inline
    total = 0
    with_truth = 0
    correct_top1 = 0
    rr_sum = 0.0
    rr_count = 0
    hit_at = {1: 0, 3: 0, 5: 0, 10: 0}
    hit_count = 0
    case_rows: List[Dict[str, Any]] = []

    for row in preds:
        total += 1
        source_uri = row.get("source_uri")
        predicted = row.get("prediction", {}) or {}
        predicted_uri = predicted.get("target_uri")
        retrieval = bool(row.get("retrieval"))
        model = row.get("model")

        true_uri = truth.get(source_uri)
        if not true_uri:
            if progress_cb is not None:
                progress_cb(1)
            continue
        with_truth += 1
        is_correct = predicted_uri == true_uri
        if is_correct:
            correct_top1 += 1
        rr = None
        hits = {1: None, 3: None, 5: None, 10: None}
        if retrieval:
            candidates = row.get("candidates", []) or []
            if any(isinstance(c, dict) and "target_uri" in c for c in candidates):
                rank = _rank_of_true(candidates, true_uri)
                if rank is not None:
                    rr = 1.0 / rank
                    rr_sum += rr
                    rr_count += 1
                hit_count += 1
                for k in hit_at.keys():
                    hits[k] = rank is not None and rank <= k
                    if hits[k]:
                        hit_at[k] += 1
        case_rows.append(
            {
                "source_uri": source_uri,
                "true_uri": true_uri,
                "predicted_uri": predicted_uri,
                "correct": bool(is_correct),
                "rr": rr,
                "hit@1": hits[1],
                "hit@3": hits[3],
                "hit@5": hits[5],
                "hit@10": hits[10],
                "model": model,
                "retrieval": retrieval,
            }
        )
        if progress_cb is not None:
            progress_cb(1)

    acc1 = (correct_top1 / with_truth) if with_truth else 0.0
    mrr = (rr_sum / rr_count) if rr_count else None
    summary = {
        "total_rows": total,
        "evaluated": with_truth,
        "accuracy@1": acc1,
        "mrr": mrr,
        "hit@1": (hit_at[1] / hit_count) if hit_count else None,
        "hit@3": (hit_at[3] / hit_count) if hit_count else None,
        "hit@5": (hit_at[5] / hit_count) if hit_count else None,
        "hit@10": (hit_at[10] / hit_count) if hit_count else None,
        "counts": {
            "rr_count": rr_count,
            "hit_count": hit_count,
            "correct_top1": correct_top1,
        },
    }

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=2)

    if cases_csv is not None:
        cp = Path(cases_csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_uri",
            "true_uri",
            "predicted_uri",
            "correct",
            "rr",
            "hit@1",
            "hit@3",
            "hit@5",
            "hit@10",
            "model",
            "retrieval",
        ]
        with cp.open("w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in case_rows:
                writer.writerow(r)


