from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import json

import typer

from .entity_io import parse_entity_pairs, write_split_sources_targets, write_parsed_jsonl
from .kg_parser import load_graph as load_rdf_graph, get_label as get_label_rdf
from .kg_tsv import load_tsv_graph, resolve_name_to_entity_uri
from .subgraph import (
    build_target_subgraph_sentences_rdf,
    build_target_subgraph_sentences_tsv,
)
from .config import models, paths
from .vectorstore import compute_ingestion_id, collection_has_ingestion, ingest_subgraph_jsonl_to_qdrant
from .predict import run_predictions
from .metrics import record_timing, new_run_id
from .timings import (
    load_predict_runs,
    summarize_by_model,
    summarize_per_run,
    format_ms,
    load_ingest_runs,
    summarize_ingest_by_collection,
    summarize_ingest_per_run,
)
from .eval import evaluate as eval_predictions, evaluate_with_pairs
from .synthetic import generate_synthetic_dataset

app = typer.Typer(no_args_is_help=True, help="Entity Alignment RAG pipeline CLI")


@app.command("split-entities")
def split_entities(
    input: Path = typer.Option(..., "--input", help="Entity pairs file (@URI<TAB>Name)"),
    sources_out: Path = typer.Option(..., "--sources-out", help="Output path for sources URIs"),
    targets_out: Path = typer.Option(..., "--targets-out", help="Output path for target names"),
    jsonl_out: Optional[Path] = typer.Option(None, "--jsonl-out", help="Optional parsed JSONL trace output"),
):
    """Step 2: Parse entity pairs and split into sources and targets."""
    import time as _time
    run_id = new_run_id("split")
    start = _time.perf_counter()
    total = sum(1 for _ in input.open("r", encoding="utf-8"))
    with typer.progressbar(length=total, label="Parsing entity pairs") as bar:
        def _progress(n: int) -> None:
            try:
                bar.update(n)
            except Exception:
                pass

        pairs = parse_entity_pairs(input, progress_cb=_progress)
    if not pairs:
        typer.echo("No valid pairs parsed; nothing to write.")
        raise typer.Exit(code=1)
    write_split_sources_targets(pairs, sources_out, targets_out)
    if jsonl_out is not None:
        write_parsed_jsonl(pairs, jsonl_out)
    typer.echo(f"Wrote {len(pairs)} pairs -> {sources_out}, {targets_out}")
    duration_ms = int((_time.perf_counter() - start) * 1000)
    record_timing("split-entities", run_id, duration_ms, {"pairs": len(pairs)})


@app.command("build-subgraphs")
def build_subgraphs(
    kg: Path = typer.Option(..., "--kg", help="Path to the Turtle KG (.ttl)"),
    attr: Optional[Path] = typer.Option(None, "--attr", help="Optional attributes TSV/Turtle path"),
    targets: Path = typer.Option(..., "--targets", help="Targets names file (one per line)"),
    out: Path = typer.Option(..., "--out", help="Output JSONL with target subgraph sentences"),
    max_triples: int = typer.Option(64, "--max-triples", help="Max triples per target"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    kg_format: Optional[str] = typer.Option(None, "--kg-format", help="One of: rdf, tsv. Auto if omitted."),
):
    """Step 3: Build 1-hop subgraphs and render sentences for targets."""
    import time as _time
    run_id = new_run_id("subgraphs")
    start = _time.perf_counter()
    g_rdf = None
    g_tsv = None
    chosen_format: Optional[str] = None
    if kg_format in {"rdf", "tsv"}:
        chosen_format = kg_format
    else:
        # Autodetect: try RDF, then TSV
        try:
            g_rdf = load_rdf_graph(kg)
            chosen_format = "rdf"
        except Exception:
            # Try TSV loader
            try:
                g_tsv = load_tsv_graph(kg)
                chosen_format = "tsv"
            except Exception as e:
                typer.echo(f"Failed to load KG '{kg}' as RDF or TSV: {e}")
                raise typer.Exit(code=1)

    if chosen_format == "rdf" and g_rdf is None:
        try:
            g_rdf = load_rdf_graph(kg)
        except Exception as e:
            typer.echo(f"Failed to load KG as RDF: {e}")
            raise typer.Exit(code=1)
    if chosen_format == "tsv" and g_tsv is None:
        try:
            g_tsv = load_tsv_graph(kg)
        except Exception as e:
            typer.echo(f"Failed to load KG as TSV: {e}")
            raise typer.Exit(code=1)

    # If attributes are provided, merge them (TSV path auto-detected; RDF merge not implemented for simplicity)
    if attr is not None and chosen_format == "tsv" and g_tsv is not None:
        try:
            g_attr = load_tsv_graph(attr)
            from .kg_tsv import merge_graphs

            g_tsv = merge_graphs(g_tsv, g_attr)
        except Exception as e:
            typer.echo(f"Failed to load/merge attributes '{attr}': {e}")
            raise typer.Exit(code=1)

    try:
        target_lines = [line.strip() for line in targets.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as e:
        typer.echo(f"Failed to read targets file '{targets}': {e}")
        raise typer.Exit(code=1)

    def is_uri(x: str) -> bool:
        return "://" in x

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with out.open("w", encoding="utf-8") as f, typer.progressbar(length=len(target_lines) or None, label="Building subgraphs") as bar:
            for item in target_lines:
                if chosen_format == "rdf":
                    g = g_rdf  # type: ignore
                    if is_uri(item):
                        target_uri = item
                    else:
                        target_uri = None
                        for s in g.subjects():  # type: ignore[attr-defined]
                            s_str = str(s)
                            if get_label_rdf(g, s_str) == item:
                                target_uri = s_str
                                break
                        if target_uri is None:
                            continue
                    row = build_target_subgraph_sentences_rdf(g, target_uri, max_triples=max_triples)
                else:
                    g = g_tsv  # type: ignore
                    if is_uri(item):
                        target_uri = item
                    else:
                        resolved = resolve_name_to_entity_uri(g, item)  # type: ignore[arg-type]
                        if not resolved:
                            continue
                        target_uri = resolved
                    row = build_target_subgraph_sentences_tsv(g, target_uri, max_triples=max_triples)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                try:
                    bar.update(1)
                except Exception:
                    pass
    except Exception as e:
        typer.echo(f"Failed while building/writing subgraphs: {e}")
        raise typer.Exit(code=1)

    if written == 0:
        typer.echo("No subgraphs were written (no resolvable targets).")
        raise typer.Exit(code=1)
    typer.echo(f"Wrote {written} subgraph rows -> {out}")
    duration_ms = int((_time.perf_counter() - start) * 1000)
    record_timing("build-subgraphs", run_id, duration_ms, {"written": written})


@app.command("build-target-subgraphs")
def build_target_subgraphs(
    kg: Path = typer.Option(..., "--kg", help="Path to the Turtle/TSV KG"),
    attr: Optional[Path] = typer.Option(None, "--attr", help="Optional attributes TSV/Turtle path"),
    targets: Path = typer.Option(..., "--targets", help="Targets names/URIs file (one per line)"),
    out: Path = typer.Option(..., "--out", help="Output JSONL with target subgraph sentences"),
    max_triples: int = typer.Option(64, "--max-triples", help="Max triples per target"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    kg_format: Optional[str] = typer.Option(None, "--kg-format", help="One of: rdf, tsv. Auto if omitted."),
):
    """Alias of build-subgraphs for clarity (targets)."""
    return build_subgraphs(kg=kg, attr=attr, targets=targets, out=out, max_triples=max_triples, seed=seed, kg_format=kg_format)


@app.command("build-source-subgraphs")
def build_source_subgraphs(
    kg: Path = typer.Option(..., "--kg", help="Path to the Turtle/TSV KG for sources"),
    attr: Optional[Path] = typer.Option(None, "--attr", help="Optional attributes TSV path (TSV mode only)"),
    sources: Path = typer.Option(..., "--sources", help="Sources URIs file (one per line)"),
    out: Path = typer.Option(..., "--out", help="Output JSONL with source subgraph sentences"),
    max_triples: int = typer.Option(64, "--max-triples", help="Max triples per source"),
    kg_format: Optional[str] = typer.Option(None, "--kg-format", help="One of: rdf, tsv. Auto if omitted."),
):
    """Build 1-hop subgraphs for sources (RDF or TSV) and render sentences."""
    import time as _time
    run_id = new_run_id("src_subgraphs")
    start = _time.perf_counter()

    # Load KG (auto-detect RDF vs TSV unless explicitly specified)
    g_rdf = None
    g_tsv = None
    chosen_format: Optional[str] = None
    if kg_format in {"rdf", "tsv"}:
        chosen_format = kg_format
    else:
        # Autodetect: try RDF first (URIs in sources typically DBpedia), then TSV
        try:
            g_rdf = load_rdf_graph(kg)
            chosen_format = "rdf"
        except Exception:
            try:
                g_tsv = load_tsv_graph(kg)
                chosen_format = "tsv"
            except Exception as e:
                typer.echo(f"Failed to load KG '{kg}' as RDF or TSV: {e}")
                raise typer.Exit(code=1)

    if chosen_format == "rdf" and g_rdf is None:
        try:
            g_rdf = load_rdf_graph(kg)
        except Exception as e:
            typer.echo(f"Failed to load KG as RDF: {e}")
            raise typer.Exit(code=1)
    if chosen_format == "tsv" and g_tsv is None:
        try:
            g_tsv = load_tsv_graph(kg)
        except Exception as e:
            typer.echo(f"Failed to load KG as TSV: {e}")
            raise typer.Exit(code=1)

    # Merge TSV attributes if provided
    if attr is not None and chosen_format == "tsv" and g_tsv is not None:
        try:
            from .kg_tsv import merge_graphs
            g_attr = load_tsv_graph(attr)
            g_tsv = merge_graphs(g_tsv, g_attr)
        except Exception as e:
            typer.echo(f"Failed to load/merge TSV attributes '{attr}': {e}")
            raise typer.Exit(code=1)
    elif attr is not None and chosen_format == "rdf":
        # For simplicity, RDF attribute merging is not supported.
        typer.echo("Note: --attr is ignored in RDF mode.")

    try:
        source_lines = [line.strip() for line in sources.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as e:
        typer.echo(f"Failed to read sources file '{sources}': {e}")
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        from .subgraph import build_target_subgraph_sentences_tsv, build_target_subgraph_sentences_rdf
        with out.open("w", encoding="utf-8") as f, typer.progressbar(length=len(source_lines) or None, label="Building source subgraphs") as bar:
            for uri in source_lines:
                if chosen_format == "rdf":
                    row = build_target_subgraph_sentences_rdf(g_rdf, uri, max_triples=max_triples)  # type: ignore[arg-type]
                else:
                    row = build_target_subgraph_sentences_tsv(g_tsv, uri, max_triples=max_triples)  # type: ignore[arg-type]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                try:
                    bar.update(1)
                except Exception:
                    pass
    except Exception as e:
        typer.echo(f"Failed while building/writing source subgraphs: {e}")
        raise typer.Exit(code=1)
    typer.echo(f"Wrote {written} source subgraph rows -> {out}")
    duration_ms = int((_time.perf_counter() - start) * 1000)
    record_timing("build-source-subgraphs", run_id, duration_ms, {"written": written})


@app.command("generate-synthetic")
def generate_synthetic(
    kg_rel: Path = typer.Option(..., "--kg-rel", help="Output TSV for relation triples"),
    kg_attr: Path = typer.Option(..., "--kg-attr", help="Output TSV for attribute triples"),
    pairs_out: Path = typer.Option(..., "--pairs", help="Output entity_pairs.txt"),
    num_entities: int = typer.Option(1000, "--num-entities", help="Number of entities to generate"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Optional seed for determinism"),
):
    """Step 1: Generate synthetic TSV KGs and entity pairs."""
    import time as _time
    run_id = new_run_id("synthetic")
    start = _time.perf_counter()
    try:
        stats = generate_synthetic_dataset(kg_rel, kg_attr, pairs_out, num_entities=num_entities, seed=seed)
    except Exception as e:
        typer.echo(f"Failed to generate synthetic dataset: {e}")
        raise typer.Exit(code=1)
    typer.echo(
        f"Generated synthetic dataset: entities={stats.num_entities}, relations={stats.relation_lines}, attributes={stats.attribute_lines}, pairs={stats.pair_lines}"
    )
    duration_ms = int((_time.perf_counter() - start) * 1000)
    record_timing(
        "generate-synthetic",
        run_id,
        duration_ms,
        {
            "num_entities": stats.num_entities,
            "relations": stats.relation_lines,
            "attributes": stats.attribute_lines,
            "pairs": stats.pair_lines,
        },
    )


@app.command("ingest")
def ingest(
    subgraphs: Path = typer.Option(..., "--subgraphs", help="Input target subgraphs JSONL"),
    collection: str = typer.Option(..., "--collection", help="Qdrant collection name"),
    qdrant_url: Optional[str] = typer.Option(None, "--qdrant-url", help="Qdrant URL (defaults to env)"),
    embed_batch_size: int = typer.Option(1024, "--embed-batch-size", min=1, help="Number of sentences per embedding request"),
    upsert_batch_size: int = typer.Option(1024, "--upsert-batch-size", min=1, help="Number of vectors per Qdrant upsert"),
):
    """Step 4: Embed sentences and ingest into Qdrant (idempotent)."""
    import time as _time
    run_id = new_run_id("ingest")
    start = _time.perf_counter()
    try:
        ingestion_id = compute_ingestion_id(subgraphs, models.embedding_model)
    except Exception as e:
        typer.echo(f"Failed to compute ingestion id: {e}")
        raise typer.Exit(code=1)

    # Quick check if ingestion already present
    try:
        from .vectorstore import get_client

        client = get_client(url=str(qdrant_url) if qdrant_url else None)
        if collection_has_ingestion(client, collection, ingestion_id):
            typer.echo(f"Collection '{collection}' already has ingestion_id {ingestion_id}; skipping.")
            raise typer.Exit(code=0)
    except Exception as e:
        typer.echo(f"Warning: could not check existing ingestion: {e}")

    try:
        total_sentences = 0
        # Estimate total sentences for progress bar
        with subgraphs.open("r", encoding="utf-8") as rf:
            for line in rf:
                if not line.strip():
                    continue
                obj = json.loads(line)
                total_sentences += len(obj.get("sentences", []))

        with typer.progressbar(length=total_sentences or None, label="Embedding & ingesting") as bar:
            def _progress(n: int) -> None:
                try:
                    bar.update(n)
                except Exception:
                    pass

            ingest_subgraph_jsonl_to_qdrant(
                subgraphs,
                collection,
                ingestion_id,
                embed_batch_size=embed_batch_size,
                upsert_batch_size=upsert_batch_size,
                progress_cb=_progress,
            )
    except Exception as e:
        typer.echo(f"Failed to ingest: {e}")
        raise typer.Exit(code=1)
    typer.echo(f"Ingested subgraphs into '{collection}' with ingestion_id {ingestion_id}")
    duration_ms = int((_time.perf_counter() - start) * 1000)
    record_timing("ingest", run_id, duration_ms, {"collection": collection, "sentences": total_sentences})


@app.command("predict")
def predict(
    sources: Path = typer.Option(..., "--sources", help="Sources URIs file (one per line)"),
    target_collection: str = typer.Option(..., "--target-collection", "--collection", help="Qdrant collection with TARGET sentences"),
    model: str = typer.Option(..., "--model", help="OpenRouter model id"),
    retrieval: bool = typer.Option(True, "--retrieval/--no-retrieval", help="Use retrieval; disable for baseline"),
    rationale: bool = typer.Option(False, "--rationale/--no-rationale", help="Include rationale in LLM output"),
    k: int = typer.Option(10, "--k", help="Top-k neighbors to retrieve"),
    max_per_uri: int = typer.Option(3, "--max-per-uri", help="Max sentences per candidate URI"),
    context_chars: Optional[int] = typer.Option(None, "--context-chars", help="Truncate each sentence to N chars"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="LLM max tokens for response"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM temperature; if omitted, not sent"),
    no_save_candidates: bool = typer.Option(False, "--no-save-candidates", help="Do not persist candidates in output"),
    source_collection: Optional[str] = typer.Option(None, "--source-collection", help="Optional Qdrant collection with SOURCE sentences"),
    max_source_sentences: int = typer.Option(3, "--max-source-sentences", help="Max source sentences to include"),
    out: Path = typer.Option(..., "--out", help="Output predictions JSONL path"),
    concurrency: int = typer.Option(8, "--concurrency", min=1, help="Concurrent requests to OpenRouter"),
    request_timeout: float = typer.Option(60.0, "--request-timeout", help="Per-request timeout seconds"),
    max_retries: int = typer.Option(5, "--max-retries", help="Max retries for transient errors (408/429/5xx/timeouts)"),
    backoff_base: float = typer.Option(0.75, "--backoff-base", help="Base seconds for exponential backoff with jitter"),
):
    """Step 5: Run LLM predictions with or without retrieval."""
    import time as _time
    run_id = new_run_id("predict")
    start = _time.perf_counter()
    try:
        total = sum(1 for _ in sources.open("r", encoding="utf-8"))
        with typer.progressbar(length=total or None, label="Predicting") as bar:
            def _progress(n: int) -> None:
                try:
                    bar.update(n)
                except Exception:
                    pass

            run_predictions(
                sources,
                target_collection,
                model,
                retrieval,
                out,
                k=k,
                include_rationale=rationale,
                max_per_uri=max_per_uri,
                context_chars=context_chars,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
                save_candidates=not no_save_candidates,
                source_collection=source_collection,
                max_source_sentences=max_source_sentences,
                progress_cb=_progress,
                concurrency=concurrency,
                request_timeout=request_timeout,
                max_retries=max_retries,
                backoff_base=backoff_base,
                # OpenRouter only
            )
    except Exception as e:
        typer.echo(f"Prediction failed: {e}")
        raise typer.Exit(code=1)
    typer.echo(f"Predictions written to {out}")
    duration_ms = int((_time.perf_counter() - start) * 1000)
    record_timing("predict", run_id, duration_ms, {"target_collection": target_collection, "model": model, "retrieval": retrieval, "total": total})


@app.command("evaluate")
def evaluate(
    pred: Path = typer.Option(..., "--pred", help="Predictions JSONL path"),
    truth: Optional[Path] = typer.Option(None, "--truth", help="Ground truth CSV path (source_uri,target_uri)"),
    pairs: Optional[Path] = typer.Option(None, "--pairs", help="entity_pairs.txt path (alternative to --truth)"),
    kg: Optional[Path] = typer.Option(None, "--kg", help="KG file used to resolve target names to URIs when using --pairs"),
    kg_format: Optional[str] = typer.Option(None, "--kg-format", help="KG format when using --pairs: rdf or tsv (auto if omitted)"),
    out: Path = typer.Option(..., "--out", help="Evaluation JSON summary output"),
    cases_csv: Optional[Path] = typer.Option(None, "--cases-csv", help="Optional per-case CSV output"),
):
    """Step 6: Evaluate predictions against ground truth."""
    import time as _time
    run_id = new_run_id("evaluate")
    start = _time.perf_counter()
    try:
        total = sum(1 for _ in pred.open("r", encoding="utf-8"))
        with typer.progressbar(length=total or None, label="Evaluating") as bar:
            def _progress(n: int) -> None:
                try:
                    bar.update(n)
                except Exception:
                    pass
            if pairs is not None:
                if kg is None:
                    raise ValueError("--kg is required when using --pairs")
                evaluate_with_pairs(pred, pairs, kg, out, cases_csv, kg_format=kg_format, progress_cb=_progress)
            else:
                if truth is None:
                    raise ValueError("Either --truth or --pairs must be provided")
                eval_predictions(pred, truth, out, cases_csv, progress_cb=_progress)
    except Exception as e:
        typer.echo(f"Evaluation failed: {e}")
        raise typer.Exit(code=1)
    typer.echo(f"Evaluation summary written to {out}")
    duration_ms = int((_time.perf_counter() - start) * 1000)
    record_timing("evaluate", run_id, duration_ms, {"pred": str(pred), "truth": str(truth) if truth else None, "pairs": str(pairs) if pairs else None})


@app.command("timings-predict")
def timings_predict(
    timings: Optional[Path] = typer.Option(None, "--timings", help="Path to timings.jsonl (defaults to artifacts/cache/timings/timings.jsonl)"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only include latest N runs (0=all)"),
    by_model: bool = typer.Option(False, "--by-model", help="Aggregate by model instead of listing per-run"),
):
    """Show prediction timing information.

    Default: list individual runs with timestamp, run_id, model, items, and duration.
    Use --by-model to aggregate totals per model.
    """
    timings_path = timings or (paths.cache_dir / "timings" / "timings.jsonl")
    if not timings_path.exists():
        typer.echo(f"No timings file found at {timings_path}")
        raise typer.Exit(code=1)
    runs = load_predict_runs(timings_path)
    # Ensure newest-first ordering when applying limit at run level
    # summarize_per_run sorts, but apply limit consistently for both modes here
    runs.sort(key=lambda r: str(r.timestamp), reverse=True)
    if limit is not None and limit >= 0:
        runs = runs[:limit or None]
    if not runs:
        typer.echo("No prediction runs found.")
        raise typer.Exit(code=0)
    if by_model:
        rows = summarize_by_model(runs)
        if json_out:
            typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
            return
        headers = ["Model", "Runs", "Items", "Total time"]
        model_w = max(len(headers[0]), max((len(str(r["model"])) for r in rows), default=0))
        runs_w = max(len(headers[1]), max((len(str(r["runs"])) for r in rows), default=0))
        items_w = max(len(headers[2]), max((len(str(r["items"])) for r in rows), default=0))
        time_w = len(headers[3])
        typer.echo(f"{headers[0]:<{model_w}}  {headers[1]:>{runs_w}}  {headers[2]:>{items_w}}  {headers[3]:>{time_w}}")
        typer.echo(f"{'-'*model_w}  {'-'*runs_w}  {'-'*items_w}  {'-'*time_w}")
        for r in rows:
            typer.echo(
                f"{str(r['model']):<{model_w}}  {int(r['runs']):>{runs_w}d}  {int(r['items']):>{items_w}d}  {format_ms(int(r['total_ms'])):>{time_w}}"
            )
        return
    # Per-run mode
    rows = summarize_per_run(runs)
    if json_out:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    headers = ["When", "Run ID", "Model", "Items", "Duration"]
    when_w = max(len(headers[0]), max((len(str(r.get("when") or "")) for r in rows), default=0))
    runid_w = max(len(headers[1]), max((len(str(r["run_id"])) for r in rows), default=0))
    model_w = max(len(headers[2]), max((len(str(r["model"])) for r in rows), default=0))
    items_w = max(len(headers[3]), max((len(str(r["items"])) for r in rows), default=0))
    dur_w = len(headers[4])
    typer.echo(f"{headers[0]:<{when_w}}  {headers[1]:<{runid_w}}  {headers[2]:<{model_w}}  {headers[3]:>{items_w}}  {headers[4]:>{dur_w}}")
    typer.echo(f"{'-'*when_w}  {'-'*runid_w}  {'-'*model_w}  {'-'*items_w}  {'-'*dur_w}")
    for r in rows:
        typer.echo(
            f"{str(r.get('when') or ''):<{when_w}}  {str(r['run_id']):<{runid_w}}  {str(r['model']):<{model_w}}  {int(r['items']):>{items_w}d}  {format_ms(int(r['duration_ms'])):>{dur_w}}"
        )


@app.command("timings-ingest")
def timings_ingest(
    timings: Optional[Path] = typer.Option(None, "--timings", help="Path to timings.jsonl (defaults to artifacts/cache/timings/timings.jsonl)"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of rows in output"),
    by_collection: bool = typer.Option(False, "--by-collection", help="Aggregate by collection instead of listing per-run"),
):
    """Show ingestion timing information.

    Default: list individual runs with timestamp, run_id, collection, sentences, and duration.
    Use --by-collection to aggregate totals per collection.
    """
    timings_path = timings or (paths.cache_dir / "timings" / "timings.jsonl")
    if not timings_path.exists():
        typer.echo(f"No timings file found at {timings_path}")
        raise typer.Exit(code=1)
    runs = load_ingest_runs(timings_path)
    if not runs:
        typer.echo("No ingestion runs found.")
        raise typer.Exit(code=0)
    if by_collection:
        rows = summarize_ingest_by_collection(runs)
        if limit is not None and limit >= 0:
            rows = rows[:limit]
        if json_out:
            typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
            return
        headers = ["Collection", "Runs", "Sentences", "Total time"]
        col_w = max(len(headers[0]), max((len(str(r["collection"])) for r in rows), default=0))
        runs_w = max(len(headers[1]), max((len(str(r["runs"])) for r in rows), default=0))
        sent_w = max(len(headers[2]), max((len(str(r["sentences"])) for r in rows), default=0))
        time_w = len(headers[3])
        typer.echo(f"{headers[0]:<{col_w}}  {headers[1]:>{runs_w}}  {headers[2]:>{sent_w}}  {headers[3]:>{time_w}}")
        typer.echo(f"{'-'*col_w}  {'-'*runs_w}  {'-'*sent_w}  {'-'*time_w}")
        for r in rows:
            typer.echo(
                f"{str(r['collection']):<{col_w}}  {int(r['runs']):>{runs_w}d}  {int(r['sentences']):>{sent_w}d}  {format_ms(int(r['total_ms'])):>{time_w}}"
            )
        return
    # Per-run mode
    rows = summarize_ingest_per_run(runs)
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    if json_out:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    headers = ["When", "Run ID", "Collection", "Sentences", "Duration"]
    when_w = max(len(headers[0]), max((len(str(r.get("when") or "")) for r in rows), default=0))
    runid_w = max(len(headers[1]), max((len(str(r["run_id"])) for r in rows), default=0))
    col_w = max(len(headers[2]), max((len(str(r["collection"])) for r in rows), default=0))
    sent_w = max(len(headers[3]), max((len(str(r["sentences"])) for r in rows), default=0))
    dur_w = len(headers[4])
    typer.echo(f"{headers[0]:<{when_w}}  {headers[1]:<{runid_w}}  {headers[2]:<{col_w}}  {headers[3]:>{sent_w}}  {headers[4]:>{dur_w}}")
    typer.echo(f"{'-'*when_w}  {'-'*runid_w}  {'-'*col_w}  {'-'*sent_w}  {'-'*dur_w}")
    for r in rows:
        typer.echo(
            f"{str(r.get('when') or ''):<{when_w}}  {str(r['run_id']):<{runid_w}}  {str(r['collection']):<{col_w}}  {int(r['sentences']):>{sent_w}d}  {format_ms(int(r['duration_ms'])):>{dur_w}}"
        )

@app.command("analyze-subgraphs")
def analyze_subgraphs(
    inputs: List[Path] = typer.Argument(..., help="Subgraph JSONL files or directories containing them"),
    pattern: str = typer.Option(
        "targets_subgraphs*.jsonl",
        "--pattern",
        help="Glob pattern used when scanning directories (recursive)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON instead of text"),
):
    """Analyze subgraph JSONL files and report sentence counts per file and in total.

    A sentence is one triple rendered to text. For each file, we count the number
    of entities (rows) and sum the per-row `triple_count` to get total sentences.
    """
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(p.rglob(pattern)))
        else:
            files.append(p)
    # De-duplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for fp in files:
        key = fp.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(fp)
    if not unique_files:
        typer.echo("No input files matched.")
        raise typer.Exit(code=1)

    per_file = []
    total_entities = 0
    total_sentences = 0
    for fp in unique_files:
        entities = 0
        sentences = 0
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    entities += 1
                    sentences += int(obj.get("triple_count") or len(obj.get("sentences", [])) or 0)
        except Exception as e:
            typer.echo(f"Failed to read {fp}: {e}")
            raise typer.Exit(code=1)
        per_file.append({"path": str(fp), "entities": entities, "sentences": sentences})
        total_entities += entities
        total_sentences += sentences

    if json_out:
        typer.echo(
            json.dumps(
                {
                    "files": per_file,
                    "total": {"entities": total_entities, "sentences": total_sentences},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    for row in per_file:
        typer.echo(f"{row['path']}: entities={row['entities']}, sentences={row['sentences']}")
    typer.echo(f"TOTAL: entities={total_entities}, sentences={total_sentences}")

def main() -> None:
    app()


if __name__ == "__main__":
    main()


