## DBP15K run step-by-step (chunked, restart-friendly)

### 0) Pre-flight env check (you set these)
```bash
uv run python - << 'PY'
import os; print({k: bool(os.getenv(k)) for k in ["OPENAI_API_KEY","OPENROUTER_API_KEY","QDRANT_URL","QDRANT_API_KEY"]})
PY
```

### 1) Combine TSVs from dbp15k folders (relations only)
- Target (2)
```bash
cat /Users/davide/dev/uni/ea-rag-2040/dbp15k/triples_2 > /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_kg2_rel.tsv
```
- Source (1) (optional; only if you plan to build source-side TSV subgraphs, otherwise use RDF `data/kg.ttl` for source)
```bash
cat /Users/davide/dev/uni/ea-rag-2040/dbp15k/triples_1 > /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_kg1_rel.tsv
```

### 2) Build entity_pairs.txt from ent_links
- Left: source URI
- Right: target token (final path segment), so TSV resolution works
```bash
uv run python - << 'PY'
import re, pathlib
src_path = pathlib.Path("/Users/davide/dev/uni/ea-rag-2040/dbp15k/ent_links")
outp = pathlib.Path("/Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_entity_pairs.txt")
outp.parent.mkdir(parents=True, exist_ok=True)

def iter_lines(p: pathlib.Path):
    if p.is_dir():
        for fp in sorted(p.glob("*")):
            if fp.is_dir():
                continue
            yield from fp.open("r", encoding="utf-8")
    else:
        yield from p.open("r", encoding="utf-8")

with outp.open("w", encoding="utf-8") as wf:
    for raw in iter_lines(src_path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        src, tgt = parts[0], parts[1]
        token = tgt.rsplit("/", 1)[-1]
        wf.write(f"@{src}\t{token}\n")
print("Wrote dbp15k_entity_pairs.txt")
PY
```

### 3) Split into sources/targets
```bash
uv run python -m ea_rag.cli split-entities \
  --input /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_entity_pairs.txt \
  --sources-out /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.txt \
  --targets-out /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_targets.txt \
  --jsonl-out /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_entity_pairs.parsed.jsonl
```

### 4) Shard targets for robustness
```bash
split -l 1000 /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_targets.txt /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_targets.part_
```

### 5) Build TARGET subgraphs per shard (TSV)
```bash
for f in /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_targets.part_*; do
  uv run python -m ea_rag.cli build-subgraphs \
    --kg /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_kg2_rel.tsv \
    --kg-format tsv \
    --targets "$f" \
    --out "/Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_targets_subgraphs.$(basename "$f").jsonl" \
    --max-triples 32
done
```

### 6) Ingest TARGET sentences (per subgraph file)
Recommended batches: 256–512.
```bash
for j in /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_targets_subgraphs.dbp15k_targets.part_*.jsonl; do
  uv run python -m ea_rag.cli ingest \
    --subgraphs "$j" \
    --collection kg_target_sentences_dbp15k \
    --embed-batch-size 512 \
    --upsert-batch-size 512
done
```

### 7) Build + ingest SOURCE sentences (for source context, sharded)
If you do not want to include source-side context in retrieval, you can skip this step and run prediction without `--source-collection`.

RDF mode (recommended for DBpedia URIs in sources):
Shard (reuse if already split for prediction):
```bash
split -l 1000 /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.txt /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.part_
```
Build SOURCE subgraphs per shard (RDF KG):
```bash
for s in /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.part_*; do
  uv run python -m ea_rag.cli build-source-subgraphs \
    --kg /Users/davide/dev/uni/ea-rag-2040/data/kg.ttl \
    --kg-format rdf \
    --sources "$s" \
    --out "/Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources_subgraphs.$(basename "$s").jsonl" \
    --max-triples 16
done
```
Ingest SOURCE sentences per subgraph file:
```bash
for j in /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources_subgraphs.dbp15k_sources.part_*.jsonl; do
  uv run python -m ea_rag.cli ingest \
    --subgraphs "$j" \
    --collection kg_source_sentences_dbp15k \
    --embed-batch-size 512 \
    --upsert-batch-size 512
done
```

TSV mode (if your sources are local tokens like `http://local/entity/...`):
```bash
for s in /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.part_*; do
  uv run python -m ea_rag.cli build-source-subgraphs \
    --kg /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_kg1_rel.tsv \
    --kg-format tsv \
    --sources "$s" \
    --out "/Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources_subgraphs.$(basename "$s").jsonl" \
    --max-triples 16
done
```

### 8) Predict in source shards (restart-friendly)
Create shards:
```bash
split -l 1000 /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.txt /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.part_
```
Run each shard:
```bash
for s in /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_sources.part_*; do
  RUN=$(date -u +%Y%m%dT%H%M%SZ)
  uv run python -m ea_rag.cli predict \
    --sources "$s" \
    --target-collection kg_target_sentences_dbp15k \
    --source-collection kg_source_sentences_dbp15k \
    --model google/gemini-2.5-flash \
    --retrieval --no-rationale \
    --k 10 --max-per-uri 3 --context-chars 400 --max-tokens 200 \
    --concurrency 20 --request-timeout 90 \
    --max-retries 8 --backoff-base 1.5 \
    --out "/Users/davide/dev/uni/ea-rag-2040/artifacts/predictions/${RUN}_dbp15k_part_$(basename "$s").jsonl"
done
```

### 9) Merge predictions
```bash
cat /Users/davide/dev/uni/ea-rag-2040/artifacts/predictions/*_dbp15k_part_dbp15k_sources.part_* \
  > /Users/davide/dev/uni/ea-rag-2040/artifacts/predictions/dbp15k_all.jsonl
```

### 10) Evaluate using ent_links via --pairs (TSV resolution)
```bash
uv run python -m ea_rag.cli evaluate \
  --pred /Users/davide/dev/uni/ea-rag-2040/artifacts/predictions/dbp15k_all.jsonl \
  --pairs /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_entity_pairs.txt \
  --kg /Users/davide/dev/uni/ea-rag-2040/artifacts/dbp15k_kg2_rel.tsv \
  --kg-format tsv \
  --out /Users/davide/dev/uni/ea-rag-2040/artifacts/eval/dbp15k.json \
  --cases-csv /Users/davide/dev/uni/ea-rag-2040/artifacts/eval/dbp15k_cases.csv
```

<!-- Removed: Gemini normalization step no longer supported -->

### Ingestion robustness note (recommended small edit)
Change the ingestion skip check to require the sentinel, so reruns after partial failures don’t skip incorrectly:
```python
# ea_rag/vectorstore.py
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
```

### Suggested defaults for large runs
- Build-subgraphs: `--max-triples 32–64`
- Ingest: `--embed-batch-size 256–512`, `--upsert-batch-size 256–512`
- Predict: `--concurrency 6`, `--request-timeout 75`, `--max-retries 6`, `--backoff-base 1.0`, `--context-chars 320–512`
