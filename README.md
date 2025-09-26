## EA-RAG

Typer-based CLI scaffolding for the Entity Alignment RAG pipeline.

### Install dependencies
Use `uv` (Python 3.13+) as the package manager. Do not add packages manually.

```bash
uv sync
```

### Environment
Copy `.env.example` to `.env` in the project root; the CLI auto-loads it via python-dotenv.

```bash
cp .env.example .env
# set OPENAI_API_KEY, OPENROUTER_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

### Verify CLI
```bash
uv run ea-rag --help
# or
uv run python -m ea_rag.cli --help
```

### Quickstart (TSV-style data)
End-to-end flow using the built-in commands. Adjust paths as needed.

```bash
# 1) Split entity pairs into sources and targets
uv run ea-rag split-entities \
  --input dbp15k/ent_links \
  --sources-out artifacts/sources.txt \
  --targets-out artifacts/targets.txt

# 2) Build target subgraphs and ingest into Qdrant
uv run ea-rag build-subgraphs \
  --kg dbp15k/triples_2 \
  --attr dbp15k/training_attrs_2 \
  --targets artifacts/targets.txt \
  --out artifacts/targets_subgraphs.jsonl
uv run ea-rag ingest \
  --subgraphs artifacts/targets_subgraphs.jsonl \
  --collection dbp15k_targets

# 3) Build + ingest source subgraphs for source-side context
uv run ea-rag build-source-subgraphs \
  --kg dbp15k/triples_1 \
  --sources artifacts/sources.txt \
  --out artifacts/sources_subgraphs.jsonl
uv run ea-rag ingest \
  --subgraphs artifacts/sources_subgraphs.jsonl \
  --collection dbp15k_sources

# 4) Predict
uv run ea-rag predict \
  --sources artifacts/sources.txt \
  --collection dbp15k_targets \
  --model google/gemini-2.5-flash \
  --source-collection dbp15k_sources \
  --out artifacts/predictions/dbp15k.jsonl

# 5) Evaluate (pairs + KG auto-resolve)
uv run ea-rag evaluate \
  --pred artifacts/predictions/dbp15k.jsonl \
  --pairs dbp15k/ent_links \
  --kg dbp15k/triples_2 \
  --out artifacts/eval/dbp15k.json
```

### Project structure
- `ea_rag/` — Python package (CLI and modules)
- `data/` — inputs and derived files
- `artifacts/` — predictions, eval, cache

### Data notes
- `artifacts/predictions/dbpyg.jsonl` was split into `dbpyg_1.jsonl`, `dbpyg_2.jsonl`, `dbpyg_3.jsonl`, and `dbpyg_4.jsonl` to keep file sizes under GitHub's limit.

For more detailed, restart-friendly runs, see `docs/cli.md`, `docs/dbp15k_run.md`, and `docs/dbp_yg_run.md`.
