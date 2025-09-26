## EA‑RAG CLI

Typer-based command line interface for the Entity Alignment RAG pipeline. It covers the full workflow: preparing data, building subgraphs, ingesting into Qdrant, running LLM predictions, and evaluating results.

### Installation

- **Python**: 3.13+
- **Install deps** (using `uv`):

```bash
uv sync
```

- **Environment**: copy and edit `.env` in the project root:

```bash
cp .env.example .env
# set OPENAI_API_KEY, OPENROUTER_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

### How to run

- From the project root, either use the installed entrypoint or module form:

```bash
uv run ea-rag --help
# or
uv run python -m ea_rag.cli --help
```

### Prerequisites and defaults

- **Embeddings**: OpenAI embeddings via LangChain. Default model: `text-embedding-3-small`.
- **Qdrant**: expected at `http://localhost:6333` (gRPC 6334). Configure via `.env`:
  - `QDRANT_URL`, `QDRANT_API_KEY` (optional for local without auth)
- **OpenRouter**: predictions use OpenRouter chat completions. Configure via `.env`:
  - `OPENROUTER_API_KEY`, optional `OPENROUTER_BASE_URL` (defaults to `https://openrouter.ai/api/v1`)

## Data formats

### Entity pairs input

- File where each non-empty line is: `@<SOURCE_URI> <TAB> <TARGET_NAME>` (leading `@` on URI is optional). Lines starting with `#` are ignored.

Example:

```text
@http://dbpedia.org/resource/Albert_Einstein	Albert Einstein
http://dbpedia.org/resource/Paris	Paris
```

### Sources and targets

- `sources.txt`: one source URI per line
- `targets.txt`: one target name or URI per line

### Subgraphs JSONL schema

Each line is a JSON object for one entity:

```json
{
  "entity_name": "Albert_Einstein",
  "entity_uri": "http://dbpedia.org/resource/Albert_Einstein",
  "sentences": ["Albert Einstein has birth place Ulm.", "Albert Einstein has field physics."],
  "triple_count": 42,
  "meta": {"incoming": 10, "outgoing": 32}
}
```

### Predictions JSONL schema

Each line corresponds to one source URI:

```json
{
  "source_uri": "http://dbpedia.org/resource/Albert_Einstein",
  "model": "openrouter/model:id",
  "retrieval": true,
  "candidates": [{"target_uri": "http://dbpedia.org/resource/Albert_Einstein", "context": "..."}],
  "prompt_text": "Task: Select the best aligned target...",
  "prediction": {"target_uri": "http://dbpedia.org/resource/Albert_Einstein", "score": 0.93, "rationale": "..."},
  "latency_ms": 1234
}
```

## Commands

### split-entities

Parse an entity pairs file and write `sources.txt` and `targets.txt` (optionally the parsed JSONL).

```bash
uv run ea-rag split-entities \
  --input data/entity_pairs.txt \
  --sources-out data/sources.txt \
  --targets-out data/targets.txt \
  --jsonl-out artifacts/cache/parsed_pairs.jsonl
```

- **--input**: entity pairs file (`@URI<TAB>Name`)
- **--sources-out**: output path for sources (URIs)
- **--targets-out**: output path for targets (names)
- **--jsonl-out**: optional parsed JSONL trace

### build-subgraphs / build-target-subgraphs

Build 1‑hop subgraphs for targets and render each triple as a sentence. Autodetects RDF vs TSV unless `--kg-format` is specified.

```bash
uv run ea-rag build-subgraphs \
  --kg dbp15k/triples_2 \
  --attr dbp15k/training_attrs_2 \
  --targets data/targets.txt \
  --out artifacts/targets_subgraphs.jsonl \
  --max-triples 64
```

- **--kg**: KG file (RDF Turtle or TSV). Autodetected.
- **--attr**: optional attributes file. Merged for TSV; ignored for RDF.
- **--targets**: file of target names or URIs (one per line)
- **--out**: output JSONL file
- **--max-triples**: max triples per entity (default 64)
- **--seed**: optional RNG seed
- **--kg-format**: `rdf` or `tsv` to override autodetection

Alias `build-target-subgraphs` is identical to `build-subgraphs`.

### build-source-subgraphs

Build 1‑hop subgraphs for sources (URIs). Useful to include source context at prediction time.

```bash
uv run ea-rag build-source-subgraphs \
  --kg dbp15k/triples_1 \
  --sources data/sources.txt \
  --out artifacts/sources_subgraphs.jsonl \
  --max-triples 64
```

- **--kg**: KG file (RDF Turtle or TSV)
- **--attr**: optional TSV attributes (TSV only)
- **--sources**: file of source URIs (one per line)
- **--out**: output JSONL file
- **--max-triples**: default 64
- **--kg-format**: `rdf` or `tsv`

### generate-synthetic

Create a synthetic TSV KG pair and a matching `entity_pairs.txt` for quick testing.

```bash
uv run ea-rag generate-synthetic \
  --kg-rel data/synth_rel.tsv \
  --kg-attr data/synth_attr.tsv \
  --pairs data/entity_pairs.txt \
  --num-entities 1000
```

- Outputs: relation and attribute TSVs, and entity pairs file

### ingest

Embed subgraph sentences and ingest into a Qdrant collection. Idempotent: skips if the collection already contains a sentinel for the same `ingestion_id` (derived from subgraphs file hash + embedding model).

```bash
uv run ea-rag ingest \
  --subgraphs artifacts/targets_subgraphs.jsonl \
  --collection dbp15k_targets \
  --embed-batch-size 1024 \
  --upsert-batch-size 1024
```

- **--subgraphs**: input JSONL created by a build-subgraphs command
- **--collection**: Qdrant collection name
- **--qdrant-url**: override URL (else read from `.env`)
- **--embed-batch-size**: sentences per embedding batch (default 1024)
- **--upsert-batch-size**: vectors per Qdrant upsert (default 1024)

Notes:
- The collection is created (or recreated) automatically with COSINE distance and vector size matching the first batch.

### predict

Run LLM predictions for each source URI, with optional retrieval from the target collection and optional source context.

```bash
uv run ea-rag predict \
  --sources data/sources.txt \
  --collection dbp15k_targets \
  --model openrouter/google/gemini-2.5-flash \
  --k 10 \
  --max-per-uri 3 \
  --context-chars 256 \
  --max-tokens 128 \
  --out artifacts/predictions/dbp15k.jsonl
```

- **--sources**: file of source URIs
- **--target-collection/--collection**: Qdrant collection with TARGET sentences
- **--model**: OpenRouter model id
- **--retrieval/--no-retrieval**: enable/disable retrieval (default enabled)
- **--rationale/--no-rationale**: include rationale in model output
- **--k**: top‑k neighbors to retrieve (default 10)
- **--max-per-uri**: max sentences per candidate URI (default 3)
- **--context-chars**: truncate each sentence to N chars (optional)
- **--max-tokens**: LLM max tokens for response (default 128)
- **--temperature**: LLM temperature (omitted if not set)
- **--no-save-candidates**: do not persist candidates/prompt text in output
- **--source-collection**: optional Qdrant collection with SOURCE sentences
- **--max-source-sentences**: max source sentences to include (default 3)
- **--out**: output predictions JSONL
- Advanced: **--concurrency**, **--request-timeout**, **--max-retries**, **--backoff-base**

### evaluate

Evaluate predictions against ground truth. Provide either a truth CSV or the original pairs with the KG to resolve names to URIs.

```bash
# Using pairs + KG (auto-resolves target names to URIs)
uv run ea-rag evaluate \
  --pred artifacts/predictions/dbp15k.jsonl \
  --pairs dbp15k/ent_links \
  --kg dbp15k/triples_2 \
  --out artifacts/eval/dbp15k_eval.json

# Using explicit truth CSV (source_uri,target_uri)
uv run ea-rag evaluate \
  --pred artifacts/predictions/dbp15k.jsonl \
  --truth artifacts/eval/truth.csv \
  --out artifacts/eval/dbp15k_eval.json \
  --cases-csv artifacts/eval/cases.csv
```

- **--pred**: predictions JSONL
- One of: **--truth** CSV (`source_uri,target_uri`) or **--pairs** file
- If using **--pairs**, provide **--kg** (and optionally `--kg-format` `rdf|tsv`)
- **--out**: JSON summary
- **--cases-csv**: optional per‑case CSV

### timings-predict

Summarize prediction timings (by run or by model). Default timings path is `artifacts/cache/timings/timings.jsonl`.

```bash
uv run ea-rag timings-predict --by-model
uv run ea-rag timings-predict --json --limit 5
```

- **--timings**: custom timings file path
- **--json**: emit JSON instead of a table
- **--limit**: include only latest N runs (0 = all)
- **--by-model**: aggregate by model

### timings-ingest

Summarize ingestion timings (by run or by collection).

```bash
uv run ea-rag timings-ingest --by-collection
uv run ea-rag timings-ingest --json --limit 10
```

- **--timings**: custom timings file path
- **--json**: emit JSON instead of a table
- **--limit**: limit rows in output
- **--by-collection**: aggregate by collection

### analyze-subgraphs

Scan one or more JSONL files (or directories) and report sentence counts per file and in total.

```bash
uv run ea-rag analyze-subgraphs artifacts --pattern "targets_subgraphs*.jsonl"
uv run ea-rag analyze-subgraphs artifacts/targets_subgraphs.jsonl --json
```

- **inputs**: files or directories (recursive) to scan
- **--pattern**: glob used when scanning directories (default `targets_subgraphs*.jsonl`)
- **--json**: output JSON summary

## End‑to‑end example (DBP15K‑style TSV)

```bash
# 1) Split entity pairs into sources and targets
uv run ea-rag split-entities \
  --input dbp15k/ent_links \
  --sources-out artifacts/dbp15k_sources.txt \
  --targets-out artifacts/dbp15k_targets.txt

# 2) Build target subgraphs and ingest into Qdrant
uv run ea-rag build-subgraphs \
  --kg dbp15k/triples_2 \
  --attr dbp15k/training_attrs_2 \
  --targets artifacts/dbp15k_targets.txt \
  --out artifacts/targets_subgraphs.jsonl
uv run ea-rag ingest \
  --subgraphs artifacts/targets_subgraphs.jsonl \
  --collection dbp15k_targets

# (Optional) Build source subgraphs and ingest for source context
uv run ea-rag build-source-subgraphs \
  --kg dbp15k/triples_1 \
  --sources artifacts/dbp15k_sources.txt \
  --out artifacts/sources_subgraphs.jsonl
uv run ea-rag ingest \
  --subgraphs artifacts/sources_subgraphs.jsonl \
  --collection dbp15k_sources

# 3) Run predictions
uv run ea-rag predict \
  --sources artifacts/dbp15k_sources.txt \
  --collection dbp15k_targets \
  --model openrouter/google/gemini-2.5-flash \
  --source-collection dbp15k_sources \
  --out artifacts/predictions/dbp15k.jsonl

# 4) Evaluate
uv run ea-rag evaluate \
  --pred artifacts/predictions/dbp15k.jsonl \
  --pairs dbp15k/ent_links \
  --kg dbp15k/triples_2 \
  --out artifacts/eval/dbp15k_eval.json
```

## Tips

- **Idempotent ingestion**: If you re‑ingest the same subgraphs file with the same embedding model, ingestion is skipped.
- **RDF vs TSV**: Autodetection tries RDF first, then TSV. Override with `--kg-format rdf|tsv` if needed.
- **Performance**: Increase `--concurrency` in `predict` to saturate OpenRouter throughput; tune `--embed-batch-size`/`--upsert-batch-size` in `ingest` for Qdrant.
- **More examples**: See `docs/dbp15k_run.md` and `docs/dbp_yg_run.md`.


