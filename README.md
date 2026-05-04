# Adapter Agent (UniRL)

Reinforcement Learning based Agent with Hierarchical Task Decomposition and Knowledge Retrieval.

## Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.12+ (managed by `uv`)
- Node.js (for visualization frontend)

### 2. Infrastructure Setup
First, start the required databases and generate the Prisma client:
```bash
just db
```

### 3. Start Visualization Dashboard
To monitor experiments in real-time, launch the dashboard:
```bash
just vis
```
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **PostgreSQL**: localhost:5432
- **Elasticsearch**: localhost:9200

### 4. Run Experiment
In a **new terminal** (while `just vis` is running), start the experiment:
```bash
uv run scripts/uniray.py
```

## Useful Commands

- **Reset Database**: Clear all existing experiment and trajectory data to start fresh.
  ```bash
  just db-clean
  ```
- **Check Infrastructure Logs**: View Docker container logs for Postgres and ES.
  ```bash
  just logs
  ```

## Benchmark Dataset Pipeline

End-to-end pipeline for generating, enhancing, classifying, and visualizing benchmark problems for a target numerical library (default: `numrs2`). Outputs land under `data/benchmarks/<name>/` unless overridden.

### 1. Generate (BigQuery → Gemini)

`scripts/build_benchmark.py` pulls source-library snippets (numpy by default) from BigQuery, asks Gemini to derive self-contained benchmark problems, then rewrites the kept problems to mandate the target library.

```bash
# numrs2 ← numpy (default)
uv run python scripts/build_benchmark.py \
  --target-name numrs2 \
  --target-summary repositories/numrs/SUMMARY.md \
  --limit 1000 \
  --difficulty Easy

# hisab ← scipy
uv run python scripts/build_benchmark.py \
  --target-name hisab \
  --target-summary repositories/hisab/SUMMARY.md \
  --source-name scipy \
  --limit 1000 \
  --difficulty Easy
```

`--source-import` accepts SQL `LIKE` substrings (repeatable, OR-ed) when the
default lookup for `--source-name` is missing.

Notable flags:
- `--skip-generate` — reuse an existing `original.csv` instead of re-querying BigQuery.
- `--skip-enhance` — stop after the generation stage.
- `--no-filter-appropriate` — keep rows the generator marked inappropriate.
- `--difficulty ""` — disable the difficulty filter when enhancing.

Requires `GEMINI_API_KEY` in `.env` and BigQuery credentials for project `dsat2-405406`.

### 2. Enhance Only (skip generation)

If you already have an `original.csv` and only want to rewrite statements for a different library or different difficulty filter:

```bash
uv run python scripts/enhance_benchmark.py \
  --input data/benchmarks/<name>/original.csv \
  --output data/benchmarks/<name>/enhanced.csv \
  --filter-appropriate \
  --difficulty Easy
```

### 3. Classify Required APIs (LLM-as-a-judge)

`scripts/classify_benchmark_apis.py` adds four columns to the dataset using a skeptical Gemini judge:
- `required_apis` — multi-label numpy/scipy categories actually needed (e.g. `linalg_basic,reduction,broadcasting`).
- `actual_complexity` — `trivial_array_ops` / `routine_numpy` / `nontrivial_algorithm`, ignoring domain framing.
- `framing_inflation` — True when the prose oversells the underlying computation.
- `classification_rationale` — short justification.

```bash
uv run python scripts/classify_benchmark_apis.py \
  --input benchmark_dataset.csv \
  --output benchmark_dataset_classified.csv \
  --max-concurrent 20
```

Useful flags: `--filter-appropriate`, `--difficulty Easy`, `--limit 200` (smoke test).

### 4. Visualize the Classification

`scripts/viz_classification.py` renders a 6-panel PNG: API frequency, complexity breakdown, difficulty × actual_complexity alignment, framing inflation, APIs-per-task histogram, and an API co-occurrence heatmap.

```bash
uv run python scripts/viz_classification.py \
  --input benchmark_dataset_classified.csv \
  --output benchmark_classification_viz.png \
  --top-cooccurrence 14
```

## System Architecture

- **`adapter_agent`**: Core RL logic, TaskNetwork, and environment wrappers.
- **`scripts/vis_server.py`**: FastAPI backend that serves experiment data from PostgreSQL.
- **`graphvis/`**: React-based interactive graph visualization.
- **PostgreSQL**: Source of Truth for trajectories, citations, and experiment metadata.
- **Elasticsearch**: Used for fast knowledge retrieval during agent rollouts.

## Visualization Nodes
- **Gray Nodes**: Queued tasks.
- **Yellow Nodes**: Currently executing tasks.
- **Green Nodes**: Successfully solved tasks (SFT candidates).
- **Purple Nodes**: Verified Knowledge extracted from successful trials.

Click on any task node to view the full reasoning trace (including thoughts, tool calls, and results) and see the specific knowledge derived from it.
