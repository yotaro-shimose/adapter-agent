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
