from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from adapter_agent.rl.postgres_db import PostgresDB

_db_manager = PostgresDB()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _db_manager.connect()
    yield
    await _db_manager.close()

app = FastAPI(title="AdapterAgent Visualization API (Prisma)", lifespan=lifespan)

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/experiments")
async def list_experiments():
    try:
        client = await _db_manager.get_client()
        experiments = await client.experiment.find_many(
            order={"created_at": "desc"}
        )
        return [e.experiment_name for e in experiments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/{experiment_name}/graph")
async def get_graph(experiment_name: str):
    client = await _db_manager.get_client()
    experiment = await client.experiment.find_unique(
        where={"experiment_name": experiment_name}
    )
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
    
    if not experiment.graph_json:
        # Return a minimal valid graph instead of nothing to avoid frontend hang
        return {
            "nodes": [
                {"id": "root", "type": "root", "metadata": {"instruction": "Initializing..."}}
            ],
            "edges": []
        }
    
    # AGGREGATE COUNTS FROM Trajectory table for SSOT
    trajectories = await client.trajectory.find_many(
        where={"experiment_name": experiment_name}
    )
    
    # Calculate counts per task_id
    stats = {}
    for t in trajectories:
        tid = t.task_id
        if tid not in stats:
            stats[tid] = {"success_count": 0, "total_count": 0}
        stats[tid]["total_count"] += 1
        if t.reward is not None and t.reward > 0:
            stats[tid]["success_count"] += 1
            
    # Inject into graph_json
    graph = experiment.graph_json
    if isinstance(graph, dict) and "nodes" in graph and isinstance(graph["nodes"], list):
        for node in graph["nodes"]:
            if not isinstance(node, dict):
                continue
            tid = node.get("id")
            if not tid or not isinstance(tid, str):
                continue
            
            metadata = node.get("metadata")
            if not isinstance(metadata, dict):
                continue

            if tid in stats:
                metadata["success_count"] = stats[tid]["success_count"]
                metadata["total_count"] = stats[tid]["total_count"]
                metadata["is_solved"] = stats[tid]["success_count"] > 0
            else:
                # If no records in DB, ensure counts are 0 to avoid drift from stale memory
                metadata["success_count"] = 0
                metadata["total_count"] = 0
                metadata["is_solved"] = False
                
    return graph

@app.get("/api/{experiment_name}/tasks")
async def list_tasks(experiment_name: str):
    """Return per-task aggregates for an experiment.

    SQL-side GROUP BY so we never ship trials_json over the wire — must stay
    cheap even with 100k+ trajectories so it doesn't compete with training
    writes on Postgres.
    """
    client = await _db_manager.get_client()
    exp = await client.experiment.find_unique(where={"experiment_name": experiment_name})
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")

    rows = await client.query_raw(
        '''
        SELECT
            task_id,
            MAX(instruction) AS instruction,
            COUNT(*)::int AS total_count,
            SUM(CASE WHEN reward IS NOT NULL AND reward > 0 THEN 1 ELSE 0 END)::int AS success_count,
            MAX(reward) AS max_reward,
            MAX(created_at) AS latest_created_at
        FROM trajectories
        WHERE experiment_name = $1
        GROUP BY task_id
        ORDER BY MAX(created_at) DESC
        ''',
        experiment_name,
    )
    return [
        {
            "task_id": r["task_id"],
            "instruction": r.get("instruction") or r["task_id"],
            "total_count": r["total_count"],
            "success_count": r["success_count"],
            "max_reward": r.get("max_reward"),
            "latest_created_at": r.get("latest_created_at"),
        }
        for r in rows
    ]

@app.get("/api/{experiment_name}/trajectory/{task_id}")
async def get_trajectory(experiment_name: str, task_id: str):
    client = await _db_manager.get_client()
    
    # Fetch experiment ID first
    exp = await client.experiment.find_unique(where={"experiment_name": experiment_name})
    if not exp:
         raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
    
    # Fetch trajectories for this experiment and task
    trajectories = await client.trajectory.find_many(
        where={
            "experiment_name": experiment_name,
            "task_id": task_id
        },
        order={"created_at": "asc"},
        include=None
    )
    
    print(f"DEBUG: Found {len(trajectories)} trajectories for task_id '{task_id}' in experiment '{experiment_name}'")

    return [
        {
            "id": t.id,
            "taskId": t.task_id,
            "instruction": t.task_id,
            "conclusion": t.conclusion if t.conclusion is not None else "n/a",
            "reward": t.reward if t.reward is not None else 0.0,
            "trials": t.trials_json,
            "knowledge_ids": t.knowledge_ids,
            "created_at": t.created_at.isoformat() if t.created_at else None
        }
        for t in trajectories
    ]


# --- WIKI ENDPOINTS ---

@app.get("/api/wiki/versions")
async def get_wiki_versions():
    """Lists all distinct versions in the wiki_articles table, sorted by latest update."""
    client = await _db_manager.get_client()
    try:
        # Fetch articles ordered by update time to find latest versions first
        articles = await client.wikiarticle.find_many(
            order={"updated_at": "desc"}
        )
        versions = []
        seen = set()
        for a in articles:
            if a.version not in seen:
                versions.append(a.version)
                seen.add(a.version)
        return versions
    except Exception:
        # Fallback: manually aggregate and sort by updated_at
        articles = await client.wikiarticle.find_many()
        v_map = {} # version -> latest_update
        for a in articles:
            if a.version not in v_map or a.updated_at > v_map[a.version]:
                v_map[a.version] = a.updated_at
        # Sort versions by latest update time (descending)
        return [v for v, _ in sorted(v_map.items(), key=lambda x: x[1], reverse=True)]

@app.get("/api/wiki/{version}/articles")
async def get_wiki_articles(version: str):
    """Lists all article titles for a specific version."""
    client = await _db_manager.get_client()
    articles = await client.wikiarticle.find_many(
        where={"version": version},
        order={"title": "asc"}
    )
    return [{"title": a.title, "updated_at": a.updated_at.isoformat()} for a in articles]

@app.get("/api/wiki/{version}/article/{title:path}")
async def get_wiki_article_content(version: str, title: str):
    """Fetches the content of a specific article."""
    client = await _db_manager.get_client()
    article = await client.wikiarticle.find_unique(
        where={"version_title": {"version": version, "title": title}}
    )
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return {
        "title": article.title,
        "content": article.content,
        "version": article.version,
        "updated_at": article.updated_at.isoformat()
    }

# --- SIMPLE RL ROLLOUT ENDPOINTS ---
# Per-script-execution view: each `run_continue_rl.py` invocation produces
# a unique `simple_train_id` ("<prefix>_<timestamp>") and writes rows to
# `simple_rl_rollouts`. These endpoints surface that table.

@app.get("/api/simple_runs")
async def list_simple_runs():
    """Per simple_train_id summary: created_at + step/rollout counts.

    Aggregates on the SQL side so the response stays small even with
    many runs and 100k+ rollouts each.
    """
    client = await _db_manager.get_client()
    rows = await client.query_raw(
        '''
        SELECT
            r.id AS simple_train_id,
            r.created_at AS created_at,
            COALESCE(s.total_rollouts, 0)::int AS total_rollouts,
            COALESCE(s.success_count, 0)::int AS success_count,
            s.max_rl_step,
            s.latest_rollout_at
        FROM simple_train_runs r
        LEFT JOIN (
            SELECT
                simple_train_id,
                COUNT(*) AS total_rollouts,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_count,
                MAX(rl_step) AS max_rl_step,
                MAX(created_at) AS latest_rollout_at
            FROM simple_rl_rollouts
            GROUP BY simple_train_id
        ) s ON s.simple_train_id = r.id
        WHERE COALESCE(s.total_rollouts, 0) > 0
        ORDER BY COALESCE(s.latest_rollout_at, r.created_at) DESC
        ''',
    )
    return [
        {
            "simple_train_id": r["simple_train_id"],
            "created_at": r.get("created_at"),
            "latest_rollout_at": r.get("latest_rollout_at"),
            "total_rollouts": r["total_rollouts"],
            "success_count": r["success_count"],
            "max_rl_step": r.get("max_rl_step"),
        }
        for r in rows
    ]


@app.get("/api/simple_runs/{simple_train_id}/summary")
async def get_simple_run_summary(simple_train_id: str):
    """Per-(rl_step, suite_name) aggregates for one run."""
    client = await _db_manager.get_client()
    rows = await client.query_raw(
        '''
        SELECT
            rl_step,
            suite_name,
            COUNT(*)::int AS total_count,
            SUM(CASE WHEN success THEN 1 ELSE 0 END)::int AS success_count,
            AVG(reward)::float AS avg_reward,
            COUNT(DISTINCT task_id)::int AS unique_tasks
        FROM simple_rl_rollouts
        WHERE simple_train_id = $1
        GROUP BY rl_step, suite_name
        ORDER BY rl_step ASC, suite_name ASC
        ''',
        simple_train_id,
    )
    return [
        {
            "rl_step": r["rl_step"],
            "suite_name": r["suite_name"],
            "total_count": r["total_count"],
            "success_count": r["success_count"],
            "avg_reward": r.get("avg_reward"),
            "unique_tasks": r["unique_tasks"],
        }
        for r in rows
    ]


@app.get("/api/simple_runs/{simple_train_id}/rollouts")
async def list_simple_run_rollouts(
    simple_train_id: str,
    rl_step: int | None = None,
    suite_name: str | None = None,
    task_id: str | None = None,
    success: str | None = None,  # "true"/"false"/None
    limit: int = 500,
):
    """List rollout rows for a run with optional filters.

    Excludes the heavy `answer`/`execution_output`/`verification_output`
    columns to keep this fast — fetch a single row via /rollout/{id}.
    """
    client = await _db_manager.get_client()
    where: dict = {"simple_train_id": simple_train_id}
    if rl_step is not None:
        where["rl_step"] = rl_step
    if suite_name is not None:
        where["suite_name"] = suite_name
    if task_id is not None:
        where["task_id"] = task_id
    if success in ("true", "false"):
        where["success"] = success == "true"

    rows = await client.simplerlrollout.find_many(
        where=where,
        order=[{"rl_step": "desc"}, {"group_idx": "asc"}, {"sample_idx": "asc"}],
        take=max(1, min(limit, 5000)),
    )
    return [
        {
            "id": r.id,
            "rl_step": r.rl_step,
            "suite_name": r.suite_name,
            "task_id": r.task_id,
            "group_idx": r.group_idx,
            "sample_idx": r.sample_idx,
            "num_samples": r.num_samples,
            "instruction": r.instruction,
            "parsed": r.parsed,
            "success": r.success,
            "reward": r.reward,
            "sampling_client_version": r.sampling_client_version,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.get("/api/simple_runs/{simple_train_id}/rollout/{rollout_id}")
async def get_simple_run_rollout(simple_train_id: str, rollout_id: int):
    """Full content for a single rollout (heavy fields included)."""
    client = await _db_manager.get_client()
    r = await client.simplerlrollout.find_unique(where={"id": rollout_id})
    if not r or r.simple_train_id != simple_train_id:
        raise HTTPException(status_code=404, detail="Rollout not found in this run")
    return {
        "id": r.id,
        "simple_train_id": r.simple_train_id,
        "rl_step": r.rl_step,
        "suite_name": r.suite_name,
        "task_id": r.task_id,
        "group_idx": r.group_idx,
        "sample_idx": r.sample_idx,
        "num_samples": r.num_samples,
        "instruction": r.instruction,
        "answer": r.answer,
        "reasoning": getattr(r, "reasoning", "") or "",
        "parsed": r.parsed,
        "success": r.success,
        "reward": r.reward,
        "execution_output": r.execution_output,
        "verification_output": r.verification_output,
        "sampling_client_version": r.sampling_client_version,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# --- SFT CACHE ENDPOINTS ---
# User-named caches of generated QRAs (replaces the old pickle cache + the
# dropped `simple_sft_qnas` audit log). Keyed by `SftCache.id`.

@app.get("/api/sft_caches")
async def list_sft_caches():
    """Per-cache summary: `(id, granular_id, library_name, total/verified counts, latest_at)`."""
    client = await _db_manager.get_client()
    rows = await client.query_raw(
        '''
        SELECT
            c.id AS id,
            c.granular_id AS granular_id,
            c.library_name AS library_name,
            c.description AS description,
            c.created_at AS created_at,
            COALESCE(s.total_items, 0)::int AS total_items,
            COALESCE(s.verified_items, 0)::int AS verified_items,
            COALESCE(s.unique_knowledges, 0)::int AS unique_knowledges,
            s.latest_item_at
        FROM sft_caches c
        LEFT JOIN (
            SELECT
                cache_id,
                COUNT(*) AS total_items,
                SUM(CASE WHEN verified THEN 1 ELSE 0 END) AS verified_items,
                COUNT(DISTINCT knowledge_id) AS unique_knowledges,
                MAX(created_at) AS latest_item_at
            FROM sft_cache_items
            GROUP BY cache_id
        ) s ON s.cache_id = c.id
        ORDER BY COALESCE(s.latest_item_at, c.created_at) DESC
        ''',
    )
    return [
        {
            "id": r["id"],
            "granular_id": r.get("granular_id"),
            "library_name": r.get("library_name"),
            "description": r.get("description"),
            "created_at": r.get("created_at"),
            "latest_item_at": r.get("latest_item_at"),
            "total_items": r["total_items"],
            "verified_items": r["verified_items"],
            "unique_knowledges": r["unique_knowledges"],
        }
        for r in rows
    ]


@app.get("/api/sft_caches/{cache_id}/items")
async def list_sft_cache_items(
    cache_id: str,
    knowledge_id: str | None = None,
    verified: str | None = None,  # "true"/"false"/None
    limit: int = 500,
):
    """Item list for a cache. Heavy fields (`answer`, `verifier_reasoning`)
    are excluded — use /item/{id} for full content."""
    client = await _db_manager.get_client()
    where: dict = {"cache_id": cache_id}
    if knowledge_id is not None:
        where["knowledge_id"] = knowledge_id
    if verified in ("true", "false"):
        where["verified"] = verified == "true"
    rows = await client.sftcacheitem.find_many(
        where=where,
        order=[{"id": "asc"}],
        take=max(1, min(limit, 5000)),
    )
    return [
        {
            "id": r.id,
            "cache_id": r.cache_id,
            "knowledge_id": r.knowledge_id,
            "knowledge_title": r.knowledge_title,
            "question": r.question,
            "verified": r.verified,
            "conclusion": r.conclusion,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.get("/api/sft_caches/{cache_id}/item/{item_id}")
async def get_sft_cache_item(cache_id: str, item_id: int):
    """Full content for a single SFT cache item, including the multi-turn
    investigation log (`trials`) when available."""
    client = await _db_manager.get_client()
    r = await client.sftcacheitem.find_unique(where={"id": item_id})
    if not r or r.cache_id != cache_id:
        raise HTTPException(status_code=404, detail="Item not found in this cache")
    return {
        "id": r.id,
        "cache_id": r.cache_id,
        "knowledge_id": r.knowledge_id,
        "knowledge_title": r.knowledge_title,
        "question": r.question,
        "reasoning": r.reasoning,
        "answer": r.answer,
        "verified": r.verified,
        "verifier_reasoning": r.verifier_reasoning,
        "conclusion": r.conclusion,
        "reward": r.reward,
        "trials": r.trials_json,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
