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

@app.get("/api/openbook/experiments")
async def list_openbook_experiments():
    try:
        client = await _db_manager.get_client()
        experiments = await client.openbookexperiment.find_many(
            order={"created_at": "desc"}
        )
        return [e.experiment_name for e in experiments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/openbook/{experiment_name}/data")
async def get_openbook_data(experiment_name: str):
    client = await _db_manager.get_client()
    
    qas = await client.openbookqa.find_many(
        where={"experiment_name": experiment_name},
        order={"knowledge_id": "asc"}
    )
    
    trajectories = await client.openbooktrajectory.find_many(
        where={"experiment_name": experiment_name},
        order={"id": "asc"}
    )
    
    return {
        "qas": [
            {
                "id": q.id,
                "knowledge_id": q.knowledge_id,
                "title": q.title,
                "question": q.question,
                "answer": q.answer
            } for q in qas
        ],
        "trajectories": [
            {
                "id": t.id,
                "qa_id": t.qa_id,
                "question": t.question,
                "hint": t.hint,
                "reasoning": t.reasoning,
                "answer": t.answer,
                "success": t.success,
                "dataset": t.dataset,
                "execution_output": t.execution_output,
                "verification_output": t.verification_output
            } for t in trajectories
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
