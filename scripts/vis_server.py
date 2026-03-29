from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from adapter_agent.rl.postgres_db import get_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB (Prisma Client)
    db = await get_db()
    yield
    # No explicit disconnect needed here if handled by PostgresDB singleton
    await db.close()

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
        db = await get_db()
        client = await db.get_client()
        experiments = await client.experiment.find_many(
            order={"created_at": "desc"}
        )
        return [e.experiment_name for e in experiments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/{experiment_name}/graph")
async def get_graph(experiment_name: str):
    db = await get_db()
    client = await db.get_client()
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
        where={"experiment_id": experiment.id}
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
    if "nodes" in graph:
        for node in graph["nodes"]:
            tid = node.get("id")
            if tid in stats:
                node["metadata"]["success_count"] = stats[tid]["success_count"]
                node["metadata"]["total_count"] = stats[tid]["total_count"]
                node["metadata"]["is_solved"] = stats[tid]["success_count"] > 0
            else:
                # If no records in DB, ensure counts are 0 to avoid drift from stale memory
                node["metadata"]["success_count"] = 0
                node["metadata"]["total_count"] = 0
                node["metadata"]["is_solved"] = False
                
    return graph

@app.get("/api/{experiment_name}/trajectory/{task_id}")
async def get_trajectory(experiment_name: str, task_id: str):
    db = await get_db()
    client = await db.get_client()
    
    # Fetch experiment ID first
    exp = await client.experiment.find_unique(where={"experiment_name": experiment_name})
    if not exp:
         raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
    
    # Fetch trajectories for this experiment and task
    trajectories = await client.trajectory.find_many(
        where={
            "experiment_id": exp.id,
            "task_id": task_id
        },
        order={"created_at": "asc"},
        include={"citations": True}
    )
    
    print(f"DEBUG: Found {len(trajectories)} trajectories for task_id '{task_id}' in experiment '{experiment_name}'")

    return [
        {
            "id": t.id,
            "taskId": t.task_id,
            "instruction": t.task_id,
            "conclusion": t.conclusion or "sft_data",
            "reward": t.reward or 1.0,
            "trials": t.trials_json,
            "citations": [
                {
                    "knowledge_id": c.knowledge_id,
                    "content": c.content,
                    "title": c.title,
                    "turn_index": c.turn_index
                } for c in (t.citations or [])
            ],
            "created_at": t.created_at.isoformat() if t.created_at else None
        }
        for t in trajectories
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
