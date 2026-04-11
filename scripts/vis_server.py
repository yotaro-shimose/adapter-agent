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
        include={"citations": True}
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
            "citations": [
                {
                    "knowledge_id": str(c.knowledge_id) if c.knowledge_id is not None else None,
                    "content": c.content,
                    "title": c.title,
                    "turn_index": c.turn_index
                } for c in (t.citations or [])
            ],
            "created_at": t.created_at.isoformat() if t.created_at else None
        }
        for t in trajectories
    ]


@app.get("/api/simple_trains")
async def list_simple_trains():
    try:
        client = await _db_manager.get_client()
        runs = await client.simpletrainrun.find_many(
            order={"created_at": "desc"}
        )
        return [r.id for r in runs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/simple_train/{simple_train_id}/knowledge")
async def get_simple_train_knowledge(simple_train_id: str):
    client = await _db_manager.get_client()
    
    # Check if run exists
    run = await client.simpletrainrun.find_unique(where={"id": simple_train_id})
    if not run:
        raise HTTPException(status_code=404, detail="SimpleTrainRun not found")
        
    trajectories = await client.simpletrajectory.find_many(
        where={"simple_train_id": simple_train_id}
    )
    
    # Aggregate by knowledge_id
    knowledge_map = {}
    for t in trajectories:
        if t.knowledge_id not in knowledge_map:
            knowledge_map[t.knowledge_id] = {
                "knowledge_id": t.knowledge_id,
                "knowledge_title": t.knowledge_title,
                "total_rollouts": 0,
                "total_success": 0,
                "steps": set()
            }
        km = knowledge_map[t.knowledge_id]
        km["total_rollouts"] += 1
        if t.success:
            km["total_success"] += 1
        km["steps"].add(t.step)
        
    for k in knowledge_map.values():
        k["steps"] = sorted(list(k["steps"]))
        
    return list(knowledge_map.values())

@app.get("/api/simple_train/{simple_train_id}/knowledge/{knowledge_id}/rollouts")
async def get_simple_train_rollouts(simple_train_id: str, knowledge_id: str):
    client = await _db_manager.get_client()
    
    trajectories = await client.simpletrajectory.find_many(
        where={
            "simple_train_id": simple_train_id,
            "knowledge_id": knowledge_id
        },
        order=[{"step": "asc"}, {"created_at": "asc"}]
    )
    
    return [
        {
            "id": t.id,
            "step": t.step,
            "question": t.question,
            "reasoning": t.reasoning,
            "answer": t.answer,
            "success": t.success,
            "execution_output": t.execution_output,
            "verification_output": t.verification_output,
            "created_at": t.created_at.isoformat() if t.created_at else None
        } for t in trajectories
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
