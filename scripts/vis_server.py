import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from adapter_agent.rl.env.sqlite_logger import SqliteLogger
import json
import os

from contextlib import asynccontextmanager

logger_db = SqliteLogger("study_logs.db")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await logger_db.initialize()
    yield
    # Shutdown (if needed)

app = FastAPI(title="AdapterAgent Visualization API", lifespan=lifespan)

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this. For research, * is fine.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/graph")
async def get_graph():
    path = "graphvis/public/data.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Graph data not found")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/trajectory/{task_id}")
async def get_trajectory(task_id: str):
    data = await logger_db.get_all_trajectories(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
