import aiosqlite
import json
from typing import Any
from loguru import logger

class SqliteLogger:
    def __init__(self, db_path: str = "study_logs.db"):
        self.db_path = db_path

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Change: task_id is no longer a PRIMARY KEY, added an autoincrement 'id'
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    instruction TEXT,
                    conclusion TEXT,
                    reward REAL,
                    trials_json TEXT,
                    final_knowledge TEXT,
                    final_knowledge_title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_task_id ON trajectories (task_id)")
            
            # Migration: Add final_knowledge_title if it doesn't exist
            try:
                await db.execute("ALTER TABLE trajectories ADD COLUMN final_knowledge_title TEXT")
            except aiosqlite.OperationalError:
                pass # Already exists

            # Migration: Add title to citations if it doesn't exist
            try:
                await db.execute("ALTER TABLE citations ADD COLUMN title TEXT")
            except aiosqlite.OperationalError:
                pass # Already exists

            # Change: linked to trajectory_id (auto-increment id from trajectories table)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id INTEGER,
                    knowledge_id TEXT,
                    content TEXT,
                    title TEXT,
                    turn_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories (id)
                )
            """)
            await db.commit()

    async def save_trajectory(
        self,
        task_id: str,
        instruction: str,
        conclusion: str,
        reward: float,
        trials: list[dict[str, Any]],
        final_knowledge: str | None = None,
        final_knowledge_title: str | None = None,
        citations: list[dict[str, Any]] | None = None
    ):
        async with aiosqlite.connect(self.db_path) as db:
            # 1. Save main trajectory data (always INSERT now to keep history)
            cursor = await db.execute(
                "INSERT INTO trajectories (task_id, instruction, conclusion, reward, trials_json, final_knowledge, final_knowledge_title) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, instruction, conclusion, reward, json.dumps(trials), final_knowledge, final_knowledge_title)
            )
            trajectory_id = cursor.lastrowid

            # 2. Save citations (snapshots) tied to trajectory_id
            if citations:
                for c in citations:
                    await db.execute(
                        "INSERT INTO citations (trajectory_id, knowledge_id, content, title, turn_index) VALUES (?, ?, ?, ?, ?)",
                        (trajectory_id, c["knowledge_id"], c.get("content"), c.get("title"), c.get("turn_index", 0))
                    )
            
            await db.commit()
            logger.info(f"💾 Trajectory saved to SQLite for task {task_id} (ID: {trajectory_id})")

    async def get_trajectory(self, task_id: str) -> dict[str, Any] | None:
        """Fetch the LATEST trajectory for a task_id."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM trajectories WHERE task_id = ? ORDER BY created_at DESC LIMIT 1", (task_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                
                trajectory_id = row["id"]
                # Fetch citations for this specific trajectory attempt
                async with db.execute("SELECT knowledge_id, content, title, turn_index FROM citations WHERE trajectory_id = ?", (trajectory_id,)) as cit_cursor:
                    citations = [dict(r) for r in await cit_cursor.fetchall()]
                
                res = dict(row)
                res["trials"] = json.loads(res["trials_json"])
                res["citations"] = citations
                return res

    async def get_all_trajectories(self, task_id: str) -> list[dict[str, Any]]:
        """Fetch ALL trajectories for a task_id."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM trajectories WHERE task_id = ? ORDER BY created_at ASC", (task_id,)) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    trajectory_id = row["id"]
                    async with db.execute("SELECT knowledge_id, content, title, turn_index FROM citations WHERE trajectory_id = ?", (trajectory_id,)) as cit_cursor:
                        citations = [dict(r) for r in await cit_cursor.fetchall()]
                    
                    res = dict(row)
                    res["trials"] = json.loads(res["trials_json"])
                    res["citations"] = citations
                    results.append(res)
                return results
