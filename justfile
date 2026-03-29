default:
  just --list


stop-docker:
    docker ps | grep "coder-mcp" | awk '{print $1}' | xargs -r docker stop

clean-tmp:
    rm -rf /tmp/*

remove-checkpoints:
    uv run scripts/remove_checkpoints.py

# Delete Cloud Run services that contain a specific pattern in their name (delegated to coder-mcp)
# Usage: just delete-cloudrun [pattern="coder-mcp-numrs2"] [region="europe-north1"]
delete-cloudrun pattern="coder-mcp-numrs2" region="europe-north1":
    just --justfile ../coder-mcp/justfile delete-cloudrun-services {{pattern}} {{region}}

# Start the database specifically
db:
    @echo "Starting PostgreSQL (Docker: adapter_agent_db)..."
    docker compose up -d postgres
    @echo "Waiting for services to be ready..."
    sleep 3
    @echo "Generating Prisma Client..."
    uv run prisma generate --schema=schema.prisma

# Start the visualization dashboard (ES + DB + Backend + Frontend)
vis:
    @echo "Starting Infrastructure (ES, Postgres)..."
    docker compose up -d
    @echo "Waiting for services to be ready..."
    sleep 3
    @echo "Generating Prisma Client..."
    uv run prisma generate --schema=schema.prisma
    @echo "Starting visualization backend (8000) and frontend (5173)..."
    npx -y concurrently -n "backend,frontend" -c "blue,green" "uv run scripts/vis_server.py" "cd graphvis && npm run dev"

# View combined infrastructure logs (Postgres, ES)
logs:
    docker compose logs -f

# Clean PostgreSQL database experiments data
db-clean:
    @echo "Cleaning PostgreSQL database..."
    uv run scripts/db_clean.py
