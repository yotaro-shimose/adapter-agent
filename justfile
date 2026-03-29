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

# Start concurrently the visualization API (port 8000) and the React frontend (port 5173)
vis:
    @echo "Starting visualization backend (8000) and frontend (5173)..."
    (cd graphvis && npm run dev) & uv run scripts/vis_server.py