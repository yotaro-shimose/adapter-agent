default:
  just --list


stop-docker:
    docker ps -q | xargs -r docker stop

clean-tmp:
    rm -rf /tmp/*

remove-checkpoints:
    uv run scripts/remove_checkpoints.py

# Delete Cloud Run services that contain a specific pattern in their name (delegated to coder-mcp)
# Usage: just delete-cloudrun [pattern="coder-mcp-numrs2"] [region="europe-north1"]
delete-cloudrun pattern="coder-mcp-numrs2" region="europe-north1":
    just --justfile ../coder-mcp/justfile delete-cloudrun-services {{pattern}} {{region}}