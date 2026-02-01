default:
  just --list


stop-docker:
    docker ps -q | xargs -r docker stop

clean-tmp:
    rm -rf /tmp/*

remove-checkpoints:
    uv run scripts/remove_checkpoints.py