stop-docker:
    docker ps -q | xargs -r docker stop