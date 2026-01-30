stop-docker:
    docker ps -q | xargs -r docker stop

clean-tmp:
    rm -rf /tmp/*