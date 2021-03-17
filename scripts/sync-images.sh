#!/bin/bash

# Syncs our docker images with the latest commit hash
# and cleans up the old ones

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo -e "\033[0;37m[DBG]:\033[0m Syncing built images..." 
find "${DIR}/../tasks" -mindepth 1 -maxdepth 1 -type d -exec sh -c '
  for task; do \
    image_name="$(whoami)/averloc--$(echo "${task}" | rev | cut -d/ -f1 | rev)"; \
    most_recent="$(docker images -q "${image_name}" | head -n1)"; \
    if [ ! -z "${most_recent}" ]; then \
        echo -e \
            "\033[0;37m[DBG]:\033[0m" \
            "  + tag" \
            "${most_recent}" \
            "-->" \
            "${image_name}:$(git rev-parse HEAD)"; \
        docker tag "${most_recent}" "${image_name}:$(git rev-parse HEAD)"; \
    fi \
  done
' sh {} +
echo -e "\033[0;37m[DBG]:\033[0m   + Sync completed!" 

echo -e "\033[0;37m[DBG]:\033[0m Cleanup old images..."
docker rmi $(
    docker images "$(whoami)/averloc--*" \
        --format "{{.ID}} {{.Repository}}:{{.Tag}}" \
    | grep -v "$(git rev-parse HEAD)" \
    | cut -d' ' -f1
)
echo -e "\033[0;37m[DBG]:\033[0m   + Cleanup completed!"
