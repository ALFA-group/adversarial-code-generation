#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

IMAGE_NAME="$(whoami)/averloc--${1}:$(git rev-parse HEAD)"
IMAGE_CONTEXT="$(mktemp -d)"

echo -e "\033[0;37m[DBG]:\033[0m" "Building '${IMAGE_NAME}'..."

function tearDown {
  rm -rf "${IMAGE_CONTEXT}"
}
trap tearDown ERR

cp -r "${DIR}/../tasks/${1}" "${IMAGE_CONTEXT}/task"
cp -r "${DIR}/../vendor" "${IMAGE_CONTEXT}"

docker build \
	-t "${IMAGE_NAME}" \
	-f "${IMAGE_CONTEXT}/task/Dockerfile" \
	"${IMAGE_CONTEXT}" \
| sed "s/^/$(printf "\r\033[37m[DBG]:\033[0m ")/"

rm -rf "${IMAGE_CONTEXT}"

echo -e "\033[0;37m[DBG]:\033[0m" "  + Image built!"
