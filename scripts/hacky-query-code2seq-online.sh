#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
THE_URL="https://djz6amor4b.execute-api.us-east-1.amazonaws.com/prod/predict"

for word in $(cat "${2}" | head -n500); do
  prediction=$(curl -s \
    --header "Content-Type: application/json" \
    --request POST \
    --data "$(cat "${3}" | sed -e "s/???/${word}/g" | "${DIR}/../results/jq" -R -s -c '{ code: . }')" \
    "${THE_URL}" \
    | tac | tac \
    | "${DIR}/../results/jq"  \
      -r '[.["0"].predictions[] | .prediction] | join("|")'
  )

  if [ "${prediction}" != "${1}" ]; then
    echo "${word}"
  fi
  sleep 2
done
