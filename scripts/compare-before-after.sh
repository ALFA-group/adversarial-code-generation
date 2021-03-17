#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

THE_OBJ=$(
  cat "${DIR}/../datasets/transformed/normalized/c2s/java-small/transforms.RenameLocalVariables/test.jsonl.gz" \
  | gzip -cd \
  | head -n1000 \
  | shuf \
  | head -n1
)

MAYBE_ID=$(
  /u/j/j/jjhenkel/jq .identifier <<< "${THE_OBJ}"
)

MATCHES=$(cat "${DIR}/../datasets/normalized/c2s/java-small/test.jsonl.gz" \
  | gzip -cd \
  | grep "\"identifier\": ${MAYBE_ID}" \
  | wc -l
)

while [[ "${MATCHES}" != "1" ]]; do
  THE_OBJ=$(
    cat "${DIR}/../datasets/transformed/normalized/c2s/java-small/transforms.RenameLocalVariables/test.jsonl.gz" \
    | gzip -cd \
    | head -n1000 \
    | shuf \
    | head -n1
  )

  MAYBE_ID=$(
    /u/j/j/jjhenkel/jq .identifier <<< "${THE_OBJ}"
  )

  MATCHES=$(cat "${DIR}/../datasets/normalized/c2s/java-small/test.jsonl.gz" \
    | gzip -cd \
    | grep "\"identifier\": ${MAYBE_ID}" \
    | wc -l
  )
done

if [[ "${MATCHES}" == "1" ]]; then
  rm -f "${DIR}/../to-compare.before.java"
  rm -f "${DIR}/../to-compare.after.java"

  /u/j/j/jjhenkel/jq -r .source_code <<< "${THE_OBJ}" > "${DIR}/../to-compare.after.java"

  cat "${DIR}/../datasets/normalized/c2s/java-small/test.jsonl.gz" \
  | gzip -cd \
  | grep "\"identifier\": ${MAYBE_ID}" \
  | /u/j/j/jjhenkel/jq -r .source_code \
  > "${DIR}/../to-compare.before.java"
fi
