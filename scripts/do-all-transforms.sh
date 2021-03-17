#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd "${DIR}/../" &> /dev/null

echo "Working on c2s/java-small..."
AVERLOC_JUST_TEST=true DEPTH=2 NUM_SAMPLES=10 \
time make apply-transforms-c2s-java-small \
  &> "${DIR}/../logs/transform-c2s-java-small-depth-2.txt"
AVERLOC_JUST_TEST=true DEPTH=3 NUM_SAMPLES=10 \
time make apply-transforms-c2s-java-small \
  &> "${DIR}/../logs/transform-c2s-java-small-depth-3.txt"
AVERLOC_JUST_TEST=true DEPTH=4 NUM_SAMPLES=10 \
time make apply-transforms-c2s-java-small \
  &> "${DIR}/../logs/transform-c2s-java-small-depth-4.txt"
echo "  + Done generating extra test sets"

echo "Working on csn/java..."
AVERLOC_JUST_TEST=true DEPTH=2 NUM_SAMPLES=10 \
time make apply-transforms-csn-java \
  &> "${DIR}/../logs/transform-csn-java-depth-2.txt"
AVERLOC_JUST_TEST=true DEPTH=3 NUM_SAMPLES=10 \
time make apply-transforms-csn-java \
  &> "${DIR}/../logs/transform-csn-java-depth-3.txt"
AVERLOC_JUST_TEST=true DEPTH=4 NUM_SAMPLES=10 \
time make apply-transforms-csn-java \
  &> "${DIR}/../logs/transform-csn-java-depth-4.txt"
echo "  + Done generating extra test sets"

echo "Working on csn/python..."
AVERLOC_JUST_TEST=true DEPTH=2 NUM_SAMPLES=10 \
time make apply-transforms-csn-python \
  &> "${DIR}/../logs/transform-csn-python-depth-2.txt"
AVERLOC_JUST_TEST=true DEPTH=3 NUM_SAMPLES=10 \
time make apply-transforms-csn-python \
  &> "${DIR}/../logs/transform-csn-python-depth-3.txt"
AVERLOC_JUST_TEST=true DEPTH=4 NUM_SAMPLES=10 \
time make apply-transforms-csn-python \
  &> "${DIR}/../logs/transform-csn-python-depth-4.txt"
echo "  + Done generating extra test sets"

echo "Working on sri/py150..."
AVERLOC_JUST_TEST=true DEPTH=2 NUM_SAMPLES=10 \
time make apply-transforms-sri-py150 \
  &> "${DIR}/../logs/transform-sri-py150-depth-2.txt"
AVERLOC_JUST_TEST=true DEPTH=3 NUM_SAMPLES=10 \
time make apply-transforms-sri-py150 \
  &> "${DIR}/../logs/transform-sri-py150-depth-3.txt"
AVERLOC_JUST_TEST=true DEPTH=4 NUM_SAMPLES=10 \
time make apply-transforms-sri-py150 \
  &> "${DIR}/../logs/transform-sri-py150-depth-4.txt"
echo "  + Done generating extra test sets"

popd &> /dev/null
echo "Finished!"
