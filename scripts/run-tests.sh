#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd "${DIR}/.." > /dev/null

PREFIX=datasets/transformed/preprocessed/ast-paths/c2s/java-small

rm -f results.stderr.txt
rm -f results-transforms.*.txt

touch results.stderr.txt

echo "Running on datasets/.../transformed.Identity:"
DATASET_NAME=${PREFIX}/transforms.Identity \
  time make test-model-code2seq > results-transforms.Identity.txt 2>> results.stderr.txt
cat results-transforms.Identity.txt | tail -n5

echo "Running on datasets/.../transformed.All:"
DATASET_NAME=${PREFIX}/transforms.All \
  time make test-model-code2seq > results-transforms.All.txt 2>> results.stderr.txt
cat results-transforms.All.txt | tail -n5

echo "Running on datasets/.../transformed.InsertPrintStatements:"
DATASET_NAME=${PREFIX}/transforms.InsertPrintStatements \
  time make test-model-code2seq > results-transforms.InsertPrintStatements.txt 2>> results.stderr.txt
cat results-transforms.InsertPrintStatements.txt | tail -n5

echo "Running on datasets/.../transformed.RenameFields:"
DATASET_NAME=${PREFIX}/transforms.RenameFields \
  time make test-model-code2seq > results-transforms.RenameFields.txt 2>> results.stderr.txt
cat results-transforms.RenameFields.txt | tail -n5

echo "Running on datasets/.../transformed.RenameLocalVariables:"
DATASET_NAME=${PREFIX}/transforms.RenameLocalVariables \
  time make test-model-code2seq > results-transforms.RenameLocalVariables.txt 2>> results.stderr.txt
cat results-transforms.RenameLocalVariables.txt | tail -n5

echo "Running on datasets/.../transformed.RenameParameters:"
DATASET_NAME=${PREFIX}/transforms.RenameParameters \
  time make test-model-code2seq > results-transforms.RenameParameters.txt 2>> results.stderr.txt
cat results-transforms.RenameParameters.txt | tail -n5

echo "Running on datasets/.../transformed.ReplaceTrueFalse:"
DATASET_NAME=${PREFIX}/transforms.ReplaceTrueFalse \
  time make test-model-code2seq > results-transforms.ReplaceTrueFalse.txt 2>> results.stderr.txt
cat results-transforms.ReplaceTrueFalse.txt | tail -n5

echo "Running on datasets/.../transformed.ShuffleLocalVariables:"
DATASET_NAME=${PREFIX}/transforms.ShuffleLocalVariables \
  time make test-model-code2seq > results-transforms.ShuffleLocalVariables.txt 2>> results.stderr.txt
cat results-transforms.ShuffleLocalVariables.txt | tail -n5

echo "Running on datasets/.../transformed.ShuffleParameters:"
DATASET_NAME=${PREFIX}/transforms.ShuffleParameters \
  time make test-model-code2seq > results-transforms.ShuffleParameters.txt 2>> results.stderr.txt
cat results-transforms.ShuffleParameters.txt | tail -n5

popd > /dev/null
