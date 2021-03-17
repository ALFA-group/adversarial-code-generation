#!/bin/bash

export GPU=0

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -ex

rm -f "${DIR}/../datasets/single-attacks-tokens/*/gradient/test.tsv"
rm -f "${DIR}/../datasets/single-attacks-tokens/*/random/test.tsv"

cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $3 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/ReplaceTrueFalse/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $4 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/RenameFields/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $5 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/RenameParameters/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $6 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/WrapTryCatch/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $7 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/InsertPrintStatements/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $8 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/UnrollWhiles/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $9 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/AddDeadCode/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/gradient-targeting/test.tsv" | awk -F'\t' '{ print $10 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/RenameLocalVariables/gradient/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $3 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/ReplaceTrueFalse/random/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $4 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/RenameFields/random/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $5 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/RenameParameters/random/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $6 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/WrapTryCatch/random/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $7 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/InsertPrintStatements/random/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $8 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/UnrollWhiles/random/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $9 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/AddDeadCode/random/test.tsv"
cat "${DIR}/../datasets/adversarial/test-depth-1-attack/tokens/sri/py150/random-targeting/test.tsv" | awk -F'\t' '{ print $10 "\t" $2 }' > "${DIR}/../datasets/single-attacks-tokens/RenameLocalVariables/random/test.tsv"

sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/ReplaceTrueFalse/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/RenameFields/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/RenameParameters/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/WrapTryCatch/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/InsertPrintStatements/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/UnrollWhiles/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/AddDeadCode/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/RenameLocalVariables/gradient/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/ReplaceTrueFalse/random/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/RenameFields/random/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/RenameParameters/random/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/WrapTryCatch/random/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/InsertPrintStatements/random/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/UnrollWhiles/random/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/AddDeadCode/random/test.tsv"
sed -i "1s/.*/src\ttgt/" "${DIR}/../datasets/single-attacks-tokens/RenameLocalVariables/random/test.tsv"

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/sri/py150/normal DATASET_NAME=datasets/single-attacks-tokens/AddDeadCode/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/AddDeadCode/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/sri/py150/normal DATASET_NAME=datasets/single-attacks-tokens/AddDeadCode/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/AddDeadCode/random time make test-model-seq2seq

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/InsertPrintStatements/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/InsertPrintStatements/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/InsertPrintStatements/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/InsertPrintStatements/random time make test-model-seq2seq

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/RenameFields/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/RenameFields/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/RenameFields/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/RenameFields/random time make test-model-seq2seq

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/RenameLocalVariables/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/RenameLocalVariables/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/RenameLocalVariables/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/RenameLocalVariables/random time make test-model-seq2seq

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/RenameParameters/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/RenameParameters/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/RenameParameters/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/RenameParameters/random time make test-model-seq2seq

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/ReplaceTrueFalse/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/ReplaceTrueFalse/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/ReplaceTrueFalse/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/ReplaceTrueFalse/random time make test-model-seq2seq

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/UnrollWhiles/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/UnrollWhiles/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/UnrollWhiles/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/UnrollWhiles/random time make test-model-seq2seq

CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/WrapTryCatch/gradient RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/WrapTryCatch/gradient time make test-model-seq2seq
CHECKPOINT=Best_F1 ARGS='--no-attack' MODELS_IN=final-models/seq2seq/c2s/java-small/normal DATASET_NAME=datasets/single-attacks-tokens/WrapTryCatch/random RESULTS_OUT=final-results/individual-attacks/c2s-java/normal-model/WrapTryCatch/random time make test-model-seq2seq
