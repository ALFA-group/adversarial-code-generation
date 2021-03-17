
set -ex

export TYPE=random
export DATA=data1
export ATTACK=ReplaceTrueFalse

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=random
export DATA=data2
export ATTACK=RenameFields

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=random
export DATA=data3
export ATTACK=RenameParameters

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=random
export DATA=data4
export ATTACK=WrapTryCatch

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=random
export DATA=data5
export ATTACK=InsertPrintStatements

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=random
export DATA=data6
export ATTACK=UnrollWhiles

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=random
export DATA=data7
export ATTACK=AddDeadCode

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=random
export DATA=data8
export ATTACK=RenameLocalVariables

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data1
export ATTACK=ReplaceTrueFalse

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data2
export ATTACK=RenameFields

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data3
export ATTACK=RenameParameters

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data4
export ATTACK=WrapTryCatch

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data5
export ATTACK=InsertPrintStatements

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data6
export ATTACK=UnrollWhiles

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data7
export ATTACK=AddDeadCode

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq

export TYPE=gradient
export DATA=data8
export ATTACK=RenameLocalVariables

GPU="3" ARGS="--individual ${DATA}" \
DATASET_NAME=datasets/adversarial/test-depth-1-attack/ast-paths/c2s/java-small/${TYPE}-targeting \
RESULTS_OUT=final-results/individual-attacks/code2seq/c2s-java/normal-model/${ATTACK}/${TYPE} \
MODELS_IN="final-models/code2seq/c2s/java-small/normal" \
  time make test-model-code2seq
