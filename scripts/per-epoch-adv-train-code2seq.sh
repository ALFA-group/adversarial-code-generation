#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "${DIR}"
if [[ -z "${GPU}" ]]; then
  echo "Error: please specify a GPU (GPU=<device>)"
  exit 1
fi

if [[ -z "${DATASET}" ]]; then
  echo "Error: please specfiy dataset (DATASET=c2s/java-small)"
  exit 1
fi

trap "echo '[ADV-TRAIN]  - CTRL-C Pressed. Quiting...'; exit;" SIGINT SIGTERM

pushd "${DIR}/.." > /dev/null

if [ "${ADV_TYPE}" = "random" ]; then
  export NO_RANDOM="false"
  export NO_GRADIENT="true"
  export TARGETING="random"
elif [ "${ADV_TYPE}" = "gradient" ]; then
  export NO_RANDOM="true"
  export NO_GRADIENT="false"
  export TARGETING="gradient"
else
  echo "Error: please specify either --random or --gradient as first arg."
  exit 1
fi

export GPU="${GPU}"
export LAMB=0.4
export NUM_T=8
export MAX_EPOCHS=10
export DATASET="${DATASET}"

SELECTED_MODEL=$(basename $(
  find "${DIR}/../final-models/code2seq/${DATASET}/normal" \
    -type f \
    -name "model_iter*" \
  | awk -F'.' '{ print $3 }' \
  | sort -t 'r' -k 3 -n -u \
  | tail -n1
))
echo "${SELECTED_MODEL}"

echo "[ADV-TRAIN] Copying over normal model for first-round targeting..."
docker run -it --rm \
  -v "${DIR}/../final-models/code2seq/${DATASET}/normal":/mnt/inputs \
  -v "${DIR}/../final-models/code2seq/pe${MAX_EPOCHS}-${TARGETING}-l${LAMB}/${DATASET}/adversarial":/mnt/outputs \
  debian:9 \
    bash -c "\
      cp /mnt/inputs/${SELECTED_MODEL}.data-00000-of-00001 /mnt/outputs/model_iter1.data-00000-of-00001 && \
      cp /mnt/inputs/${SELECTED_MODEL}.dict /mnt/outputs/model_iter1.dict && \
      cp /mnt/inputs/${SELECTED_MODEL}.index /mnt/outputs/model_iter1.index && \
      cp /mnt/inputs/${SELECTED_MODEL}.meta /mnt/outputs/model_iter1.meta  \
    "
echo "[ADV-TRAIN]   + Done!"

echo "[ADV-TRAIN] Starting loop..."

for epoch in $(seq 1 ${MAX_EPOCHS}); do
echo "[ADV-TRAIN]   + Epoch ${epoch} targeting begins..."

NO_RANDOM="${NO_RANDOM}" \
NO_GRADIENT="${NO_GRADIENT}" \
NO_TEST="true" \
SHORT_NAME="d1-train-epoch-${epoch}-l${LAMB}" \
GPU="${GPU}" \
DATASET="${DATASET}" \
MODELS_IN=final-models/code2seq/pe${MAX_EPOCHS}-${TARGETING}-l${LAMB}/${DATASET}/adversarial \
TRANSFORMS='transforms\.\w+' \
  time make extract-adv-dataset-ast-paths

echo "[ADV-TRAIN]     + Targeting complete!"
echo "[ADV-TRAIN]   + Epoch ${epoch} training begins..."

ARGS="${DO_FINETUNE} ${NUM_T} --epochs 2 --lamb ${LAMB}" \
SHORT_NAME="d1-train-epoch-${epoch}-l${LAMB}" \
GPU="${GPU}" \
MODELS_OUT=final-models/code2seq/pe${MAX_EPOCHS}-${TARGETING}-l${LAMB}/${DATASET} \
DATASET_NAME=datasets/adversarial/${SHORT_NAME}/ast-paths/${DATASET}/${TARGETING}-targeting \
  time make adv-train-model-code2seq
echo "[ADV-TRAIN]     + Training epoch complete!"

rm -rf ${DIR}/../datasets/adversarial/d1-train-epoch-${epoch}-l${LAMB}/
export DO_FINETUNE="--adv_fine_tune"
done

echo "[ADV-TRAIN]  + Training finished!"

popd > /dev/null
