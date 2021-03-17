
ARGS="--epochs 20" GPU=1 MODELS_OUT=final-models/code2seq/sri/py150 DATASET_NAME=datasets/transformed/preprocessed/ast-paths/sri/py150/transforms.Identity DATASET_SHORT_NAME=py150 time make train-model-code2seq

