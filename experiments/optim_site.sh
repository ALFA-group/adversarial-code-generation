# get normalized sri/py150 transformed dataset
echo 'Applying transforms...'
#make apply-transforms-sri-py150

# tokenize the normalized datasest
echo 'Tokenizing transforms...'
#python3 tasks/preprocess-dataset-tokens/preprocess.py
#make extract-transformed-tokens

# handle replacement tokens
echo 'Handling replacement tokens'
#python3 tasks/extract-adv-dataset-tokens/app_site.py test transforms.InsertHoles

# do gradient targeting
echo 'Doing gradient targeting...'
#python3 models/pytorch-seq2seq/gradient_attack_site.py --data_path datasets/adversarial/sri/py150/test.tsv --expt_dir final-models/seq2seq/sri/py150/normal/lstm --load_checkpoint Best_F1 --replacement_tokens "the value of x and y in for and if is 123 5321 sum total avg" --save_path datasets/adversarial/sri/py150/testing.json --random

# insert replacement tokens at the chosen site
echo 'Inserting tokens...'
#python3 models/pytorch-seq2seq/replace_site.py --source_data_path datasets/adversarial/sri/py150/test.tsv --dest_data_path datasets/adversarial/sri/py150/gradient-attack/test.tsv --mapping_json datasets/adversarial/sri/py150/testing-gradient.json
#python3 models/pytorch-seq2seq/replace_site.py --source_data_path datasets/adversarial/sri/py150/test.tsv --dest_data_path datasets/adversarial/sri/py150/random-attack/test.tsv --mapping_json datasets/adversarial/sri/py150/testing-random.json

CHECKPOINT="Best_F1" \
NO_RANDOM="false" \
NO_GRADIENT="false" \
NO_TEST="false" \
AVERLOC_JUST_TEST="true" \
GPU=0 \
SHORT_NAME="renamevar-param" \
DATASET=sri/py150 \
MODELS_IN=final-models/seq2seq/sri/py150/normal/ \
TRANSFORMS="renamevar-param" \
	time make extract-adv-dataset-tokens
