export SHELL:=/bin/bash
export SHELLOPTS:=$(if $(SHELLOPTS),$(SHELLOPTS):)pipefail:errexit

.ONESHELL:

ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Cross-platform realpath from 
# https://stackoverflow.com/a/18443300
# NOTE: Adapted for Makefile use
define BASH_FUNC_realpath%%
() {
  OURPWD=$PWD
  cd "$(dirname "$1")"
  LINK=$(readlink "$(basename "$1")")
  while [ "$LINK" ]; do
    cd "$(dirname "$LINK")"
    LINK=$(readlink "$(basename "$1")")
  done
  REALPATH="$PWD/$(basename "$1")"
  cd "$OURPWD"
  echo "$REALPATH"
}
endef
export BASH_FUNC_realpath%%

define echo_debug
	echo -e "\033[0;37m[DBG]:\033[0m" ${1}
endef
define echo_info
	echo -e "[INF]:" ${1}
endef
define echo_warn
	echo -e "\033[0;33m[WRN]:\033[0m" ${1}
endef
define echo_error
	echo -e "\033[0;31m[ERR]:\033[0m" ${1}
endef

define mkdir_cleanup_on_error
	function tearDown {
		rm -rf ${1}
	}
	trap tearDown ERR
	mkdir -p ${1}
endef

define run_dataset_ast_paths_preprocessing
	@NORM_PATH="${ROOT_DIR}/${1}/${2}"
	@PROC_PATH="${ROOT_DIR}/$(subst normalized,preprocessed/ast-paths,${1})/${2}"
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	mkdir -p "$${PROC_PATH}"
	docker run -it --rm \
		-v "$${NORM_PATH}":/analysis/inputs/public/source-code \
		-v "$${PROC_PATH}":/analysis/output/fs/ast-paths/${3} \
		"$${IMAGE_NAME}"
	$(call echo_debug,"    + Done!")
endef

export echo_debug
export echo_info
export echo_warn
export echo_error

export mkdir_cleanup_on_error
export run_dataset_ast_paths_preprocessing

.DEFAULT_GOAL := help

#######################################################################################################################
#######################################################################################################################

.PHONY: help
help: ## (MISC) This help.
	@grep -E \
		'^[\/\.0-9a-zA-Z_-]+:.*?## .*$$' \
		$(MAKEFILE_LIST) \
		| grep -v '<!PRIVATE>' \
		| sort -t'#' -k3,3 \
		| awk 'BEGIN {FS = ":.*?## "}; \
		       {printf "\033[36m%-34s\033[0m %s\n", $$1, $$2}'

.PHONY: docker-cleanup
.SILENT: docker-cleanup
docker-cleanup: ## (MISC) Cleans up old and out-of-sync Docker images.
	$(call echo_debug,"Removing exited containers...")
	docker rm $(docker ps -aqf status=exited)
	$(call echo_debug,"  + Exited containers removed!")
	$(call echo_debug,"Removing dangling images...")
	docker rmi $(docker images -qf dangling=true)
	$(call echo_debug,"  + Dangling images removed!")
	"${ROOT_DIR}/scripts/sync-images.sh"

.PHONY: submodules
.SILENT: submodules
submodules: ## (MISC) Ensures that submodules are setup.
	## https://stackoverflow.com/a/52407662
	if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo -e "\033[0;37m[DBG]:\033[0m" "Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi

#######################################################################################################################
#######################################################################################################################

.PHONY: build-image-apply-targeting-seq2seq
build-image-apply-targeting-seq2seq: submodules ## Build tasks/apply-targeting-seq2seq <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		apply-targeting-seq2seq

.PHONY: build-image-astor-apply-transforms
build-image-astor-apply-transforms: submodules ## Builds our baseline generator docker image  <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		astor-apply-transforms

.PHONY: build-image-augment-dataset-tokens
build-image-augment-dataset-tokens: submodules ## Builds our dataset-augmentation processor (for tokens) <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		augment-dataset-tokens

.PHONY: build-image-depth-k-test-seq2seq
build-image-depth-k-test-seq2seq: submodules ## Build tasks/depth-k-test-seq2seq <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		depth-k-test-seq2seq

.PHONY: build-image-download-c2s-dataset
build-image-download-c2s-dataset: submodules ## Builds tasks/download-c2s-dataset <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		download-c2s-dataset

.PHONY: build-image-download-csn-dataset
build-image-download-csn-dataset: submodules ## Builds tasks/download-csn-dataset <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		download-csn-dataset

.PHONY: build-image-download-onedrive-dataset
build-image-download-onedrive-dataset: submodules ## Builds tasks/download-onedrive-dataset <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		download-onedrive-dataset

.PHONY: build-image-extract-adv-dataset-c2s
build-image-extract-adv-dataset-c2s: submodules ## Builds our adversarial dataset extractor (representation: ast-paths). <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		extract-adv-dataset-c2s

.PHONY: build-image-extract-adv-dataset-tokens
build-image-extract-adv-dataset-tokens: submodules ## Builds our adversarial dataset extractor (representation: tokens). <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		extract-adv-dataset-tokens

.PHONY: build-image-generate-baselines
build-image-generate-baselines: submodules ## Builds our baseline generator docker image  <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		generate-baselines

.PHONY: build-image-integrated-gradients-seq2seq
build-image-integrated-gradients-seq2seq: submodules ## Builds our IG for seq2seq docker image  <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		integrated-gradients-seq2seq

.PHONY: build-image-normalize-raw-dataset
build-image-normalize-raw-dataset: submodules ## Builds our dataset normalizer docker image  <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		normalize-raw-dataset

.PHONY: build-image-preprocess-dataset-c2s
build-image-preprocess-dataset-c2s: submodules ## Builds a preprocessor for generating code2seq style data <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		preprocess-dataset-c2s

.PHONY: build-image-preprocess-dataset-tokens
build-image-preprocess-dataset-tokens: submodules ## Builds our tokens dataset preprocessor (for seq2seq model)  <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		preprocess-dataset-tokens

.PHONY: build-image-spoon-apply-transforms
build-image-spoon-apply-transforms: submodules ## Builds our dockerized version of spoon. <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-image.sh" \
		spoon-apply-transforms

.PHONY: build-image-test-model-code2seq
build-image-test-model-code2seq: submodules ## Build tasks/test-model-code2seq <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		test-model-code2seq

.PHONY: build-image-train-model-code2seq
build-image-train-model-code2seq: submodules ## Build tasks/train-model-code2seq <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		train-model-code2seq

.PHONY: build-image-test-model-seq2seq
build-image-test-model-seq2seq: submodules ## Build tasks/test-model-seq2seq <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		test-model-seq2seq

.PHONY: build-image-train-model-seq2seq
build-image-train-model-seq2seq: submodules ## Build tasks/train-model-seq2seq <!PRIVATE>
	@"${ROOT_DIR}/scripts/build-model-image.sh" \
		train-model-seq2seq

#######################################################################################################################
#######################################################################################################################

.PHONY: check-dataset-path
check-dataset-path: ## Ensures DATASET="<blah>" parameter is present. <!PRIVATE>
ifndef DATASET
	$(error DATASET is a required parameter for this target.)
endif

.PHONY: requires-confirmation
requires-confirmation: ## Forces user to confirm action. <!PRIVATE>
	@docker run -it --rm \
		-v "${ROOT_DIR}/$${DATASET}:/mnt" \
		-v "${ROOT_DIR}/scripts:/scripts" \
		debian:stretch \
		  bash /scripts/list-files-before-delete.sh
	@echo -n "Are you sure you want to remove these files: [y/N] " \
		&& read ANSWER \
		&& [ $${ANSWER:-N} = y ]

.PHONY: clear-dataset
danger-clear-dataset: check-dataset-path requires-confirmation ## (DANGER) Removes the datset specified by DATASET="datasets/<name>".
	docker run -it --rm \
		-v "${ROOT_DIR}/$${DATASET}:/mnt" \
		debian:stretch \
		  bash -c 'rm -rf /mnt/c2s/* /mnt/csn/* /mnt/sri/*'

#######################################################################################################################
#######################################################################################################################

datasets/raw/c2s/java-small: ## Downloads code2seq's Java small dataset (non-preprocessed sources) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--download-c2s-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/c2s/java-small:/mnt" \
		-e DATASET_URL=https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz \
		"$${IMAGE_NAME}"

datasets/raw/csn/java: ## Downloads CodeSearchNet's Java data (GitHub's code search dataset) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--download-csn-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/csn/java:/mnt" \
		-e DATASET_URL=https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip \
		"$${IMAGE_NAME}"

datasets/raw/csn/python: ## Downloads CodeSearchNet's Python data (GitHub's code search dataset) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--download-csn-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/csn/python:/mnt" \
		-e DATASET_URL=https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip \
		"$${IMAGE_NAME}"

datasets/raw/sri/py150: ## Downloads SRI Lab's py150k dataset (our semi-preprocessed version on One Drive) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--download-onedrive-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/sri/py150:/mnt" \
		-e TEST_FILE_URL='https://onedrive.live.com/download?cid=2F634444193E8221&resid=2F634444193E8221%2182214&authkey=AOQXaY9GP1OUK-4' \
		-e TRAIN_FILE_URL='https://onedrive.live.com/download?cid=2F634444193E8221&resid=2F634444193E8221%2182215&authkey=ACnuAlzcKJ_tBQI' \
		-e VALID_FILE_URL='https://onedrive.live.com/download?cid=2F634444193E8221&resid=2F634444193E8221%2182213&authkey=ABAUlzP9kkM_3f0' \
		"$${IMAGE_NAME}"

DD_DEPS := datasets/raw/c2s/java-small
DD_DEPS += datasets/raw/csn/java
DD_DEPS += datasets/raw/csn/python
DD_DEPS += datasets/raw/sri/py150

.PHONY: download-datasets
download-datasets: build-image-download-csn-dataset build-image-download-c2s-dataset build-image-download-onedrive-dataset | $(DD_DEPS) ## (DS-1) Downloads all prerequisite datasets
	@$(call echo_info,"Downloaded all datasets to './datasets/raw/' directory.")

#######################################################################################################################
#######################################################################################################################

datasets/normalized/debug/java: ## Generate a normalized version of our Java debugging dataset <!PRIVATE>
	@$(call echo_debug,"Normalizing dataset 'raw/c2s/java-small' (to debug/java)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--normalize-raw-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/c2s/java-small:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/normalized/debug/java:/mnt/outputs" \
		"$${IMAGE_NAME}" java gz 3000 500 1000
	@$(call echo_debug,"  + Normalization complete!")

datasets/normalized/c2s/java-small: ## Generate a normalized version of code2seq's Java small dataset <!PRIVATE>
	@$(call echo_debug,"Normalizing dataset 'raw/c2s/java-small'...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--normalize-raw-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/c2s/java-small:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/normalized/c2s/java-small:/mnt/outputs" \
		"$${IMAGE_NAME}" java gz 150000 10000 20000
	@$(call echo_debug,"  + Normalization complete!")

datasets/normalized/csn/java: ## Generates a normalized version of CodeSearchNet's Java dataset <!PRIVATE>
	@$(call echo_debug,"Normalizing dataset 'raw/csn/java'...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--normalize-raw-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/csn/java:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/normalized/csn/java:/mnt/outputs" \
		"$${IMAGE_NAME}" java gz 150000 10000 20000
	@$(call echo_debug,"  + Normalization complete!")

datasets/normalized/csn/python: ## Generates a normalized version of CodeSearchNet's Python dataset <!PRIVATE>
	@$(call echo_debug,"Normalizing dataset 'raw/csn/python'...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--normalize-raw-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/csn/python:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/normalized/csn/python:/mnt/outputs" \
		"$${IMAGE_NAME}" python gz 150000 10000 20000
	@$(call echo_debug,"  + Normalization complete!")

datasets/normalized/sri/py150: ## Generates a normalized version of SRI Lab's py150k dataset <!PRIVATE>
	@$(call echo_debug,"Normalizing dataset 'raw/sri/py150'...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--normalize-raw-dataset:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/raw/sri/py150:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/normalized/sri/py150:/mnt/outputs" \
		"$${IMAGE_NAME}" python gz 150000 10000 20000
	@$(call echo_debug,"  + Normalization complete!")

ND_DEPS := datasets/normalized/c2s/java-small
ND_DEPS += datasets/normalized/csn/java
ND_DEPS += datasets/normalized/csn/python
ND_DEPS += datasets/normalized/sri/py150

.PHONY: normalize-datasets
normalize-datasets: build-image-normalize-raw-dataset | $(ND_DEPS) ## (DS-2) Normalizes all downloaded datasets
	@$(call echo_info,"Normalized all datasets to './datasets/normalized/' directory.")

#######################################################################################################################
#######################################################################################################################

datasets/preprocessed/ast-paths/c2s/java-small: ## Generate a preprocessed (representation: ast-paths) version of code2seq's Java small dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/ast-paths/c2s/java-small' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/vendor/code2seq:/code2seq" \
		-v "${ROOT_DIR}/datasets/normalized/c2s/java-small:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/ast-paths/c2s/java-small:/mnt/outputs" \
		"$${IMAGE_NAME}" java
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

datasets/preprocessed/ast-paths/csn/java: ## Generate a preprocessed (representation: ast-paths) version of CodeSearchNet's Java dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/ast-paths/csn/java' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/normalized/csn/java:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/ast-paths/csn/java:/mnt/outputs" \
		"$${IMAGE_NAME}" java
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

datasets/preprocessed/ast-paths/csn/python: ## Generate a preprocessed (representation: ast-paths) version of CodeSearchNet's Python dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/ast-paths/csn/python' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	docker run -it  \
		-v "${ROOT_DIR}/datasets/normalized/csn/python:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/ast-paths/csn/python:/mnt/outputs" \
		"$${IMAGE_NAME}" python
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

datasets/preprocessed/ast-paths/sri/py150: ## Generate a preprocessed (representation: ast-paths) version of SRI Lab's py150k dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/ast-paths/sri/py150' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/normalized/sri/py150:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/ast-paths/sri/py150:/mnt/outputs" \
		"$${IMAGE_NAME}" python
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

EAP_DEPS := datasets/preprocessed/ast-paths/c2s/java-small
EAP_DEPS += datasets/preprocessed/ast-paths/csn/java
EAP_DEPS += datasets/preprocessed/ast-paths/csn/python
EAP_DEPS += datasets/preprocessed/ast-paths/sri/py150

extract-ast-paths: build-image-preprocess-dataset-c2s | $(EAP_DEPS) ## (DS-3) Generate preprocessed data in a form usable by code2seq style models. 
	@$(call echo_info,"AST Paths (code2seq style) preprocessed representations extracted!")

#######################################################################################################################
#######################################################################################################################

datasets/preprocessed/tokens/c2s/java-small: ## Generate a preprocessed (representation: tokens) version of code2seq's java-small dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/tokens/c2s/java-small' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/normalized/c2s/java-small:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/tokens/c2s/java-small:/mnt/outputs" \
		"$${IMAGE_NAME}"
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/preprocessed/tokens/csn/java: ## Generate a preprocessed (representation: tokens) version of CodeSearchNet's Java dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/tokens/csn/java' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/normalized/csn/java:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/tokens/csn/java:/mnt/outputs" \
		"$${IMAGE_NAME}"
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/preprocessed/tokens/csn/python: ## Generate a preprocessed (representation: tokens) version of CodeSearchNet's Python dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/tokens/csn/python' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/normalized/csn/python:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/tokens/csn/python:/mnt/outputs" \
		"$${IMAGE_NAME}"
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/preprocessed/tokens/sri/py150: ## Generate a preprocessed (representation: tokens) version of SRI Lab's py150k dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'preprocessed/tokens/sri/py150' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-v "${ROOT_DIR}/datasets/normalized/sri/py150:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/preprocessed/tokens/sri/py150:/mnt/outputs" \
		"$${IMAGE_NAME}"
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

ETOK_DEPS := datasets/preprocessed/tokens/c2s/java-small
ETOK_DEPS += datasets/preprocessed/tokens/csn/java
ETOK_DEPS += datasets/preprocessed/tokens/csn/python
ETOK_DEPS += datasets/preprocessed/tokens/sri/py150

extract-tokens: build-image-preprocess-dataset-tokens | $(ETOK_DEPS) ## (DS-3) Generate preprocessed data in a form usable by seq2seq style models. 
	@$(call echo_info,"Tokens (seq2seq style) preprocessed representations extracted!")

#######################################################################################################################
#######################################################################################################################

# ALL_TRANSFORMS:="renamevar-param"
ALL_TRANSFORMS:="transforms.Identity"
ALL_TRANSFORMS+="transforms.Insert"
ALL_TRANSFORMS+="transforms.Replace"
ALL_TRANSFORMS+="transforms.Combined"
# ALL_TRANSFORMS+="transforms.AddDeadCode"
# ALL_TRANSFORMS+="transforms.InsertPrintStatements"
# ALL_TRANSFORMS+="transforms.RenameFields"
# ALL_TRANSFORMS+="transforms.RenameLocalVariables"
# ALL_TRANSFORMS+="transforms.RenameParameters"
# ALL_TRANSFORMS+="transforms.ReplaceTrueFalse"
# ALL_TRANSFORMS+="transforms.UnrollWhiles"
# ALL_TRANSFORMS+="transforms.WrapTryCatch"
# ALL_TRANSFORMS+="depth-2-sample-1"
# ALL_TRANSFORMS+="depth-2-sample-2"
# ALL_TRANSFORMS+="depth-2-sample-3"
# ALL_TRANSFORMS+="depth-2-sample-4"
# ALL_TRANSFORMS+="depth-2-sample-5"
# ALL_TRANSFORMS+="depth-2-sample-6"
# ALL_TRANSFORMS+="depth-2-sample-7"
# ALL_TRANSFORMS+="depth-2-sample-8"
# ALL_TRANSFORMS+="depth-2-sample-9"
# ALL_TRANSFORMS+="depth-2-sample-10"
# ALL_TRANSFORMS+="depth-3-sample-1"
# ALL_TRANSFORMS+="depth-3-sample-2"
# ALL_TRANSFORMS+="depth-3-sample-3"
# ALL_TRANSFORMS+="depth-3-sample-4"
# ALL_TRANSFORMS+="depth-3-sample-5"
# ALL_TRANSFORMS+="depth-3-sample-6"
# ALL_TRANSFORMS+="depth-3-sample-7"
# ALL_TRANSFORMS+="depth-3-sample-8"
# ALL_TRANSFORMS+="depth-3-sample-9"
# ALL_TRANSFORMS+="depth-3-sample-10"
# ALL_TRANSFORMS+="depth-4-sample-1"
# ALL_TRANSFORMS+="depth-4-sample-2"
# ALL_TRANSFORMS+="depth-4-sample-3"
# ALL_TRANSFORMS+="depth-4-sample-4"
# ALL_TRANSFORMS+="depth-4-sample-5"
# ALL_TRANSFORMS+="depth-4-sample-6"
# ALL_TRANSFORMS+="depth-4-sample-7"
# ALL_TRANSFORMS+="depth-4-sample-8"
# ALL_TRANSFORMS+="depth-4-sample-9"
# ALL_TRANSFORMS+="depth-4-sample-10"
# ALL_TRANSFORMS+="depth-5-sample-1"
# ALL_TRANSFORMS+="depth-5-sample-2"
# ALL_TRANSFORMS+="depth-5-sample-3"
# ALL_TRANSFORMS+="depth-5-sample-4"
# ALL_TRANSFORMS+="depth-5-sample-5"
# ALL_TRANSFORMS+="depth-5-sample-6"
# ALL_TRANSFORMS+="depth-5-sample-7"
# ALL_TRANSFORMS+="depth-5-sample-8"
# ALL_TRANSFORMS+="depth-5-sample-9"
# ALL_TRANSFORMS+="depth-5-sample-10"

datasets/transformed/preprocessed/ast-paths/debug/java: ## <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/ast-paths/debug/java' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/vendor/code2seq:/code2seq" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/debug/java/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/ast-paths/debug/java/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}" java
	done
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

datasets/transformed/preprocessed/ast-paths/c2s/java-small: ## <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/ast-paths/c2s/java-small' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/vendor/code2seq:/code2seq" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-small/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/ast-paths/c2s/java-small/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}" java
	done
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

datasets/transformed/preprocessed/ast-paths/csn/java: ## <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/ast-paths/csn/java' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/vendor/code2seq:/code2seq" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/java/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/ast-paths/csn/java/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}" java
	done
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

datasets/transformed/preprocessed/ast-paths/csn/python: ## <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/ast-paths/csn/python' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/vendor/code2seq:/code2seq" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/python/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/ast-paths/csn/python/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}" python
	done
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

datasets/transformed/preprocessed/ast-paths/sri/py150: ## <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/ast-paths/sri/py150' (using 'ast-paths' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-c2s:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/vendor/code2seq:/code2seq" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/sri/py150/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/ast-paths/sri/py150/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}" python
	done
	@$(call echo_debug,"  + Finalizing (using 'ast-paths' representation) complete!")

ETAP_DEPS := datasets/transformed/preprocessed/ast-paths/c2s/java-small
ETAP_DEPS += datasets/transformed/preprocessed/ast-paths/csn/java
ETAP_DEPS += datasets/transformed/preprocessed/ast-paths/csn/python
ETAP_DEPS += datasets/transformed/preprocessed/ast-paths/sri/py150

extract-transformed-ast-paths: build-image-preprocess-dataset-c2s | $(ETAP_DEPS) ## (DS-6) Extract preprocessed representations (ast-paths) from our transfromed (normalized) datasets 
	@$(call echo_info,"AST Paths (code2seq style) preprocessed representations extracted (for transformed datasets)!")

#######################################################################################################################
#######################################################################################################################

datasets/transformed/preprocessed/tokens/c2s/debug: ## Generate a preprocessed (representation: tokens) version of the debug dataset <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/tokens/c2s/debug' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/debug/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/c2s/debug/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}"
	done
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/transformed/preprocessed/tokens/c2s/java-small: ## Generate a preprocessed (representation: tokens) version of code2seq's java-small dataset  <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/tokens/c2s/java-small' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-small/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/c2s/java-small/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}"
	done
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/transformed/preprocessed/tokens/c2s/java-med: ## Generate a preprocessed (representation: tokens) version of code2seq's java-med dataset  <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/tokens/c2s/java-med' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-med/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/c2s/java-med/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}"
	done
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/transformed/preprocessed/tokens/csn/java: ## Generate a preprocessed (representation: tokens) version of csn's java dataset  <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/tokens/csn/java' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/java/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/csn/java/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}"
	done
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/transformed/preprocessed/tokens/csn/python: ## Generate a preprocessed (representation: tokens) version of csn's python dataset  <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/tokens/csn/python' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/python/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/csn/python/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}"
	done
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

datasets/transformed/preprocessed/tokens/sri/py150: ## Generate a preprocessed (representation: tokens) version of SRI Lab's py150k dataset  <!PRIVATE>
	@$(call echo_debug,"Finalizing dataset 'transformed/preprocessed/tokens/sri/py150' (using 'tokens' representation)...")
	@$(call mkdir_cleanup_on_error,$@)
	@IMAGE_NAME="$(shell whoami)/averloc--preprocess-dataset-tokens:$(shell git rev-parse HEAD)"
	@for transform in ${ALL_TRANSFORMS}; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/sri/py150/$${transform}:/mnt/inputs" \
			-v "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/sri/py150/$${transform}:/mnt/outputs" \
			"$${IMAGE_NAME}"
	done
	@$(call echo_debug,"  + Finalizing (using 'tokens' representation) complete!")

ETT_DEPS := datasets/transformed/preprocessed/tokens/c2s/java-small
ETT_DEPS += datasets/transformed/preprocessed/tokens/csn/java
ETT_DEPS += datasets/transformed/preprocessed/tokens/csn/python
ETT_DEPS += datasets/transformed/preprocessed/tokens/sri/py150

.PHONY: extract-transformed-tokens
extract-transformed-tokens: build-image-preprocess-dataset-tokens | $(ETT_DEPS) ## (DS-6) Extract preprocessed representations (tokens) from our transfromed (normalized) datasets 
	@$(call echo_info,"Tokens preprocessed representations extracted (for transformed datasets)!")


#######################################################################################################################
#######################################################################################################################

.PHONY: generate-baselines-c2s-java-small
generate-baselines-c2s-java-small: build-image-generate-baselines ## Generate baselines (projected test sets) for our various transforms (c2s/java-small) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--generate-baselines:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Generating transforms.*/baseline.jsonl.gz files...")
	@for transform in All InsertPrintStatements RenameFields RenameLocalVariables RenameParameters ReplaceTrueFalse ShuffleLocalVariables ShuffleParameters; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-small/transforms.Identity:/mnt/identity" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-small/transforms.$${transform}:/mnt/inputs" \
			"$${IMAGE_NAME}"
		@$(call echo_debug,"  + transforms.$${transform}/baseline.jsonl.gz generated")
	done
	@$(call echo_debug,"  + Baselines generated!")

.PHONY: generate-baselines-c2s-java-med
generate-baselines-c2s-java-med: build-image-generate-baselines ## Generate baselines (projected test sets) for our various transforms (c2s/java-med) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--generate-baselines:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Generating transforms.*/baseline.jsonl.gz files...")
	@for transform in All InsertPrintStatements RenameFields RenameLocalVariables RenameParameters ReplaceTrueFalse ShuffleLocalVariables ShuffleParameters; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-med/transforms.Identity:/mnt/identity" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-med/transforms.$${transform}:/mnt/inputs" \
			"$${IMAGE_NAME}"
		@$(call echo_debug,"  + transforms.$${transform}/baseline.jsonl.gz generated")
	done
	@$(call echo_debug,"  + Baselines generated!")

.PHONY: generate-baselines-csn-java
generate-baselines-csn-java: build-image-generate-baselines ## Generate baselines (projected test sets) for our various transforms (csn/java) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--generate-baselines:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Generating transforms.*/baseline.jsonl.gz files...")
	@for transform in All InsertPrintStatements RenameFields RenameLocalVariables RenameParameters ReplaceTrueFalse ShuffleLocalVariables ShuffleParameters; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/java/transforms.Identity:/mnt/identity" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/java/transforms.$${transform}:/mnt/inputs" \
			"$${IMAGE_NAME}"
		@$(call echo_debug,"  + transforms.$${transform}/baseline.jsonl.gz generated")
	done
	@$(call echo_debug,"  + Baselines generated!")

.PHONY: generate-baselines-csn-python
generate-baselines-csn-python: build-image-generate-baselines ## Generate baselines (projected test sets) for our various transforms (csn/python) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--generate-baselines:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Generating transforms.*/baseline.jsonl.gz files...")
	@for transform in All InsertPrintStatements RenameFields RenameLocalVariables RenameParameters ReplaceTrueFalse ShuffleLocalVariables ShuffleParameters; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/python/transforms.Identity:/mnt/identity" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/csn/python/transforms.$${transform}:/mnt/inputs" \
			"$${IMAGE_NAME}"
		@$(call echo_debug,"  + transforms.$${transform}/baseline.jsonl.gz generated")
	done
	@$(call echo_debug,"  + Baselines generated!")

.PHONY: generate-baselines-sri-py150
generate-baselines-sri-py150: build-image-generate-baselines ## Generate baselines (projected test sets) for our various transforms (sri/py150) <!PRIVATE>
	@IMAGE_NAME="$(shell whoami)/averloc--generate-baselines:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Generating transforms.*/baseline.jsonl.gz files...")
	@for transform in All InsertPrintStatements RenameFields RenameLocalVariables RenameParameters ReplaceTrueFalse ShuffleLocalVariables ShuffleParameters; do 
		docker run -it --rm \
			-v "${ROOT_DIR}/datasets/transformed/normalized/sri/py150/transforms.Identity:/mnt/identity" \
			-v "${ROOT_DIR}/datasets/transformed/normalized/sri/py150/transforms.$${transform}:/mnt/inputs" \
			"$${IMAGE_NAME}"
		@$(call echo_debug,"  + transforms.$${transform}/baseline.jsonl.gz generated")
	done
	@$(call echo_debug,"  + Baselines generated!")

#######################################################################################################################
#######################################################################################################################

.PHONY: check-dataset-name
check-dataset-name:
ifndef DATASET_NAME
	$(error DATASET_NAME is a required parameter for this target.)
endif

.PHONY: check-models-out
check-models-out:
ifndef MODELS_OUT
	$(error MODELS_OUT is a required parameter for this target.)
endif

.PHONY: check-results-out
check-results-out:
ifndef RESULTS_OUT
	$(error RESULTS_OUT is a required parameter for this target.)
endif

.PHONY: check-models-in
check-models-in:
ifndef MODELS_IN
	$(error MODELS_IN is a required parameter for this target.)
endif

.PHONY: check-gpu
check-gpu:
ifndef GPU
	$(error GPU is a required parameter for this target.)
endif

.PHONY: check-checkpoint
check-checkpoint:
ifndef CHECKPOINT
	$(error CHECKPOINT is a required parameter for this target.)
endif

.PHONY: depth-k-test-seq2seq
depth-k-test-seq2seq: check-checkpoint check-dataset-name check-gpu check-models-in build-image-depth-k-test-seq2seq
	@IMAGE_NAME="$(shell whoami)/averloc--depth-k-test-seq2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		-v "${ROOT_DIR}/$${RESULTS_OUT}:/mnt/outputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: test-model-code2seq
test-model-code2seq: check-dataset-name check-results-out check-gpu check-models-in build-image-test-model-code2seq ## (TEST) Tests the code2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--test-model-code2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		-v "${ROOT_DIR}/$${RESULTS_OUT}:/mnt/outputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: train-model-code2seq
train-model-code2seq: check-dataset-name check-gpu check-models-out build-image-train-model-code2seq ## (TRAIN) Trains the code2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--train-model-code2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_OUT}/normal:/mnt/outputs" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: adv-train-model-code2seq
adv-train-model-code2seq: check-dataset-name check-gpu check-models-out build-image-train-model-code2seq  ## (TRAIN) Trains the code2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--train-model-code2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_OUT}/adversarial:/mnt/outputs" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: aug-train-model-code2seq
aug-train-model-code2seq: check-dataset-name check-gpu check-models-out build-image-train-model-code2seq  ## (TRAIN) Trains the code2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--train-model-code2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_OUT}/augmented:/mnt/outputs" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: test-model-seq2seq
test-model-seq2seq: check-dataset-name check-results-out check-gpu check-models-in build-image-test-model-seq2seq ## (TEST) Tests the seq2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--test-model-seq2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-e CHECKPOINT="$${CHECKPOINT}" \
		-e SRC_FIELD="$${SRC_FIELD}" \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		-v "${ROOT_DIR}/$${RESULTS_OUT}:/mnt/outputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: train-model-seq2seq
train-model-seq2seq: check-dataset-name check-gpu check-models-out build-image-train-model-seq2seq  ## (TRAIN) Trains the seq2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--train-model-seq2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_OUT}/normal:/mnt/outputs" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: adv-train-model-seq2seq
adv-train-model-seq2seq: check-dataset-name check-gpu check-models-out build-image-train-model-seq2seq  ## (TRAIN) Trains the seq2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--train-model-seq2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_OUT}/adversarial:/mnt/outputs" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		"$${IMAGE_NAME}" $${ARGS}

.PHONY: aug-train-model-seq2seq
aug-train-model-seq2seq: check-dataset-name check-gpu check-models-out build-image-train-model-seq2seq  ## (TRAIN) Trains the seq2seq model on a selected dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--train-model-seq2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_OUT}/augmented:/mnt/outputs" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs" \
		"$${IMAGE_NAME}" $${ARGS}

#######################################################################################################################
#######################################################################################################################

.PHONY: check-transforms
check-transforms:
ifndef TRANSFORMS
	$(error TRANSFORMS (regex) is a required parameter for this target.)
endif

.PHONY: check-short-name
check-short-name:
ifndef SHORT_NAME
	$(error SHORT_NAME is a required parameter for this target.)
endif

.PHONY: check-dataset
check-dataset: ## Ensures DATASET="<blah>" parameter is present. <!PRIVATE>
ifndef DATASET
	$(error DATASET is a required parameter for this target.)
endif

.PHONY: extract-adv-dataset-ast-paths
extract-adv-dataset-ast-paths: | check-dataset check-short-name check-transforms check-gpu check-models-in build-image-extract-adv-dataset-c2s
	@IMAGE_NAME="$(shell whoami)/averloc--extract-adv-dataset-c2s:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-e AVERLOC_JUST_TEST="$${AVERLOC_JUST_TEST}" \
		-e NO_GRADIENT="$${NO_GRADIENT}" \
		-e NO_RANDOM="$${NO_RANDOM}" \
		-e NO_TEST="$${NO_TEST}" \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/tasks/extract-adv-dataset-c2s:/app" \
		-v "${ROOT_DIR}/datasets/transformed/preprocessed/ast-paths/$${DATASET}:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/adversarial/$${SHORT_NAME}/ast-paths/$${DATASET}:/mnt/outputs" \
		"$${IMAGE_NAME}" $$(find "${ROOT_DIR}/datasets/transformed/preprocessed/ast-paths/$${DATASET}" -type d | grep -Po "$${TRANSFORMS}")

#######################################################################################################################
#######################################################################################################################

.PHONY: extract-adv-dataset-tokens
extract-adv-dataset-tokens: | check-dataset check-checkpoint check-short-name check-transforms check-gpu check-models-in build-image-extract-adv-dataset-tokens
	@IMAGE_NAME="$(shell whoami)/averloc--extract-adv-dataset-tokens:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-e N_ALT_ITERS=$${N_ALT_ITERS} \
		-e Z_OPTIM=$${Z_OPTIM} \
		-e U_OPTIM=$${U_OPTIM} \
		-e Z_INIT=$${Z_INIT} \
		-e U_PGD_EPOCHS=$${U_PGD_EPOCHS} \
		-e U_ACCUMULATE_BEST_REPLACEMENTS=$${U_ACCUMULATE_BEST_REPLACEMENTS} \
		-e USE_LOSS_SMOOTHING=$${USE_LOSS_SMOOTHING} \
		-e U_RAND_UPDATE_PGD=$${U_RAND_UPDATE_PGD} \
		-e Z_EPSILON=$${Z_EPSILON} \
		-e ATTACK_VERSION=$${ATTACK_VERSION} \
		-e U_LEARNING_RATE=$${U_LEARNING_RATE} \
		-e Z_LEARNING_RATE=$${Z_LEARNING_RATE} \
		-e SMOOTHING_PARAM=$${SMOOTHING_PARAM} \
		-e VOCAB_TO_USE=$${VOCAB_TO_USE} \
		-e EXACT_MATCHES=$${EXACT_MATCHES} \
		-e AVERLOC_JUST_TEST="$${AVERLOC_JUST_TEST}" \
		-e CHECKPOINT="$${CHECKPOINT}" \
		-e NO_GRADIENT="$${NO_GRADIENT}" \
		-e NO_RANDOM="$${NO_RANDOM}" \
		-e NO_TEST="$${NO_TEST}" \
		-e NUM_REPLACEMENTS=$${NUM_REPLACEMENTS} \
		-e SHORT_NAME=$${SHORT_NAME} \
		-e CURRENT_ATTACK_EPOCH=$${CURRENT_ATTACK_EPOCH} \
 		-v "${ROOT_DIR}/final-results:/final-results" \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/$${DATASET}:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/adversarial/$${SHORT_NAME}/tokens/$${DATASET}:/mnt/outputs" \
		"$${IMAGE_NAME}" $$(find "${ROOT_DIR}/datasets/transformed/preprocessed/tokens/$${DATASET}" -type d | grep -Po "$${TRANSFORMS}")

#######################################################################################################################
#######################################################################################################################

.PHONY: apply-transforms-debug-java
apply-transforms-debug-java: build-image-spoon-apply-transforms ## (DS-4) Apply our suite of transforms to our Java debugging dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--spoon-apply-transforms:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-e DEPTH="$${DEPTH}" \
		-e NUM_SAMPLES="$${NUM_SAMPLES}" \
		-e AVERLOC_JUST_TEST="$${AVERLOC_JUST_TEST}" \
		-v "${ROOT_DIR}/datasets/normalized/debug/java:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/transformed/normalized/debug/java:/mnt/outputs" \
	  -v "${ROOT_DIR}/vendor/CodeSearchNet/function_parser/function_parser:/src/function-parser/function_parser" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/jars/spoon.jar:/app/spoon.jar" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/Transforms.java:/app/Transforms.java" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/transforms:/app/transforms" \
		"$${IMAGE_NAME}"

.PHONY: apply-transforms-c2s-java-small
apply-transforms-c2s-java-small: build-image-spoon-apply-transforms ## (DS-4) Apply our suite of transforms to code2seq's java-small dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--spoon-apply-transforms:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-e DEPTH="$${DEPTH}" \
		-e NUM_SAMPLES="$${NUM_SAMPLES}" \
		-e AVERLOC_JUST_TEST="$${AVERLOC_JUST_TEST}" \
		-v "${ROOT_DIR}/datasets/normalized/c2s/java-small:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/transformed/normalized/c2s/java-small:/mnt/outputs" \
	  -v "${ROOT_DIR}/vendor/CodeSearchNet/function_parser/function_parser:/src/function-parser/function_parser" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/jars/spoon.jar:/app/spoon.jar" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/Transforms.java:/app/Transforms.java" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/transforms:/app/transforms" \
		"$${IMAGE_NAME}"

.PHONY: apply-transforms-csn-java
apply-transforms-csn-java: build-image-spoon-apply-transforms ## (DS-4) Apply our suite of transforms to CodeSearchNet's java dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--spoon-apply-transforms:$(shell git rev-parse HEAD)"
	docker run -it --rm \
		-e DEPTH="$${DEPTH}" \
		-e NUM_SAMPLES="$${NUM_SAMPLES}" \
		-e AVERLOC_JUST_TEST="$${AVERLOC_JUST_TEST}" \
		-v "${ROOT_DIR}/datasets/normalized/csn/java:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/transformed/normalized/csn/java:/mnt/outputs" \
	  -v "${ROOT_DIR}/vendor/CodeSearchNet/function_parser/function_parser:/src/function-parser/function_parser" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/jars/spoon.jar:/app/spoon.jar" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/Transforms.java:/app/Transforms.java" \
		-v "${ROOT_DIR}/tasks/spoon-apply-transforms/transforms:/app/transforms" \
		"$${IMAGE_NAME}"

.PHONY: apply-transforms-csn-python
apply-transforms-csn-python: build-image-astor-apply-transforms ## (DS-4) Apply our suite of transforms to CodeSearchNet's python dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--astor-apply-transforms:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Testing astor on normalized csn/python files...")
	docker run -it --rm \
		-e DEPTH="$${DEPTH}" \
		-e NUM_SAMPLES="$${NUM_SAMPLES}" \
		-e AVERLOC_JUST_TEST="$${AVERLOC_JUST_TEST}" \
		-v "${ROOT_DIR}/datasets/normalized/csn/python:/mnt/inputs" \
		-v "${ROOT_DIR}/tasks/astor-apply-transforms:/app" \
		-v "${ROOT_DIR}/datasets/transformed/normalized/csn/python:/mnt/outputs" \
		"$${IMAGE_NAME}"
	
.PHONY: apply-transforms-sri-py150
apply-transforms-sri-py150: build-image-astor-apply-transforms ## (DS-4) Apply our suite of transforms to SRI Lab's py150k dataset.
	@IMAGE_NAME="$(shell whoami)/averloc--astor-apply-transforms:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Testing astor on normalized sri/py150 files...")
	docker run -it --rm \
		-e DEPTH="$${DEPTH}" \
		-e NUM_SAMPLES="$${NUM_SAMPLES}" \
		-e AVERLOC_JUST_TEST="$${AVERLOC_JUST_TEST}" \
		-v "${ROOT_DIR}/datasets/normalized/sri/py150:/mnt/inputs" \
		-v "${ROOT_DIR}/tasks/astor-apply-transforms:/app" \
		-v "${ROOT_DIR}/datasets/transformed/normalized/sri/py150:/mnt/outputs" \
		"$${IMAGE_NAME}"

#######################################################################################################################
#######################################################################################################################

.PHONY: apply-targeting-seq2seq-c2s-java-small
apply-targeting-seq2seq-c2s-java-small: check-gpu check-models-in build-image-apply-targeting-seq2seq ## (TARG-1) Apply semi-targeting to c2s/java-small (tokens) dataset
	@IMAGE_NAME="$(shell whoami)/averloc--apply-targeting-seq2seq:$(shell git rev-parse HEAD)"
	@$(call echo_debug,"Applying semi-targeting to targeted/c2s/java-small...")
	docker run -it --rm \
		-v "${ROOT_DIR}/tasks/apply-targeting-seq2seq:/app" \
		-v "${ROOT_DIR}/models/pytorch-seq2seq:/model" \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/datasets/adversarial/targeting/tokens/c2s/java-small:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/adversarial/targeted/tokens/c2s/java-small:/mnt/outputs" \
		"$${IMAGE_NAME}"


#######################################################################################################################
#######################################################################################################################

.PHONY: do-integrated-gradients-seq2seq
do-integrated-gradients-seq2seq: check-dataset-name check-results-out check-gpu check-models-in build-image-integrated-gradients-seq2seq ## (IG) Do IG for our seq2seq model
	@IMAGE_NAME="$(shell whoami)/averloc--integrated-gradients-seq2seq:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		--gpus "device=$${GPU}" \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/$${DATASET_NAME}:/mnt/inputs.tsv" \
		-v "${ROOT_DIR}/$${RESULTS_OUT}:/mnt/outputs" \
		"$${IMAGE_NAME}" $${ARGS}


#######################################################################################################################
#######################################################################################################################

.PHONY: aug-dataset-tokens-c2s-java-small
aug-dataset-tokens-c2s-java-small: build-image-augment-dataset-tokens ## (AUG) Do dataset augmentation for c2s/java-small
	@IMAGE_NAME="$(shell whoami)/averloc--augment-dataset-tokens:$(shell git rev-parse HEAD)"
	DOCKER_API_VERSION=1.40 docker run -it --rm \
		-v "${ROOT_DIR}/$${MODELS_IN}:/models" \
		-v "${ROOT_DIR}/datasets/adversarial/all-attacks/tokens/c2s/java-small:/mnt/inputs" \
		-v "${ROOT_DIR}/datasets/augmented/all-attacks/tokens/c2s/java-small:/mnt/outputs" \
		"$${IMAGE_NAME}"
