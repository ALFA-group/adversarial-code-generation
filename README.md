# Generating Adversarial Computer Programs using Optimized Obfuscations

Code repository for the paper Generating Adversarial Computer Programs using Optimized Obfuscations, published at ICLR 2021.

Link to paper - https://openreview.net/forum?id=PH5PH9ZO_4 

Citation
```
@inproceedings{
shashank2021generating,
title={Generating Adversarial Computer Programs using Optimized Obfuscations},
author={Shashank Srikant and Sijia Liu and Tamara Mitrovska and Shiyu Chang and Quanfu Fan and Gaoyuan Zhang and Una-May O{'R}eilly},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=PH5PH9ZO_4}
}
```

## Abstract 
Machine learning (ML) models that learn and predict properties of computer programs are increasingly being adopted and deployed. 
These models have demonstrated success in applications such as auto-completing code, summarizing large programs, and detecting bugs and malware in programs. 
In this work, we investigate principled ways to adversarially perturb a computer program to fool such learned models, and thus determine their adversarial robustness. We use program obfuscations, which have conventionally been used to avoid attempts at reverse engineering programs, as adversarial perturbations. These perturbations modify programs in ways that do not alter their functionality but can be crafted to deceive an ML model when making a decision. We provide a general formulation for an adversarial program that allows applying multiple obfuscation transformations to a program in any language. We develop first-order optimization algorithms to  efficiently determine two key aspects -- which parts of the program to transform, and what transformations to use. We show that it is important to optimize both these aspects to generate the best adversarially perturbed program. Due to the discrete nature of this problem, we also propose using randomized smoothing to improve the attack loss landscape to ease optimization. 
We evaluate our work on Python and Java programs on the problem of program summarization. 
We show that our best attack proposal achieves a  improvement over a state-of-the-art attack generation approach for programs trained on a seq2seq model.
We further show that our formulation is better at training models that are robust to adversarial attacks.

## Authors

[Shashank Srikant](https://shashank-srikant.github.io/),  [Sijia Liu](https://lsjxjtu.github.io/), [Tamara Mitrovska](https://superurop.mit.edu/scholars/tamara-mitrovska/), [Shiyu Chang](http://people.csail.mit.edu/chang87/), [Quanfu Fan](https://mitibmwatsonailab.mit.edu/people/quanfu-fan/), [Gaoyuan Zhang](https://researcher.watson.ibm.com/researcher/view.php?person=ibm-Gaoyuan.Zhang), and  [Una-May O’Reilly](https://alfagroup.csail.mit.edu/unamay).

For any queries, please contact Shashank (shash@mit.edu), Sijia (liusiji5@msu.edu), or Una-May (unamay@csail.mit.edu)

## Instructions
Our codebase builds on the well documented codebase released by the authors of `Semantic Robustness of Models of Source Code`

Paper - https://arxiv.org/abs/2002.03043

Github - https://github.com/jjhenkel/averloc

The repository provides a number of Makefile commands to download datasets, transform their ASTs, train code models, and finally attack and evaluate the performance on a test set. For full details on the directory structure and the organization of the files in this repository, see section [Directory Structure](#directory-structure).

Our attack formulation is mainly implemented in the following files -
- Alternate optimization - [`./models/pytorch-seq2seq/gradient_attack.py`](https://github.com/ALFA-group/adversarial-code-generation/blob/master/models/pytorch-seq2seq/gradient_attack.py#L302)
- Joint optimization - [`./models/pytorch-seq2seq/gradient_attack_v3.py`](https://github.com/ALFA-group/adversarial-code-generation/blob/master/models/pytorch-seq2seq/gradient_attack_v3.py#L43)
- Helper functions - [`./models/pytorch-seq2seq/gradient_attack_utils.py`](https://github.com/ALFA-group/adversarial-code-generation/blob/master/models/pytorch-seq2seq/gradient_attack_utils.py)

The following instructions help reproduce the results from our work
- Download and normalize datasets
```
make download-datasets
make normalize-datasets
```
- Create the transformed datasets
```
make apply-transforms-sri-py150
make apply-transforms-c2s-java-small
make extract-transformed-tokens
```
- Run normal training for seq2seq
```
./experiments/normal_seq2seq_train.sh
```
This command will train the model on the Python dataset. To train on Java, replace `sri/py150` with `c2s/java-small` inside the script.
- Create adversarial datasets and evaluate
```
./experiments/run_attack_0.sh
./experiments/run_attack_1.sh
```
These scripts contain commands for various experiments. For each experiment we first create an adversarial dataset by using our attack and then evaluate the trained seq2seq model on the generated dataset. The numbers 0 and 1 in the script names refer to the GPU on which the experiments are run.
- Collect all results into a csv table and generate plots
```
python experiments/collect_results.py 1
```

## Directory Structure

The instructions that follow have been adopted from [Ramakrishnan et al.'s codebase](https://github.com/jjhenkel/averloc).
In this repository, we have the following directories:

### `./datasets`

**Note:** the datasets are all much too large to be included in this GitHub repo. This is simply the
structure as it would exist on disk once our framework is setup.

```bash
./datasets
  + ./raw            # The four datasets in "raw" form
  + ./normalized     # The four datasets in the "normalized" JSON-lines representation 
  + ./preprocess
    + ./tokens       # The four datasets in a representation suitable for token-level models
    + ./ast-paths    # The four datasets in a representation suitable for code2seq
  + ./transformed    # The four datasets transformed via our code-transformation framework 
    + ./normalized   # Transformed datasets normalized back into the JSON-lines representation
    + ./preprocessed # Transformed datasets preprocessed into:
      + ./tokens     # ... a representation suitable for token-level models
      + ./ast-paths  # ... a representation suitable for code2seq
  + ./adversarial    # Datasets in the format < source, target, tranformed-variant #1, #2, ..., #K >
    + ./tokens       # ... in a token-level representation
    + ./ast-paths    # ... in an ast-paths representation
```

## `./models`

We have two Machine Learning on Code models. Both of them are trained on the Code Summarization task. The
seq2seq model has been modified to incorporate our attack formulation, and includes an adversarial training loop.
The branch `pytorch-code2seq` implements our attack formulation on `code2seq`. This is work in progress.

```bash
./models
  + ./pytorch-seq2seq   # seq2seq model implementation
  + ./pytorch-code2seq  # code2seq model implementation, available on branch pytorch-code2seq. This is WIP.
```

## `./results`

This directory stores results that are small-enough to be checked into GitHub. This is automatically generated once the codebase is set up.

## `./scripts`

In this directory there are a large number of scripts for doing various chores related to running and maintaing
this code transformation infrastructure.

## `./tasks`

This directory houses the implementations of various pieces of our core framework:

```bash
./tasks
  + ./astor-apply-transforms
  + ./depth-k-test-seq2seq
  + ./download-c2s-dataset
  + ./download-csn-dataset
  + ./extract-adv-dataset-c2s
  + ./extract-adv-dataset-tokens
  + ./generate-baselines
  + ./integrated-gradients-seq2seq
  + ./normalize-raw-dataset
  + ./preprocess-dataset-c2s
  + ./preprocess-dataset-tokens
  + ./spoon-apply-transforms
  + ./test-model-seq2seq
  + ./train-model-seq2seq
```

## `./vendor`

This directory contains dependencies in the form of git submodukes.

## `Makefile`

We have one overarching `Makefile` that can be used to drive a number of the data generation, training, testing, adn evaluation tasks.

```
download-datasets                  (DS-1) Downloads all prerequisite datasets
normalize-datasets                 (DS-2) Normalizes all downloaded datasets
extract-ast-paths                  (DS-3) Generate preprocessed data in a form usable by code2seq style models. 
extract-tokens                     (DS-3) Generate preprocessed data in a form usable by seq2seq style models. 
apply-transforms-c2s-java-med      (DS-4) Apply our suite of transforms to code2seq's java-med dataset.
apply-transforms-c2s-java-small    (DS-4) Apply our suite of transforms to code2seq's java-small dataset.
apply-transforms-csn-java          (DS-4) Apply our suite of transforms to CodeSearchNet's java dataset.
apply-transforms-csn-python        (DS-4) Apply our suite of transforms to CodeSearchNet's python dataset.
apply-transforms-sri-py150         (DS-4) Apply our suite of transforms to SRI Lab's py150k dataset.
extract-transformed-ast-paths      (DS-6) Extract preprocessed representations (ast-paths) from our transfromed (normalized) datasets 
extract-transformed-tokens         (DS-6) Extract preprocessed representations (tokens) from our transfromed (normalized) datasets 
extract-adv-datasets-tokens        (DS-7) Extract preprocessed adversarial datasets (representations: tokens)
docker-cleanup                     (MISC) Cleans up old and out-of-sync Docker images.
submodules                         (MISC) Ensures that submodules are setup.
help                               (MISC) This help.
test-model-seq2seq                 (TEST) Tests the seq2seq model on a selected dataset.
train-model-seq2seq                (TRAIN) Trains the seq2seq model on a selected dataset.
```
