#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

OUTPUT_DIR="results/01-25-2020"

SETS="c2s/java-small csn/java csn/python sri/py150"

MODELS="adversarial-one-step adversarial-all normal"

TESTS_SOURCE="datasets/transformed/preprocessed/tokens"

trap "echo 'CTRL-C Pressed. Quiting...'; exit;" SIGINT SIGTERM

for the_set in ${SETS}; do
  for the_model in ${MODELS}; do
    for the_test in $(find ${TESTS_SOURCE}/${the_set} -mindepth 1 -maxdepth 1 -type d | sort); do
      results_dir="${OUTPUT_DIR}/${the_set}/${the_model}/$(basename ${the_test})"
    
      if [ "$(basename ${the_test})" != "transforms.Identity" ]; then 

        f1=$(cat "${results_dir}/results-test_stats.txt" | grep -Po 'f1: \d+\.\d+' | awk '{ print $2}')
        f1_b=$(cat "${results_dir}/results-baseline_stats.txt" | grep -Po 'f1: \d+\.\d+' | awk '{ print $2}')

        echo "${the_set},${the_model},$(basename ${the_test}),$(awk "BEGIN{print ${f1}-${f1_b}}")"

      else

        f1=$(cat "${results_dir}/results-test_stats.txt" | grep -Po 'f1: \d+\.\d+' | awk '{ print $2}')
        echo "${the_set},${the_model},$(basename ${the_test}),$(awk "BEGIN{print ${f1}}")"

      fi

    done
  done
done
