#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd "${DIR}/../vendor/code2seq/JavaExtractor/JPredict" > /dev/null

docker run -it --rm -v `pwd`:/mnt -w /mnt --entrypoint mvn maven:3.6.3-jdk-8 package 

popd > /dev/null
