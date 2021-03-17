#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd "${DIR}/../vendor/spoon" > /dev/null

docker run -it --rm -v `pwd`:/mnt -w /mnt --entrypoint mvn maven:3.5.4-jdk-9 -Dmaven.repo.local=/mnt/maven-cache -DskipTests=true -Dmaven.javadoc.skip=true package 

yes | cp "${DIR}/../vendor/spoon/target/spoon-core-8.2.0-SNAPSHOT-jar-with-dependencies.jar" "${DIR}/../tasks/spoon-apply-transforms/jars/spoon.jar" 

popd > /dev/null
