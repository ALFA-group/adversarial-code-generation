#!/bin/bash

trap "echo Exited!; exit 1;" SIGINT SIGTERM

/jdk-9.0.4+11/bin/javac -cp /app/spoon.jar:/app/gson.jar:/app transforms/*.java && \
  /jdk-9.0.4+11/bin/javac -cp /app/spoon.jar:/app/gson.jar:/app Transforms.java

/jdk-9.0.4+11/bin/java -XX:-UsePerfData -Xmx128g -d64 -cp /app/spoon.jar:/app/log4j-api.jar:/app/gson.jar:/app Transforms

echo "Starting normalizer:"
find /mnt/raw-outputs/ -mindepth 2 -maxdepth 2 -type d | \
    python3 /src/function-parser/function_parser/parser_cli.py java raw 
echo "  + Done!"
