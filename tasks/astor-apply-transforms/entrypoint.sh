#!/bin/bash

trap "echo Exited!; exit 1;" SIGINT SIGTERM

python3 /app/app.py

echo "Starting normalizer:"
find /mnt/raw-outputs/ -mindepth 2 -maxdepth 2 -type d | \
    python3 /src/function-parser/function_parser/parser_cli.py python raw 
echo "  + Done!"
