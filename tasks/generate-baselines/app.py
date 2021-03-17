import os
import sys
import gzip
import json


if __name__ == "__main__":
  KEEP_SET = set()
  for from_file in sys.stdin:
    KEEP_SET.add(from_file.strip().lower())
  
  for line in gzip.open('/mnt/identity/test.jsonl.gz'):
    as_json = json.loads(line)
    if as_json['from_file'].strip().lower() in KEEP_SET:
      print(json.dumps(as_json))
