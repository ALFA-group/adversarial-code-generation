import os
import sys
import json
import tqdm


if __name__ == '__main__':
  all_lines = sys.stdin.readlines()
  for line in tqdm.tqdm(all_lines, file=sys.stderr, desc="  + Processing"):
    try:
      with open(line.strip(), 'r') as f_handle:
        json_obj = {
          "granularity": "file",
          "language": "java",
          "code": f_handle.read().strip()
        }
        print(json.dumps(json_obj))
    except Exception as ex:
      print("  - Failed to load sample, reason: {}".format(ex), file=sys.stderr)
      print("    - This sample will be excluded. Continuing...", file=sys.stderr)
      continue
