import os
import re
import sys
import random


MAX_REPLACE = 5
REPLACE_REGEX = re.compile(r'replace? me [a-z]+ \d+')


def random_rename(WORDBANK, min_len, max_len):
  return ' '.join(random.sample(WORDBANK, random.randint(min_len, max_len)))


if __name__ == '__main__':
  WORDBANK = []
  with open('/app/wordbank.txt', 'r') as wb_f:
    WORDBANK = [ x.strip() for x in wb_f.readlines() ]

  first_line = True
  for line in sys.stdin:
    if first_line:
      print(line.strip())
      first_line = False
      continue

    parts = [ x.strip() for x in line.split('\t') ]
    new_parts = []
    for part in parts:
      REPLACEMENTS = {}
      index = 1
      for replace_me in re.findall(REPLACE_REGEX, part):
        if replace_me not in REPLACEMENTS:
          if index > MAX_REPLACE:
            REPLACEMENTS[replace_me] = random_rename(WORDBANK, 3, 6)
            continue
          
          REPLACEMENTS[replace_me] = "@R_" + str(index) + "@"
          index += 1
      
      new_part = part
      for key in REPLACEMENTS.keys():
        new_part = new_part.replace(key, REPLACEMENTS[key])
      new_parts.append(new_part)

    print('\t'.join(new_parts))
    

