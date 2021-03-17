import os
import re
import sys

if __name__ == '__main__':
  TARGETS = []
  with open(sys.argv[1], 'r') as tgts_f:
    TARGETS = [ [ y.strip() for y in x.strip().split('\t') ] for x in tgts_f.readlines() ]

  first_line = True
  for j, line in enumerate(sys.stdin):
    if first_line:
      print(line.strip())
      first_line = False
      continue
    
    parts = [ x.strip() for x in line.split('\t') ]
    new_parts = []
    for i, part in enumerate(parts):
      if i <= 1:
        new_parts.append(part)
        continue
      targets = TARGETS[j - 1][i - 1]
      new_part = part
      for k, trg in enumerate(targets.split(' ')):
        new_part = new_part.replace('@R_{}@'.format(k+1), trg)
      new_parts.append(new_part)
    
    print('\t'.join(new_parts))
