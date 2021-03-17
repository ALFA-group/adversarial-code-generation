import os
import sys
import math
import tqdm
import random
import subprocess


if __name__ == '__main__':
  ID_MAP = {}
  TRANSFORMS = [ x.strip() for x in sys.argv[2:] if x.strip().lower() != 'transforms.identity' ]
  print(TRANSFORMS)

  print("Loading identity transform...")
  with open("/mnt/inputs/transforms.Identity/data.{}.c2s".format(sys.argv[1]), 'r') as identity:
    for line in identity:
      the_hash = line.split()[0]
      the_rest = line.replace(the_hash + " ", "")
      ID_MAP[the_hash] = the_rest
  print("  + Loaded {} samples".format(len(ID_MAP)))

  print("Loading transformed samples...")
  TRANSFORMED = {}
  for transform_name in TRANSFORMS:
    TRANSFORMED[transform_name] = {}
    with open("/mnt/inputs/{}/data.{}.c2s".format(transform_name, sys.argv[1]), 'r') as current:
      for line in current:
        the_hash = line.split()[0]
        the_rest = line.replace(the_hash + " ", "")
        if the_hash not in TRANSFORMED:
          TRANSFORMED[transform_name][the_hash] = the_rest
    print("  + Loaded {} samples from '{}'".format(
      len(TRANSFORMED[transform_name]), transform_name
    ))

  OUT_MAPS = {
    'transforms.Identity': open('/mnt/staging/data0.{}.c2s'.format(sys.argv[1]), 'w')
  }

  for i, transform_name in enumerate(TRANSFORMED.keys()):
    OUT_MAPS[transform_name] = open(
      '/mnt/staging/data{}.{}.c2s'.format(i+1, sys.argv[1]), 'w'
    )
    print('{} is in: data{}.{}.c2s'.format(transform_name, i+1, sys.argv[1]))

  index = 0
  for key in tqdm.tqdm(ID_MAP.keys(), desc="  + Progress"):
    for transform_name in TRANSFORMS:
      if key in TRANSFORMED[transform_name]:
        OUT_MAPS[transform_name].write(str(index) + ' ' + TRANSFORMED[transform_name][key])
      else:
        OUT_MAPS[transform_name].write(str(index) + ' ' + ID_MAP[key])
    OUT_MAPS['transforms.Identity'].write(ID_MAP[key])
    index += 1
  print("  + Adversarial {}ing file generation complete!".format(sys.argv[1]))
