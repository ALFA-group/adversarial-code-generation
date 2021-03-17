import tqdm
import random


if __name__ == "__main__":
  print("Loading inputs...")

  out_file = open('/mnt/outputs/train.tsv', 'w')
  out_file.write("src\ttgt\n")
  first = True
  for line in tqdm.tqdm(open('/mnt/inputs/train.tsv').readlines()):
    if first:
      first = False
      continue
    parts = line.split('\t')
    label = parts[1]
    selection = random.randint(2, 9)
    out_file.write("{}\t{}\n".format(
      parts[selection].strip(), label.strip()
    ))
