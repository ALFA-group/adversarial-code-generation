import tqdm
import random


if __name__ == "__main__":
  print("Creating augmented dataset...")

  out_file = open('/mnt/outputs/train.tsv', 'w')
  out_file.write("src\ttgt\trandomAttack\n")
  first = True
  for line in tqdm.tqdm(open('/mnt/inputs/train.tsv').readlines()):
    if first:
      first = False
      continue
    parts = line.split('\t')
    label = parts[1]
    selection = random.randint(2, len(parts) - 1) 
    out_file.write("{}\t{}\t{}\n".format(
      parts[0].strip(), label.strip(), parts[selection].strip()
    )) 
