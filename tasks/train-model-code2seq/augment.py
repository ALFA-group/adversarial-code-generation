import tqdm
import random


if __name__ == "__main__":
  print("Creating augmented dataset...")

  out_file = open('/mnt/outputs/data1.train.c2s', 'w')

  FILES = [
    open('/mnt/inputs/data1.train.c2s', 'r').readlines(),
    open('/mnt/inputs/data2.train.c2s', 'r').readlines(),
    open('/mnt/inputs/data3.train.c2s', 'r').readlines(),
    open('/mnt/inputs/data4.train.c2s', 'r').readlines(),
    open('/mnt/inputs/data5.train.c2s', 'r').readlines(),
    open('/mnt/inputs/data6.train.c2s', 'r').readlines(),
    open('/mnt/inputs/data7.train.c2s', 'r').readlines(),
    open('/mnt/inputs/data8.train.c2s', 'r').readlines()
  ]

  for i, line in tqdm.tqdm(enumerate(open('/mnt/inputs/data0.train.c2s').readlines())):
    selection = random.randint(1, 8)
    out_file.write(FILES[selection-1][i]) 
