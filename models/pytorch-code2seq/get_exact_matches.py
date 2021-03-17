import os
import json
import torch
import pickle
import numpy as np
import argparse

from gradient_attack_utils import get_exact_matches
from dataset import Vocabulary, create_dataloader
from model import Code2Seq
from pytorch_lightning import Trainer, seed_everything
from data_preprocessing.preprocess_code2seq_data import preprocess

SEED=7

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_data_path', action='store', dest='orig_data_path', help='Path to original data')
    parser.add_argument('--data_path', action='store', dest='data_path', help='Path to transformed dataset')
    parser.add_argument('--checkpoint', action='store', dest='checkpoint', help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, action='store', dest='batch_size')
    parser.add_argument('--split', action='store', dest='split')
    parser.add_argument('--vocab_path', action='store')

    args = parser.parse_args()
    return args

def create_datafile(data_path, exact_matches, split, fname_to_idx, skip, indexed=True):

    new_data_path = os.path.join(data_path, 'small.{}.c2s'.format(split))
    if indexed:
        lines = open(os.path.join(data_path, 'indexed.{}.c2s'.format(split)), 'r')
    else:
        lines = open(os.path.join(data_path, 'data.{}.c2s'.format(split)), 'r')
    new_file = open(new_data_path, 'w')
    seen = set()
    count = 0
    all_lines = {}
    for line in lines:
        if indexed:
            index = line.split()[0]
        else:
            split_line = line.split()
            fname, method_name = split_line[0], split_line[1]
            index = fname_to_idx[fname]
            if fname in skip:
                index = None

        if index != None and index in exact_matches:
            new_file.write(line)
            count +=1

    print("wrote {} lines".format(count))
    print("Saved exact matches.")

def add_indices(data_path, split):
    new_data_path = os.path.join(data_path, 'indexed.{}.c2s'.format(split))
    lines = open(os.path.join(data_path, 'data.{}.c2s'.format(split)), 'r')
    new_file = open(new_data_path, 'w')
    fname_to_idx = {}
    count = 0
    seen = {}
    skip = set()
    for line in lines:
        split_line = line.split()
        fname, method_name = split_line[0], split_line[1]
        new_line = line.replace(fname, str(count))
        new_file.write(new_line)
        # p = (fname, method_name)
        p = fname
        if p in fname_to_idx:
            skip.add(p)
        else:
            fname_to_idx[p] = str(count)
        count += 1

    print(count, len(fname_to_idx))
    print('saved indexed identity file')
    return fname_to_idx, skip

if __name__ == '__main__':

    args = parse_args()
    seed_everything(SEED)
    model = Code2Seq.load_from_checkpoint(checkpoint_path=args.checkpoint)

    fname_to_idx, skip = add_indices(args.orig_data_path, args.split)
    preprocess("code2seq", "data", False, 1, args.orig_data_path, True, "indexed", 1000)

    data_loader, n_samples = create_dataloader(
        os.path.join(args.orig_data_path, args.split), model.hyperparams.max_context, False, False, args.batch_size, 1
    )
    vocab = pickle.load(open(args.vocab_path, 'rb'))
    label_to_id = vocab['label_to_id']
    id_to_label = {label_to_id[l]:l for l in label_to_id}

    li_exact_matches = get_exact_matches(data_loader, n_samples, model, id_to_label)
    print(len(li_exact_matches))
    create_datafile(args.orig_data_path, li_exact_matches, args.split, fname_to_idx, skip)
    create_datafile(args.data_path, li_exact_matches, args.split, fname_to_idx, skip, indexed=False)




