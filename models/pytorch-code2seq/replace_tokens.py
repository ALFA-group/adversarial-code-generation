import json
import pandas as pd 
import argparse
import tqdm
import re
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data_path', required=True)
    parser.add_argument('--dest_data_path', required=True)
    parser.add_argument('--mapping_json', required=True)
    parser.add_argument('--transform_name', required=True)
    opt = parser.parse_args()
    return opt

opt = parse_args()

mapping = json.load(open(opt.mapping_json))
mapping = mapping[opt.transform_name]

with open(opt.source_data_path, 'r') as in_f:
    with open(opt.dest_data_path, 'w') as dst_f:
        for line in tqdm.tqdm(in_f):
            fname = line.split(' ')[0]
            new_line = line
            if fname in mapping:
                for repl_tok in mapping[fname]:
                    new_line = new_line.replace(repl_tok, '|'.join(mapping[fname][repl_tok].split(' ')))
            dst_f.write(new_line)
