from argparse import ArgumentParser
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True

from config import Config
from model import Model

import os
import re
import sys
import json

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", dest="test_path", help="path to preprocessed dataset", required=True)
    parser.add_argument("--load", dest="load_path", help="path to saved file", metavar="FILE", required=True)
    parser.add_argument("--batch_size", type=int, help="size of batch in training", required=False, default=32)
    parser.add_argument('--num_replace_tokens', default=20, type=int)
    parser.add_argument('--random', default=False, action='store_true')
    parser.add_argument('--output_json_path', required=True)
    args = parser.parse_args()

    args.data_path = None
    args.save_path_prefix = None
    args.release = None

    config = Config.get_default_config(args)

    assert args.output_json_path[-5:]==".json", "output_json_path doesn't appear to be a json"

    model = Model(config)
    print('Created model')

    replace_tokens = ["@R_%d@"%i for i in range(1, args.num_replace_tokens+1)]

    print('Running gradient %sattack'%('and random ' if args.random else ''))

    gradient_replacements, random_replacements = model.run_gradient_attack(replace_tokens, batch_size=args.batch_size, random=args.random)

    json.dump(gradient_replacements, open(args.output_json_path, 'w'), indent=4)
    print('Saved:', args.output_json_path)

    if args.random:
        output_json_path = args.output_json_path[:-5] + "-random.json"
        json.dump(random_replacements, open(output_json_path, 'w'), indent=4)
        print('Saved:', output_json_path)

    model.close_session()
