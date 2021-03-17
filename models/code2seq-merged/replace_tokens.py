import json
import pandas as pd 
import argparse
import tqdm
import re

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source_data_path', required=True)
	parser.add_argument('--dest_data_path', required=True)
	parser.add_argument('--mapping_json', required=True)
	opt = parser.parse_args()
	return opt

opt = parse_args()

mapping = json.load(open(opt.mapping_json))

data = pd.read_csv(opt.source_data_path, sep='\t', index_col=0)

with open(opt.source_data_path, 'r') as in_f:
	with open(opt.dest_data_path, 'w') as dst_f:
		for line in tqdm.tqdm(in_f):
			index = line.split(' ')[0]
			new_line = re.sub(r'^\d+ ', '', line)
			if index in mapping:
				for repl_tok in mapping[index]:
					new_line = new_line.replace(repl_tok, mapping[index][repl_tok])
			dst_f.write(new_line)
