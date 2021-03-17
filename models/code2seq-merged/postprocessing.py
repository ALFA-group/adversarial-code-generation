import os
from subprocess import call
import random
from math import ceil

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-b", dest="batch", type=int, help="size of batch", required=False)
parser.add_argument("-sd", dest="src_dir", help="directory containing all the training data", required=False)
parser.add_argument("-dd", dest="d_dir", help="directory where the processed data to store", required=False)
args = parser.parse_args()

batch_size = args.batch
src_dir = args.src_dir
dest_dir = args.d_dir

def get_subdirs(a_dir):
    subdirs = [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    return subdirs

def build_dict(d):
	trans_dict = {}
	with open(d+"/data.train.c2s") as f:
		line = f.readline().strip()
		line_num = 0
		while line:
			key = line.split(" ")[0]
			if not key in trans_dict:
				trans_dict[key] = line_num
			line_num += 1
			line = f.readline().strip()
	return trans_dict
	

#build unique key dict for each subdir#
def build_dict_dirs(des_dir, subdirs, a_dir):
	#num_functions = 0
	trans_dicts = []
	#id_index = subdirs.index("identity")
	for d in subdirs:
		a_dict = build_dict(a_dir+"/"+d)
		#with open(des_dir+"/"+d+".dict", 'w') as f:
		#	for key in a_dict:
		#		f.write(key+":"+str(a_dict[key])+"\n")
		trans_dicts.append(a_dict)
	return trans_dicts


def build_adv_data(src_dir, des_dir, dirs, dicts):
	id_index = dirs.index("identity")
	if not os.path.exists(des_dir):
		call("mkdir "+des_dir, shell=True)	
	id_dict = dicts[id_index]
	id_lines = open(src_dir+"/identity/data.train.c2s").readlines()
	for i in range(len(dicts)):
		#open(des_dir+"/data"+str(i)+".trans.c2s",'w').close()
		subdir = dirs[i]
		a_dict = dicts[i]
		print(subdir+" has "+str(len(a_dict))+" unique hashes")
		lines = open(src_dir+"/"+subdir+"/data.train.c2s").readlines()
		#open(des_dir+"/data"+str(i)+".trans.c2s",'w').close()
		with open(des_dir+"/data"+str(i)+".trans.c2s",'w') as f:
			for key in id_dict:
				if key in a_dict:
				#	print("In orig data")
					line_num = a_dict[key]
					content = " ".join(lines[line_num].split(" ")[1:])
				else:
				#	print("Not in orig data")
					line_num = id_dict[key]
					content = " ".join(id_lines[line_num].split(" ")[1:])
				#print(content)
				f.write(content)

def build_batches(dirs, des_dir, batch_size):
	for i in range(len(dirs)):
		if not os.path.exists(des_dir+"/"+str(i)):
			call("mkdir "+des_dir+"/"+str(i), shell=True)
		batch_id = 0
		with open(des_dir+"/data"+str(i)+".trans.c2s") as f:
			line = f.readline()
			ct = 0
			while line:
				with open(des_dir+"/"+str(i)+"/"+str(batch_id)+".train.c2s", 'a+') as g:
					g.write(line)
					ct += 1
					if ct >= batch_size:
						batch_id += 1
						ct = 0
				line = f.readline()

def build_val_data(src_dir, dirs, des_dir):
    open(des_dir+"/"+"data.val.c2s", 'w').close()
    ratio = 1.0/float(len(dirs))
    with open(des_dir+"/"+"data.val.c2s", 'a') as f:
        for d in dirs:
            lines = open(src_dir+"/"+d+"/data.val.c2s").readlines()
            num_lines = ceil(len(lines) * ratio)
            sampled_lines = random.sample(lines, num_lines)
            for line in sampled_lines:
                f.write(line)

def remove_hash(a_f):
	with open(a_f) as f:
		with open("temp", 'w') as g:
			line = f.readline()
			while line:
				content = " ".join(line.split(" ")[1:])
				g.write(content)
				line = f.readline()
	call("mv -f temp "+af,shell=True)

subdirs = get_subdirs(src_dir)
subdir_dicts = build_dict_dirs(dest_dir, subdirs, src_dir)
build_adv_data(src_dir, dest_dir, subdirs, subdir_dicts)
call("cp "+src_dir+"/identity/data.dict.c2s "+dest_dir, shell=True)
build_val_data(src_dir, subdirs, dest_dir)
remove_hash(des_dir+"/"+"data.val.c2s")
