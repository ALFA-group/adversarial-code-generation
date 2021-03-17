import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

def loss_plot(batch_loss_list_per_iter, outfile, extension='.png'):
	# plt.clf()
	# plt.plot(batch_loss_list_per_iter)
	# plt.savefig(outfile+extension)
	with open(outfile+".pkl", 'wb') as fp:
		pkl.dump(batch_loss_list_per_iter, fp)

def loss_multiple_plots(dir, extension="pkl"):
	get_uq_names = set()

	for f in os.listdir(dir):
		if f.endswith(extension):
			get_uq_names.add('_'.join(f.split('_')[:1]))
	
	for s in get_uq_names:
		plt.clf()
		legend_names, fnames = [], []
		for f in os.listdir(dir):
			if f.endswith(extension):
				if '_'.join(f.split('_')[:1]) == s:
					fnames.append(f)
		fnames = sorted(fnames)
		for f in fnames:
			with open(os.path.join(dir, f), 'rb') as fp:
				lst = pkl.load(fp)
				if 'rndinit-0' in f:
					plt.plot(lst, marker='x')
				elif 'rndinit-1' in f:
					plt.plot(lst, marker='+')
				elif 'rndinit-2' in f:
					plt.plot(lst, marker='o')
				else:
					plt.plot(lst)
				strname = f.split(extension)[0]
				split_name = strname.split('_')
				lgnd_name = '_'.join(split_name[3:]+split_name[1:3])
				legend_names.append(lgnd_name)
	
		plt.legend(legend_names, bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0)
		# plt.tight_layout(rect=[0,0,0.75,1])
		plt.savefig(os.path.join(dir, 'results_{}.jpg'.format(s)), bbox_inches="tight")

		plt.clf()
		per_alt_iter_loss = []
		for f in fnames:
			with open(os.path.join(dir, f), 'rb') as fp:
				lst = pkl.load(fp)
				per_alt_iter_loss.append(lst[-1])
		# plt.legend(legend_names, bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0)
		# plt.tight_layout(rect=[0,0,0.75,1])
		plt.plot(per_alt_iter_loss)
		plt.savefig(os.path.join(dir, 'results_best_{}.jpg'.format(s)), bbox_inches="tight")

if __name__ == "__main__":
	loss_multiple_plots(sys.argv[1])