import json
import matplotlib.pyplot as plt
import numpy as np

def get_vals(filename):

	with open(filename, 'r') as f:
		data = json.load(f)

	x = []
	data = data['transforms.InsertHoles']
	for idx in data:
		repl_tokens = data[idx]
		for r in repl_tokens:
			if repl_tokens[r] != '':
				line_num = int(r[3:-1])
				val = line_num/(len(repl_tokens)-1)
				x.append(val)
				break
	return x

x = get_vals('../datasets/adversarial/sri/py150/testing-gradient.json')
plt.hist(np.asarray(x, dtype='float'), bins=20)
# plt.savefig('plot.png')
# plt.show()

