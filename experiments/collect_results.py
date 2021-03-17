import os
import csv
import json
import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

common = ['Method', 'ver', '\# sites', 'zopt', 'zsites', '\# PGD iters', 'optimal', 'smooth']
ao, jo, uwisc = "\\ao", "\\jo", "\\uwisc"
ao_ltx, jo_ltx, uwisc_ltx = "AO", "JO", "Baseline"

def merge_results_latex_v1(dfs, basepath, fname='results_datasets'):
	result = pd.merge(dfs[0], dfs[1], on=common, how='right')
	results = result.drop(['ver', 'smooth'], axis=1)
	result.to_csv(basepath+fname+".csv", index=False)
	print('Merged CSV saved to', basepath)
	
	latex_text = "\\renewcommand{\\arraystretch}{1.2}\n\
\\begin{table}[]\n\
\\centering\n\
\\resizebox{0.9\\textwidth}{!}{\n\
\\begin{tabular}{l l r c c c c }\n\
\\toprule\n\
\\multicolumn{3}{l}{} & \\multicolumn{2}{c}{\\textbf{Python}} & \\multicolumn{2}{c}{\\textbf{Java}} \\\\ \\cline{4-7}\\cline{4-7}\n \
"
	
	with open(basepath+fname+".csv", 'r') as fp:
		lines = fp.read().split('\n')
	
	for cnt, l  in enumerate(lines):
		if len(l) < 1:
			continue
		if cnt == 0:
			cs = l.split(",")
			replace = ["F1_x", "F1_y", "ASR_x", "ASR_y"]
			for i, c in enumerate(cs):
				if c in replace:
					cs[i] = c.split("_")[0]
			for i, c in enumerate(cs):
				cs[i] = "\\textbf{" + cs[i] + "}"
			latex_text += " & ".join(cs)
			
		else:
			cs = l.split(",")
			if ('\\jo' in cs[0] or '\\ao' in cs[0]) and ("hline" not in latex_text[-8:]):
				latex_text += "\\hline\n"
			latex_text += " & ".join(cs)
		
		if cnt == 0:
			latex_text += "\\\\ \\bottomrule \n"
		elif cnt <= 3:
			latex_text += "\\\\ \\hline \n"
		else:
			latex_text += "\\\\ \n"
	
	# latex_text = latex_text[:-8] # remove the last \\\\ \n
	latex_text += "\n\\bottomrule \n\\end{tabular}\n} \n\\end{table}"
	with open(basepath+fname+".tex", 'w') as fp:
		fp.write(latex_text)
	
	return result

def get_full_table_latex(dfs, common_cols, basepath, fname='results_datasets'):
	result = pd.merge(dfs[0], dfs[1], on=common_cols[0], how='right')
	result.to_csv(basepath+fname+".csv", index=False)
	print('Merged CSV saved to', basepath)
	
	latex_text = "\\renewcommand{\\arraystretch}{1.2}\n\
\\begin{table}[]\n\
\\centering\n\
\\resizebox{0.9\\textwidth}{!}{\n\
\\begin{tabular}{l l r c c c c c c c c}\n\
\\toprule\n\
\\multicolumn{3}{l}{} & \\multicolumn{4}{c}{\\textbf{Python}} & \\multicolumn{4}{c}{\\textbf{Java}} \\\\ \\cline{4-11}\n \
\\multicolumn{3}{l}{} & \\multicolumn{2}{c}{\\textbf{ASR}} & \\multicolumn{2}{c}{\\textbf{F1}} & \\multicolumn{2}{c}{\\textbf{ASR}} & \\multicolumn{2}{c}{\\textbf{F1}}\\\\ \\cline{4-11}\n \
"
	
	with open(basepath+fname+".csv", 'r') as fp:
		lines = fp.read().split('\n')
	
	for cnt, l  in enumerate(lines):
		if len(l) < 1:
			continue
		cs = l.split(",")
		if cnt == 0:
			replace1 = ["F1_x", "F1_y", "ASR_x", "ASR_y"]
			replace2 = ["F1_S_x", "F1_S_y", "ASR_S_x", "ASR_S_y"]
			for i, c in enumerate(cs):
				if c in replace1:
					cs[i] = "L"
				elif c in replace2:
					cs[i] = "L+S"
			for i, c in enumerate(cs):
				cs[i] = "\\textbf{" + cs[i] + "}"
			latex_text += " & ".join(cs)
		else:
			if ((jo in cs[0]) or (ao in cs[0]) or (uwisc in cs[0])) and ("hline" not in latex_text[-8:]):
				latex_text += "\\hline\n"
			latex_text += " & ".join(cs)
		
		if cnt == 0:
			latex_text += "\\\\ \\bottomrule \n"
		elif cnt <= 2:
			latex_text += "\\\\ \\hline \n"
		else:
			latex_text += "\\\\ \n"
	
	# latex_text = latex_text[:-8] # remove the last \\\\ \n
	latex_text += "\n\\bottomrule \n\\end{tabular}\n} \n\\end{table}"
	with open(basepath+fname+".tex", 'w') as fp:
		fp.write(latex_text)
	
	return result

def get_res1_table_latex_single(dfs, common_cols, basepath, fname='table1'):
	result = pd.merge(dfs[0], dfs[1], on=['Method', 'Smooth'], how='right')
	result.to_csv(basepath+fname+".csv", index=False)
	print('Merged CSV saved to', basepath)
	latex_text = "\\renewcommand{\\arraystretch}{1.2}\n\
	\\begin{table}[]\n\
	\\hspace*{-0in}\n\
	\\centering\n\
	\\resizebox{1\\textwidth}{!}{\n\
	\\begin{tabular}{l  c r c r | c r c r}\n\
	\\toprule\n\
	\multirow{2}{*}{\\textbf{Method}} \
		& \\multicolumn{4}{c}{\\textbf{$\\boldsymbol{k=1}$ site}} &  \\multicolumn{4}{c}{\\textbf{$\\boldsymbol{k=5}$  sites}}\\\\ \\cline{2-9}\n \
		& \\textbf{ASR} & & {\\textbf{F1}} & \\multicolumn{1}{c|}{} \
		& \\textbf{ASR} & & {\\textbf{F1}} & \\multicolumn{1}{c}{} \
		\\\\ \\hline\n \
	"
	
	with open(basepath+fname+".csv", 'r') as fp:
		lines = fp.read().split('\n')
	
	up = "$\\begingroup\\color{green}\\blacktriangle\endgroup$"
	down = "$\\begingroup\\color{red}\\blacktriangledown\endgroup$"
	color_blue, color_red = "\\textcolor{blue}", "\\textcolor{red}"
	for cnt, l  in enumerate(lines):
		if len(l) < 1:
			continue
		cs = l.split(",")
		cs = cs[:-8]
		row = []
		if cnt == 0:
			continue
		else:
			for i, c in enumerate(cs):
				if cs[0] == 'Random' and cs[1] == 'Yes':
					# Removes Random+smoothing row
					continue
				elif i == 0:
					if c == 'Random' and cs[1] == 'No':
						c = '\\textsc{Baseline*}'# uwisc + '~(Baseline)'
					elif c == 'Baseline':
						c = 'Random replace'
					if cs[1] == 'Yes':
						s = "\\begin{tabular}[l]{@{}l@{}}"+c+" + \\rsmo\\end{tabular}"
						row.append(s)
					else:
						row.append(c)
				elif i == 1:
					continue
				elif (i-1)%4 == 0:
					if cnt <= 3:
						row.append(" ")
					elif "-" in c:
						# improvement in F1
						row.append(""+color_blue+"{"+c+"}~"+up)
					else:
						row.append(""+color_red+"{"+c+"}~"+down)
				elif i%2 == 1:
					if cnt <= 3:
						row.append(" ")
					elif "-" in c:
						row.append(""+color_red+"{"+c+"}~"+down)
					else:
						# improvement in ASR
						row.append(""+color_blue+"{"+c+"}~"+up)
				else:
					row.append(c)

			if len(row) > 0:
				latex_text += " & ".join(row)
				#latex_text = latex_text[:-1]
		if len(row) > 0:
			if cnt == 0:
				latex_text += "\\\\ \\bottomrule \n\n"
			elif cnt > 1 and cnt <= 3:
				latex_text += "\\\\ \\hline \n\n"
			else:
				latex_text += "\\\\ \n\n"
		
	# latex_text = latex_text[:-8] # remove the last \\\\ \n
	caption_text = "\\protect\\input{tablecaption.tex}"
	latex_text += "\n\\hline\n\\end{tabular}\n} \n\\caption{"+caption_text+"}\n\\label{tbl:results}\n\\end{table}"

	with open(basepath+fname+"_single.tex", 'w') as fp:
		fp.write(latex_text)
	
	return result

def get_res1_table_latex(dfs, common_cols, basepath, fname='table1'):
	result = pd.merge(dfs[0], dfs[1], on=['Method', 'Smooth'], how='right')
	result.to_csv(basepath+fname+".csv", index=False)
	print('Merged CSV saved to', basepath)
	latex_text = "\\renewcommand{\\arraystretch}{1.25}\n\
	\\begin{table}[]\n\
	\\hspace*{-0.6in}\n\
	\\centering\n\
	\\resizebox{1.25\\textwidth}{!}{\n\
	\\begin{tabular}{l  c r c r | c r c r | c r c r | c r c r}\n\
	\\toprule\n\
	\multirow{3}{*}{\\textbf{Method}} & \\multicolumn{8}{c|}{\\textbf{Python}} & \\multicolumn{8}{c}{\\textbf{Java}} \\\\ \n \
		& \\multicolumn{4}{c}{\\textbf{$\\boldsymbol{k=1}$ site}} &  \\multicolumn{4}{c|}{\\textbf{$\\boldsymbol{k=5}$  sites}} &  \\multicolumn{4}{c}{\\textbf{$\\boldsymbol{k=1}$ site}} &  \\multicolumn{4}{c}{\\textbf{$\\boldsymbol{k=5}$ sites}}\\\\ \\cline{2-17}\n \
		& \\multicolumn{2}{c}{\\textbf{ASR}} & \\multicolumn{2}{c|}{\\textbf{F1}} & \
		\\multicolumn{2}{c}{\\textbf{ASR}} & \\multicolumn{2}{c|}{\\textbf{F1}} & \
		\\multicolumn{2}{c}{\\textbf{ASR}} & \\multicolumn{2}{c|}{\\textbf{F1}} & \
		\\multicolumn{2}{c}{\\textbf{ASR}} & \\multicolumn{2}{c}{\\textbf{F1}} \\\\ \\hline\n \
	"
	
	with open(basepath+fname+".csv", 'r') as fp:
		lines = fp.read().split('\n')
	
	up = "$\\begingroup\\color{green}\\blacktriangle\endgroup$"
	down = "$\\begingroup\\color{red}\\blacktriangledown\endgroup$"
	color_blue, color_red = "\\textcolor{blue}", "\\textcolor{red}"
	for cnt, l  in enumerate(lines):
		if len(l) < 1:
			continue
		cs = l.split(",")
		row = []
		if cnt == 0:
			continue
		else:
			for i, c in enumerate(cs):
				if cs[0] == 'Random' and cs[1] == 'Yes':
					# Removes Random+smoothing row
					continue
				elif i == 0:
					if c == 'Random' and cs[1] == 'No':
						c = '\\textsc{Baseline*}'# uwisc + '~(Baseline)'
					elif c == 'Baseline':
						c = 'Random replace'
					if cs[1] == 'Yes':
						s = "\\begin{tabular}[l]{@{}l@{}}"+c+" + \\rsmo\\end{tabular}"
						row.append(s)
					else:
						row.append(c)
				elif i == 1:
					continue
				elif (i-1)%4 == 0:
					if cnt <= 3:
						row.append(" ")
					elif "-" in c:
						# improvement in F1
						row.append(""+color_blue+"{"+c+"}~"+up)
					else:
						row.append(""+color_red+"{"+c+"}~"+down)
				elif i%2 == 1:
					if cnt <= 3:
						row.append(" ")
					elif "-" in c:
						row.append(""+color_red+"{"+c+"}~"+down)
					else:
						# improvement in ASR
						row.append(""+color_blue+"{"+c+"}~"+up)
				else:
					row.append(c)

			if len(row) > 0:
				latex_text += " & ".join(row)
		if len(row) > 0:
			if cnt == 0:
				latex_text += "\\\\ \\bottomrule \n\n"
			elif cnt > 1 and cnt <= 3:
				latex_text += "\\\\ \\hline \n\n"
			else:
				latex_text += "\\\\ \n\n"
		
	# latex_text = latex_text[:-8] # remove the last \\\\ \n
	caption_text = "\\protect\\input{tablecaption.tex}"
	latex_text += "\n\\hline\n\\end{tabular}\n\\label{tbl:results}\n} \n\\caption{"+caption_text+"}\n\\end{table}"

	with open(basepath+fname+".tex", 'w') as fp:
		fp.write(latex_text)
	
	return result


def process_results(df, dir, fname, flag, latex_version=1):
	if flag == 0:
		init = 1
	else:
		init = 0
	entries = []
	versions = ["v2", "v3"]
	n_rows1 = 3
	n_rows2 = 6
	for v in versions:
		done_naming, first_smooth = False, True
		for index, row in df.iterrows():
			entry = []
			names = row['config'].split('-')            
			# print(names)
			if len(names) < 6:
				continue
			version = names[init]
			z = names[init+2]
			pgd = names[init+3]
			if version != v:
				continue
			zs = z.split('_')
			z_opt = zs[1]
			z_sites = zs[2]
			pgds = pgd.split('_')
			pgd_iter = pgds[1]
			pgd_smooth = pgds[2]
			if z_opt=='no' and pgd_iter=='no':
				name = "No attack"
				optimal = name
			elif z_opt== "rand" and z_sites=="1" and pgd_iter=='rand':
				name = "Random"
				optimal = 'Baseline'
			elif z_opt== "rand" and z_sites=="1" and pgd_iter!='rand':
				name = "\\multirow{3}{*}{"+uwisc+"}"
				optimal = 'Random'
			#elif z_opt== "rand":
			#	name = '~'
			#	optimal = 'Random'
			elif z_opt=="o":
				if v == 'v2':
					name_orig = ao
					rowz = n_rows1
				elif v == 'v3':
					name_orig = jo
					rowz = n_rows2
				name = name_orig # "\\multirow{"+ str(rowz) +"}{*}{\\begin{tabular}[l]{@{}l@{}}"+ name_orig + "\\end{tabular}}"
				optimal = name_orig
				# done_naming = True
			
			if z_opt == 'o':
				z_opt = 'Optimal'
				z_sites = z_sites
			elif z_opt == 'rand':
				z_opt = 'Random'
				z_sites = z_sites
				if optimal != 'Baseline':
					optimal = 'Random'
			elif z_opt == 'no':
				z_opt = 'NA'
				z_sites = "0"

			if pgd_iter == 'no':
				pgd_iter = 'NA'
			elif pgd_iter == 'rand':
				pgd_iter = 'Random'
			
			if pgd_smooth == 'no':
				pgd_smooth = 'No'
			else:
				pgd_smooth = 'Yes'
				if first_smooth:
					name = "\multirow{"+ str(n_rows1) +"}{*}{\\begin{tabular}[l]{@{}l@{}}"+ name_orig + "+\\\\ smoothing \end{tabular}}"
					first_smooth = False

			if z_sites == '0':
				z_opt_name = z_opt
			else:
				z_opt_name =  z_opt + "-" + z_sites
			entry = [name, v, z_opt_name, z_opt, z_sites, pgd_iter, optimal, pgd_smooth, row['asr_dataset'], row['f1']]
			
			if len(entry) > 0:
				entries.append(entry)
	df = pd.DataFrame(entries, columns = common + ['ASR', 'F1'])
	return df
	
def get_full_table(df):
	##### v2 begins here ########
	# common = ['Method', 'ver', '\# sites', '\# PGD iters', 'smooth']
	common_cols = common[:]
	df = df.drop(['zopt', 'zsites'], axis=1)
	dropcols = ['zopt', 'zsites', 'smooth', 'ver', 'optimal']
	for d in dropcols:
		common_cols.remove(d)
	vers = df['ver'].unique()
	rows = []
	vers = ['v2', 'v3']
	for v in vers:
		ms = df.loc[(df['ver'] == v)].reset_index()['optimal']
		ms = ms.unique()
		for m in ms:
			if m not in [ao, jo]:
				site_names = df.loc[(df['ver'] == v) & (df['optimal'] == m)].reset_index()['\# sites'].unique()
			else:
				site_names = df.loc[(df['ver'] == v) & (df['optimal'] == m) & (df['smooth'] == 'Yes')]['\# sites'].unique()
			# print(site_names)
			for s in site_names:
				its = df.loc[(df['ver'] == v)  & (df['optimal'] == m) & (df['\# sites'] == s)].reset_index()['\# PGD iters'].unique()
				for it in its:
					mn = df.loc[(df['ver'] == v)  & (df['optimal'] == m) & (df['\# sites'] == s) & (df['\# PGD iters'] == it)].reset_index()['Method'].unique()[0]
					if 'smooth' in mn:
						continue
					vals = []
					add_me = False
					for metrics in ['ASR', 'F1']:
						for c in ['No', 'Yes']:
							val = df.loc[(df['ver'] == v) & (df['\# sites'] == s)  & (df['optimal'] == m) & (df['smooth'] == c) & (df['\# PGD iters'] == it)].reset_index()[metrics]
							if len(val) > 0:
								# print(val)
								vals.append(val[0])
							else:
								vals.append(-1)
					rows.append([mn, s, it]+vals)
					
	common_cols = common_cols+['ASR','ASR_S', 'F1', 'F1_S']
	df = pd.DataFrame(rows, columns=common_cols)
	return df, common_cols

def get_table_results(df):
	common_cols = common[:]
	order = ['No attack', 'Baseline', 'Random', ao, jo]
	smoothing = ['No', 'Yes']
	rows = []
	ba, bf = {'0':-1, '1': -1, '5': -1}, {'0':-1, '1': -1, '5': -1}
	for sm in smoothing:
		for o in order:
			if o == 'No attack':
				sites = ['0', '0']
				iters = 'NA'
			elif o == 'Baseline':
				sites = ['1', '1']
				iters = 'Random'
			elif o == 'Random' or o == ao:
				iters = '3'
				sites = ['1', '5']
			elif o == jo:
				iters = '10'
				sites = ['1', '5']
			row = []
			for s in sites:
				# print('{}:{}:{}:{}'.format(o, iters, sm, s))
				d = df.loc[(df['optimal'] == o) & (df['zsites'] == s) & (df['smooth'] == sm) & (df['\# PGD iters'] == iters)].reset_index()
				if o == 'Random' and sm == 'No':
					ba[s] = d['ASR'][0]
					bf[s] = d['F1'][0]
					
				if len(d) > 0:
					if ba[s] == -1:
						d1 = -1
					else:
						d1 = "{:+.2f}".format(d['ASR'][0] - ba[s])
					
					if bf[s] == - 1:
						d2 = -1
					else:
						d2 = "{:+.2f}".format(d['F1'][0] - bf[s])
				
					asr, f1 = d['ASR'][0], d['F1'][0]
					if d['ASR'][0] < 0.05:
						asr = 0
					if d['F1'][0] > 99:
						f1 = 100
					row.extend(["{:.2f}".format(asr), d1, "{:.2f}".format(f1), d2])
				
			if len(row) > 0:
				rows.append([o, sm] + row)
	colnames = ['Method', 'Smooth', 'ASR_1', 'diffa_1', 'F1_1', 'difff_1', 'ASR_5', 'diffa_5', 'F1_5', 'difff_5']
	df1 = pd.DataFrame(rows, columns=colnames)
	# print(df1)
	return df1, colnames


def save_results_pd(src_dir, dest_dir, outname, flag=1):
	notlookfor = ['test-']
	if flag == 1:
		lookfor = ['v2-', 'v3-']
	elif flag == 0:
		lookfor = ['test-']
		notlookfor = ['$'] # some random string
	elif flag == 2:
		lookfor = ['v2-']
	elif flag == 3:
		lookfor = ['v3-']

	configs = os.listdir(src_dir)
	configs_filtered = []
	for cfg in configs:
		for l in lookfor:
			for nl in notlookfor:
				if l in cfg and nl not in cfg:
					configs_filtered.append(cfg)
	configs = configs_filtered

	col_names = ['config', 'asr_dataset', 'f1', 'asr_sample_mean', 'asr_sample_std']
	li_of_dicts = []
	for cfg in configs:
		try:
			config_res = json.load(open(os.path.join(src_dir, cfg, 'results.json'), 'r'))
			for c in col_names:
				if c == 'config':
					config_res[c] = cfg.split('-gradient-attack')[0]
				else:
					config_res[c] = round(config_res[c],2)
			li_of_dicts.append(config_res)
		except Exception as e:
			print('Not found ', os.path.join(src_dir, cfg, 'results.json'))

	df = pd.DataFrame(li_of_dicts)
	df = df[col_names]
	df = df.sort_values(by=['config'])
	df.to_csv(os.path.join(dest_dir, outname), index=False)

	print('{} saved in {}'.format(outname, dest_dir))
	return df


def save_plots(basepath, dataset, df, asr_vs_iter=True, asr_vs_site=True):

	colors = {('v2', 'no'):'purple',
				('v2','smooth'):'brown',
				('v3', 'no'):'blue',
				('v3', 'smooth'):'green',
				('baseline', 'no'):'red',
				('baseline', 'smooth'):'orange',
				'avg_n_sites':'teal'}
	dataset_small_name = 'py150' if 'py150' in dataset else 'javasmall'
	gen_dataset_name = 'Python' if 'py150' in dataset else 'Java'
	dest_dir = 'scratch/plots/'
	mk_sz, f_sz1, f_sz2, f_sz3 = 7, 17, 18, 18
	
	'''
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "sans-serif",
		"font.sans-serif": ["Helvetica"]
	})
	'''

	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)
	
	if asr_vs_iter:

		configs = {'sites':['1', '5'], 'pgd_epochs':[1, 3, 10, 20], 'version':['v3', 'v2'], 'smoothing':['no', 'smooth']}
		all_expts = [df.at[i, 'config'] for i in range(len(df.index))]
		for s in configs['sites']:
			opt_experiments = [expt for expt in all_expts if ('-z_o_'+s+'-' in expt and dataset_small_name in expt)]
			rand_experiments = [expt for expt in all_expts if ('-z_rand_'+s+'-' in expt and dataset_small_name in expt and '31' in expt)]

			if s == '1':
				fig_title = '{}; $k = {}$ site'.format(gen_dataset_name, s)
			else:
				fig_title = '{}; $k = {}$ sites'.format(gen_dataset_name, s)
			fig_name_save = os.path.join(dest_dir, dataset_small_name+'-z_o_'+s+'.png')
			
			plt.clf()
			plt.xlabel('Iterations', fontsize=f_sz2)
			plt.ylabel('Attack Success Rate (ASR)', fontsize=f_sz2)
			plt.ylim(0, 101)
			plt.xlim(0, 21)
			plt.xticks(list(range(0, 21, 5)), fontsize=f_sz2)
			plt.yticks(list(range(0, 101, 10)), fontsize=f_sz2)

			legend_names = []
			plot_baseline = False

			for v in configs['version']:
				for smoothing in configs['smoothing']:
				
					for i,experiments in enumerate([rand_experiments, opt_experiments]):
						asr_dataset, n_iters = [], []
						for iters in configs['pgd_epochs']:

							for exp_name in experiments:
								if smoothing not in exp_name or v not in exp_name or '-pgd_'+str(iters)+'_' not in exp_name:
									continue	
								n_iters.append(iters)
								d = df.at[all_expts.index(exp_name), 'asr_dataset']
								asr_dataset.append(d)

						
						if i == 0 and v == 'v2' and smoothing == 'no': # plot Baseline

							plt.plot(n_iters, asr_dataset, marker='o', markersize=mk_sz, linestyle='dashed', color=colors[('baseline', smoothing)])
							v_name = 'Baseline'
							smoothing_name = '' if smoothing == 'no' else ' + RS'
							legend_names.append(v_name + smoothing_name)

						elif i == 1:
							plt.plot(n_iters, asr_dataset, marker='o', markersize=mk_sz, color=colors[(v, smoothing)])
							v_name = ao_ltx if v == 'v2' else jo_ltx
							smoothing_name = '' if smoothing == 'no' else ' + RS'
							legend_names.append(v_name + smoothing_name)
						
			if s == '1':
				plt.legend(legend_names,loc="upper right", frameon=False, prop={'size': f_sz1}, ncol=2)

			plt.tight_layout()
			plt.title(fig_title, fontsize=f_sz3)
			plt.savefig(fig_name_save)
			print('saved plot: '+fig_title)

	if asr_vs_site:

		avg_n_sites = {'Python':22.16, 'Java':7.62} # these values need to ideally come from experiments/get_stats.py
		max_n_sites = {'Python':140, 'Java':56}
		sites = [1, 5, 10, 20, 'all']
		version = ['v2', 'v3']
		smoothing = ['no', 'smooth']
		all_expts = [df.at[i, 'config'] for i in range(len(df.index))]

		plt.clf()
		plt.xlabel('Perturbation Strength ($k$)', fontsize=f_sz2)
		plt.ylabel('Attack Success Rate (ASR)', fontsize=f_sz2)
		plt.ylim(0, 101)
		plt.xlim(0, 31)

		x_labels = list(range(0, 21, 5)) + ['...', max_n_sites[gen_dataset_name]]
		plt.xticks(list(range(0, 31, 5)), fontsize=f_sz2)
		plt.yticks(list(range(0, 101, 10)), fontsize=f_sz2)

		legend_names = []
		opt_experiments = [expt for expt in all_expts if ('-z_o_' in expt and dataset_small_name in expt)]
		rand_experiments = [expt for expt in all_expts if ('-z_rand_' in expt and dataset_small_name in expt and '31' in expt)]
		for smooth in smoothing:
			for v in version:

				for i, expt in enumerate([rand_experiments, opt_experiments]):
					asr_dataset, n_sites = [], []
					for s in sites:
						for exp_name in expt:
							if v == 'v2':
								iters = '-pgd_3_'
							else:
								iters = '-pgd_10_'
							if i == 0:
								iters = '-pgd_1_' # use pgd 1 for baseline
								site = '-z_rand_'+str(s)+'-'
							else:
								site = '-z_o_'+str(s)+'-'
							if site in exp_name and v in exp_name and smooth in exp_name and iters in exp_name:
								d = df.at[all_expts.index(exp_name), 'asr_dataset']
								asr_dataset.append(d)
								if s == 'all':
									n_sites.append(30) # plot all sites as last point
								else:
									n_sites.append(s)

					if i == 1 and smooth == 'smooth':
						plt.plot(n_sites, asr_dataset, marker='o', markersize=mk_sz, color=colors[(v, smooth)])
						v_name = ao_ltx if v == 'v2' else jo_ltx
						smoothing_name = '' if smooth == 'no' else ' + RS'
						legend_names.append(v_name + smoothing_name)

					elif i == 0 and v == 'v2' and smooth == 'no':  # plot Baseline
						plt.plot(n_sites, asr_dataset, marker='o', markersize=mk_sz, linestyle='dashed', color=colors[('baseline', smooth)])
						v_name = 'Baseline'
						smoothing_name = '' if smooth == 'no' else ' + RS'
						legend_names.append(v_name + smoothing_name)
		
		plt.legend(legend_names,loc="upper left", frameon=False, prop={'size': f_sz1}, ncol=1)
		plt.tight_layout()
		plt.title(gen_dataset_name, fontsize=f_sz3)
		fig_name_save = os.path.join(dest_dir, dataset_small_name+'-asr_vs_sites.png')
		plt.savefig(fig_name_save)
		print('saved plot: ', fig_name_save)

if __name__ == '__main__':
	flag = int(sys.argv[1])
	basepath = './final-results/seq2seq/'
	datasets = ['sri/py150/', 'c2s/java-small']
	csv_exists = [True, False]
	latex_version = 2
	dfs, dfs_full, dfs_results1 = [], [], []
	merge_csv = False

	for dataset, ex in zip(datasets, csv_exists):
		src_dir = basepath+dataset
		dest_dir = basepath+dataset
		outname = 'final-results.csv'
		if flag == 0:
			outname = outname.split('.')[0] + '_test.' + outname.split('.')[1]
		if ex:
			df = pd.read_csv(os.path.join(dest_dir, outname))
			print('Found csv: ', os.path.join(dest_dir, outname))
		else:
			df = save_results_pd(src_dir, dest_dir, outname, flag)

		if merge_csv and dataset=='sri/py150/':
			fname = 'final-results-2.csv'
			try:
				df2 = pd.read_csv(os.path.join(dest_dir, fname))
				df = pd.concat([df,df2], axis=0, ignore_index=True)
			except:
				print(fname, " not found.")
			
		if df is not None:
			save_plots(basepath, dataset, df, asr_vs_iter=True, asr_vs_site=True)

		df = process_results(df, dest_dir, outname, flag)
		df.to_csv(os.path.join(basepath+dataset,'processed_results.csv'))
		df_full, commoncols_full = get_full_table(df)
		dfs_full.append((df_full, commoncols_full))
		
		(df_results12, commoncols_res1) = get_table_results(df)
		dfs_results1.append((df_results12, commoncols_res1))

	# df = merge_results_latex_v1(dfs, basepath)
	if True: # full results
		dfull, cfull = zip(*dfs_full)
		df = get_full_table_latex(dfull, cfull, basepath)
	
	if True: # full results
		dres1, cres1 = zip(*dfs_results1)
		df = get_res1_table_latex(dres1, cres1, basepath)
		df = get_res1_table_latex_single(dres1, cres1, basepath)

