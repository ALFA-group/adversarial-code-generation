import sys, os
import numpy as np
import tqdm
try:
	from bleu import moses_multi_bleu
except:
	from seq2seq.evaluator.bleu import moses_multi_bleu


def calculate_metrics_from_files(pred_file, labels_file, verbose=False):
	f_pred = open(pred_file, 'r')
	f_true = open(labels_file, 'r')

	hypotheses = f_pred.readlines()
	references = f_true.readlines()

	f_pred.close()
	f_true.close()

	a = calculate_metrics(hypotheses, references, verbose)
	for m in a:
		print('%s: %.3f'%(m,a[m]))
	print()

def get_freqs(pred, true):
	all_words = set(pred+true)
	d_pred = {x: pred.count(x) for x in all_words}
	d_true = {x: true.count(x) for x in all_words}
	return d_pred, d_true

def calculate_metrics(y_pred, y_true, orig_y_pred=None, verbose=False, bleu=False):
	''' 
	Calculate exact match accuracy, precision, recall, F1 score, word-level accuracy
	y_pred and y_true are lists of strings
	function returns dict with the calculated metrics
	'''

	N = min(len(y_pred),len(y_true))
	# N = 4500
	if len(y_pred)!=len(y_true):
		print('Warning: The number of predictions and ground truths are not equal, calculating metrics over %d points'%N)

	# for precision, recall, f1
	tp = 0
	fp = 0
	fn = 0

	# for exact match
	exact_match, exact_match_idx, exact_predicted, good_match_idx, good_match_idx_extended , exact_match_idx_orig = 0, [], [], [], [], []
	li_exact_match, li_orig_match, err_idx = [], [], []
	
	# for word-level accuracy
	correct_words = 0
	total_words = 0

	if verbose:
		a = tqdm.tqdm(range(N))
	else:
		a = range(N)

	for i in a:
		# print(i)
		pred = y_pred[i].split()
		true = y_true[i].split()

		total_words += len(true)
		correct_matches = 0
		for j in range(min(len(true), len(pred))):
			if pred[j]==true[j]:
				correct_words += 1
				correct_matches += 1


		d_pred, d_true = get_freqs(pred, true)
		
		if pred == true:
			exact_match += 1
			if len(pred) > 1 and ('<unk>' not in pred):
				exact_match_idx.append(i)
				# exact_predicted.append(pred)
				# print(pred)
		
		if orig_y_pred is not None:
			orig_pred = orig_y_pred[i].split()
			orig_d_pred, _ = get_freqs(orig_pred, true)
			exact_matches, orig_exact_match_cnt = 0, 0
			for j in range(min(len(true), len(pred), len(orig_pred))):
				if true[j] == orig_pred[j] and pred[j]!=true[j]:
					exact_matches += 1
					'''
					print(orig_pred)
					print(true)
					print(pred)
					err_idx.append(i)
					print('=====')
					'''
				
				if true[j] == orig_pred[j]:
					orig_exact_match_cnt += 1

			li_exact_match.append(exact_matches)
			li_orig_match.append(orig_exact_match_cnt)
		
		# print(d_pred, d_true)

		calc_type = 2

		if calc_type==1:
			# this is my implementation
			for word in d_pred: 
				tp += min(d_pred[word], d_true[word])
				fp += max(0, d_pred[word]-d_true[word])
				fn += max(0, d_true[word]-d_pred[word])
		else:
			# this is the code2seq implementation
			orig_80, pred_80 = 0 , 0
			for word in d_pred: 
				if d_pred[word]>0:
					if d_true[word]>0:
						tp += 1
					else:
						fp += 1
				if d_true[word]>0 and d_pred[word]==0:
					fn += 1

			if orig_y_pred is not None:
				for word in orig_d_pred:
					if orig_d_pred[word] > 0:
						if word in d_true and d_true[word]>0:
							if word in d_pred and d_pred[word] > 0:							
								pred_80 += 1
							orig_80 += 1

			# if tp > 0.8*len(d_pred) and  len(pred) > 1 and ('unk' not in y_pred[i]):
			#	good_match_idx.append(i)

	# print(tp, fp, fn)
	precision = tp / (tp+fp+0.0000000001)
	recall = tp / (tp+fn+0.0000000001)
	f1 = 2*precision*recall / (precision+recall+0.0000000001)
	exact_match /= N
	word_level_accuracy = correct_words / total_words

	if sum(li_orig_match) == 0:
		sum_li = 1
	else:
		sum_li = sum(li_orig_match)
	asr_dataset = round(sum(li_exact_match)/sum_li * 100, 2) 
	ax = [e/o if o!=0 else 0 for e, o in zip(li_exact_match, li_orig_match)]
	asr_sample_mean = round(sum(ax)/sum_li, 2)
	asr_sample_std =  round(np.std(np.array(ax)), 2)

	d = {
			'precision': precision*100, 
			'recall': recall*100, 
			'f1': f1*100, 
			'exact_match':exact_match*100, 
			'word-level accuracy': word_level_accuracy*100, 
			'total_samples': N,
			'asr_dataset': asr_dataset,
			'asr_sample_mean': asr_sample_mean,
			'asr_sample_std': asr_sample_std,
			'li_exact_match': li_exact_match,
			'li_orig_match': li_orig_match,
			'exact_match_idx': exact_match_idx
		}

	if bleu:
		bleu_score = moses_multi_bleu(np.array(y_pred), np.array(y_true))
		d['BLEU'] = bleu_score

	return d

def parse_args():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--f_true', help='File with ground truth labels', required=True)
	parser.add_argument('--f_pred', help='File with predicted labels', required=True)
	parser.add_argument('--verbose', action='store_true', help='verbosity')


	args = parser.parse_args()
	assert os.path.exists(args.f_true), 'Invalid file for ground truth labels'
	assert os.path.exists(args.f_pred), 'Invalid file for predicted labels'
	return args


if __name__=="__main__":
	args = parse_args()
	calculate_metrics_from_files(args.f_pred, args.f_true, args.verbose)
