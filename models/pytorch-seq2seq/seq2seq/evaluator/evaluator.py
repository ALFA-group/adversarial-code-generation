from __future__ import print_function, division

import torch
import torchtext
import itertools
from torch.nn.utils.rnn import pad_packed_sequence

import seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.evaluator.metrics import calculate_metrics

import tqdm
import time

class Evaluator(object):
	""" Class to evaluate models with given datasets.

	Args:
		loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
		batch_size (int, optional): batch size for evaluator (default: 64)
	"""

	def __init__(self, loss=NLLLoss(), batch_size=64):
		self.loss = loss
		self.batch_size = batch_size

	def evaluate_adaptive_batch(self, model, data_sml, data_med, data_lrg, verbose=False, src_field_name=seq2seq.src_field_name):
		""" Evaluate a model on given dataset and return performance.

		Args:
			model (seq2seq.models): model to evaluate
			data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

		Returns:
			loss (float): loss of the given model on the given dataset
		"""
		model.eval()

		loss = self.loss
		loss.reset()
		match = 0
		total = 0

		print('Adapting batch sizes:')
		print('  - Max: batch_size={}'.format(self.batch_size))
		print('  - Med: batch_size={}'.format(self.batch_size // 2))
		print('  - Min: batch_size=1')
		print('Selected field: {}'.format(src_field_name))

		# device = None if torch.cuda.is_available() else -1
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		use_small = False
		if not use_small:
			buckets = [
				torchtext.data.BucketIterator(
					dataset=data_sml, batch_size=self.batch_size,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				),
				torchtext.data.BucketIterator(
					dataset=data_med, batch_size=self.batch_size // 2,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				),
				torchtext.data.BucketIterator(
					dataset=data_lrg, batch_size=1,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				)
			]
			tot_len = len(buckets[0]) + len(buckets[1]) + len(buckets[2])
		else:
			buckets = [
				torchtext.data.BucketIterator(
					dataset=data_sml, batch_size=self.batch_size,
					sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
					device=device, train=False
				),
			]
			tot_len = len(buckets[0])

		batch_iterator = itertools.chain(*buckets)

		src_vocab = data_sml.fields[seq2seq.src_field_name].vocab
		tgt_vocab = data_sml.fields[seq2seq.tgt_field_name].vocab
		pad = tgt_vocab.stoi[data_sml.fields[seq2seq.tgt_field_name].pad_token]
		eos = tgt_vocab.stoi[data_sml.fields[seq2seq.tgt_field_name].SYM_EOS]

		output_seqs, orig_output_seqs = [], []
		input_seqs, orig_input_seqs = [], []
		ground_truths = []
		all_idxs = []

		cnt = 0
		with torch.no_grad():

			if verbose:
				batch_iterator = tqdm.tqdm(
					batch_iterator,
					total=tot_len
				)

			for batch in batch_iterator:
				cnt += 1
			
				# a = time.time()
				input_variables, input_lengths  = getattr(batch, src_field_name)
				orig_input_variables, orig_input_lengths  = getattr(batch, 'src')
				target_variables = getattr(batch, seq2seq.tgt_field_name)
				indices = getattr(batch, 'index')
				# b = time.time()

				decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)
				orig_decoder_outputs, orig_decoder_hidden, orig_other = model(orig_input_variables, orig_input_lengths.tolist(), target_variables)

				# c = time.time()
				# decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist())

				# print(target_variables, target_variables.size())
				# print(len(decoder_outputs), decoder_outputs[0].size())

				# Evaluation
				seqlist = other['sequence']
				for step, step_output in enumerate(decoder_outputs):
					target = target_variables[:, step + 1]
					loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

					non_padding = target.ne(pad)
					non_eos = target.ne(eos)
					mask = torch.mul(non_padding, non_eos)
					# mask = non_padding 

					correct = seqlist[step].view(-1).eq(target).masked_select(mask).sum().item()
					match += correct
					total += mask.sum().item()

				# d = time.time()

				# print(other['length'])
				# print(other['sequence'])
				oh_no = 0
				for i, (output_seq_len, orig_output_seq_len, idxs) in enumerate(zip(other['length'], orig_other['length'], indices)):
					# print(i,output_seq_len)
					tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
					tgt_seq = [tgt_vocab.itos[tok] for tok in tgt_id_seq]
					# print(tgt_seq)
					output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
					gt = [tgt_vocab.itos[tok] for tok in target_variables[i]]
					ground_truths.append(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))
					
					orig_tgt_id_seq = [orig_other['sequence'][di][i].data[0] for di in range(orig_output_seq_len)]
					orig_tgt_seq = [tgt_vocab.itos[tok] for tok in orig_tgt_id_seq]
					orig_output_seqs.append(' '.join([x for x in orig_tgt_seq if x not in ['<sos>','<eos>','<pad>']]))

					all_idxs.append(idxs.unsqueeze(0).detach().cpu().numpy()[0].item())

					'''
					orig_input_seqs.append([src_vocab.itos[t] for t in orig_input_variables[i]])
					input_seqs.append([src_vocab.itos[t] for t in input_variables[i]])
					
					if orig_output_seqs != output_seqs:
						# print([src_vocab.itos[t] for t in output_seqs[i]])
						# print([src_vocab.itos[t] for t in orig_output_seqs[i]])
						pass
					'''

				# e = time.time()
				# print(cnt, b-a, c-b, d-c, e-d)
				torch.cuda.empty_cache()

				# model.encoder.embedded[0].detach()
		# print(output_seqs)
		# print(ground_truths)
		# ground_truths = [' '.join(l[1:-1]) for l in data.tgt]
		# for i in range(len(ground_truths)):
			# print(ground_truths[i], " ---- ", output_seqs[i])
		# print([a for a in data.tgt])
		
		def add_idx(inp_list):
			di = {}
			for i, j in zip(all_idxs, inp_list):
				di[i] = j
			return di
		
		other_metrics = calculate_metrics(output_seqs, ground_truths, orig_output_seqs)

		other_metrics['li_exact_match'] = add_idx(other_metrics['li_exact_match'])
		other_metrics['li_orig_match'] = add_idx(other_metrics['li_orig_match'])
		output_seqs = add_idx(output_seqs)
		ground_truths = add_idx(ground_truths)
		
		'''
		for e in other_metrics['err_idx']:
			print(input_seqs[e])
			print(orig_input_seqs[e])
			print('*')
			print(orig_output_seqs[e])
			print(output_seqs[e])
			print('---')
		'''

		# match_idxs = [all_idxs[eidx] for eidx in other_metrics['exact_match_idx_orig']]
		# other_metrics['exact_match_idx_orig'] = match_idxs
		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total

		other_metrics.update({'Loss':loss.get_loss(), 'accuracy (torch)': accuracy*100})
		d = {
				'metrics': other_metrics,
				'output_seqs': output_seqs,
				'ground_truths': ground_truths
			}

		return d

	def evaluate(self, model, data, verbose=False, src_field_name=seq2seq.src_field_name):
		""" Evaluate a model on given dataset and return performance.

		Args:
			model (seq2seq.models): model to evaluate
			data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

		Returns:
			loss (float): loss of the given model on the given dataset
		"""
		model.eval()

		loss = self.loss
		loss.reset()
		match = 0
		total = 0


		# device = None if torch.cuda.is_available() else -1
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		batch_iterator = torchtext.data.BucketIterator(
			dataset=data, batch_size=self.batch_size,
			sort=True, sort_key=lambda x: len(getattr(x,src_field_name)),
			device=device, train=False)
		src_vocab = data.fields[seq2seq.src_field_name].vocab
		tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
		pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]
		eos = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].SYM_EOS]

		output_seqs = []
		ground_truths = []

		cnt = 0
		with torch.no_grad():

			if verbose:
				batch_iterator = tqdm.tqdm(batch_iterator)

			for batch in batch_iterator:
				cnt += 1
				# a = time.time()

				# print(src_field_name)
				input_variables, input_lengths  = getattr(batch, src_field_name)
				target_variables = getattr(batch, seq2seq.tgt_field_name)
				# b = time.time()

				decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

				# c = time.time()
				# decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist())

				# print(target_variables, target_variables.size())
				# print(len(decoder_outputs), decoder_outputs[0].size())

				# Evaluation
				seqlist = other['sequence']
				for step, step_output in enumerate(decoder_outputs):
					target = target_variables[:, step + 1]
					loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

					non_padding = target.ne(pad)
					non_eos = target.ne(eos)
					mask = torch.mul(non_padding, non_eos)
					# mask = non_padding 

					correct = seqlist[step].view(-1).eq(target).masked_select(mask).sum().item()
					match += correct
					total += mask.sum().item()

				# d = time.time()

				# print(other['length'])
				# print(other['sequence'])

				for i,output_seq_len in enumerate(other['length']):
					# print(i,output_seq_len)
					tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
					tgt_seq = [tgt_vocab.itos[tok] for tok in tgt_id_seq]
					# print(tgt_seq)
					output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
					gt = [tgt_vocab.itos[tok] for tok in target_variables[i]]
					ground_truths.append(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))

					# if get_attributions:
					#     a
					#     exit() 

				# e = time.time()

				# print(cnt, b-a, c-b, d-c, e-d)
				torch.cuda.empty_cache()

				# model.encoder.embedded[0].detach()

		# print(output_seqs)
		# print(ground_truths)
		# ground_truths = [' '.join(l[1:-1]) for l in data.tgt]
		# for i in range(len(ground_truths)):
			# print(ground_truths[i], " ---- ", output_seqs[i])
		# print([a for a in data.tgt])
		other_metrics = calculate_metrics(output_seqs, ground_truths)


		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total

		other_metrics.update({'Loss':loss.get_loss(), 'accuracy (torch)': accuracy*100})
		d = {
				'metrics': other_metrics,
				'output_seqs': output_seqs,
				'ground_truths': ground_truths
			}

		return d