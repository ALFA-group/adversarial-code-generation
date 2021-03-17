import os
import re
import sys
import json
import os.path
import pprint
from collections import OrderedDict
import os
import torchtext
import torch
import argparse
import json
import csv
import tqdm
import numpy as np
import random
import itertools

from seq2seq.evaluator.metrics import calculate_metrics
from seq2seq.loss import Perplexity, AttackLoss
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.util.plots import loss_plot

from gradient_attack_utils import get_valid_token_mask
from gradient_attack_utils import valid_replacement
from gradient_attack_utils import get_all_replacement_toks
from gradient_attack_utils import calculate_loss
from gradient_attack_utils import replace_toks_batch
from gradient_attack_utils import get_all_replacements
from gradient_attack_utils import bisection
from gradient_attack_utils import convert_to_onehot
from gradient_attack_utils import get_random_token_replacement
from gradient_attack_utils import get_random_token_replacement_2
from gradient_attack_utils import modify_onehot

from torch.autograd import Variable

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def apply_gradient_attack_v3(data, model, input_vocab, replace_tokens, field_name, opt, orig_tok_map, idx_to_fname, output_vocab=None, device='cpu'):
	########################################
	# Parameters that ideally need to come in from opt

	pgd_epochs = opt.u_pgd_epochs
	
	z_optim = opt.z_optim
	z_epsilon = int(opt.z_epsilon)
	z_init = opt.z_init # 0: initialize with all zeros; 1: initialize with uniform; 2: debug
	z_learning_rate = opt.z_learning_rate

	u_optim = opt.u_optim
	u_learning_rate = opt.u_learning_rate
	
	li_use_loss_smoothing = [opt.use_loss_smoothing]
	smoothing_param = opt.smoothing_param

	evaluate_only_on_good_samples = False
	matches_json = '/mnt/outputs/exact_matches_idxs.json'

	vocab_to_use = opt.vocab_to_use
	##########################################
	u_rand_update_pgd = False # Optimal site is randomly selected instead of argmax
	u_projection = 2 # 1: simple 0, 1 projection; 2: simplex projection

	li_u_optim_technique = [1] # 1: PGD: SGD with relaxation; 2: signed gradient
	li_u_init_pgd = [3] #list(range(5)) # 0: Original (fixed) init; 1: randomly initalize all tokens; 2: pick PGD optimal randomly instead of argmax; >2: randomly initialize only z=true;	
	li_use_u_discrete = [True]
	smooth_iters = 10
	
	use_cw_loss = False
	choose_best_loss_among_iters = True

	analyze_exact_match_sample = False
	samples_to_analyze = 1
	zlen_debug = 4
	plt_fname = '/mnt/outputs/loss_batch.pkl'
	outpth = '/mnt/outputs/'

	stats = {}
	config_dict = OrderedDict([
		('version', 'v3'),
		('pgd_epochs', pgd_epochs),
		('z_optim', z_optim),
		('z_epsilon', z_epsilon),
		('z_init', z_init),
		('z_learning_rate', z_learning_rate),
		('evaluate_only_on_good_samples', evaluate_only_on_good_samples),
		('u_optim', u_optim),
		('u_learning_rate', u_learning_rate),
		('u_rand_update_pgd', u_rand_update_pgd),
		('smooth_iters', smooth_iters),
		('use_cw_loss', use_cw_loss),
		('choose_best_loss_among_iters', choose_best_loss_among_iters),
		('analyze_exact_match_sample', analyze_exact_match_sample),
	])
	stats['config_dict'] = config_dict
	########################################
	
	# This datastructure is meant to return best replacements only for *one* set of best params
	# If using in experiment mode (i.e. itertools.product has mutliple combinations), don't expect consistent
	# results from best_replacements_dataset
	best_replacements_dataset = {}
	
	'''
	with open(matches_json, 'r') as f:
		exact_matches_file_names = json.load(f) # mapping of file/sample index to file name
	exact_matches_file_names = set([str(e) for e in exact_matches_file_names])
	'''

	for params in itertools.product(li_u_optim_technique, li_u_init_pgd, li_use_loss_smoothing, li_use_u_discrete):
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(config_dict)
		(u_optim_technique, u_init_pgd, use_loss_smoothing, use_u_discrete) = params
		od = OrderedDict([
			('u_optim_technique', u_optim_technique),
			('u_init_pgd', u_init_pgd),
			('use_loss_smoothing', use_loss_smoothing),
			('use_u_discrete', use_u_discrete),
		])
		pp.pprint(od)
		stats['config_dict2'] = od
		batch_iterator = torchtext.data.BucketIterator(
			dataset=data, batch_size=opt.batch_size,
			sort=True, sort_within_batch=True,
			sort_key=lambda x: len(x.src),
			device=device, repeat=False
			)
		batch_generator = batch_iterator.__iter__()
		if use_cw_loss:
			loss_obj = AttackLoss(device=device)
		else:
			weight = torch.ones(len(output_vocab.vocab)).half()
			pad = output_vocab.vocab.stoi[output_vocab.pad_token]
			loss_obj = Perplexity(weight, pad)
			if torch.cuda.is_available():
				loss_obj.cuda()
		model.train()
		
		best_loss_among_iters, best_loss_among_iters_status = {}, {}
		nothing_to_attack, rand_replacement_too_long, tot_attacks, tot_samples = 0, 0, 0, 0
		sample_to_select_idx, pred_to_select, sample_to_select_idx_cnt, sname = None, None, 0, ''

		# a mask of length len(input_vocab) which lists which are valid/invalid tokens
		if vocab_to_use == 1:
			invalid_tokens_mask = get_valid_token_mask(negation=True, vocab=input_vocab, exclude=[])
		elif vocab_to_use == 2:
			invalid_tokens_mask = [False]*len(input_vocab)


		for bid, batch in enumerate(tqdm.tqdm(batch_generator, total=len(batch_iterator))):
			if analyze_exact_match_sample and (sample_to_select_idx_cnt >= samples_to_analyze):
				continue
				
			found_sample, zlen, plen, zstr = False, 0, 0, None
			indices = getattr(batch, 'index')
			input_variables, input_lengths = getattr(batch, field_name)
			target_variables = getattr(batch, 'tgt')
			orig_input_variables, orig_lens = getattr(batch, 'src')
			tot_samples += len(getattr(batch, field_name)[1])

			# Do random attack if inputs are too long and will OOM under gradient attack
			if max(getattr(batch, field_name)[1]) > 250:
				rand_replacement_too_long += len(getattr(batch, field_name)[1])
				rand_replacements = get_random_token_replacement_2(
					input_variables.cpu().numpy(),
					input_vocab,
					indices.cpu().numpy(),
					replace_tokens,
					opt.distinct,
					z_epsilon
				)

				best_replacements_dataset.update(rand_replacements)
				continue

			# too update replacement-variables with max-idx in case this is the iter with the best optimized loss
			update_this_iter = False
			
			indices = indices.cpu().numpy()
			inputs_oho = Variable(convert_to_onehot(input_variables, vocab_size=len(input_vocab), device=device), requires_grad=True).half()
			
			#### To compute which samples have exact matches with ground truth in this batch
			if analyze_exact_match_sample or evaluate_only_on_good_samples:
				# decoder_outputs: List[(max_length x decoded_output_sz)]; List length -- batch_sz
				# These steps are common for every batch.
				decoder_outputs, decoder_hidden, other = model(inputs_oho, input_lengths, target_variables, already_one_hot=True)

				output_seqs, ground_truths = [], []
				
				for i,output_seq_len in enumerate(other['length']):
					# print(i,output_seq_len)
					tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
					tgt_seq = [output_vocab.vocab.itos[tok] for tok in tgt_id_seq]
					output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
					gt = [output_vocab.vocab.itos[tok] for tok in target_variables[i]]
					ground_truths.append(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))

				other_metrics = calculate_metrics(output_seqs, ground_truths)

				if len(other_metrics['exact_match_idx']) > 0:
					sample_to_select_idx = other_metrics['exact_match_idx'][0]
				
				if evaluate_only_on_good_samples:
					pass
					if len(other_metrics['good_match_idx']) == 0:
						continue
					attack_sample_set = other_metrics['good_match_idx']
				elif sample_to_select_idx is None:
					continue
			
			###############################################
			# Initialize z for the batch
			status_map, z_map, z_all_map, z_np_map, site_map_map, site_map_lookup_map, z_initialized_map,invalid_tokens_mask_map = {}, {}, {}, {}, {}, {}, {}, {}
			
			for ii in range(inputs_oho.shape[0]):
				replace_map_i, site_map, status = get_all_replacement_toks(input_variables.cpu().numpy()[ii], None, input_vocab, replace_tokens)
				
				if not status:
					continue

				site_map_lookup = []
				for cnt, k in enumerate(site_map.keys()):
					site_map_lookup.append(k)
				
				if z_init == 0:
					z_np = np.zeros(len(site_map_lookup)).astype(float)
				elif z_init == 1:
					z_np = (1/len(site_map_lookup))*np.ones(len(site_map_lookup)).astype(float)
				elif z_init == 2:
					z_np = np.zeros(len(site_map_lookup)).astype(float)
					z_np[0] = 1
				
				z = torch.tensor(z_np, requires_grad=True, device=device)
				if len(z.shape) == 1:
					z = z.unsqueeze(dim=1)
				
				mask = np.array(input_variables.cpu().numpy()[ii]*[False]).astype(bool)
				for kk in range(len(site_map_lookup)):
					if not z[kk]:
						continue
					m = site_map[site_map_lookup[kk]]
					mask = np.array(m) | mask
				
				status_map[ii] = status
				z_map[ii] = z
				z_np_map[ii] = z_np
				z_all_map[ii] = list(mask)
				site_map_map[ii] = site_map
				site_map_lookup_map[ii] = site_map_lookup
				z_initialized_map[ii] = [False]*z_np.shape[0]
				# selected_toks = torch.sum(z * embed, dim=0)  # Element-wise mult

			if analyze_exact_match_sample and (sample_to_select_idx not in z_np_map or len(z_np_map[sample_to_select_idx]) < zlen_debug):
				continue
			
			new_inputs, site_map_map, z_all_map, input_lengths, sites_to_fix_map = replace_toks_batch(input_variables.cpu().numpy(), indices, z_map, site_map_map, site_map_lookup_map, {}, field_name, input_vocab, orig_tok_map, idx_to_fname)
			input_lengths = torch.tensor(input_lengths, device=device)
			inputs_oho_orig = Variable(convert_to_onehot(torch.tensor(new_inputs, device=device), vocab_size=len(input_vocab), device=device), requires_grad=True).half()
			inputs_oho_orig = modify_onehot(inputs_oho_orig, site_map_map, sites_to_fix_map, device)	
			
			# Initialize input_hot_grad
			# This gets updated for each i with (not z_all_map) tokens being switched to x_orig
			if u_init_pgd == 1:
				input_h = inputs_oho_orig[0][0].clone().detach()
			elif u_init_pgd == 2:
				input_h = torch.zeros(inputs_oho_orig[0][0].shape).half()
			elif u_init_pgd == 3:
				valid_tokens = [not t for t in invalid_tokens_mask[:]]
				input_h = inputs_oho_orig[0][0].clone().detach()
				input_h[valid_tokens] = 1/sum(valid_tokens)
				input_h[invalid_tokens_mask] = 0
			elif u_init_pgd == 4:
				input_h = (1 - inputs_oho_orig[0][0].clone())/(len(invalid_tokens_mask)-1)				
			input_hot_grad = input_h.clone().detach().requires_grad_(True).repeat(inputs_oho_orig.shape[0], inputs_oho_orig.shape[1]).view(inputs_oho_orig.shape)
			
			##################################################
			for i in range(inputs_oho_orig.shape[0]):
				if i not in status_map:
					continue

				if analyze_exact_match_sample and  (i != sample_to_select_idx):
					continue

				fn_name = str(indices[i])

				input_hot_orig_i = inputs_oho_orig[i].unsqueeze(0) # is not affected by gradients; okay to copy by reference
				input_hot_grad_i = input_hot_grad[i].unsqueeze(0)
				il_i = input_lengths[i].unsqueeze(0)
				tv_i = target_variables[i].unsqueeze(0)
				site_map_lookup = site_map_lookup_map[i]
				z = z_map[i]
				site_map = site_map_map[i]
				z_all = z_all_map[i]
				
				if z_epsilon == 0:
					z_epsilon = z.shape[0]

				if i not in status_map:
					nothing_to_attack += 1
					continue

				tot_attacks += 1

				if analyze_exact_match_sample:
					sample_to_select_idx_cnt += 1
					sname = fn_name
					found_sample = True
					print('found {}; z len {}'.format(sname, len(z_np_map[i])))
					print([input_vocab.itos[t] for t in new_inputs[i]])
					print([input_vocab.itos[t] for t in input_variables[i]])
					zlen = sum(z_all_map[i])
					plen = len(z_all_map[i])
					zstr = str(z_np_map[i])
					print(zstr)

				# Revert all (not z_mask) tokens to x_orig
				# Take care with cloning to ensure gradients are not shared.
				not_z_all = [not t for t in z_all]
				input_hot_grad_i[0][not_z_all] = input_hot_orig_i[0][not_z_all].detach().clone().requires_grad_(True)
				
				embed = None
				for sm in site_map_lookup:
					if embed is None:
						embed = np.array(site_map[sm]).astype(float)
					else:
						embed = np.vstack((embed, np.array(site_map[sm]).astype(float)))
				embed = torch.tensor(embed, requires_grad=True, device=device) # values don't get updated/modified
				if len(embed.shape) == 1:
					embed = embed.unsqueeze(dim=0)
				
				batch_loss_list_per_iter, best_replacements_sample = [], {}

				# Begin optim iters
				for j in range(pgd_epochs):
					# Forward propagation   
					# decoder_outputs: List[(max_length x decoded_output_sz)]; List length -- batch_sz
					selected_toks = torch.sum(z * embed, dim=0)  # Element-wise mult
					selected_toks = selected_toks.repeat(input_hot_grad_i.shape[2],1).T.unsqueeze(0).half()
					perturbed_sample = selected_toks * input_hot_grad_i + (1-selected_toks) * input_hot_orig_i
					
					# Calculate loss
					if use_u_discrete:
						a = perturbed_sample.argmax(2)
						m = torch.zeros(perturbed_sample.shape, requires_grad=True, device=device).scatter(2, a.unsqueeze(2), 1.0).half()
						decoder_outputs, decoder_hidden, other = model(m, il_i, tv_i, already_one_hot=True)
					else:
						decoder_outputs, decoder_hidden, other = model(perturbed_sample, il_i, tv_i, already_one_hot=True)
					loss, l_scalar, sample_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_outputs, other, tv_i)

					if analyze_exact_match_sample: # sample_to_select_idx is not None at this stage
						batch_loss_list_per_iter.append(sample_wise_loss_per_batch)
					else:
						batch_loss_list_per_iter.append(sample_wise_loss_per_batch)

					if (fn_name not in best_loss_among_iters) or (best_loss_among_iters[fn_name] < sample_wise_loss_per_batch[0]):
						best_loss_among_iters_status[fn_name] = True
						best_loss_among_iters[fn_name] = sample_wise_loss_per_batch[0]
					else:
						best_loss_among_iters_status[fn_name] = False
					
					invalid_tokens_mask_ij = invalid_tokens_mask[:]

					# Forward propagation   
					# Calculate loss on the continuous value vectors
					if not use_loss_smoothing:
						decoder_outputs, decoder_hidden, other = model(perturbed_sample, il_i, tv_i, already_one_hot=True)
						loss, l_scalar, sample_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_outputs, other, tv_i)
						
						# update loss and backprop
						model.zero_grad()
						input_hot_grad_i.retain_grad()
						z.retain_grad()
						loss.backward(retain_graph=True)
						
						grads_oh_i = input_hot_grad_i.grad
						gradients = grads_oh_i.detach().cpu().numpy()[0]
						grads_z_i = z.grad
					else:
						b_loss, smooth_grads_oh, smooth_grads_z = [], None, None
						mask_optimisee = torch.sum(z * embed, dim=0).cpu().detach().numpy().astype(bool)
						for si in range(smooth_iters):
							smooth_hot_grad_i = input_hot_grad_i.clone()
							noise = smoothing_param * torch.empty(input_hot_grad_i.shape, device=device).normal_(mean=0,std=1).half()
							smooth_hot_grad_i[:, mask_optimisee, :] = smooth_hot_grad_i[:, mask_optimisee, :] + noise[:, mask_optimisee, :]
							smooth_hot_grad_i = input_hot_grad_i + noise 
							smooth_input = selected_toks * smooth_hot_grad_i + (1-selected_toks) * input_hot_orig_i
							smooth_decoder_outputs, smooth_decoder_hidden, smooth_other = model(smooth_input, il_i, tv_i, already_one_hot=True)
							loss, l_scalar, sample_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, smooth_decoder_outputs, smooth_other, tv_i)

							# update loss and backprop
							model.zero_grad()
							smooth_hot_grad_i.retain_grad()
							z.retain_grad()
							loss.backward(retain_graph=True)

							if smooth_grads_oh is None:
								smooth_grads_oh = smooth_hot_grad_i.grad
								smooth_grads_z = z.grad
							else:
								smooth_grads_oh += smooth_hot_grad_i.grad
								smooth_grads_z += z.grad
						
						grads_oh_i = smooth_grads_oh/smooth_iters
						gradients = grads_oh_i.detach().cpu().numpy()[0]
						grads_z_i = smooth_grads_z/smooth_iters

					# Optimize input_hot_grad_i
					if u_optim:
						if analyze_exact_match_sample:
							print('-- u optim --')
						for idx in range(z.shape[0]):
							# if z_np[idx] == 0:
							#	continue
							mask = site_map[site_map_lookup[idx]]
							# Can take a mean across all tokens for which z=1
							# Currently, this mean is for all tokens for which z_i=1
							avg_tok_grads = np.mean(gradients[mask], axis=0)
							repl_tok_idx = site_map_lookup[idx]
							# print(repl_tok_idx)
							repl_tok = input_vocab.itos[repl_tok_idx]
							# print("repl tok: {}".format(repl_tok))
							nabla = avg_tok_grads
							
							if u_optim_technique == 2:
								nabla = np.sign(nabla)

							# PGD
							step = u_learning_rate/np.sqrt(j+1) * nabla
							if use_cw_loss:
								step = -1 * step
							
							# any one entry of the masked entries
							# initalize to 0s for first entry
							input_h = input_hot_grad_i[0][mask,:][0,:].detach().cpu().numpy()
							'''
							print("z idx {}".format(idx))
							print(np.expand_dims(input_h, axis=0).shape)
							print(np.argmax(np.expand_dims(input_h, axis=0), axis=1))
							'''
							input_h = input_h + step

							# projection
							if u_projection == 1:
								optim_input = np.clip(input_h, 0, 1)
							elif u_projection == 2:
								# simplex projection
								fmu = lambda mu, a=input_h: np.sum(np.maximum(0, a - mu )) - 1
								mu_opt = bisection(fmu, -1, 1, 30)
								if mu_opt is None:
									mu_opt = 0 # assigning randomly to 0
								optim_input = np.maximum(0, input_h - mu_opt)
								# print(fmu(mu_opt))

							# projection onto only valid tokens. Rest are set to 0
							optim_input[invalid_tokens_mask_ij] = 0
							# print(sum(invalid_tokens_mask_map))

							if u_rand_update_pgd:
								max_idx = random.randrange(optim_input.shape[0])
							else:
								max_idx = np.argmax(optim_input)
							
							# Update to replacements with best loss so far
							if choose_best_loss_among_iters:
								if best_loss_among_iters_status[fn_name]:
									best_replacements_sample[repl_tok] = input_vocab.itos[max_idx]
							else:
								best_replacements_sample[repl_tok] = input_vocab.itos[max_idx]
							
							# Ensure other z's for this index don't use this replacement token
							invalid_tokens_mask_ij[max_idx] = True # setting it as invalid being True
							
							# Update optim_input							
							input_hot_grad_i[0][mask,:] = torch.tensor(optim_input, requires_grad=True, device=device)
						
						if analyze_exact_match_sample:
							print('Best loss: ', best_loss_among_iters[fn_name])
							print("Loss: {}".format(batch_loss_list_per_iter))
							print(best_replacements_sample)

					# Optimize z
					if z_optim:
						# print('Optimizing z')
						if analyze_exact_match_sample:
							print('-- z optim --')
							print(z.squeeze().cpu().detach().numpy())
							print("Constraint: {}".format(z_epsilon))
						
						# Gradient ascent. Maximize CE loss
						a = z + z_learning_rate/np.sqrt(j+1) * grads_z_i
						if analyze_exact_match_sample:
							print(a.squeeze().cpu().detach().numpy())
						a_np = a.cpu().detach().numpy()
						fmu = lambda mu, a=a_np, epsilon=z_epsilon: np.sum( a - mu ) - epsilon
						mu_opt = bisection(fmu, 0, np.max(a_np), 50)
						if mu_opt is None:
							mu_opt = 0 # assigning randomly to 0
						if mu_opt > 0:
							z = torch.clamp(a-mu_opt, 0, 1)
						else:
							z = torch.clamp(a, 0, 1)
						# one = torch.ones(z.shape, device=device, requires_grad=True)
						# zero = torch.zeros(z.shape, device=device, requires_grad=True)
						# z = torch.where(z > 0.5, one, zero)
						if analyze_exact_match_sample:
							print(z.squeeze().cpu().detach().numpy())
							print('---')

				# end optim iterations

				# Select a final z
				z_final_soft = z.squeeze(dim=1).detach().cpu().numpy()

				z_final = np.random.binomial(1, z_final_soft)
				if analyze_exact_match_sample:
					print('Final z -- ')
					print(z_final_soft)
					print(z_final)

				if sum(z_final) == 0 or sum(z_final) > z_epsilon:
					if sum(z_final) == 0:
						z_final_soft_idx = np.argsort(z_final_soft)[::-1][0]
					elif sum(z_final) > z_epsilon:
						z_final_soft_idx = np.argsort(z_final_soft)[::-1][:z_epsilon]
					z_final = np.zeros(z_final.shape)
					z_final[z_final_soft_idx] = 1
				
				if analyze_exact_match_sample:
					print('constraint: {}'.format(z_epsilon))
					print('after constraint: {}'.format(z_final))

				for ix in range(z_final.shape[0]):
					if z_final[ix] == 0:
						# Find out the replace token corresponding to this site
						remove_key = input_vocab.itos[site_map_lookup[ix]]
						# Remove this token from best_replacements_sample
						best_replacements_sample.pop(remove_key, None)
				
				if analyze_exact_match_sample:
					print('Final best repalcements' , best_replacements_sample)

				
				if analyze_exact_match_sample:
					if found_sample:
						if len(batch_loss_list_per_iter) > 0:
							out_str = 'ss{}_zlen-{}_n-{}_zstr-{}_opt-{}_lr-{}_uinit-{}_smooth-{}_udisc-{}'.format(sname, zlen, plen, zstr, u_optim_technique, u_learning_rate, u_init_pgd, int(use_loss_smoothing), int(use_u_discrete))
							print(out_str)
							loss_plot(batch_loss_list_per_iter, os.path.join(outpth, out_str))

				best_replacements_dataset[fn_name] = best_replacements_sample

	print('Skipped and reverted to random attacks: {}/{} ({})'.format(rand_replacement_too_long, tot_samples, round(100*rand_replacement_too_long/tot_samples, 2)))
	print('Nothing to attack: {}/{} ({})'.format(nothing_to_attack, tot_attacks, round(100*nothing_to_attack/tot_attacks, 2)))
	print('----------------')
	print("# of samples attacked: {}".format(len(best_replacements_dataset.keys())))

	stats['reverted_to_random_attacks_pc'] = round(100*rand_replacement_too_long/tot_samples, 2)
	stats['nothing_to_attack_pc'] = round(100*nothing_to_attack/tot_attacks, 2)
	stats['n_samples_attacked'] = len(best_replacements_dataset.keys())
	
	if analyze_exact_match_sample:
		kzs = best_replacements_dataset.keys()
		for kz in kzs:
			print("{}::{}".format(kz, best_replacements_dataset[kz]))
		print('====')
	
	best_replacements_dataset, avg_replaced = get_all_replacements(best_replacements_dataset, field_name, orig_tok_map, idx_to_fname, True)
	
	if analyze_exact_match_sample:
		for kz in kzs:
			print("{}::{}".format(kz, best_replacements_dataset[kz]))
	
	print('\n# tokens optimized on an average: {}'.format(avg_replaced))
	stats['n_tokens_optimized_avg'] = avg_replaced
	print("\n# of samples attacked post processing: {}\n=======".format(len(best_replacements_dataset.keys())))
	stats['n_samples_attacked_post_processing'] = len(best_replacements_dataset.keys())

	return best_replacements_dataset, stats