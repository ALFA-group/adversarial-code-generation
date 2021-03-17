import os
import re
import sys
import json
import os.path
import pprint
import time

from seq2seq.util.checkpoint import Checkpoint
from seq2seq.loss import Perplexity, AttackLoss

from gradient_attack_v3 import apply_gradient_attack_v3
from gradient_attack_utils import get_valid_token_mask
from gradient_attack_utils import valid_replacement
from gradient_attack_utils import get_all_replacement_toks
from gradient_attack_utils import calculate_loss
from gradient_attack_utils import replace_toks_batch
from gradient_attack_utils import get_all_replacements
from gradient_attack_utils import bisection
from gradient_attack_utils import convert_to_onehot
from gradient_attack_utils import get_random_token_replacement_2
from gradient_attack_utils import get_exact_matches
from gradient_attack_utils import modify_onehot
from gradient_attack_utils import remove_padding_and_flatten

from dataset import Vocabulary, create_dataloader
from model import Code2Seq

from torch.autograd import Variable
from collections import OrderedDict
import seq2seq
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
import pickle
import math

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path', help='Path to data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', required=True,
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default='Best_F1')
    parser.add_argument('--num_replacements', type=int, default=1500)
    parser.add_argument('--distinct', action='store_true', dest='distinct', default=True)
    parser.add_argument('--no-distinct', action='store_false', dest='distinct')
    parser.add_argument('--no_gradient', action='store_true', dest='no_gradient', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--random', action='store_true', default=False, help='Also generate random attack')
    parser.add_argument('--n_alt_iters', type=int)
    parser.add_argument('--z_optim', action='store_true', default=False)
    parser.add_argument('--z_epsilon', type=int)
    parser.add_argument('--z_init', type=int)
    parser.add_argument('--u_optim', action='store_true', default=False)
    parser.add_argument('--u_pgd_epochs', type=int)
    parser.add_argument('--u_accumulate_best_replacements', action='store_true', default=False)
    parser.add_argument('--u_rand_update_pgd', action='store_true', default=False)
    parser.add_argument('--use_loss_smoothing', action='store_true', default=False)
    parser.add_argument('--attack_version', type=int)
    parser.add_argument('--z_learning_rate', type=float)
    parser.add_argument('--u_learning_rate', type=float)
    parser.add_argument('--smoothing_param', type=float)
    parser.add_argument('--vocab_to_use', type=int)
    parser.add_argument('--exact_matches', action='store_true', default=False)
    parser.add_argument('--vocab', action='store', help='Vocabulary file')

    opt = parser.parse_args()

    return opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_best_site(inputs, grads, vocab, indices, replace_tokens, tokens, distinct):

    """
    inputs: numpy array with indices (batch, max_len)
    grads: numpy array (batch, max_len, vocab_size)
    vocab: Vocab object
    indices: numpy array of size batch
    replace_tokens: tokens representing sites
    tokens: tokens to replace the site
    """
    token_indices = [vocab.stoi[token] for token in tokens if vocab.stoi[token] != 0]
    #token_ind = {tok:vocab.stoi[tok] for tok in tokens}
    #print('tokens: ', token_ind)
    if token_indices == []:
        # none of the tokens are in the input vocab
        return get_random_site(inputs, vocab, indices, replace_tokens, tokens, distinct)
    replacements = {}
    for i in range(inputs.shape[0]):
        inp = inputs[i] # shape (max_len, )
        gradients = grads[i] # shape (max_len, vocab_size)
        index = str(indices[i])
        max_grad = None
        best_site = None
        sites = {}
        for repl_token in replace_tokens:
            repl_token_idx = vocab.stoi[repl_token]

            if repl_token_idx not in inp:
                continue
            # if repl_token_idx==0:
            #     sites[repl_token] = ''
            #     continue
            
            idx = inp.tolist().index(repl_token_idx)
            avg_grad = 0
            for t in token_indices:
                avg_grad += gradients[idx][t]
            avg_grad /= len(token_indices)
            if max_grad == None or avg_grad > max_grad:
                max_grad = avg_grad
                best_site = repl_token
            sites[repl_token] = ''

        sites[best_site] = ' '.join(tokens)
        replacements[index] = sites
    return replacements


def get_random_site(inputs, vocab, indices, replace_tokens, tokens, distinct):

    """
    Choose a site at random to be replaced with token.
    """
    replacements = {}
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        index = str(indices[i])
        sites = {}
        for repl_token in replace_tokens:
            repl_token_idx = vocab.stoi[repl_token]
            if repl_token_idx in inp:
                sites[repl_token] = ''
        best_site = random.choice(list(sites.keys()))
        sites[best_site] = ' '.join(tokens)
        replacements[index] = sites
    return replacements

def get_best_token_replacement(inputs, grads, vocab, indices, replace_tokens, distinct):
    '''
    inputs is numpy array with indices (batch, max_len)
    grads is numpy array (batch, max_len, vocab_size)
    vocab is Vocab object
    indices is numpy array of size batch
    '''
    
    best_replacements = {}    
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        gradients = grads[i]
        index = str(indices[i])
        d = {}              
        for repl_tok in replace_tokens:
            repl_tok_idx = vocab.stoi[repl_tok]
            if repl_tok_idx not in inp:
                continue
                
            mask = inp==repl_tok_idx

            # Is mean the right thing to do here? 
            avg_tok_grads = np.mean(gradients[mask], axis=0)

            exclude = list(d.values()) if distinct else []
            
            max_idx = np.argmax(avg_tok_grads)
            if not valid_replacement(vocab.itos[max_idx], exclude=exclude):
                idxs = np.argsort(avg_tok_grads)[::-1]
                for idx in idxs:
                    if valid_replacement(vocab.itos[idx], exclude=exclude):
                        max_idx = idx
                        break
            d[repl_tok] = vocab.itos[max_idx]

        if len(d)>0:
            best_replacements[index] = d
    
    return best_replacements

def apply_gradient_attack(data, model, input_vocab, replace_tokens, field_name, opt, output_vocab=None):
    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=opt.batch_size,
        sort=True, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False
        )
    batch_generator = batch_iterator.__iter__()

    weight = torch.ones(len(output_vocab.vocab)).half()
    pad = output_vocab.vocab.stoi[output_vocab.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()
    model.train()

    d = {}

    for batch in tqdm.tqdm(batch_generator, total=len(batch_iterator)):
#       print('batch attr: ', batch.__dict__.keys())
        indices = getattr(batch, 'index')
        input_variables, input_lengths = getattr(batch, field_name)
        target_variables = getattr(batch, 'tgt')

        # Do random attack if inputs are too long and will OOM under gradient attack
        if max(getattr(batch, field_name)[1]) > 250:
            rand_replacements = get_random_token_replacement(
                    input_variables.cpu().numpy(),
                    input_vocab,
                    indices.cpu().numpy(),
                    replace_tokens,
                    opt.distinct
            )
            d.update(rand_replacements)
            continue

        # convert input_variables to one_hot
        input_onehot = Variable(convert_to_onehot(input_variables, vocab_size=len(input_vocab), device=device), requires_grad=True).half()
      
        # Forward propagation       
        decoder_outputs, decoder_hidden, other = model(input_onehot, input_lengths, target_variables, already_one_hot=True)

        # print outputs for debugging
        # for i,output_seq_len in enumerate(other['length']):
        #   print(i,output_seq_len)
        #   tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
        #   tgt_seq = [output_vocab.itos[tok] for tok in tgt_id_seq]
        #   print(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]), end=', ')
        #   gt = [output_vocab.itos[tok] for tok in target_variables[i]]
        #   print(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))
        
        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variables.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])
            # Backward propagation
        model.zero_grad()
        input_onehot.retain_grad()
        loss.backward(retain_graph=True)
        grads = input_onehot.grad
        del input_onehot
        best_replacements = get_best_token_replacement(input_variables.cpu().numpy(), grads.cpu().numpy(), input_vocab, indices.cpu().numpy(), replace_tokens, opt.distinct)
        d.update(best_replacements)

    return d


def apply_gradient_attack_v2(data, model, token_to_id, id_to_token, replace_tokens, field_name, opt, orig_tok_map, idx_to_fname, label_to_id, id_to_label, n_samples, device='cpu'):
    ########################################
    # Parameters that ideally need to come in from opt
    use_orig_tokens = True
    n_alt_iters = opt.n_alt_iters
    n_alt_iters = 2*n_alt_iters

    z_optim = opt.z_optim
    z_epsilon = opt.z_epsilon
    z_init = opt.z_init # 0: all sites are picked; 1: 1 rand site is picked; 2: epsilon sites are picked.; >= 3, say x: (x-1) sites are picked
    z_step = 1
    
    u_optim = opt.u_optim
    u_pgd_epochs = opt.n_alt_iters
    u_rand_update_pgd = opt.u_rand_update_pgd # Optimal site is randomly selected instead of argmax
    u_accumulate_best_replacements = opt.u_accumulate_best_replacements
    u_projection = 2 # 1: simple 0, 1 projection; 2: simplex projection

    li_u_optim_technique = [1] # 1: PGD: SGD with relaxation; 2: signed gradient
    li_u_init_pgd = [3] #list(range(5)) # 0: Original (fixed) init; 1: randomly initalize all tokens; 2: pick PGD optimal randomly instead of argmax; >2: randomly initialize only z=true; 
    li_learning_rate = [1]
    li_use_u_discrete = [True]
    li_use_loss_smoothing = [opt.use_loss_smoothing]
    smooth_iters = 10
    smoothing_param = opt.smoothing_param

    vocab_to_use = opt.vocab_to_use
    use_cw_loss = False
    choose_best_loss_among_iters = True

    analyze_exact_match_sample = False
    samples_to_analyze = 1
    zlen_debug = 4
    plt_fname = '/mnt/outputs/loss_batch.pkl'
    outpth = '/mnt/outputs/'

    stats = {}
    
    config_dict = OrderedDict([
        ('version', 'v2'),
        ('n_alt_iters', n_alt_iters),
        ('z_optim', z_optim),
        ('z_epsilon', z_epsilon),
        ('z_init', z_init),
        ('u_optim', u_optim),
        ('u_pgd_epochs', u_pgd_epochs),
        ('u_accumulate_best_replacements', u_accumulate_best_replacements),
        ('u_rand_update_pgd', u_rand_update_pgd),
        ('smooth_iters', smooth_iters),
        ('use_cw_loss', use_cw_loss),
        ('choose_best_loss_among_iters', choose_best_loss_among_iters),
        ('analyze_exact_match_sample', analyze_exact_match_sample),
        ('use_orig_tokens', use_orig_tokens),
    ])

    stats['config_dict'] = config_dict

    ########################################
    
    # This datastructure is meant to return best replacements only for *one* set of best params
    # If using in experiment mode (i.e. itertools.product has mutliple combinations), don't expect consistent
    # results from best_replacements_dataset
    best_replacements_dataset = {}

    def print_flat_tokens(tokens, vocab):
        print([vocab[l] for l in tokens])

    def get_tok_list(tokens, vocab):
        tokens1 = tokens.permute(1,0)
        for i in range(tokens1.shape[0]):
            print([vocab[l] for l in tokens1[i].cpu().numpy().tolist()])

    def element_wise_pad_flat(tokens, contexts_per_label, tok_map):
        start_tokens = []
        start_idx = 0
        for i in range(len(contexts_per_label)):
            end_idx = start_idx + contexts_per_label[i]
            start_tokens.append(tokens[:, start_idx:end_idx])
            start_idx = end_idx

        # print(len(start_tokens))
        # start_tokens_flat, start_tokens_lens = remove_padding_and_flatten(start_tokens, tok_map)
        # print(len(start_tokens_flat))
        return start_tokens

    def get_context_batch_i(batch, i, context_lens):
        d = {}
        if i == 0:
            st = 0
        else:
            st = sum(context_lens[:i])
        
        en = st + context_lens[i]
        # print(context_lens)
        # print(st, en)
        d['from_token'] = batch['from_token'][:, st:en]
        d['to_token'] = batch['to_token'][:, st:en]
        d['path_types'] = batch['path_types'][:, st:en]
        return d

    def get_input_oho(contexts_oho, contexts_per_label, i):
        """
        Concat the from_tokens and to_tokens one-hot reps 
        for the ith context in the batch.
        """
        start_idx = sum(contexts_per_label[:i])
        n_paths = contexts_per_label[i]
        from_toks = contexts_oho['from_token'][:, start_idx:start_idx+n_paths]
        to_toks = contexts_oho['to_token'][:, start_idx:start_idx+n_paths]
        input_oho = torch.cat((from_toks, to_toks), 1)
        return input_oho

    def update_context_oho(context_oho, contexts_per_label, i, input_oho):
        """
        Change the one-hot of the ith context to input_oho.
        """
        start_idx = sum(contexts_per_label[:i])
        n_paths = contexts_per_label[i]
        from_toks, to_toks = input_oho[:,:n_paths], input_oho[:,n_paths:]
        context_oho['from_token'] = context_oho['from_token'].clone()
        context_oho['to_token'] = context_oho['to_token'].clone()
        context_oho['from_token'][:, start_idx:start_idx+n_paths] = from_toks
        context_oho['to_token'][:, start_idx:start_idx+n_paths] = to_toks
        return context_oho

    def concat_from_to_tokens(context, contexts_per_label):
        new_input = []
        for i in range(len(contexts_per_label)):
            new_input.append(get_input_oho(context, contexts_per_label, i))
        return new_input

    def get_context_oho(contexts, token_to_id):
        context_oho = {"from_token":torch.nn.functional.one_hot(contexts['from_token'], num_classes=len(token_to_id)).float(),
                            "to_token":torch.nn.functional.one_hot(contexts['from_token'], num_classes=len(token_to_id)).float(),
                            "path_types":path_types}
        context_oho['from_token'].requires_grad=True
        context_oho['to_token'].requires_grad=True
        return context_oho

    for params in itertools.product(li_u_optim_technique, li_u_init_pgd, li_learning_rate, li_use_loss_smoothing, li_use_u_discrete):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(config_dict)
        (u_optim_technique, u_init_pgd, learning_rate, use_loss_smoothing, use_u_discrete) = params
        od = OrderedDict([
            ('u_optim_technique', u_optim_technique),
            ('u_init_pgd', u_init_pgd),
            ('learning_rate', learning_rate),
            ('use_loss_smoothing', use_loss_smoothing),
            ('use_u_discrete', use_u_discrete),
        ])
        pp.pprint(od)
        stats['config_dict2'] = od
        
        batch_generator = iter(data)
        if use_cw_loss:
            loss_obj = AttackLoss(device=device)
        else:
            weight = torch.ones(len(label_to_id))
            pad = token_to_id["<PAD>"]
            loss_obj = Perplexity(weight, pad)
            if torch.cuda.is_available():
                loss_obj.cuda()
        model.train()
        
        nothing_to_attack, rand_replacement_too_long, tot_attacks, tot_samples = 0, 0, 1, 0
        sample_to_select_idx, pred_to_select, sample_to_select_idx_cnt, sname = None, None, 0, ''
        all_fnames = []

        # a mask of length len(id_to_token) which lists which are valid/invalid tokens
        if vocab_to_use == 1:
            invalid_tokens_mask = get_valid_token_mask(True, id_to_token, [])
        elif vocab_to_use == 2:
            invalid_tokens_mask = [False]*len(id_to_token) 

        for bid, batch in enumerate(tqdm.tqdm(batch_generator, total=math.ceil(n_samples/opt.batch_size))):
            found_sample, zlen, plen, zstr = False, 0, 0, None
            # max name parts x total paths
            from_tokens = batch.context['from_token']
            to_tokens = batch.context['to_token']
            path_types = batch.context['path_types']
            context = batch.context
            # labels: max label length x batch size
            labels = batch.labels
            contexts_per_label = batch.contexts_per_label
            filenames = batch.filenames
            batch_sz = len(filenames)
            # print(path_types.shape, len(filenames), len(contexts_per_label), labels.shape)
            all_fnames += filenames

            # from_token, to_token in context_oho: max name parts x total paths x |V|
            context_oho = get_context_oho(context, token_to_id)

            from_tokens_list= element_wise_pad_flat(from_tokens, contexts_per_label, token_to_id)
            to_tokens_list = element_wise_pad_flat(to_tokens, contexts_per_label, token_to_id)
            toks = []
            best_replacements_batch, best_losses_batch, continue_z_optim = {}, {}, {}

            # too update replacement-variables with max-idx in case this is the iter with the best optimized loss
            update_this_iter = False
            
            ###############################################
            # Initialize z for the batch
            # all_toks_i: max name parts x 2*(# of paths for i)
            # site_map: dict; replace token index --> mask showing the occurence of the replace token 
            # status: bool; shows if this sample has any replace toks
            # site_map_lookup: list of vocab indices of all replace tokens in this sample
            # z_np: numpy array with length len(site_map_lookup); z[i] is 1 or 0 - site chosen for optim or not
            status_map, z_map, z_all_map, z_np_map, site_map_map, site_map_lookup_map, z_initialized_map, invalid_tokens_mask_map, input_var_map, context_map = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            for ii in range(batch_sz):
                # all_toks_i = from_tokens_list[ii][:]
                # all_toks_i.concat(to_tokens_list[ii][:])
                all_toks_i = torch.cat((from_tokens_list[ii], to_tokens_list[ii]), 1)
                site_map, status = get_all_replacement_toks(np.array(all_toks_i), token_to_id, replace_tokens)
                if not status:
                    continue
                site_map_lookup = []
                for cnt, k in enumerate(site_map.keys()):
                    site_map_lookup.append(k)
                
                if z_epsilon == 0: # select all sites
                    z_np = np.ones(len(site_map_lookup)).astype(float)
                elif z_epsilon > 0 : # select z_epsilon sites
                    # defaults to a random 0-1 distribution
                    rdm_idx_list = list(range(len(site_map_lookup)))
                    if z_epsilon == 1:
                        rdm_idx = 0
                    else:
                        rdm_idx = random.sample(rdm_idx_list, min(len(rdm_idx_list), z_epsilon))
                    z_np = np.zeros(len(site_map_lookup)).astype(float)
                    z_np[rdm_idx] = 1
                z = torch.tensor(z_np, requires_grad=True, device=device)
                if len(z.shape) == 1:
                    z = z.unsqueeze(dim=1)
                
                mask = np.zeros(all_toks_i.shape).astype(bool)
                for kk in range(len(site_map_lookup)):
                    if not z[kk]:
                        continue
                    m = site_map[site_map_lookup[kk]]
                    mask = np.array(m) | mask

                # inputs_torch = torch.as_tensor(np.array(all_toks), device=device)
                # inputs_torch = inputs_torch.unsqueeze(dim=1)
                # inputs_oho = Variable(convert_to_onehot(inputs_torch, vocab_size=len(token_to_id), device=device), requires_grad=True)
                # dim inputs_oho: num_tokens x batch sz x vocab sz
                # input_var_map[ii] = inputs_oho.squeeze()
                status_map[ii] = status
                z_map[ii] = z
                z_np_map[ii] = z_np
                z_all_map[ii] = list(mask)
                site_map_map[ii] = site_map
                site_map_lookup_map[ii] = site_map_lookup
                best_replacements_batch[filenames[ii]] = {}
                best_losses_batch[filenames[ii]] = None
                continue_z_optim[filenames[ii]] = True
                context_i = get_context_batch_i(batch.context, ii, contexts_per_label)
                context_map[ii] = context_i

            if (u_optim or z_optim) and use_orig_tokens:
                new_context, site_map_map, z_all_map = replace_toks_batch(context, filenames, contexts_per_label, z_map, site_map_map, site_map_lookup_map, best_replacements_batch, token_to_id, id_to_token, orig_tok_map, device)
                context_oho = get_context_oho(new_context, token_to_id)


            ##################################################
            u_has_been_init = False
            for alt_iters in range(n_alt_iters):
                batch_loss_list_per_iter = []
                best_loss_among_iters, best_replace_among_iters = {}, {}
                
                # Iterative optimization
                if u_optim and alt_iters%2 == 0:
                    # Updates x based on the latest z
                    if analyze_exact_match_sample:
                        print('u-step')
                    # If current site has not been initialized, then initialize it with u_init for PGD
                    if not u_has_been_init:
                        for i in range(batch_sz):
                            if i not in status_map:
                                continue
                            fn_name = str(filenames[i])
                            # input_hot = input_var_map[i].detach().cpu().numpy()
                            input_hot = get_input_oho(context_oho, contexts_per_label, i).detach().cpu().numpy()

                            # Ensure the replacements for the sample are unique and have not already been picked
                            # during another z-site's optimization
                            
                            for z in range(z_np_map[i].shape[0]):
                                if z_np_map[i][z] == 0:
                                    continue
                                
                                # Make input_oho[i] zero for tokens which correspond to
                                # - sites z_i = True
                                # - and haven't been initialized before
                                mask = site_map_map[i][site_map_lookup_map[i][z]]
                                if u_init_pgd == 1:
                                    input_h = input_hot[mask,:][0,:]
                                elif u_init_pgd == 2:
                                    input_h = np.zeros(input_hot[mask,:][0,:].shape)
                                elif u_init_pgd == 3:
                                    valid_tokens_i = [not t for t in invalid_tokens_mask]
                                    # print(mask.any())
                                    input_h = input_hot[mask,:][0,:]
                                    input_h[valid_tokens_i] = 1/sum(valid_tokens_i)
                                    input_h[invalid_tokens_mask] = 0
                                elif u_init_pgd == 4:
                                    input_h = (1 - input_hot[mask,:][0,:])/(len(invalid_tokens_mask)-1)
                                input_hot[mask,:] = input_h
                            # input_var_map[i] = torch.tensor(input_hot, requires_grad=True, device=device)
                            context_oho = update_context_oho(context_oho, contexts_per_label, i, torch.tensor(input_hot, requires_grad=True, device=device))
                        u_has_been_init = True

                    for j in range(u_pgd_epochs):
                        # Forward propagation   
                        # decoder_output: torch.Size([max_len, batch_sz, vocab_sz])
                        if use_u_discrete:
                            decoder_output = model(batch.context, contexts_per_label, labels.shape[0], labels, use_embedding_layer=False, already_one_hot=False)
                        else:
                            decoder_output = model(context_oho, contexts_per_label, labels.shape[0], labels, use_embedding_layer=False, already_one_hot=True)

                        loss, l_scalar, token_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_output.to(device), labels.to(device))

                        if analyze_exact_match_sample: # sample_to_select_idx is not None at this stage
                            batch_loss_list_per_iter.append(token_wise_loss_per_batch[sample_to_select_idx])
        
                        for dxs in range(len(filenames)):
                            fname = filenames[dxs]
                            if fname not in best_loss_among_iters:
                                best_loss_among_iters[fname] = [token_wise_loss_per_batch[dxs]]
                            else:
                                best_loss_among_iters[fname].append(token_wise_loss_per_batch[dxs])
                            
                        # Forward propagation   
                        # Calculate loss on the continuous value vectors

                        decoder_output = model(context_oho, contexts_per_label, labels.shape[0], labels, use_embedding_layer=False, already_one_hot=True)
                        loss, l_scalar, token_wise_loss_per_batch = calculate_loss(use_cw_loss, loss_obj, decoder_output.to(device), labels.to(device))
                        
                        # update loss and backprop
                        from_token_oho, to_token_oho = context_oho['from_token'], context_oho['to_token']
                        model.zero_grad()
                        from_token_oho.retain_grad()
                        to_token_oho.retain_grad()
                        loss.backward(retain_graph=True)
                        grads_oh = {'from_token':from_token_oho.grad, 'to_token':to_token_oho.grad}

                        if use_loss_smoothing:
                            b_loss, smooth_grads_oh = [], None
                            for si in range(smooth_iters):
                                smooth_input = inputs_oho + smoothing_param * torch.empty(inputs_oho.shape, device=device).normal_(mean=0,std=1).half()
                                smooth_decoder_outputs, smooth_decoder_hidden, smooth_other = model(smooth_input, input_lengths, target_variables, already_one_hot=True)
                                if use_cw_loss:
                                    loss, token_wise_loss_per_batch = loss_obj.get_loss(smooth_other['logits'], target_variables)
                                else:
                                    loss_obj.reset()
                                    token_wise_loss_per_batch = None
                                    for step, step_output in enumerate(smooth_decoder_outputs):
                                        batch_size = target_variables.size(0)
                                        l = torch.nn.NLLLoss(reduction='none')(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1]).unsqueeze(dim=1)
                                        # dim of l: batch_sz x token_i of output
                                        if token_wise_loss_per_batch is None:
                                            token_wise_loss_per_batch = l
                                        else:
                                            token_wise_loss_per_batch = torch.cat((token_wise_loss_per_batch, l), 1)
                                        loss_obj.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])

                                    # dim of token_wise_loss_per_batch = batch_sz x 1 
                                    token_wise_loss_per_batch = torch.mean(token_wise_loss_per_batch, dim=1).detach().cpu().numpy()
                                    
                                    if analyze_exact_match_sample: # sample_to_select_idx is not None at this stage
                                        b_loss.append(token_wise_loss_per_batch[sample_to_select_idx])
                                    else:
                                        b_loss.append(token_wise_loss_per_batch)
                                    loss = loss_obj

                                # update loss and backprop
                                model.zero_grad()
                                smooth_input.retain_grad()
                                loss.backward(retain_graph=True)
                                if smooth_grads_oh is None:
                                    smooth_grads_oh = smooth_input.grad
                                else:
                                    smooth_grads_oh += smooth_input.grad
                            
                            grads_oh = smooth_grads_oh/smooth_iters

    
                        for i in range(batch_sz):
                            if analyze_exact_match_sample and i != sample_to_select_idx:
                                continue
                            
                            # additional_check = False
                            # if additional_check:
                            #     tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
                            #     tgt_seq = [output_vocab.vocab.itos[tok] for tok in tgt_id_seq]
                            #     output_seqs.append(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]))
                            #     assert output_seqs == pred_to_select

                            filename = filenames[i]
                        
                            input_hot = get_input_oho(context_oho, contexts_per_label, i).detach().cpu().numpy()
                            optim_input = None
                            best_replacements_sample = {} # Map per sample
                            gradients = get_input_oho(grads_oh, contexts_per_label, i).cpu().numpy()

                            # This does not get updated across PGD iters
                            # Gets updated only across alt-iters so that a newly found z-map can avoid
                            # reusing replacements that have been found in previous iters
                            
                            if i not in status_map:
                                if alt_iters == 0 and j == 0:
                                    nothing_to_attack += 1
                                continue

                            if alt_iters == 0 and j == 0:
                                tot_attacks += 1

                            if analyze_exact_match_sample and j == 0:
                                if alt_iters == 0:
                                    sample_to_select_idx_cnt += 1
                                    sname = filename
                                    found_sample = True
                                    print('found {}; z len {}'.format(sname, len(z_np_map[i])))
                                    # print([input_vocab.itos[t] for t in new_inputs[i]])
                                    # print([input_vocab.itos[t] for t in input_variables[i]])

                                zlen = sum(z_all_map[i])
                                plen = len(z_all_map[i])
                                zstr = str(alt_iters) +"::"+ str(z_np_map[i])
                                print(zstr)
                            
                            site_map_lookup = site_map_lookup_map[i]
                            z = z_map[i]
                            z_np = z_np_map[i]
                            site_map = site_map_map[i]
                            invalid_tokens_mask_i = invalid_tokens_mask[:]
                            # print('sample {}'.format(i))
                            # Fixed z, optimize u
                            # Apply a map such that z=1 sites are selected
                            # Apply gradient-based token replacement on these sites
                            for idx in range(z_np.shape[0]):
                                if z_np[idx] == 0:
                                    continue
                                mask = site_map[site_map_lookup[idx]]
                                # Can take a mean across all tokens for which z=1
                                # Currently, this mean is for all tokens for which z_i=1
                                avg_tok_grads = np.mean(gradients[mask], axis=0)
                                repl_tok_idx = site_map_lookup[idx]
                                # print(repl_tok_idx)
                                repl_tok = id_to_token[repl_tok_idx]
                                # print("repl tok: {}".format(repl_tok))
                                nabla = avg_tok_grads
                                
                                if u_optim_technique == 2:
                                    nabla = np.sign(nabla)

                                # PGD
                                step = learning_rate/np.sqrt(j+1) * nabla
                                if use_cw_loss:
                                    step = -1 * step
                                
                                # any one entry of the masked entries
                                # initalize to 0s for first entry
                                input_h = input_hot[mask,:][0,:]
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
                                    mu_opt = bisection(fmu, -1, 1, 20)
                                    if mu_opt is None:
                                        mu_opt = 0 # assigning randomly to 0
                                    optim_input = np.maximum(0, input_h - mu_opt)
                                    # print(fmu(mu_opt))

                                # projection onto only valid tokens. Rest are set to 0
                                optim_input[invalid_tokens_mask_i] = 0
                                # print(sum(invalid_tokens_mask_map))

                                if u_rand_update_pgd:
                                    max_idx = random.randrange(optim_input.shape[0])
                                else:
                                    max_idx = np.argmax(optim_input)
                                
                                # This ds is reset in every PGD iter. 
                                # This is for the current PGD iter across z sites.
                                best_replacements_sample[repl_tok] = id_to_token[max_idx]
                                
                                # Ensure other z's for this index don't use this replacement token
                                invalid_tokens_mask_i[max_idx] = True # setting it as invalid being True
                                
                                # Update optim_input
                                input_hot[mask,:] = optim_input
                            
                            # inputs_oho[i] = torch.tensor(input_hot, requires_grad=True, device=device)
                            context_oho = update_context_oho(context_oho, contexts_per_label, i, torch.tensor(input_hot, requires_grad=True, device=device))

                            # Done optimizing
                            if filename not in best_replace_among_iters:
                                best_replace_among_iters[filename] = [best_replacements_sample]
                            else:
                                best_replace_among_iters[filename].append(best_replacements_sample)

                    if analyze_exact_match_sample:
                        print(batch_loss_list_per_iter)
                        if found_sample:
                            if len(batch_loss_list_per_iter) > 0:
                                out_str = 'ss{}_zlen-{}_n-{}_zstr-{}_opt-{}_lr-{}_uinit-{}_smooth-{}_udisc-{}'.format(sname, zlen, plen, zstr, u_optim_technique, learning_rate, u_init_pgd, int(use_loss_smoothing), int(use_u_discrete))
                                print(out_str)
                                loss_plot(batch_loss_list_per_iter, os.path.join(outpth, out_str))

                        print(best_replace_among_iters)
                        print(best_loss_among_iters)
                        print('****')

                elif z_optim and alt_iters%2 == 1 and z_step == 1:
                    if analyze_exact_match_sample:
                        print('z-step')

                    # Mask current replaced tokens with a zero vector
                    # find best sites and sort greedily to get top-k
                    for i in range(batch_sz):
                        if i not in status_map:
                            continue
                        
                        if analyze_exact_match_sample and i != sample_to_select_idx:
                            continue
                        
                        fname = filenames[i]

                        if  not u_accumulate_best_replacements and not continue_z_optim[fname]:
                            if analyze_exact_match_sample:
                                print('not optimizing z ..')
                            continue

                        # inputs_oho is the latest updated input from the u step
                        # for each token in the current z map, replace it with a zero vector
                        # run the forward pass of the model, and pick the most sensitive z sites
                        z_losses, token_losses = [], []
                        
                        for j in range(z_np_map[i].shape[0]):
                            mask = site_map_map[i][site_map_lookup_map[i][j]]
                            inputs_oho_i = get_input_oho(context_oho, contexts_per_label, i).clone()
                            inputs_oho_i[mask,:] = torch.zeros(inputs_oho_i[mask,:].shape, requires_grad=True)
                            context_i = get_context_batch_i(context_oho, i, contexts_per_label)
                            n_paths = contexts_per_label[i]
                            context_i['from_token'], context_i['to_token'] = inputs_oho_i[:, :n_paths], inputs_oho_i[:, n_paths:]
                            decoder_output = model(context_i, [contexts_per_label[i]], labels.shape[0], labels[:,i:i+1], use_embedding_layer=False, already_one_hot=True)
                            loss, l_scalar, token_wise_loss = calculate_loss(use_cw_loss, loss_obj, decoder_output.to(device), labels[:, i:i+1].to(device))
                            z_losses.append(l_scalar)
                            token_losses.append(token_wise_loss)
                        
                        # Sorts by highest loss first
                        loss_order = np.argsort(np.array(z_losses))[::-1]
                        '''
                        if i == sample_to_select_idx:
                            print(z_losses)
                            print(loss_order)
                            print(token_losses)
                        '''
                        if z_epsilon == 0:
                            toselect = len(z_losses)
                        elif z_epsilon > 0:
                            toselect = z_epsilon
                    
                        idxs = loss_order[:toselect]
                        notidxs = loss_order[toselect:]
                        z_np_map[i][idxs] = 1
                        z_np_map[i][notidxs] = 0

                        #if z_np_map[i].shape[0] > 2:
                        #   print(z_np_map[i])
                        #   print('----')
                    
                        if analyze_exact_match_sample:
                            print(z_np_map[i])
                            print('****')
                    
                    if not u_accumulate_best_replacements:
                        new_context, site_map_map, z_all_map = replace_toks_batch(context, filenames, contexts_per_label, z_map, site_map_map, site_map_lookup_map, best_replacements_batch, token_to_id, id_to_token, orig_tok_map, device)
                        context_oho = get_context_oho(new_context, token_to_id)

                # Choose the best loss from u optim
                if u_optim and alt_iters%2 == 0:
                    for i in range(batch_sz):
                        if i not in status_map:
                            continue

                        if analyze_exact_match_sample and i != sample_to_select_idx:
                            continue
                        fname = filenames[i]
                        best_idx, best_loss_u = max(enumerate(best_loss_among_iters[fname]), key=lambda x: x[1])
                        best_replace = best_replace_among_iters[fname][best_idx]
                        
                        if best_losses_batch[fname] is None or best_loss_u > best_losses_batch[fname]:
                            best_losses_batch[fname] = best_loss_u
                            if not u_accumulate_best_replacements:
                                best_replacements_batch[fname] = best_replace   
                            else:
                                best_replacements_batch[fname].update(best_replace)
                        else:
                            continue_z_optim[fname] = False

                best_replacements_dataset.update(best_replacements_batch)

                if analyze_exact_match_sample:
                    print(best_replacements_batch)
                    print(best_losses_batch)
                    print(best_replacements_dataset)
                    print('-----')

        # print('Nothing to attack: {}/{} ({})'.format(nothing_to_attack, tot_attacks, round(100*nothing_to_attack/tot_attacks, 2)))
        # print('----------------')

        # stats['nothing_to_attack_pc'] = round(100*nothing_to_attack/tot_attacks, 2)

    if analyze_exact_match_sample:
        kzs = best_replacements_dataset.keys()
        print(best_replacements_dataset)

    
    print("# of samples attacked: {}".format(len(best_replacements_dataset.keys())))
    stats['n_samples_attacked'] = len(best_replacements_dataset.keys())
    best_replacements_dataset, avg_replaced = get_all_replacements(best_replacements_dataset, orig_tok_map, all_fnames, True)
    print('\n# tokens optimized on an average: {}'.format(avg_replaced))
    stats['n_tokens_optimized_avg'] = avg_replaced
    print("\n# of samples attacked post processing: {}\n=======".format(len(best_replacements_dataset.keys())))
    stats['n_samples_attacked_post_processing'] = len(best_replacements_dataset.keys())



    if analyze_exact_match_sample:
        for kz in kzs:
            print("{}::{}".format(kz, best_replacements_dataset[kz]))

    return best_replacements_dataset, stats

def apply_random_attack(data, model, input_vocab, replace_tokens, field_name, opt):
    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=opt.batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False)
    batch_generator = batch_iterator.__iter__()

    d = {}

    for batch in tqdm.tqdm(batch_generator, total=len(batch_iterator)):
        indices = getattr(batch, 'index')
        input_variables, input_lengths = getattr(batch, field_name)
        target_variables = getattr(batch, 'tgt')
        rand_replacements = get_random_token_replacement(input_variables.cpu().numpy(),input_vocab, indices.cpu().numpy(), replace_tokens, opt.distinct)

        d.update(rand_replacements)

    return d

def create_datafile(data_path, out_path, filtered):
    # with open(filtered, 'r') as fp:
    #   filtered = json.load(fp)
    filtered = set(map(str, filtered))

    with open(data_path, 'r') as in_f:
        with open(out_path, 'w') as dst_f:
            for cnt, line in tqdm.tqdm(enumerate(in_f)):
                if cnt == 0:
                    dst_f.write(line) 
                else:
                    parts = line.strip().split('\t')
                    index = parts[0]
                    if index in filtered:
                        dst_f.write(line) 
                
    print('Done dumping reduced data set')
    return out_path


if __name__=="__main__":
    opt = parse_args()
    print(opt)
    print('data path: ', opt.data_path)
    data_split = opt.data_path.split('/')[-2]
    print('data_split', data_split)

    # replace_tokens = ["@R_%d@"%x for x in range(0,opt.num_replacements+1)]
    replace_tokens = ["@R_%d@"%x for x in range(1000)]
    
    model = Code2Seq.load_from_checkpoint(checkpoint_path=opt.expt_dir)
    data_loader, n_samples = create_dataloader(
        opt.data_path, model.hyperparams.max_context, False, False, opt.batch_size, 1,
    )
    print('total samples: ', n_samples)

    vocab = pickle.load(open(opt.vocab, 'rb'))
    token_to_id = vocab['token_to_id']
    id_to_token = {token_to_id[t]:t for t in token_to_id}
    print('length: ', len(id_to_token))
    label_to_id = vocab['label_to_id']
    id_to_label = {label_to_id[t]:t for t in label_to_id}


    if opt.random:
        rand_d = {}

        for field_name, _ in fields_inp:
            if field_name in ['src', 'tgt', 'index', 'transforms.Identity']:
                continue

            print('Random Attack', field_name)
            rand_d[field_name] = apply_random_attack(data, model, input_vocab, replace_tokens, field_name, opt)

        save_path = opt.save_path
        if save_path is None:
            fname = opt.data_path.replace('/', '|').replace('.','|') + "%s.json"%("-distinct" if opt.distinct else "")
            save_path = os.path.join(opt.expt_dir, fname)

        # Assuming save path ends with '.json'
        save_path = save_path[:-5] + '-random.json'
        json.dump(rand_d, open(save_path, 'w'), indent=4)
        print('  + Saved:', save_path)

    if  opt.attack_version == 1:
        attack_fname = apply_gradient_attack
    elif opt.attack_version == 2:
        attack_fname = apply_gradient_attack_v2
    elif  opt.attack_version == 3:
        attack_fname = apply_gradient_attack_v3

    if not opt.no_gradient:
        d = {}
        
        # for field_name, _ in fields_inp:
        field_name = 'transforms.Rename' # need to fix this
        # if field_name in ['src', 'tgt', 'index', 'transforms.Identity']:
        # 	continue

        print('Attacking using Gradient', field_name)
        
        # load original tokens that were replaced by replace tokens
        site_map_path = '/mnt/inputs/{}/{}_site_map.json'.format(field_name, data_split)
        with open(site_map_path, 'r') as f:
            orig_tok_map = json.load(f) # mapping of fnames to {replace_tokens:orig_tokens}
        
        # with open('/mnt/outputs/{}_idx_to_fname.json'.format(data_split), 'r') as f:
        # 	idx_to_fname = json.load(f) # mapping of file/sample index to file name
        idx_to_fname = None
        
        t_start = time.time()
        d[field_name], stats = attack_fname(data_loader, model, token_to_id, id_to_token, replace_tokens, field_name, opt, orig_tok_map, idx_to_fname, label_to_id, id_to_label, n_samples, device)
        t_elapsed = time.gmtime(time.time() - t_start)
        t_elapsed = time.strftime("%H:%M:%S", t_elapsed)
        stats['time_taken_to_attack(h:m:s)'] = t_elapsed

        if data_split == 'test':
            with open('/mnt/outputs/stats.json', 'w') as f:
                json.dump(stats, f)
            
        save_path = opt.save_path
        # Assuming save path ends with '.json'
        save_path = save_path[:-5] + '-gradient.json'
        json.dump(d, open(save_path, 'w'), indent=4)
        print('  + Saved:', save_path)