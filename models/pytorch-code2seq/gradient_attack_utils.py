import re
import numpy as np
import torch
import random
import tqdm
import copy

import torchtext
from torch.autograd import Variable

def classify_tok(tok):
    PY_KEYWORDS = re.compile(
      r'^(False|class|finally|is|return|None|continue|for|lambda|try|True|def|from|nonlocal|while|and|del|global|not|with|as|elif|if|or|yield|assert|else|import|pass|break|except|in|raise)$'
    )

    JAVA_KEYWORDS = re.compile(
      r'^(abstract|assert|boolean|break|byte|case|catch|char|class|continue|default|do|double|else|enum|exports|extends|final|finally|float|for|if|implements|import|instanceof|int|interface|long|module|native|new|package|private|protected|public|requires|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|void|volatile|while)$'
    )

    NUMBER = re.compile(
      r'^\d+(\.\d+)?$'
    )

    BRACKETS = re.compile(
      r'^(\{|\(|\[|\]|\)|\})$'
    )

    OPERATORS = re.compile(
      r'^(=|!=|<=|>=|<|>|\?|!|\*|\+|\*=|\+=|/|%|@|&|&&|\||\|\|)$'
    )

    PUNCTUATION = re.compile(
      r'^(;|:|\.|,)$'
    )

    WORDS = re.compile(
      r'^(\w+)$'
    )


    if PY_KEYWORDS.match(tok):
        return 'KEYWORD'
    elif JAVA_KEYWORDS.match(tok):
        return 'KEYWORD'
    elif NUMBER.match(tok):
        return 'NUMBER'
    elif BRACKETS.match(tok):
        return 'BRACKET'
    elif OPERATORS.match(tok):
        return 'OPERATOR'
    elif PUNCTUATION.match(tok):
        return 'PUNCTUATION'
    elif WORDS.match(tok):
        return 'WORDS'
    else:
        return 'OTHER'

def normalize_subtoken(subtoken):
    normalized = re.sub(
        r'[^\x00-\x7f]', r'',  # Get rid of non-ascii
        re.sub(
            r'["\',`]', r'',     # Get rid of quotes and comma 
            re.sub(
                r'\s+', r'',       # Get rid of spaces
                subtoken.lower()
                .replace('\\\n', '')
                .replace('\\\t', '')
                .replace('\\\r', '')
            )
        )
    )

    return normalized.strip()

def camel_case_split(identifier):
    matches = re.finditer(
    '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
    identifier
    )
    return [m.group(0) for m in matches]


def subtokens(in_list):
    good_list = []
    for tok in in_list:
        for subtok in tok.replace('_', ' ').split(' '):
            if subtok.strip() != '':
                good_list.extend(camel_case_split(subtok))

    return good_list

def tokenize(name):
    return [normalize_subtoken(subtok) for subtok in subtokens([name])]

def get_valid_token_mask(negation, id_to_token, exclude):
    mask_valid = []
    for i in range(len(id_to_token)):
        if negation:
            mask_valid.append(not valid_replacement(id_to_token[i], exclude=exclude))
        else:
            mask_valid.append(valid_replacement(id_to_token[i], exclude=exclude))
    return mask_valid

def valid_replacement(s, exclude=[]):
    return classify_tok(s)=='WORDS' and s not in exclude

def convert_to_onehot(inp, vocab_size, device):
    return torch.zeros(inp.size(0), inp.size(1), vocab_size, device=device).scatter_(2, inp.unsqueeze(2), 1.)

def get_all_replacement_toks(input_var, token_to_id, replace_tokens):
    site_map, status = {}, False
    input_var_flat = [input_var[i][j] for i in range(input_var.shape[0]) for j in range(input_var.shape[1])]
    for repl_tok in replace_tokens:
        if (repl_tok not in token_to_id) or (token_to_id[repl_tok] not in input_var_flat):
            continue
        repl_tok_idx = token_to_id[repl_tok]
        status = True
        mask = input_var==repl_tok_idx
        assert mask.shape == input_var.shape
        site_map[repl_tok_idx] = mask
    return site_map, status

def calculate_loss(use_cw_loss, loss_obj, decoder_outputs, target_variables):
    token_wise_loss_per_batch = None
    if use_cw_loss:
        loss, token_wise_loss_per_batch = loss_obj.get_loss(decoder_outputs, target_variables)
        l_scalar = loss
    else:
        loss_obj.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variables.size(1)
            l = torch.nn.NLLLoss(reduction='none')(step_output.contiguous().view(batch_size, -1), target_variables[step, :]).unsqueeze(dim=1)
            # dim of l: batch_sz x token_i of output
            if token_wise_loss_per_batch is None:
                token_wise_loss_per_batch = l
            else:
                token_wise_loss_per_batch = torch.cat((token_wise_loss_per_batch, l), 1)
            loss_obj.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[step, :])

        # dim of token_wise_loss_per_batch = batch_sz x 1 
        token_wise_loss_per_batch = torch.mean(token_wise_loss_per_batch, dim=1).detach().cpu().numpy()
        loss = loss_obj
        l_scalar = loss_obj.get_loss()                    
    
    return loss, l_scalar, token_wise_loss_per_batch

def pad_inputs(new_inputs, new_site_map_map, z_all_map, token_to_id, max_size, updated_lengths):
    """
    Pad new inputs, site maps, and z maps to length max_size.
    """

    res_inputs, res_site_map_map, res_z_all_map = [], {}, {}
    
    for i in range(len(new_inputs)):
        
        if len(new_inputs[i]) == max_size:
            res_inputs.append(new_inputs[i])
            if i in new_site_map_map:
                res_site_map_map[i] = new_site_map_map[i]
                res_z_all_map[i] = z_all_map[i]
        else:
            res_inputs.append(new_inputs[i]+[input_vocab.stoi['<pad>'] for j in range(max_size-len(new_inputs[i]))])
            if i in new_site_map_map:
                res_site_map_map[i] = {}
                for r in new_site_map_map[i]:
                    res_site_map_map[i][r] = new_site_map_map[i][r]+[False for j in range(max_size-len(new_inputs[i]))]
            if i in z_all_map:
                res_z_all_map[i] = z_all_map[i] + [False for j in range(max_size-len(new_inputs[i]))]
                
    return res_inputs, res_site_map_map, res_z_all_map

def pad_sample(input_var, pad_idx):
    """Pad each path in a sample to the max length"""
    # for p in input_var:
    #     print(p)
    max_path_len = max([len(p) for p in input_var])
    padded_input = []
    for p in input_var:
        padded_p = p + [pad_idx for i in range(max_path_len-len(p))]
        padded_input.append(padded_p)
    assert len(set([len(p) for p in padded_input])) == 1
    return np.array(padded_input)

def pad_batch(input_vars, pad_idx, batch_sz, max_len=None):
    max_name_len = max(input_vars[i].shape[1] for i in range(batch_sz)) if max_len is None else max_len
    new_inputs = {}
    for i in range(batch_sz):
        if i in input_vars:
            sample = input_vars[i]
            n_paths, path_len = sample.shape
            padding = np.full((n_paths, max_name_len-path_len), pad_idx)
            padded_sample = np.concatenate((sample, padding), 1)
            # padded_sample = np.array([np.concatenate((p, np.array([pad_idx for j in range(max_name_len-len(p))]))) for p in sample])
            new_inputs[i] = padded_sample
    if len(new_inputs) > 0:
        assert len(set([len(sample[0]) for sample in new_inputs.values()])) == 1
    return new_inputs, max_name_len

def pad_batch_site_map(site_map_map, pad_idx, batch_size, max_len):
    new_site_map_map = {}
    for i in range(batch_size):
        if i in site_map_map:
            new_site_map_map[i] = {}
            for site_idx in site_map_map[i]:
                sample = site_map_map[i][site_idx]
                n_paths, path_len = sample.shape
                padding = np.full((n_paths, max_len-path_len), pad_idx)
                padded_sample = np.concatenate((sample, padding), 1)
                new_site_map_map[i][site_idx] = padded_sample
    return new_site_map_map
    




def replace_toks_sample(input_var, z, site_map_lookup, best_replacements_sample, orig_replacements, id_to_token, token_to_id):

    """
    input_var: np array (max name parts, number of paths)
    z: list of 0 or 1 showing which sites are selected
    site_map: replace token --> mask of size (max name parts, number of paths)
    site_map_lookup: list of replace tokens in input_var, same len as z
    best_replacements_sample: replace token --> token to put at that site
    orig_replacements: replace_token --> original token at that site
    token_to_id, id_to_token: vocab

    If z[i]=1 replace the site z[i] with the best replacement token,
    otherwise with the original token. Works only for rename transforms.
    """

    input_var = input_var.tolist()
    # find replacement tokens for sites
    toks_to_be_replaced = {}
    for i in range(len(z)):
        repl_tok_idx = site_map_lookup[i]
        repl_tok = id_to_token[repl_tok_idx]
        if z[i] == 1 and repl_tok in best_replacements_sample:
            toks_to_be_replaced[repl_tok_idx] = [token_to_id[best_replacements_sample[repl_tok]]]
        else:
            toks_to_be_replaced[repl_tok_idx] = [token_to_id[tok] for tok in tokenize(orig_replacements[repl_tok][0]) if tok in token_to_id]
            if toks_to_be_replaced[repl_tok_idx] == []:
                toks_to_be_replaced[repl_tok_idx] = [token_to_id["<UNK>"]]
    
    pad_idx = token_to_id["<PAD>"]
    input_var_no_pad = []
    for p in input_var:
        end = None
        if pad_idx in p:
            end = p.index(pad_idx)
        input_var_no_pad.append(p[:end])


    # update input (replace @R tokens)
    new_input, new_input_li = [], []
    for path in input_var_no_pad:
        new_path, new_path_li = [], []
        for tok_idx in path:
            if tok_idx not in toks_to_be_replaced:
                new_path.append(tok_idx)
                new_path_li.append([tok_idx])
            else:
                new_path += toks_to_be_replaced[tok_idx]
                new_path_li.append(toks_to_be_replaced[tok_idx])
        new_input.append(new_path)
        new_input_li.append(new_path_li)

    # update site_map
    new_site_map = {}
    for r_idx in site_map_lookup:
        new_site_map_li = []
        for k in range(len(new_input_li)):
            path = new_input_li[k]
            path_map = []
            for i in range(len(path)):
                tok_idx = input_var[k][i]
                path_map += [tok_idx == r_idx for j in range(len(path[i]))]
            new_site_map_li.append(path_map)
        new_site_map[r_idx] = new_site_map_li
    

    new_input = pad_sample(new_input, pad_idx)
    for i in new_site_map:
        new_site_map[i] = pad_sample(new_site_map[i], False)
    if len(new_site_map) > 0:
        assert new_input.shape == list(new_site_map.values())[0].shape

    # if not all([new_site_map[i].any() for i in new_site_map]):
    #     print("found all 0 mask after pad")

    # update z_map
    mask = np.array(np.array(new_input)*[False]).astype(bool)
    for kk in range(len(site_map_lookup)):
        if not z[kk]:
            continue
        m = new_site_map[site_map_lookup[kk]]
        mask = np.array(m) | mask
    assert new_input.shape == mask.shape
    # if not mask.any():
    #     print("found all 0 mask")


    return new_input, new_site_map, mask

def replace_toks_batch(contexts, filenames, paths_per_label, z_map, site_map_map, site_map_lookup_map, best_replacements_batch, token_to_id, id_to_token, orig_tok_map, device):

    new_inputs, new_site_map_map, z_all_map = {}, {}, {}

    # replace the tokens in each sample
    batch_size = len(paths_per_label)
    for i in range(batch_size):
        fname = filenames[i]
        n_paths = paths_per_label[i]
        start_idx = sum(paths_per_label[:i])
        end_idx = start_idx + n_paths
        start_toks = contexts['from_token'][:, start_idx:end_idx]
        end_toks = contexts['to_token'][:, start_idx:end_idx]
        input_var = torch.cat((start_toks, end_toks), 1)
        input_var = input_var.permute(1,0).detach().cpu().numpy()
        if i in site_map_map:
            new_input, new_site_map, new_mask = replace_toks_sample(input_var, z_map[i], site_map_lookup_map[i], best_replacements_batch[fname], orig_tok_map[fname.split(".")[0]], id_to_token, token_to_id)
            new_inputs[i] = new_input
            new_site_map_map[i] = new_site_map
            z_all_map[i] = new_mask
        else:
            new_inputs[i] = copy.deepcopy(input_var)

    # pad all batch samples to the same length
    new_inputs, max_len = pad_batch(new_inputs, token_to_id["<PAD>"], batch_size, None)
    new_site_map_map = pad_batch_site_map(new_site_map_map, False, batch_size, max_len)
    z_all_map, _ = pad_batch(z_all_map, False, batch_size, max_len)

    # concat all samples
    from_token, to_token = None, None
    for i in range(batch_size):
        n_paths = paths_per_label[i]
        if i in new_site_map_map:
            z_all_map[i] = np.transpose(z_all_map[i])
            for site_idx in new_site_map_map[i]:
                new_site_map_map[i][site_idx] = np.transpose(new_site_map_map[i][site_idx])
        new_input_np = np.transpose(new_inputs[i])
        if from_token is None:
            from_token, to_token = new_input_np[:, :n_paths], new_input_np[:, n_paths:]
        else:
            from_token = np.concatenate((from_token, new_input_np[:, :n_paths]), 1)
            to_token = np.concatenate((to_token, new_input_np[:, n_paths:]), 1)
    from_token = torch.from_numpy(from_token)
    to_token = torch.from_numpy(to_token)
    new_contexts = {'from_token': from_token.to(torch.int64), 'to_token': to_token.to(torch.int64), 'path_types':contexts['path_types']}
    return new_contexts, new_site_map_map, z_all_map
    

def replace_toks_sample_2(input_var, z, site_map, site_map_lookup, best_replacements_sample, orig_replacements, id_to_token, token_to_id):

    """
    input_var: np array (max name parts, number of paths)
    z: list of 0 or 1 showing which sites are selected
    site_map: replace token --> mask of size (max name parts, number of paths)
    site_map_lookup: list of replace tokens in input_var, same len as z
    best_replacements_sample: replace token --> token to put at that site
    orig_replacements: replace_token --> original token at that site
    token_to_id, id_to_token: vocab

    If z[i]=1 replace the site z[i] with the best replacement token,
    otherwise with the original token. Works only for replace transforms.
    """
    max_len, n_paths = input_var.shape
    new_input = copy.deepcopy(input_var)
    for i in range(len(z)):
        repl_tok_idx = site_map_lookup[i]
        repl_tok = id_to_token[repl_tok_idx]
        mask = site_map[repl_tok_idx]
        if z[i] == 1 and repl_tok in best_replacements_sample:
            tok = best_replacements_sample[repl_tok]
        else:
            tok = orig_replacements[repl_tok][0]
        simple_tok = tok if '_' not in tok else tok.split('_')[0] # change this!!
        if simple_tok not in token_to_id:
            print(simple_tok, "not in vocab")
            continue
        new_input[mask] = token_to_id[simple_tok]
    return new_input

def replace_toks_batch_2(contexts, filenames, paths_per_label, z_map, site_map_map, site_map_lookup_map, best_replacements_batch, token_to_id, id_to_token, orig_tok_map, device):

    """
    If z[i]=1 replace the site z[i] with the best replacement token,
    otherwise with the original token. Works only for replace transforms.
    """

    batch_size = len(paths_per_label)
    from_token, to_token = None, None
    for i in range(batch_size):
        fname = filenames[i]
        n_paths = paths_per_label[i]
        start_idx = sum(paths_per_label[:i])
        end_idx = start_idx + n_paths
        start_toks = contexts['from_token'][:, start_idx:end_idx]
        end_toks = contexts['to_token'][:, start_idx:end_idx]
        input_var = torch.cat((start_toks, end_toks), 1)
        new_input = replace_toks_sample(input_var.detach().cpu().numpy(), z_map[i], site_map_map[i], site_map_lookup_map[i], best_replacements_batch[fname], orig_tok_map[fname.split(".")[0]], id_to_token, token_to_id)
        if from_token is None:
            from_token, to_token = torch.from_numpy(new_input[:, :n_paths]), torch.from_numpy(new_input[:, n_paths:])
        else:
            from_token = torch.cat((from_token, torch.from_numpy(new_input[:, :n_paths])), 1)
            to_token = torch.cat((to_token, torch.from_numpy(new_input[:, n_paths:])), 1)
    new_contexts = {'from_token': from_token.to(torch.int64), 'to_token': to_token.to(torch.int64), 'path_types':contexts['path_types']}
    return new_contexts

def modify_onehot(inputs_oho, site_map_map, sites_to_fix_map, device):

    for i in range(inputs_oho.shape[0]):

        if i in site_map_map:
            site_map = site_map_map[i]
            sites_to_fix = sites_to_fix_map[i]

            for site in sites_to_fix:
                mask = site_map[site]
                inputs_oho[i][mask] = torch.zeros(inputs_oho[i][mask].shape, requires_grad=True, device=device).half()

    return inputs_oho


def get_all_replacements(best_replacements, orig_tok_map, filenames, only_processed=False):
    """
    Creates a dictionary where optimized sites map to their best replacements
    and unoptimized ones map to their original tokens. This dictionary should be returned
    by apply_gradient_attack_v2 and should be used in replace_tokens.py
    """

    all_replacements = {}
    avg_replaced, tot_replaced = 0, 0

    for fname in filenames:

        # add optimized site replacements
        if fname in best_replacements:
            all_replacements[fname] = {site:best_replacements[fname][site] for site in best_replacements[fname]}
            avg_replaced += len(best_replacements[fname])
        else:
            all_replacements[fname] = {}

        # find keys in orig_tok_map[fname] that don't contain optimized R sites
        valid_keys = []
        for key in orig_tok_map[fname.split(".")[0]]:
            valid = True
            for repl_tok in all_replacements[fname]:
                if repl_tok in key:
                    valid = False
                    break
            if valid:
                valid_keys.append(key)

        # add unoptimized site replacements
        to_add = {s:' '.join(tokenize(orig_tok_map[fname.split(".")[0]][s][0])) for s in valid_keys}
        all_replacements[fname].update(to_add)
        tot_replaced += 1

    if tot_replaced == 0:
        avg_replaced = 0
    else:
        avg_replaced /= tot_replaced
    return all_replacements, avg_replaced

def remove_padding_and_flatten(inp_tokens, token_to_id):
    inp_tokens_flat, inp_tokens_lengths = [], []
    pad_idx = token_to_id['<PAD>']
    for sample in inp_tokens:
        flat_inp, inp_lens = [], []
        sample = sample.cpu().numpy()
        for col in range(sample.shape[1]):
            inp = sample[:,col].tolist()
            if pad_idx in inp:
                inp = inp[:inp.index(pad_idx)]
            flat_inp += inp
            inp_lens.append(len(inp))
        inp_tokens_flat.append(flat_inp)
        inp_tokens_lengths.append(inp_lens)
    return inp_tokens_flat, inp_tokens_lengths


def bisection(f,a,b,N):	
    # From https://www.math.ubc.ca/~pwalls/math-python/roots-optimization/bisection/	
    # '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.	
    #	
    # Parameters	
    # ----------	
    # f : function	
    #     The function for which we are trying to approximate a solution f(x)=0.	
    # a,b : numbers	
    #     The interval in which to search for a solution. The function returns	
    #     None if f(a)*f(b) >= 0 since a solution is not guaranteed.	
    # N : (positive) integer	
    #     The number of iterations to implement.	
    #	
    # Returns	
    # -------	
    # x_N : number	
    #     The midpoint of the Nth interval computed by the bisection method. The	
    #     initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some	
    #     midpoint m_n = (a_n + b_n)/2, then the function returns this solution.	
    #     If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any	
    #     iteration, the bisection method fails and return None.	
    while 1:
        try:
            if f(a)*f(b) >= 0:	
                a = a - 10	
                b = b + 10	
                # print("Bisection method fails.")	
                continue	
            else:	
                break
        except Exception as e:
            return None
    a_n = a	
    b_n = b	
    for n in range(1,N+1):	
        m_n = (a_n + b_n)/2	
        f_m_n = f(m_n)	
        if f(a_n)*f_m_n < 0:	
            a_n = a_n	
            b_n = m_n	
        elif f(b_n)*f_m_n < 0:	
            a_n = m_n	
            b_n = b_n	
        elif np.abs(f_m_n) <= 1e-5:	
            # print("Found exact solution.")	
            return m_n	
        else:	
            # print("Bisection method fails.")	
            return None	
    return (a_n + b_n)/2	


def get_random_token_replacement(inputs, vocab, indices, replace_tokens, distinct):
    '''
    inputs is numpy array with indices (batch, max_len)
    grads is numpy array (batch, max_len, vocab_size)
    vocab is Vocab object
    indices is numpy array of size batch
    '''
    rand_replacements = {}    
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        index = str(indices[i])
        
        d = {}      
        for repl_tok in replace_tokens:
            repl_tok_idx = vocab.stoi[repl_tok]
            if repl_tok_idx not in inp:
                continue
                
            exclude = list(d.values()) if distinct else []
            
            rand_idx = random.randint(0, len(vocab)-1)
            while not valid_replacement(vocab.itos[rand_idx], exclude=exclude):
                rand_idx = random.randint(0, len(vocab)-1)

            d[repl_tok] = vocab.itos[rand_idx]

        if len(d)>0:
            rand_replacements[index] = d
    
    return rand_replacements

def get_random_token_replacement_2(inputs, vocab, indices, replace_tokens, distinct, z_epsilon):

    rand_replacements = {}
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        index = str(indices[i])
        d = {}

        # find all replace tokens in input i
        replace_tokens_i = []
        for repl_tok in replace_tokens:
            repl_tok_idx = vocab.stoi[repl_tok]
            if repl_tok_idx in inp:
                replace_tokens_i.append(repl_tok)

        if z_epsilon == 0:
            sites_picked = len(replace_tokens_i)
        else:
            sites_picked = min(len(replace_tokens_i), z_epsilon)

        random_sites = random.sample(replace_tokens_i, sites_picked)

        # replace sites with random tokens
        for site in random_sites:
            exclude = list(d.values()) if distinct else []
            rand_idx = random.randint(0, len(vocab)-1)
            while not valid_replacement(vocab.itos[rand_idx], exclude=exclude):
                rand_idx = random.randint(0, len(vocab)-1)
            d[site] = vocab.itos[rand_idx]

        rand_replacements[index] = d

    return rand_replacements
            
            

def get_exact_matches(data_loader, n_samples, model, id_to_label):

    """
    Returns the filenames of samples whose predicted target sequence
    is equal to the actual target sequence.
    """
    batch_iterator = iter(data_loader)
    model.eval()
    exact_matches = []
    special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    errors = 0
    partial_matches = []

    for bid, batch in enumerate(tqdm.tqdm(batch_iterator, total=n_samples)):

        filenames = batch.filenames
        try:
            # outputs_oho shape: [target len; batch size; vocab size]
            outputs_oho = model(batch.context, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
        except:
            errors += 1
            continue

        for i, fname in enumerate(filenames):
            output_i = outputs_oho[:,i,:]
            label = batch.labels[:,i].cpu().numpy().tolist()
            pred_tgt_idxs = output_i.argmax(dim=-1).detach().numpy().tolist()
            pred_tgt_seq = [id_to_label[j] for j in pred_tgt_idxs]
            pred_tgt_seq = [tok for tok in pred_tgt_seq if tok not in special_tokens]
            ground_truth = [id_to_label[tok] for tok in label if id_to_label[tok] not in special_tokens]

            if pred_tgt_seq == ground_truth:
                exact_matches.append(fname)
            if len(pred_tgt_seq) >= len(ground_truth) and pred_tgt_seq[:len(ground_truth)] == ground_truth:
                partial_matches.append(fname)

            
    print('exact matches: ', len(exact_matches))
    return exact_matches
            


"""
def get_best_token_replacement(inputs, grads, vocab, indices, replace_tokens, distinct):
    '''
    inputs is numpy array with input vocab indices (batch, max_len)
    grads is numpy array (batch, max_len, vocab_size)
    vocab is Vocab object
    indices is numpy array of size batch
    returns a dict with {index: {"@R_1@":'abc', ...}}
    '''
    def valid_replacement(s, exclude=[]):
        return classify_tok(s)=='WORDS' and s not in exclude
    
    best_replacements = {}    
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        gradients = grads[i]
        index = str(indices[i])
        
        d = {}				
        for repl_tok in replace_tokens:
            repl_tok_idx = input_vocab.stoi[repl_tok]
            if repl_tok_idx not in inp:
                continue
                
            inp[0] = repl_tok_idx
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


def get_random_token_replacement(inputs, vocab, indices, replace_tokens, distinct):
    '''
    inputs is numpy array with input vocab indices (batch, max_len)
    grads is numpy array (batch, max_len, vocab_size)
    vocab is Vocab object
    indices is numpy array of size batch
    '''
    def valid_replacement(s, exclude=[]):
        return classify_tok(s)=='WORDS' and s not in exclude
    
    rand_replacements = {}    
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        index = str(indices[i])
        
        d = {}		
        for repl_tok in replace_tokens:
            repl_tok_idx = input_vocab.stoi[repl_tok]
            if repl_tok_idx not in inp:
                continue
                
            inp[0] = repl_tok_idx

            exclude = list(d.values()) if distinct else []
            
            rand_idx = random.randint(0, len(vocab)-1)
            while not valid_replacement(vocab.itos[rand_idx], exclude=exclude):
                rand_idx = random.randint(0, len(vocab)-1)

            d[repl_tok] = vocab.itos[rand_idx]

        if len(d)>0:
            rand_replacements[index] = d
    
    return rand_replacements
"""