from seq2seq.loss import Perplexity
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
import seq2seq
from seq2seq.evaluator.metrics import calculate_metrics


import os
import torchtext
import torch
import argparse
import json
import csv
import tqdm
import torch.nn as nn
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path',
          help='Path to test data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--output_dir', action='store', dest='output_dir', default=None)
    parser.add_argument('--batch_size', action='store', default=5, type=int)

    opt = parser.parse_args()

    return opt

def load_model(expt_dir, model_name):
    checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    seq2seq.eval()
    return seq2seq, input_vocab, output_vocab

def load_data(data_path):
    src = SourceField()
    tgt = TargetField()

    fields = []
    with open(data_path, 'r') as f:
        cols = f.readline()[:-1].split('\t')
        for col in cols:
            if col=='tgt':
                fields.append(('tgt', tgt))
            else:
                fields.append((col, src))

    dev = torchtext.data.TabularDataset(
                                    path=data_path, format='tsv',
                                    fields=fields,
                                    skip_header=True, 
                                    csv_reader_params={'quoting': csv.QUOTE_NONE}
                                    )

    return dev, fields, src, tgt


def get_best_attack(batch, model, attacks, src_vocab, tgt_vocab):

    with torch.no_grad():
        target_variables = getattr(batch, seq2seq.tgt_field_name)
        batch_size = target_variables.size(0)
        l_d = [{'best_attack':'', 'best_attack_depth':-1, 'output_seq':'', 'ground_truth':'', 'max_loss':0} for i in range(batch_size)]

        norm_term = target_variables.data.ne(pad).sum(axis=1).cpu().numpy()

        for attack_num,attack in enumerate(attacks):
            input_variables, input_lengths  = getattr(batch, attack)
            cur_losses = np.zeros(batch_size)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            for step, step_output in enumerate(decoder_outputs):
                x = loss_criterion(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])
                cur_losses += x.cpu().detach().numpy()

            cur_losses = np.divide(cur_losses, norm_term)

            for i,output_seq_len in enumerate(other['length']):
                if cur_losses[i]>l_d[i]['max_loss']:
                    l_d[i]['max_loss'] = cur_losses[i]
                    tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
                    tgt_seq = [tgt_vocab.itos[tok] for tok in tgt_id_seq]
                    l_d[i]['output_seq'] = ' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']])
                    l_d[i]['best_attack'] = attack
                    l_d[i]['best_attack_depth'] = 0 if attack == 'src' else attack.count(',')+1
                
                # only need to store these once
                if attack_num==0:
                    gt = [tgt_vocab.itos[tok] for tok in target_variables[i]]
                    l_d[i]['ground_truth']  = ' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']])

    return l_d

def attack_model(model, data, attacks, src_vocab, tgt_vocab, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # batch size of 1 used as we need to find the worst attack for each data point individually
    batch_iterator = torchtext.data.BucketIterator(dataset=data, batch_size=opt.batch_size,sort=False, sort_within_batch=True,sort_key=lambda x: len(x.src),device=device, repeat=False)
    batch_generator = batch_iterator.__iter__()
    outputs = []
    gts = []
    attack_counts = {}
    attack_depth_counts = {}

    attack_counts = {}

    with open(os.path.join(opt.output_dir,'attacked.txt'), 'w') as f:
        for batch in tqdm.tqdm(batch_generator, total=len(batch_iterator)):
            d_l  = get_best_attack(batch, model, attacks, src_vocab, tgt_vocab)
            for d in d_l:
                outputs.append(d['output_seq'])
                gts.append(d['ground_truth'])            
                if d['best_attack'] not in attack_counts:
                    attack_counts[d['best_attack']] = 0
                attack_counts[d['best_attack']] += 1

                if d['best_attack_depth'] not in attack_depth_counts:
                    attack_depth_counts[d['best_attack_depth']] = 0
                attack_depth_counts[d['best_attack_depth']] += 1
                f.write(json.dumps(d)+'\n')

        f.write(json.dumps(attack_counts)+'\n')


    print('Details written to', os.path.join(opt.output_dir,'attacked.txt'))

    metrics = calculate_metrics(outputs, gts)

    print(metrics)
    print(attack_counts)
    print(attack_depth_counts)

    with open(os.path.join(opt.output_dir,'attacked_metrics.txt'), 'w') as f:
        f.write(json.dumps(metrics)+'\n')

    print('Metrics written to', os.path.join(opt.output_dir,'attacked_metrics.txt'))


if __name__=="__main__":
    opt = parse_args()
    print(opt)

    if opt.output_dir is None:
        opt.output_dir = opt.expt_dir

    data, fields, src, tgt = load_data(opt.data_path)
    attacks = [field[0] for field in fields if field[0] not in ['tgt', 'index']]
    print(attacks)

    print('Loaded Data')

    model_name = opt.load_checkpoint

    print(opt.expt_dir, model_name)      

    model, input_vocab, output_vocab = load_model(opt.expt_dir, model_name)

    print('Loaded model')

    src.vocab = input_vocab
    tgt.vocab = output_vocab

    # weight = torch.ones(len(tgt.vocab))
    # pad = tgt.vocab.stoi[tgt.pad_token]
    # loss = Perplexity(weight, pad)
    # if torch.cuda.is_available():
    #     loss.cuda()

    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    weight[pad] = 0
    loss_criterion = nn.NLLLoss(weight=weight, reduction='none')
    if torch.cuda.is_available():
        loss_criterion.cuda()


    attack_model(model, data, attacks, input_vocab, output_vocab, opt)




