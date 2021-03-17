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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path',
          help='Path to test data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--output_dir', action='store', dest='output_dir', default=None)

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
    l = []
    with torch.no_grad():
        target_variables = getattr(batch, seq2seq.tgt_field_name)

        for attack in attacks:
            input_variables, input_lengths  = getattr(batch, attack)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            loss.reset()
            for step, step_output in enumerate(decoder_outputs):
                batch_size = target_variables.size(0)
                loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])

            # other['length'] should be a list of length 1 only, so the loop is redundant
            for i,output_seq_len in enumerate(other['length']):
                tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
                tgt_seq = [tgt_vocab.itos[tok] for tok in tgt_id_seq]
                output_seq = ' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']])
                gt = [tgt_vocab.itos[tok] for tok in target_variables[i]]
                ground_truth = ' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']])

            attack_depth = 0 if attack == 'src' else attack.count(',')+1
            l.append((loss.get_loss(), attack_depth, attack, output_seq))

    l = sorted(l, key=lambda x:x[0], reverse=True)
    d = {}
    d['best_attack_depth'] = l[0][1]
    d['best_attack'] = l[0][2]
    d['output_seq'] = l[0][3]
    d['ground_truth'] = ground_truth
    d['logs'] = l

    return d

def attack_model(model, data, attacks, src_vocab, tgt_vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # batch size of 1 used as we need to find the worst attack for each data point individually
    batch_iterator = torchtext.data.BucketIterator(dataset=data, batch_size=1,sort=False, sort_within_batch=True,sort_key=lambda x: len(x.src),device=device, repeat=False)
    batch_generator = batch_iterator.__iter__()
    outputs = []
    gts = []
    attack_counts = {}
    attack_depth_counts = {}

    attack_counts = {}

    with open(os.path.join(opt.output_dir,'attacked.txt'), 'w') as f:
        for batch in tqdm.tqdm(batch_generator):
            d  = get_best_attack(batch, model, attacks, src_vocab, tgt_vocab)
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

    print(attack_counts)

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

    if opt.output_dir is None:
        opt.output_dir = opt.expt_dir

    data, fields, src, tgt = load_data(opt.data_path)
    attacks = [field[0] for field in fields if field[0] not in ['tgt']]
    # print(attacks)

    print('Loaded Data')

    model_name = opt.load_checkpoint

    print(opt.expt_dir, model_name)      

    model, input_vocab, output_vocab = load_model(opt.expt_dir, model_name)

    print('Loaded model')

    src.vocab = input_vocab
    tgt.vocab = output_vocab

    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()


    attack_model(model, data, attacks, input_vocab, output_vocab)




