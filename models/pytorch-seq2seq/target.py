import argparse
import csv
import json
import os
import random
import seq2seq
import sys
import torch
import torchtext
import tqdm
import gc

from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.evaluator.metrics import calculate_metrics
from seq2seq.loss import Perplexity
from seq2seq.util.checkpoint import Checkpoint


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data_path',
    action='store',
    dest='data_path',
    help='Path to test data'
  )
  parser.add_argument(
    '--expt_dir',
    action='store',
    dest='expt_dir',
    default='./experiment',
    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided'
  )
  parser.add_argument(
    '--load_checkpoint',
    action='store',
    dest='load_checkpoint',
    help='The name of the checkpoint to load, usually an encoded time string'
  )
  parser.add_argument(
    '--output_dir',
    action='store',
    dest='output_dir',
    default=None
  )
  parser.add_argument(
    '--batch_size',
    action='store',
    dest='batch_size',
    default=128,
    type=int
  )

  return parser.parse_args()


def load_model(expt_dir, model_name):
  checkpoint = Checkpoint.load(os.path.join(
    expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name
  ))

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
    path=data_path,
    format='tsv',
    fields=fields,
    skip_header=True, 
    csv_reader_params={ 'quoting': csv.QUOTE_NONE }
  )

  return dev, fields, src, tgt


def generate_assignments(batch_size, safe_keys):
  assignments = []
  for i in range(batch_size):
    assignment = {}
    assignment[-1] = random.choice(safe_keys)
    assignment[-2] = random.choice(safe_keys)
    assignment[-3] = random.choice(safe_keys)
    assignment[-4] = random.choice(safe_keys)
    assignment[-5] = random.choice(safe_keys)
    assignments.append(assignment)
  return assignments


def apply_replacements_to_batch(batch, batch_size, replacements, safe_keys):
  try:
    copy = batch.clone()
    for i in range(batch_size):
      copy[i][copy[i] == -1] = replacements[i][-1]
      copy[i][copy[i] == -2] = replacements[i][-2]
      copy[i][copy[i] == -3] = replacements[i][-3]
      copy[i][copy[i] == -4] = replacements[i][-4]
      copy[i][copy[i] == -5] = replacements[i][-5]
  except Exception:
    return apply_replacements_to_batch(batch, batch_size, generate_assignments(batch_size, safe_keys))
  
  return copy, replacements


def find_best_replacement(batch_size, batch, model, attacks, src_vocab, tgt_vocab, safe_replacements):
  safe_keys = list(safe_replacements.keys())
  with torch.no_grad():
    best_replacements = {}

    for attack in attacks:
      input_variables, input_lengths  = getattr(batch, attack)
      target_variables = getattr(batch, seq2seq.tgt_field_name)

      assignments = generate_assignments(batch_size, safe_keys)
      
      worst = 0.0
      best = 100000.0
      best_assignments = {}
      for attempt in range(10):
        new_input, assignments = apply_replacements_to_batch(input_variables, batch_size, assignments, safe_keys)

        decoder_outputs, decoder_hidden, other = model(new_input, input_lengths.tolist(), target_variables)

        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
          loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])

        the_loss = loss.get_loss()
        if the_loss > worst:
          worst = the_loss
          best_assignments = assignments.copy()
        if the_loss < best:
          best = the_loss

        del new_input
        torch.cuda.empty_cache()

        assignments = generate_assignments(batch_size, safe_keys)


      best_replacement = []
      for i, replacement in enumerate(best_assignments):
        best_replacement.append(' '.join([
          src_vocab.itos[replacement[-1]],
          src_vocab.itos[replacement[-2]],
          src_vocab.itos[replacement[-3]],
          src_vocab.itos[replacement[-4]],
          src_vocab.itos[replacement[-5]]
        ]))
      best_replacements[attack] = best_replacement
    
    for i in range(batch_size):
      reordered = []
      for attack in attacks:
        reordered.append(best_replacements[attack][i])
      print('\t'.join(reordered))

    del batch


def find_best_replacements(batch_size, model, data, attacks, src_vocab, tgt_vocab, safe_replacements):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Currently doing this one datapoint at a time
  batch_iterator = torchtext.data.BucketIterator(
    dataset=data,
    batch_size=batch_size,
    sort=False,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
    repeat=False
  )

  for batch in tqdm.tqdm(batch_iterator.__iter__(), file=sys.stderr, total=len(data) // batch_size):
    find_best_replacement(batch_size, batch, model, attacks, src_vocab, tgt_vocab, safe_replacements)


if __name__=="__main__":
  opt = parse_args()

  if opt.output_dir is None:
    opt.output_dir = opt.expt_dir

  data, fields, src, tgt = load_data(opt.data_path)
  attacks = [
    field[0] for field in fields if field[0] not in ['tgt']
  ]

  model, input_vocab, output_vocab = load_model(
    opt.expt_dir, opt.load_checkpoint
  )

  src.vocab = input_vocab
  tgt.vocab = output_vocab

  safe_replacements = {}
  for idx, word in enumerate(input_vocab.itos):
    if word[0].isalpha() and word.isalnum():
      safe_replacements[idx] = word

  src.vocab.stoi['@R_1@'] = -1
  src.vocab.stoi['@R_2@'] = -2
  src.vocab.stoi['@R_3@'] = -3
  src.vocab.stoi['@R_4@'] = -4
  src.vocab.stoi['@R_5@'] = -5

  weight = torch.ones(len(tgt.vocab))
  pad = tgt.vocab.stoi[tgt.pad_token]
  loss = Perplexity(weight, pad)
  if torch.cuda.is_available():
    loss.cuda()

  find_best_replacements(
    opt.batch_size,
    model,
    data, attacks,
    input_vocab,
    output_vocab,
    safe_replacements
  )
