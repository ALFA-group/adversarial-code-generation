import os
import argparse
import torch
import torchtext
import seq2seq 
from seq2seq.util.checkpoint import Checkpoint 
from seq2seq.dataset.fields import SourceField, TargetField, FnameField
from seq2seq.util.concat import torch_concat


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='Path to checkpoint.')
parser.add_argument('--data_path', dest='data_path', help='Path to train, test, valid files.')
parser.add_argument('--reps_path', dest='reps_path', help='Path to representations.')

args = parser.parse_args()

params = {
    'src_vocab_size': 15000, 
    'tgt_vocab_size': 5000, 
    'max_len': 128, 
}

def load_data(data_path, src, tgt, fname):

	
	fields = [('from_file', fname), ('src', src), ('tgt', tgt)]
	len_filter = lambda x: len(x.src) <= params['max_len'] and len(x.tgt) <= params['max_len']
	train = torchtext.data.TabularDataset(
		path=os.path.join(data_path, 'train.tsv'),
		format='tsv',
		fields=fields,
		filter_pred=len_filter
		)
	valid = torchtext.data.TabularDataset(
		path=os.path.join(data_path, 'valid.tsv'),
		format='tsv',
		fields=fields,
		filter_pred=len_filter
		)
	test = torchtext.data.TabularDataset(
		path=os.path.join(data_path, 'test.tsv'),
		format='tsv',
		fields=fields,
		filter_pred=len_filter
		)
	src.build_vocab(train, max_size=params['src_vocab_size'])
	tgt.build_vocab(train, max_size=params['tgt_vocab_size'])
	fname.build_vocab(train, valid, test)

	return train, valid, test

def save_data(path, fname, rep):

	with open(path+fname+'.torch', 'wb') as f:
		torch.save(rep, f)




def get_reps(model, data):

	model.eval()
	device = 'cuda:0' if torch.cuda.is_available() else -1
	batch_iterator = torchtext.data.BucketIterator(
		dataset=data, batch_size=1,
		sort=True, sort_key=lambda x: len(x.src),
		device=device, train=False
		)
	with torch.no_grad():
		all_hidden, all_fnames = None, None
		# first = True
		for batch in batch_iterator:
			input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
			target_variables = getattr(batch, seq2seq.tgt_field_name)
			fnames = getattr(batch, seq2seq.fname_field_name)
			_, (encoder_output, encoder_hidden) = model(input_variables, input_lengths.tolist(), target_variables, get_reps=True)

			encoder_hidden = torch.mean(torch.stack(encoder_hidden), 0)
			# if first:
			# 	print(encoder_hidden.shape)
			# 	first = False
			all_hidden = torch_concat(all_hidden, encoder_hidden)
			all_fnames = torch_concat(all_fnames, fnames)
	return all_hidden, all_fnames


def denumericalize(all_fnames, fname):
    with torch.cuda.device_of(all_fnames):
        all_fnames = all_fnames.tolist()
    all_fnames = [fname.vocab.itos[ex] for ex in all_fnames]
    return all_fnames

src, tgt, fname = SourceField(), TargetField(), FnameField()
train, valid, test = load_data(args.data_path, src, tgt, fname)


checkpoint = Checkpoint.load(args.checkpoint_path)
model = checkpoint.model

test_reps, test_fnames = get_reps(model, test)
test_fnames = denumericalize(test_fnames, fname)
# print(test_reps.shape, len(test_fnames))
train_reps, train_fnames = get_reps(model, train)
train_fnames = denumericalize(train_fnames, fname)

valid_reps, valid_fnames = get_reps(model, valid)
valid_fnames = denumericalize(valid_fnames, fname)


reps_path = args.reps_path


save_data(reps_path, 'train_reps', [train_reps, train_fnames])
save_data(reps_path, 'valid_reps', [valid_reps, valid_fnames])
save_data(reps_path, 'test_reps', [test_reps, test_fnames])


