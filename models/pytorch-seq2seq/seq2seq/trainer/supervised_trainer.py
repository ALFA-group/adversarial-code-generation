from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim
from torch.utils.tensorboard import SummaryWriter


import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint



class SupervisedTrainer(object):
	""" The SupervisedTrainer class helps in setting up a training framework in a
	supervised setting.

	Args:
		expt_dir (optional, str): experiment Directory to store details of the experiment,
			by default it makes a folder in the current directory to store the details (default: `experiment`).
		loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
		batch_size (int, optional): batch size for experiment, (default: 64)
		checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
	"""
	def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
				 random_seed=None,
				 checkpoint_every=1000, print_every=100, tensorboard=True):
		self._trainer = "Simple Trainer"
		self.random_seed = random_seed
		if random_seed is not None:
			random.seed(random_seed)
			torch.manual_seed(random_seed)
		self.loss = loss
		self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.batch_size = batch_size

		self.logger = logging.getLogger(__name__)

		self.writer = SummaryWriter(log_dir=expt_dir) if tensorboard else None

	def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
		loss = self.loss
		# Forward propagation
		decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable, teacher_forcing_ratio=teacher_forcing_ratio)
		# Get loss
		loss.reset()
		for step, step_output in enumerate(decoder_outputs):
			batch_size = target_variable.size(0)
			loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
		# Backward propagation
		model.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.get_loss()

	def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
					   dev_data=None, teacher_forcing_ratio=0):
		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# device = None if torch.cuda.is_available() else -1
		batch_iterator = torchtext.data.BucketIterator(
			dataset=data, batch_size=self.batch_size,
			sort=False, sort_within_batch=True,
			sort_key=lambda x: len(x.src),
			device=device, repeat=False)

		steps_per_epoch = len(batch_iterator)
		total_steps = steps_per_epoch * n_epochs

		self.print_every = steps_per_epoch // 25

		log.info('Steps per epoch: %d'%steps_per_epoch)
		log.info('Total steps: %d'%total_steps)

		step = start_step
		step_elapsed = 0

		# num_checkpoints = 25
		# self.checkpoint_every = (total_steps+1)//num_checkpoints

		if start_step>0 and dev_data is not None:
			d = self.evaluator.evaluate(model, dev_data)
			dev_loss = d['metrics']['Loss']
			accuracy = d['metrics']['accuracy (torch)']
			other_metrics=d['metrics']
			best_f1 = other_metrics['f1']
			best_acc = accuracy
		else:
			best_f1 = 0.0
			best_acc = 0.0

		Checkpoint(model=model,
				   optimizer=self.optimizer,
				   epoch=0, step=step,
				   input_vocab=data.fields[seq2seq.src_field_name].vocab,
				   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name='Best_F1')

		for epoch in range(start_epoch, n_epochs + 1):
			log.debug("Epoch: %d, Step: %d" % (epoch, step))

			batch_generator = batch_iterator.__iter__()
			# consuming seen batches from previous training
			for _ in range((epoch - 1) * steps_per_epoch, step):
				next(batch_generator)


			model.train(True)
			for batch in batch_generator:
				step += 1
				step_elapsed += 1

				input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
				target_variables = getattr(batch, seq2seq.tgt_field_name)
				
				loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model, teacher_forcing_ratio)

				# Record average loss
				print_loss_total += loss
				epoch_loss_total += loss


				if step % self.print_every == 0 and step_elapsed >= self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					log_msg = 'Epoch: %d, Step: %d, Progress: %d%%, Train %s: %.4f' % (epoch, step,
						step / total_steps * 100,
						self.loss.name,
						print_loss_avg)
					log.info(log_msg)

					if self.writer:
						self.writer.add_scalar('Train/loss_step', print_loss_avg, step)

				# Checkpoint
				# if step % self.checkpoint_every == 0 or step == total_steps:
				#     Checkpoint(model=model,
				#                optimizer=self.optimizer,
				#                epoch=epoch, step=step,
				#                input_vocab=data.fields[seq2seq.src_field_name].vocab,
				#                output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)
				#     log.info('Checkpoint saved')

			if step_elapsed == 0: 
				continue

			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
			self.writer.add_scalar('Train/loss_epoch', epoch_loss_avg, epoch)

			other_metrics = {}
			if dev_data is not None:
				d = self.evaluator.evaluate(model, dev_data)
				dev_loss = d['metrics']['Loss']
				accuracy = d['metrics']['accuracy (torch)']
				other_metrics=d['metrics']
				self.optimizer.update(dev_loss, epoch)
				log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
				self.writer.add_scalar('Val/loss', dev_loss, epoch)
				self.writer.add_scalar('Val/acc', accuracy, epoch)

				for metric in other_metrics:
					try:
						log_msg += ", %s: %.4f"%(metric.replace(' ','_').replace('-','_'), other_metrics[metric])
						self.writer.add_scalar('Val/%s'%metric.replace(' ','_').replace('-','_'), other_metrics[metric], epoch)
					except:
						continue

				log_msg = log_msg[:-1]
				log.info(log_msg)

				if other_metrics['f1'] > best_f1:
					Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=data.fields[seq2seq.src_field_name].vocab,
								   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name='Best_F1')
					log_msg = 'Checkpoint saved, Epoch %d, Prev Val F1: %.4f, New Val F1: %.4f' % (epoch, best_f1, other_metrics['f1'])
					log.info(log_msg)
					best_f1 = other_metrics['f1']

				# if accuracy > best_acc:
				#     Checkpoint(model=model,
				#                    optimizer=self.optimizer,
				#                    epoch=epoch, step=step,
				#                    input_vocab=data.fields[seq2seq.src_field_name].vocab,
				#                    output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name='Best_Acc')
				#     log_msg = 'Checkpoint saved, Epoch %d, Prev Val Acc: %.4f, New Val Acc: %.4f' % (epoch, best_acc, accuracy)
				#     log.info(log_msg)
				#     best_acc = accuracy

				model.train(mode=True)

			else:
				self.optimizer.update(epoch_loss_avg, epoch)

			# Checkpoint(model=model,
			#                    optimizer=self.optimizer,
			#                    epoch=epoch, step=step,
			#                    input_vocab=data.fields[seq2seq.src_field_name].vocab,
			#                    output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name='Latest')
			# log_msg = 'Latest Checkpoint saved, Epoch %d, %s' % (epoch, str(other_metrics))
			# log.info(log_msg)


	def train(self, model, data, num_epochs=5,
			  resume=False, dev_data=None,
			  optimizer=None, teacher_forcing_ratio=0, load_checkpoint=None):
		""" Run training for a given model.

		Args:
			model (seq2seq.models): model to run training on, if `resume=True`, it would be
			   overwritten by the model loaded from the latest checkpoint.
			data (seq2seq.dataset.dataset.Dataset): dataset object to train on
			num_epochs (int, optional): number of epochs to run (default 5)
			resume(bool, optional): resume training with the latest checkpoint, (default False)
			dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
			optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
			   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
			teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
		Returns:
			model (seq2seq.models): trained model.
		"""
		# If training is set to resume
		if resume:
			if load_checkpoint is None:
				load_checkpoint = Checkpoint.get_latest_checkpoint(self.expt_dir)
			resume_checkpoint = Checkpoint.load(load_checkpoint)
			model = resume_checkpoint.model
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step

			self.logger.info("Resuming training from %d epoch, %d step" % (start_epoch, step))
		else:
			start_epoch = 1
			step = 0
			if optimizer is None:
				optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
			self.optimizer = optimizer

		self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

		self._train_epoches(data, model, num_epochs,
							start_epoch, step, dev_data=dev_data,
							teacher_forcing_ratio=teacher_forcing_ratio)
		return model
