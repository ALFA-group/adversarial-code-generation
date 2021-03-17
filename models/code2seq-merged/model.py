import _pickle as pickle
import os
import time
import random

import numpy as np
import shutil
import tensorflow as tf

import sys
from math import ceil

import reader
from common import Common
import time
import datetime

######Auxilary function for formatting evaluation outputs####
def format_orig(target):
    result = " ".join(target.split('|'))
    return result
    
def format_pred(target):
    result = ""
    for tk in target:
        if not tk == '<PAD>':
            result += tk + " "
    return result[:-1]
    
def format_attention(word_attention):
    result = ""
    for predicted_word, attention_timestep in word_attention:
        result = result + predicted_word + "$"
        for context, attention in [(key, attention_timestep[key]) for key in sorted(attention_timestep, key=attention_timestep.get, reverse=True)]:
            context_string = ""
            for c in context:
                context_string += c+","
            result = result + context_string + ":" + str(attention.item())+" "
            #current_timestep_paths = .append((attention.item(), context))
        result += " & "
    return result
    

class Model:
    topk = 10
    num_batches_to_log = 100

    def __init__(self, config, repl_tokens=[]):
        self.config = config
        self.sess = tf.Session()

        self.eval_queue = None
        self.predict_queue = None
        self.queue = None
        

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_predicted_indices_op, self.eval_top_values_op, self.eval_true_target_strings_op, self.eval_topk_values = None, None, None, None
        self.predict_top_indices_op, self.predict_top_scores_op, self.predict_target_strings_op = None, None, None
        self.subtoken_to_index = None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        else:
            with open('{}.dict.c2s'.format(config.TRAIN_PATH), 'rb') as file:
                subtoken_to_count = pickle.load(file)
                node_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                max_contexts = pickle.load(file)
                self.num_training_examples = pickle.load(file)
                print('Dictionaries loaded.')

            if self.config.DATA_NUM_CONTEXTS <= 0:
                self.config.DATA_NUM_CONTEXTS = max_contexts
            self.subtoken_to_index, self.index_to_subtoken, self.subtoken_vocab_size = \
                Common.load_vocab_from_dict(subtoken_to_count, add_values=[Common.PAD, Common.UNK]+repl_tokens,
                                            max_size=config.SUBTOKENS_VOCAB_MAX_SIZE)
            print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)
            if len(repl_tokens)>0:
                print('Added replace tokens to subtoken vocabulary:', repl_tokens)

            self.target_to_index, self.index_to_target, self.target_vocab_size = \
                Common.load_vocab_from_dict(target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS],
                                            max_size=config.TARGET_VOCAB_MAX_SIZE)
            print('Loaded target word vocab. size: %d' % self.target_vocab_size)

            self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                Common.load_vocab_from_dict(node_to_count, add_values=[Common.PAD, Common.UNK], max_size=None)
            print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)
            self.epochs_trained = 0

    def close_session(self):
        self.sess.close()


    def train(self, lamb=0.0):

        assert (self.config.TRANSFS is None and self.config.TRAIN_DIR is None) or (self.config.TRANSFS is not None and self.config.TRAIN_DIR is not None), "For adversarial training, TRANSFS and TRAIN_DIR must not be None. For regular training, both must be None"

        if self.config.TRANSFS is not None:
            self.adv_train(lamb=lamb)
            return

        print('Starting normal training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        best_f1 = 0
        best_epoch = 0
        best_f1_precision = 0
        best_f1_recall = 0
        epochs_no_improve = 0

        self.queue_thread = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                          node_to_index=self.node_to_index,
                                          target_to_index=self.target_to_index,
                                          config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.get_output())
        self.print_hyperparams()
        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        time.sleep(1)
        print('Started reader...')

        multi_batch_start_time = time.time()
        for iteration in range(1, (self.config.NUM_EPOCHS // self.config.SAVE_EVERY_EPOCHS) + 1):
            self.queue_thread.reset(self.sess)
            try:
                batch_num = 0
                while True:
                    batch_num += 1
                    _, batch_loss = self.sess.run([optimizer, train_loss])
                    sum_loss += batch_loss
                    # print('SINGLE BATCH LOSS', batch_loss)
                    if batch_num % self.num_batches_to_log == 0:
                        print(datetime.datetime.now(), end=':   ')
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        sum_loss = 0
                        multi_batch_start_time = time.time()


            except tf.errors.OutOfRangeError:
                self.epochs_trained += self.config.SAVE_EVERY_EPOCHS
                print('Finished %d epochs, number of batches evaluated: %d' % (self.config.SAVE_EVERY_EPOCHS, batch_num))
                results, precision, recall, f1 = self.evaluate()
                if self.config.BEAM_WIDTH == 0:
                    print('Accuracy after %d epochs: %.5f' % (self.epochs_trained, results))
                else:
                    print('Accuracy after {} epochs: {}'.format(self.epochs_trained, results))
                print('After %d epochs: Precision: %.5f, recall: %.5f, F1: %.5f' % (
                    self.epochs_trained, precision, recall, f1))
                # print('Rouge: ', rouge)
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_precision = precision
                    best_f1_recall = recall
                    best_epoch = self.epochs_trained
                    epochs_no_improve = 0
                    self.save_model(self.sess, self.config.SAVE_PATH)
                    print('New model saved, F1:', f1)
                else:
                    epochs_no_improve += self.config.SAVE_EVERY_EPOCHS
                    if epochs_no_improve >= self.config.PATIENCE:
                        print('Not improved for %d epochs, stopping training' % self.config.PATIENCE)
                        print('Best scores - epoch %d: ' % best_epoch)
                        print('Precision: %.5f, recall: %.5f, F1: %.5f' % (best_f1_precision, best_f1_recall, best_f1))
                        return

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH + '.final')
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sh%sm%ss\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))


    def adv_train(self, lamb=0.0):

        print('Starting adversarial training')
        start_time = time.time()

        if lamb>0:
            print('Lambda:', lamb)

        batch_num = 0
        sum_loss = 0
        best_f1 = 0
        best_epoch = 0
        best_f1_precision = 0
        best_f1_recall = 0
        epochs_no_improve = 0

        self.eval_queues = []
           
    #####build a training graph inside, not calling the function####
        target_index = tf.placeholder(tf.int32, [None, None])
        target_lengths = tf.placeholder(tf.int64, [None,])
        path_source_indices = tf.placeholder(tf.int32, [None, 200, 5])
        node_indices = tf.placeholder(tf.int32, [None, 200, 9])
        path_target_indices = tf.placeholder(tf.int32, [None, 200, 5])
        valid_context_mask = tf.placeholder(tf.float32, [None, 200])
        path_source_lengths = tf.placeholder(tf.int32, [None, 200])
        path_lengths = tf.placeholder(tf.int32, [None, 200])
        path_target_lengths = tf.placeholder(tf.int32, [None, 200])

        with tf.variable_scope('model'):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                        mode='FAN_OUT',
                                                                                                        uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            nodes_vocab = tf.get_variable('NODES_VOCAB', shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            # (batch, max_contexts, decoder_size)
            batched_contexts = self.compute_contexts(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_context_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths)

            batch_size = tf.shape(target_index)[0]
            outputs, final_states = self.decode_outputs(target_words_vocab=target_words_vocab,
                                                        target_input=target_index, batch_size=batch_size,
                                                        batched_contexts=batched_contexts,
                                                        valid_mask=valid_context_mask)
            step = tf.Variable(0, trainable=False)

            logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
            target_words_nonzero = tf.sequence_mask(target_lengths + 1,
                                                    maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            
            loss_weight = tf.placeholder(tf.float32)

            train_loss = loss_weight*(tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size))

            if self.config.USE_MOMENTUM:
                learning_rate = tf.train.exponential_decay(0.01, step * self.config.BATCH_SIZE,
                                                           self.num_training_examples,
                                                           0.95, staircase=True)
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
                train_op = optimizer.minimize(train_loss, global_step=step)
            else:
                params = tf.trainable_variables()
                gradients = tf.gradients(train_loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

            self.saver = tf.train.Saver(max_to_keep=10)
        ##################################end of build training graph###########################
        
        ############################building eval graph#########################################
        eval_target_index = tf.placeholder(tf.int32, [None, None])
        eval_target_lengths = tf.placeholder(tf.int64, [None,])
        eval_path_source_indices = tf.placeholder(tf.int32, [None, 200, 5])
        eval_node_indices = tf.placeholder(tf.int32, [None, 200, 9])
        eval_path_target_indices = tf.placeholder(tf.int32, [None, 200, 5])
        eval_valid_context_mask = tf.placeholder(tf.float32, [None, 200])
        eval_path_source_lengths = tf.placeholder(tf.int32, [None, 200])
        eval_path_lengths = tf.placeholder(tf.int32, [None, 200])
        eval_path_target_lengths = tf.placeholder(tf.int32, [None, 200])

        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            eval_subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32, trainable=False)
            eval_target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            eval_nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            eval_batched_contexts = self.compute_contexts(subtoken_vocab=eval_subtoken_vocab, nodes_vocab=eval_nodes_vocab,
                                                     source_input=eval_path_source_indices, nodes_input=eval_node_indices,
                                                     target_input=eval_path_target_indices,
                                                     valid_mask=eval_valid_context_mask,
                                                     path_source_lengths=eval_path_source_lengths,
                                                     path_lengths=eval_path_lengths, path_target_lengths=eval_path_target_lengths, is_evaluating=True)

            eval_batch_size = tf.shape(eval_target_index)[0]
            #print("batch size is "+str(batch_size)
            eval_outputs, eval_final_states = self.decode_outputs(target_words_vocab=eval_target_words_vocab,
                                                        target_input=eval_target_index, batch_size=eval_batch_size,
                                                        batched_contexts=eval_batched_contexts,
                                                        valid_mask=eval_valid_context_mask, is_evaluating=False, dropout=0)


            eval_logits = eval_outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)
            eval_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_target_index, logits=eval_logits)
            eval_target_words_nonzero = tf.sequence_mask(eval_target_lengths + 1,
                                                    maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            eval_graph_loss = tf.reduce_sum(eval_crossent * eval_target_words_nonzero) / tf.to_float(eval_batch_size)
        ############################end of building eval graph##################################
        
        #####Read the adv_data######
        for transf in range(self.config.TRANSFS):
            adv = reader.Reader(subtoken_to_index=self.subtoken_to_index, node_to_index=self.node_to_index,target_to_index=self.target_to_index,config=self.config, adv_training = True, is_evaluating=True, adv_transf = transf)
            self.eval_queues.append(adv)
                                     
        
        open("training_timing_new.log",'w').close()
        
        self.print_hyperparams()
        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        time.sleep(1)
        print('Started reader...')

        multi_batch_start_time = time.time()
        
        my_training_start_time = time.time()
        for iteration in range(1, (self.config.NUM_EPOCHS // self.config.SAVE_EVERY_EPOCHS) + 1):
            my_training_elapse = time.time() - my_training_start_time
            my_training_start_time = time.time()
            
            with open("training_timing_new.log",'a+') as trf:
                trf.write(str(my_training_elapse)+"\n")
            #open(self.config.TRAIN_PATH+str(iteration)+".train.c2s",'w').close()
            print("start evaluation")
            a_time = time.time()
            for transf in range(self.config.TRANSFS):
                self.eval_queues[transf].reset(self.sess)
            try:
                while True:
                    batch_num += 1
                    worst_loss = 0.0
                    worst_tensors = None
                    print("Evaluating and training batch %d"%batch_num, end=', Time taken: ')
                    print(time.time()-a_time)
                    a_time = time.time()

                    for transf in range(self.config.TRANSFS):
                        eval_input_tensors_data = self.eval_queues[transf].get_output()
                        eval_input_tensors = self.sess.run(eval_input_tensors_data)
                        if lamb>0 and transf==0:
                            orig_tensors = eval_input_tensors
                            continue
                        eval_curr_loss = self.sess.run(eval_graph_loss, feed_dict={eval_target_index: eval_input_tensors[reader.TARGET_INDEX_KEY], eval_target_lengths: eval_input_tensors[reader.TARGET_LENGTH_KEY], eval_path_source_indices: eval_input_tensors[reader.PATH_SOURCE_INDICES_KEY], eval_node_indices: eval_input_tensors[reader.NODE_INDICES_KEY], eval_path_target_indices: eval_input_tensors[reader.PATH_TARGET_INDICES_KEY],  eval_valid_context_mask: eval_input_tensors[reader.VALID_CONTEXT_MASK_KEY], eval_path_source_lengths: eval_input_tensors[reader.PATH_SOURCE_LENGTHS_KEY], eval_path_lengths: eval_input_tensors[reader.PATH_LENGTHS_KEY], eval_path_target_lengths: eval_input_tensors[reader.PATH_TARGET_LENGTHS_KEY]})
                        # print(eval_curr_loss)
                        if eval_curr_loss > worst_loss:
                            worst_loss = eval_curr_loss
                            worst_tensors = eval_input_tensors
                    
                    # for composite adversarial training loss
                    if lamb>0:
                        _, batch1_loss = self.sess.run([train_op, train_loss],feed_dict={loss_weight: lamb, target_index: orig_tensors[reader.TARGET_INDEX_KEY], target_lengths: orig_tensors[reader.TARGET_LENGTH_KEY], path_source_indices: orig_tensors[reader.PATH_SOURCE_INDICES_KEY], node_indices: orig_tensors[reader.NODE_INDICES_KEY], path_target_indices: orig_tensors[reader.PATH_TARGET_INDICES_KEY],  valid_context_mask: orig_tensors[reader.VALID_CONTEXT_MASK_KEY], path_source_lengths: orig_tensors[reader.PATH_SOURCE_LENGTHS_KEY], path_lengths: orig_tensors[reader.PATH_LENGTHS_KEY], path_target_lengths: orig_tensors[reader.PATH_TARGET_LENGTHS_KEY]})

                    _, batch_loss = self.sess.run([train_op, train_loss],feed_dict={loss_weight: (1.0-lamb), target_index: worst_tensors[reader.TARGET_INDEX_KEY], target_lengths: worst_tensors[reader.TARGET_LENGTH_KEY], path_source_indices: worst_tensors[reader.PATH_SOURCE_INDICES_KEY], node_indices: worst_tensors[reader.NODE_INDICES_KEY], path_target_indices: worst_tensors[reader.PATH_TARGET_INDICES_KEY],  valid_context_mask: worst_tensors[reader.VALID_CONTEXT_MASK_KEY], path_source_lengths: worst_tensors[reader.PATH_SOURCE_LENGTHS_KEY], path_lengths: worst_tensors[reader.PATH_LENGTHS_KEY], path_target_lengths: worst_tensors[reader.PATH_TARGET_LENGTHS_KEY]})
                    
                    sum_loss += batch1_loss + batch_loss
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        sum_loss = 0
                        multi_batch_start_time = time.time()    


            except tf.errors.OutOfRangeError:
                self.epochs_trained += self.config.SAVE_EVERY_EPOCHS
                print('Finished %d epochs' % self.config.SAVE_EVERY_EPOCHS)
               
               
                results, precision, recall, f1 = self.evaluate()
                if self.config.BEAM_WIDTH == 0:
                    print('Accuracy after %d epochs: %.5f' % (self.epochs_trained, results))
                else:
                    print('Accuracy after {} epochs: {}'.format(self.epochs_trained, results))
                print('After %d epochs: Precision: %.5f, recall: %.5f, F1: %.5f' % (
                    self.epochs_trained, precision, recall, f1))
                # if f1 > best_f1:
                #     best_f1 = f1
                #     best_f1_precision = precision
                #     best_f1_recall = recall
                #     best_epoch = self.epochs_trained
                #     epochs_no_improve = 0
                #     self.save_model(self.sess, self.config.SAVE_PATH)
                # else:
                #     epochs_no_improve += self.config.SAVE_EVERY_EPOCHS
                #     if epochs_no_improve >= self.config.PATIENCE:
                #         print('Not improved for %d epochs, stopping training' % self.config.PATIENCE)
                #         print('Best scores - epoch %d: ' % best_epoch)
                #         print('Precision: %.5f, recall: %.5f, F1: %.5f' % (best_f1_precision, best_f1_recall, best_f1))
                #         return


        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sh%sm%ss\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / self.num_batches_to_log
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (
            batch_num,
            avg_loss,
            self.config.BATCH_SIZE * self.num_batches_to_log * (
                self.config.TRANSFS + 1 if self.config.TRANSFS is not None else 1
            ) / (
                multi_batch_elapsed if multi_batch_elapsed > 0 else 1
            )
        ))

    
    def adv_eval_batched(self):
        test_queues = []
        print("\n\n")
    ################Build the eval graph to select the worst data###############
        eval_target_index = tf.placeholder(tf.int32, [None, None])
        eval_target_lengths = tf.placeholder(tf.int64, [None,])
        eval_path_source_indices = tf.placeholder(tf.int32, [None, 200, 5])
        eval_node_indices = tf.placeholder(tf.int32, [None, 200, 9])
        eval_path_target_indices = tf.placeholder(tf.int32, [None, 200, 5])
        eval_valid_context_mask = tf.placeholder(tf.float32, [None, 200])
        eval_path_source_lengths = tf.placeholder(tf.int32, [None, 200])
        eval_path_lengths = tf.placeholder(tf.int32, [None, 200])
        eval_path_target_lengths = tf.placeholder(tf.int32, [None, 200])

        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            eval_subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32, trainable=False)
            eval_target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            eval_nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            eval_batched_contexts = self.compute_contexts(subtoken_vocab=eval_subtoken_vocab, nodes_vocab=eval_nodes_vocab,
                                                     source_input=eval_path_source_indices, nodes_input=eval_node_indices,
                                                     target_input=eval_path_target_indices,
                                                     valid_mask=eval_valid_context_mask,
                                                     path_source_lengths=eval_path_source_lengths,
                                                     path_lengths=eval_path_lengths, path_target_lengths=eval_path_target_lengths, is_evaluating=True)

            eval_batch_size = tf.shape(eval_target_index)[0]
            #print("batch size is "+str(batch_size)
            eval_outputs, eval_final_states = self.decode_outputs(target_words_vocab=eval_target_words_vocab,
                                                        target_input=eval_target_index, batch_size=eval_batch_size,
                                                        batched_contexts=eval_batched_contexts,
                                                        valid_mask=eval_valid_context_mask, is_evaluating=False, dropout=0, adv_testing = True)


            #pred_outputs, pred_final_states = self.decode_outputs(target_words_vocab=eval_target_words_vocab,
            #                                            target_input=eval_target_index, batch_size=eval_batch_size,
            #                                            batched_contexts=eval_batched_contexts, valid_mask=eval_valid_context_mask,
            #                                            is_evaluating=True)
            
            eval_logits = eval_outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)
            eval_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_target_index, logits=eval_logits)
            eval_target_words_nonzero = tf.sequence_mask(eval_target_lengths + 1,
                                                    maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            eval_graph_loss = tf.reduce_sum(eval_crossent * eval_target_words_nonzero, axis=1)
            # eval_graph_loss = tf.reduce_sum(eval_crossent * eval_target_words_nonzero) / tf.to_float(eval_batch_size)
            
            
            predicted_indices_op = eval_outputs.sample_id
            self.saver = tf.train.Saver(max_to_keep=10)
            # topk_values = tf.constant(1, shape=(1, 1), dtype=tf.float32)
            # predicted_indices_attention_weights = tf.squeeze(eval_final_states.alignment_history.stack(), 1)
  
    ###############################end of build graph######################################
        for transf in range(self.config.TRANSFS):
            adv = reader.Reader(subtoken_to_index=self.subtoken_to_index, node_to_index=self.node_to_index,target_to_index=self.target_to_index,config=self.config, adv_training = True, adv_testing=True, adv_transf = transf)
            #adv.reset(self.sess)
            test_queues.append(adv)
        
        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
        for transf in range(self.config.TRANSFS):
            test_queues[transf].reset(self.sess)
        
        path_prefix = '/mnt/outputs/' if "PATH_PREFIX" in os.environ else ''
        open("{}predicted_target".format(path_prefix), 'w').close()
        open("{}true_target".format(path_prefix), 'w').close()
        batch_num = 0 
        try:
            while True:
                print('Next batch', batch_num)
                worst_loss = 0.0
                worst_tensors = None
                for transf in range(self.config.TRANSFS):
                    

                    test_input_tensors_data = test_queues[transf].get_output()
                    test_input_tensors = self.sess.run(test_input_tensors_data)
                    test_curr_loss, predicted_indices = self.sess.run([eval_graph_loss, predicted_indices_op], feed_dict={eval_target_index: test_input_tensors[reader.TARGET_INDEX_KEY], eval_target_lengths: test_input_tensors[reader.TARGET_LENGTH_KEY], eval_path_source_indices: test_input_tensors[reader.PATH_SOURCE_INDICES_KEY], eval_node_indices: test_input_tensors[reader.NODE_INDICES_KEY], eval_path_target_indices: test_input_tensors[reader.PATH_TARGET_INDICES_KEY],  eval_valid_context_mask: test_input_tensors[reader.VALID_CONTEXT_MASK_KEY], eval_path_source_lengths: test_input_tensors[reader.PATH_SOURCE_LENGTHS_KEY], eval_path_lengths: test_input_tensors[reader.PATH_LENGTHS_KEY], eval_path_target_lengths: test_input_tensors[reader.PATH_TARGET_LENGTHS_KEY]})
                    path_target_strings = test_input_tensors[reader.PATH_TARGET_STRINGS_KEY]

                    if transf==0:
                        batch_size = test_curr_loss.shape[0]
                        # list of lists: [worst_loss, true_target, predicted_target]
                        list_of_worst = []
                        for i in range(batch_size):
                            target_string = format_orig(Common.binary_to_string(test_input_tensors[reader.TARGET_STRING_KEY][i]))
                            predicted_string = format_pred([self.index_to_target[idx] for idx in predicted_indices[i]])
                            list_of_worst.append([test_curr_loss[i], target_string, predicted_string])


                    for i in range(batch_size):
                        if test_curr_loss[i] > list_of_worst[i][0]:
                            list_of_worst[i][0] = test_curr_loss[i]
                            # if this assertion fails, then it means there is something wrong with the data alignment
                            assert format_orig(Common.binary_to_string(test_input_tensors[reader.TARGET_STRING_KEY][i]))==list_of_worst[i][1]
                            list_of_worst[i][2] = format_pred([self.index_to_target[idx] for idx in predicted_indices[i]])

                    # print(list_of_worst)
                
                with open('{}true_target'.format(path_prefix),'a+') as f, open("{}predicted_target".format(path_prefix), 'a+') as g:
                    for _, true_target, predicted_target in list_of_worst:
                        f.write(true_target+"\n")
                        g.write(predicted_target +"\n")

                batch_num += 1
                
        except tf.errors.OutOfRangeError:
            return


    def adv_eval(self):
        test_queues = []
        print("\n\n",1)
    ################Build the eval graph to select the worst data###############
        eval_target_index = tf.placeholder(tf.int32, [None, None])
        eval_target_lengths = tf.placeholder(tf.int64, [None,])
        eval_path_source_indices = tf.placeholder(tf.int32, [None, 200, 5])
        eval_node_indices = tf.placeholder(tf.int32, [None, 200, 9])
        eval_path_target_indices = tf.placeholder(tf.int32, [None, 200, 5])
        eval_valid_context_mask = tf.placeholder(tf.float32, [None, 200])
        eval_path_source_lengths = tf.placeholder(tf.int32, [None, 200])
        eval_path_lengths = tf.placeholder(tf.int32, [None, 200])
        eval_path_target_lengths = tf.placeholder(tf.int32, [None, 200])

        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            eval_subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32, trainable=False)
            eval_target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            eval_nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            eval_batched_contexts = self.compute_contexts(subtoken_vocab=eval_subtoken_vocab, nodes_vocab=eval_nodes_vocab,
                                                     source_input=eval_path_source_indices, nodes_input=eval_node_indices,
                                                     target_input=eval_path_target_indices,
                                                     valid_mask=eval_valid_context_mask,
                                                     path_source_lengths=eval_path_source_lengths,
                                                     path_lengths=eval_path_lengths, path_target_lengths=eval_path_target_lengths, is_evaluating=True)

            eval_batch_size = tf.shape(eval_target_index)[0]
            #print("batch size is "+str(batch_size)
            eval_outputs, eval_final_states = self.decode_outputs(target_words_vocab=eval_target_words_vocab,
                                                        target_input=eval_target_index, batch_size=eval_batch_size,
                                                        batched_contexts=eval_batched_contexts,
                                                        valid_mask=eval_valid_context_mask, is_evaluating=False, dropout=0, adv_testing = True)


            #pred_outputs, pred_final_states = self.decode_outputs(target_words_vocab=eval_target_words_vocab,
            #                                            target_input=eval_target_index, batch_size=eval_batch_size,
            #                                            batched_contexts=eval_batched_contexts, valid_mask=eval_valid_context_mask,
            #                                            is_evaluating=True)
            
            eval_logits = eval_outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)
            eval_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_target_index, logits=eval_logits)
            eval_target_words_nonzero = tf.sequence_mask(eval_target_lengths + 1,
                                                    maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            eval_graph_loss = tf.reduce_sum(eval_crossent * eval_target_words_nonzero) / tf.to_float(eval_batch_size)
            
            
            predicted_indices_op = eval_outputs.sample_id
            self.saver = tf.train.Saver(max_to_keep=10)
            topk_values = tf.constant(1, shape=(1, 1), dtype=tf.float32)
            predicted_indices_attention_weights = tf.squeeze(eval_final_states.alignment_history.stack(), 1)
  
    ###############################end of build graph######################################
        for transf in range(self.config.TRANSFS):
            adv = reader.Reader(subtoken_to_index=self.subtoken_to_index, node_to_index=self.node_to_index,target_to_index=self.target_to_index,config=self.config, adv_training = True, adv_testing=True, adv_transf = transf)
            #adv.reset(self.sess)
            test_queues.append(adv)
        
        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
        for transf in range(self.config.TRANSFS):
            test_queues[transf].reset(self.sess)
        
        path_prefix = '/mnt/outputs/' if "PATH_PREFIX" in os.environ else ''
        open("{}predicted_target".format(path_prefix), 'w').close()
        open("{}true_target".format(path_prefix), 'w').close()
        open("{}attention_weights".format(path_prefix), 'w').close()
        try:
            while True:
                worst_loss = 0.0
                worst_tensors = None
                for transf in range(self.config.TRANSFS):
                    test_input_tensors_data = test_queues[transf].get_output()
                    test_input_tensors = self.sess.run(test_input_tensors_data)
                    test_curr_loss = self.sess.run(eval_graph_loss, feed_dict={eval_target_index: test_input_tensors[reader.TARGET_INDEX_KEY], eval_target_lengths: test_input_tensors[reader.TARGET_LENGTH_KEY], eval_path_source_indices: test_input_tensors[reader.PATH_SOURCE_INDICES_KEY], eval_node_indices: test_input_tensors[reader.NODE_INDICES_KEY], eval_path_target_indices: test_input_tensors[reader.PATH_TARGET_INDICES_KEY],  eval_valid_context_mask: test_input_tensors[reader.VALID_CONTEXT_MASK_KEY], eval_path_source_lengths: test_input_tensors[reader.PATH_SOURCE_LENGTHS_KEY], eval_path_lengths: test_input_tensors[reader.PATH_LENGTHS_KEY], eval_path_target_lengths: test_input_tensors[reader.PATH_TARGET_LENGTHS_KEY]})
                    if test_curr_loss > worst_loss:
                        worst_loss = test_curr_loss
                        worst_tensors = test_input_tensors
                        
                predicted_indices, top_scores, attention_weights = self.sess.run([predicted_indices_op, topk_values, predicted_indices_attention_weights], feed_dict={eval_target_index: worst_tensors[reader.TARGET_INDEX_KEY], eval_target_lengths: worst_tensors[reader.TARGET_LENGTH_KEY], eval_path_source_indices: worst_tensors[reader.PATH_SOURCE_INDICES_KEY], eval_node_indices: worst_tensors[reader.NODE_INDICES_KEY], eval_path_target_indices: worst_tensors[reader.PATH_TARGET_INDICES_KEY],  eval_valid_context_mask: worst_tensors[reader.VALID_CONTEXT_MASK_KEY], eval_path_source_lengths: worst_tensors[reader.PATH_SOURCE_LENGTHS_KEY], eval_path_lengths: worst_tensors[reader.PATH_LENGTHS_KEY], eval_path_target_lengths: worst_tensors[reader.PATH_TARGET_LENGTHS_KEY]})
                ####These are used to retrive the attention weights####
                path_source_string = worst_tensors[reader.PATH_SOURCE_STRINGS_KEY]
                path_strings = worst_tensors[reader.PATH_STRINGS_KEY]
                path_target_string = worst_tensors[reader.PATH_TARGET_STRINGS_KEY]
                path_source_string = path_source_string.reshape((-1))
                path_strings = path_strings.reshape((-1))
                path_target_string = path_target_string.reshape((-1))
                
                #####These are used in output predicted string#####
                true_target = Common.binary_to_string(worst_tensors[reader.TARGET_STRING_KEY][0])
                predicted_indices = np.squeeze(predicted_indices, axis=0)
                top_scores = np.squeeze(top_scores, axis=0)
                predicted_strings = [self.index_to_target[idx]
                                     for idx in predicted_indices]
                                     
                attention_per_path = self.get_attention_per_path(path_source_string, path_strings, path_target_string, attention_weights)                     
               
                
                result = [(true_target, predicted_strings, top_scores, attention_per_path)]
                word_attention_pairs = [(word, attention) for word, attention in
                                        zip(predicted_strings, attention_per_path) if
                                        Common.legal_method_names_checker(word)]
                
                #print("true target {}".format(true_target))
                #print("predicted target {}".format(predicted_strings))
                true_target = format_orig(true_target)
                predicted_target = format_pred(predicted_strings)
                word_attention = format_attention(word_attention_pairs)
                
                with open('{}true_target'.format(path_prefix),'a+') as f:
                    f.write(true_target+"\n")
                with open("{}predicted_target".format(path_prefix), 'a+') as g:
                    g.write(predicted_target +"\n")
                with open("{}attention_weights".format(path_prefix), 'a+') as g:
                    g.write(word_attention +"\n") 
                
        except tf.errors.OutOfRangeError:
            return    
    
    
    
    
    def evaluate(self, release=False):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                            node_to_index=self.node_to_index,
                                            target_to_index=self.target_to_index,
                                            config=self.config, is_evaluating=True)
            reader_output = self.eval_queue.get_output()
            self.eval_predicted_indices_op, self.eval_topk_values, _, _ = \
                self.build_test_graph(reader_output)
            self.eval_true_target_strings_op = reader_output[reader.TARGET_STRING_KEY]
            self.saver = tf.train.Saver(max_to_keep=10)

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
            if release:
                release_name = self.config.LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name)
                self.saver.save(self.sess, release_name)
                shutil.copyfile(src=self.config.LOAD_PATH + '.dict', dst=release_name + '.dict')
                return None
        model_dirname = os.path.dirname(self.config.SAVE_PATH if self.config.SAVE_PATH else self.config.LOAD_PATH)
        ref_file_name = model_dirname + '/ref.txt'
        predicted_file_name = model_dirname + '/pred.txt'
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)

        with open(model_dirname + '/log.txt', 'w') as output_file, open(ref_file_name, 'w') as ref_file, open(
                predicted_file_name,
                'w') as pred_file:
            num_correct_predictions = 0 if self.config.BEAM_WIDTH == 0 \
                else np.zeros([self.config.BEAM_WIDTH], dtype=np.int32)
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            self.eval_queue.reset(self.sess)
            start_time = time.time()

            try:
                while True:
                    predicted_indices, true_target_strings, top_values = self.sess.run(
                        [self.eval_predicted_indices_op, self.eval_true_target_strings_op, self.eval_topk_values],
                    )
                    true_target_strings = Common.binary_to_string_list(true_target_strings)
                    ref_file.write(
                        '\n'.join(
                            [name.replace(Common.internal_delimiter, ' ') for name in true_target_strings]) + '\n')
                    if self.config.BEAM_WIDTH > 0:
                        # predicted indices: (batch, time, beam_width)
                        predicted_strings = [[[self.index_to_target[i] for i in timestep] for timestep in example] for
                                             example in predicted_indices]
                        predicted_strings = [list(map(list, zip(*example))) for example in
                                             predicted_strings]  # (batch, top-k, target_length)
                        pred_file.write('\n'.join(
                            [' '.join(Common.filter_impossible_names(words)) for words in predicted_strings[0]]) + '\n')
                    else:
                        predicted_strings = [[self.index_to_target[i] for i in example]
                                             for example in predicted_indices]
                        pred_file.write('\n'.join(
                            [' '.join(Common.filter_impossible_names(words)) for words in predicted_strings]) + '\n')

                    num_correct_predictions = self.update_correct_predictions(num_correct_predictions, output_file,
                                                                              zip(true_target_strings,
                                                                                  predicted_strings))
                    true_positive, false_positive, false_negative = self.update_per_subtoken_statistics(
                        zip(true_target_strings, predicted_strings),
                        true_positive, false_positive, false_negative)

                    total_predictions += len(true_target_strings)
                    total_prediction_batches += 1
                    if total_prediction_batches % self.num_batches_to_log == 0:
                        elapsed = time.time() - start_time
                        self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed)
            except tf.errors.OutOfRangeError:
                pass

            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')
            # Common.compute_bleu(ref_file_name, predicted_file_name)

        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        print("Evaluation time: %sh%sm%ss" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        return num_correct_predictions / total_predictions, precision, recall, f1

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_name, predicted in results:
            original_name_parts = original_name.split(Common.internal_delimiter) # list
            filtered_original = Common.filter_impossible_names(original_name_parts) # list
            predicted_first = predicted
            if self.config.BEAM_WIDTH > 0:
                predicted_first = predicted[0]
            filtered_predicted_first_parts = Common.filter_impossible_names(predicted_first) # list

            if self.config.BEAM_WIDTH == 0:
                output_file.write('Original: ' + Common.internal_delimiter.join(original_name_parts) +
                                  ' , predicted 1st: ' + Common.internal_delimiter.join(filtered_predicted_first_parts) + '\n')
                if filtered_original == filtered_predicted_first_parts or Common.unique(filtered_original) == Common.unique(
                        filtered_predicted_first_parts) or ''.join(filtered_original) == ''.join(filtered_predicted_first_parts):
                    num_correct_predictions += 1
            else:
                filtered_predicted = [Common.internal_delimiter.join(Common.filter_impossible_names(p)) for p in predicted]

                true_ref = original_name
                output_file.write('Original: ' + ' '.join(original_name_parts) + '\n')
                for i, p in enumerate(filtered_predicted):
                    output_file.write('\t@{}: {}'.format(i + 1, ' '.join(p.split(Common.internal_delimiter)))+ '\n')
                if true_ref in filtered_predicted:
                    index_of_correct = filtered_predicted.index(true_ref)
                    update = np.concatenate(
                        [np.zeros(index_of_correct, dtype=np.int32),
                         np.ones(self.config.BEAM_WIDTH - index_of_correct, dtype=np.int32)])
                    num_correct_predictions += update
        return num_correct_predictions

    def update_per_subtoken_statistics(self, results, true_positive, false_positive, false_negative):
        for original_name, predicted in results:
            if self.config.BEAM_WIDTH > 0:
                predicted = predicted[0]
            filtered_predicted_names = Common.filter_impossible_names(predicted)
            filtered_original_subtokens = Common.filter_impossible_names(original_name.split(Common.internal_delimiter))

            if ''.join(filtered_original_subtokens) == ''.join(filtered_predicted_names):
                true_positive += len(filtered_original_subtokens)
                continue

            for subtok in filtered_predicted_names:
                if subtok in filtered_original_subtokens:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtok in filtered_original_subtokens:
                if not subtok in filtered_predicted_names:
                    false_negative += 1
        return true_positive, false_positive, false_negative

    def print_hyperparams(self):
        print('Training batch size:\t\t\t', self.config.BATCH_SIZE)
        print('Dataset path:\t\t\t\t', self.config.TRAIN_PATH)
        print('Training file path:\t\t\t', self.config.TRAIN_PATH + '.train.c2s')
        print('Validation path:\t\t\t', self.config.TEST_PATH)
        print('Taking max contexts from each example:\t', self.config.MAX_CONTEXTS)
        print('Random path sampling:\t\t\t', self.config.RANDOM_CONTEXTS)
        print('Embedding size:\t\t\t\t', self.config.EMBEDDINGS_SIZE)
        if self.config.BIRNN:
            print('Using BiLSTMs, each of size:\t\t', self.config.RNN_SIZE // 2)
        else:
            print('Uni-directional LSTM of size:\t\t', self.config.RNN_SIZE)
        print('Decoder size:\t\t\t\t', self.config.DECODER_SIZE)
        print('Decoder layers:\t\t\t\t', self.config.NUM_DECODER_LAYERS)
        print('Max path lengths:\t\t\t', self.config.MAX_PATH_LENGTH)
        print('Max subtokens in a token:\t\t', self.config.MAX_NAME_PARTS)
        print('Max target length:\t\t\t', self.config.MAX_TARGET_PARTS)
        print('Embeddings dropout keep_prob:\t\t', self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)
        print('LSTM dropout keep_prob:\t\t\t', self.config.RNN_DROPOUT_KEEP_PROB)
        print('============================================')

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed):
        accuracy_message = str(correct_predictions / total_predictions)
        throughput_message = "Prediction throughput: %d" % int(total_predictions / (elapsed if elapsed > 0 else 1))
        output_file.write(accuracy_message + '\n')
        output_file.write(throughput_message)
        # print(accuracy_message)
        print(throughput_message)

    def build_training_graph(self, input_tensors):
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_context_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model'):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                        mode='FAN_OUT',
                                                                                                        uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            nodes_vocab = tf.get_variable('NODES_VOCAB', shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            # (batch, max_contexts, decoder_size)
            batched_contexts = self.compute_contexts(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_context_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths)

            batch_size = tf.shape(target_index)[0]
            outputs, final_states = self.decode_outputs(target_words_vocab=target_words_vocab,
                                                        target_input=target_index, batch_size=batch_size,
                                                        batched_contexts=batched_contexts,
                                                        valid_mask=valid_context_mask)
            step = tf.Variable(0, trainable=False)

            logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
            target_words_nonzero = tf.sequence_mask(target_lengths + 1,
                                                    maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)

            if self.config.USE_MOMENTUM:
                learning_rate = tf.train.exponential_decay(0.01, step * self.config.BATCH_SIZE,
                                                           self.num_training_examples,
                                                           0.95, staircase=True)
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
                train_op = optimizer.minimize(loss, global_step=step)
            else:
                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

            self.saver = tf.train.Saver(max_to_keep=10)

        return train_op, loss

    def decode_outputs(self, target_words_vocab, target_input, batch_size, batched_contexts, valid_mask,
                       is_evaluating=False, dropout=None, adv_testing=False):
        if not dropout:
            dropout = self.config.RNN_DROPOUT_KEEP_PROB
        num_contexts_per_example = tf.count_nonzero(valid_mask, axis=-1)

        start_fill = tf.fill([batch_size],
                             self.target_to_index[Common.SOS])  # (batch, )
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.LSTMCell(self.config.DECODER_SIZE) for _ in range(self.config.NUM_DECODER_LAYERS)
        ])
        contexts_sum = tf.reduce_sum(batched_contexts * tf.expand_dims(valid_mask, -1),
                                     axis=1)  # (batch_size, dim * 2 + rnn_size)
        contexts_average = tf.divide(contexts_sum, tf.to_float(tf.expand_dims(num_contexts_per_example, -1)))
        fake_encoder_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(contexts_average, contexts_average) for _ in
                                   range(self.config.NUM_DECODER_LAYERS))
        projection_layer = tf.layers.Dense(self.target_vocab_size, use_bias=False)
        if is_evaluating and self.config.BEAM_WIDTH > 0:
            batched_contexts = tf.contrib.seq2seq.tile_batch(batched_contexts, multiplier=self.config.BEAM_WIDTH)
            num_contexts_per_example = tf.contrib.seq2seq.tile_batch(num_contexts_per_example,
                                                                     multiplier=self.config.BEAM_WIDTH)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.config.DECODER_SIZE,
            memory=batched_contexts
        )
        # TF doesn't support beam search with alignment history
        should_save_alignment_history = (is_evaluating or adv_testing) and self.config.BEAM_WIDTH == 0
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=self.config.DECODER_SIZE,
                                                           alignment_history=should_save_alignment_history)
        if is_evaluating:
            if self.config.BEAM_WIDTH > 0:
                decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                                batch_size=batch_size * self.config.BEAM_WIDTH)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=tf.contrib.seq2seq.tile_batch(fake_encoder_state, multiplier=self.config.BEAM_WIDTH))
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=target_words_vocab,
                    start_tokens=start_fill,
                    end_token=self.target_to_index[Common.PAD],
                    initial_state=decoder_initial_state,
                    beam_width=self.config.BEAM_WIDTH,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_words_vocab, start_fill, 0)
                initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state,
                                                          output_layer=projection_layer)

        else:
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell,
                                                         output_keep_prob=dropout)
            target_words_embedding = tf.nn.embedding_lookup(target_words_vocab,
                                                            tf.concat([tf.expand_dims(start_fill, -1), target_input],
                                                                      axis=-1))  # (batch, max_target_parts, dim * 2 + rnn_size)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_words_embedding,
                                                       sequence_length=tf.ones([batch_size], dtype=tf.int32) * (
                                                           self.config.MAX_TARGET_PARTS + 1))

            initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state,
                                                      output_layer=projection_layer)
        outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                          maximum_iterations=self.config.MAX_TARGET_PARTS + 1)
        return outputs, final_states

    def calculate_path_abstraction(self, path_embed, path_lengths, valid_contexts_mask, is_evaluating=False):
        return self.path_rnn_last_state(is_evaluating, path_embed, path_lengths, valid_contexts_mask)

    def path_rnn_last_state(self, is_evaluating, path_embed, path_lengths, valid_contexts_mask):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH,
                                                   self.config.EMBEDDINGS_SIZE])  # (batch * max_contexts, max_path_length+1, dim)
        flat_valid_contexts_mask = tf.reshape(valid_contexts_mask, [-1])  # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]),
                              tf.cast(flat_valid_contexts_mask, tf.int32))  # (batch * max_contexts)
        if self.config.BIRNN:
            rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            if not is_evaluating:
                rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw,
                                                            output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_fw,
                cell_bw=rnn_cell_bw,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths)
            final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1)  # (batch * max_contexts, rnn_size)  
        else:
            rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE)
            if not is_evaluating:
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths
            )
            final_rnn_state = state.h  # (batch * max_contexts, rnn_size)

        return tf.reshape(final_rnn_state,
                          shape=[-1, max_contexts, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)

    def compute_contexts(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,
                         target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths,
                         is_evaluating=False):

        source_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=source_input)  # (batch, max_contexts, max_name_parts, dim)
        path_embed = tf.nn.embedding_lookup(params=nodes_vocab,
                                            ids=nodes_input)  # (batch, max_contexts, max_path_length+1, dim)
        target_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=target_input)  # (batch, max_contexts, max_name_parts, dim)

        source_word_mask = tf.expand_dims(
            tf.sequence_mask(path_source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(
            tf.sequence_mask(path_target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)

        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask,
                                         axis=2)  # (batch, max_contexts, dim)
        path_nodes_aggregation = self.calculate_path_abstraction(path_embed, path_lengths, valid_mask,
                                                                 is_evaluating)  # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum],
                                  axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        batched_embed = tf.layers.dense(inputs=context_embed, units=self.config.DECODER_SIZE,
                                        activation=tf.nn.tanh, trainable=not is_evaluating, use_bias=False)

        return batched_embed

    def compute_contexts_onehot(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths, is_evaluating=False):

        # REPLACE the embedding lookup layer with a matrix multiplication between a one hot input and an embedding matrix
        print('Custom linear embedding layer')

        # source_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab, ids=source_input)  # (batch, max_contexts, max_name_parts, dim)
        source_word_onehot = tf.one_hot(source_input, depth=self.subtoken_vocab_size, axis=-1) # (batch, max_contexts, max_name_parts, |self.subtoken_vocab_size|)
        source_word_embed = tf.matmul(source_word_onehot, subtoken_vocab)
        # (batch, max_contexts, max_name_parts, embedding_dim)

        path_embed = tf.nn.embedding_lookup(params=nodes_vocab, ids=nodes_input)  # (batch, max_contexts, max_path_length+1, dim)

        # target_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab, ids=target_input)  # (batch, max_contexts, max_name_parts, dim)
        target_word_onehot = tf.one_hot(target_input, depth=self.subtoken_vocab_size, axis=-1) # (batch, max_contexts, max_name_parts, |self.subtoken_vocab_size|)
        target_word_embed = tf.matmul(target_word_onehot, subtoken_vocab)
        # (batch, max_contexts, max_name_parts, embedding_dim)

        source_word_mask = tf.expand_dims(
            tf.sequence_mask(path_source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(
            tf.sequence_mask(path_target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)

        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask,
                                         axis=2)  # (batch, max_contexts, dim)
        path_nodes_aggregation = self.calculate_path_abstraction(path_embed, path_lengths, valid_mask,
                                                                 is_evaluating)  # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum],
                                  axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        batched_embed = tf.layers.dense(inputs=context_embed, units=self.config.DECODER_SIZE,
                                        activation=tf.nn.tanh, trainable=not is_evaluating, use_bias=False)

        return batched_embed, source_word_onehot, target_word_onehot

    def build_test_graph(self, input_tensors):
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            batched_contexts = self.compute_contexts(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths,
                                                     is_evaluating=True)

            outputs, final_states = self.decode_outputs(target_words_vocab=target_words_vocab,
                                                        target_input=target_index, batch_size=tf.shape(target_index)[0],
                                                        batched_contexts=batched_contexts, valid_mask=valid_mask,
                                                        is_evaluating=True)

        if self.config.BEAM_WIDTH > 0:
            predicted_indices = outputs.predicted_ids
            topk_values = outputs.beam_search_decoder_output.scores
            attention_weights = [tf.no_op()]
        else:
            predicted_indices = outputs.sample_id
            topk_values = tf.constant(1, shape=(1, 1), dtype=tf.float32)
            attention_weights = tf.squeeze(final_states.alignment_history.stack(), 1)

        return predicted_indices, topk_values, target_index, attention_weights

    def predict(self, predict_data_lines):
        if self.predict_queue is None:
            self.predict_queue = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                               node_to_index=self.node_to_index,
                                               target_to_index=self.target_to_index,
                                               config=self.config, is_evaluating=True)
            self.predict_placeholder = tf.placeholder(tf.string)
            reader_output = self.predict_queue.process_from_placeholder(self.predict_placeholder)
            reader_output = {key: tf.expand_dims(tensor, 0) for key, tensor in reader_output.items()}
            self.predict_top_indices_op, self.predict_top_scores_op, _, self.attention_weights_op = \
                self.build_test_graph(reader_output)
            self.predict_source_string = reader_output[reader.PATH_SOURCE_STRINGS_KEY]
            self.predict_path_string = reader_output[reader.PATH_STRINGS_KEY]
            self.predict_path_target_string = reader_output[reader.PATH_TARGET_STRINGS_KEY]
            self.predict_target_strings_op = reader_output[reader.TARGET_STRING_KEY]

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        results = []
        for line in predict_data_lines:
            predicted_indices, top_scores, true_target_strings, attention_weights, path_source_string, path_strings, path_target_string = self.sess.run(
                [self.predict_top_indices_op, self.predict_top_scores_op, self.predict_target_strings_op,
                 self.attention_weights_op,
                 self.predict_source_string, self.predict_path_string, self.predict_path_target_string],
                feed_dict={self.predict_placeholder: line})

            top_scores = np.squeeze(top_scores, axis=0)
            path_source_string = path_source_string.reshape((-1))
            path_strings = path_strings.reshape((-1))
            path_target_string = path_target_string.reshape((-1))
            predicted_indices = np.squeeze(predicted_indices, axis=0)
            true_target_strings = Common.binary_to_string(true_target_strings[0])

            if self.config.BEAM_WIDTH > 0:
                predicted_strings = [[self.index_to_target[sugg] for sugg in timestep]
                                     for timestep in predicted_indices]  # (target_length, top-k)  
                predicted_strings = list(map(list, zip(*predicted_strings)))  # (top-k, target_length)
                top_scores = [np.exp(np.sum(s)) for s in zip(*top_scores)]
            else:
                predicted_strings = [self.index_to_target[idx]
                                     for idx in predicted_indices]  # (batch, target_length)  

            attention_per_path = None
            if self.config.BEAM_WIDTH == 0:
                attention_per_path = self.get_attention_per_path(path_source_string, path_strings, path_target_string,
                                                                 attention_weights)

            results.append((true_target_strings, predicted_strings, top_scores, attention_per_path))
        return results

    @staticmethod
    def get_attention_per_path(source_strings, path_strings, target_strings, attention_weights):
        # attention_weights:  (time, contexts)
        results = []
        for time_step in attention_weights:
            attention_per_context = {}
            for source, path, target, weight in zip(source_strings, path_strings, target_strings, time_step):
                string_triplet = (
                    Common.binary_to_string(source), Common.binary_to_string(path), Common.binary_to_string(target))
                attention_per_context[string_triplet] = weight
            results.append(attention_per_context)
        return results

    def save_model(self, sess, path):
        save_target = path + '_iter%d' % self.epochs_trained
        dirname = os.path.dirname(save_target)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.saver.save(sess, save_target)

        dictionaries_path = save_target + '.dict'
        with open(dictionaries_path, 'wb') as file:
            pickle.dump(self.subtoken_to_index, file)
            pickle.dump(self.index_to_subtoken, file)
            pickle.dump(self.subtoken_vocab_size, file)

            pickle.dump(self.target_to_index, file)
            pickle.dump(self.index_to_target, file)
            pickle.dump(self.target_vocab_size, file)

            pickle.dump(self.node_to_index, file)
            pickle.dump(self.index_to_node, file)
            pickle.dump(self.nodes_vocab_size, file)

            pickle.dump(self.num_training_examples, file)
            pickle.dump(self.epochs_trained, file)
            pickle.dump(self.config, file)
        print('Saved after %d epochs in: %s' % (self.epochs_trained, save_target))

    def load_model(self, sess):
        if not sess is None:
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done loading model')
        print("Loading model from: '" + self.config.LOAD_PATH + "'")
        with open(self.config.LOAD_PATH + '.dict', 'rb') as file:
            if self.subtoken_to_index is not None:
                return
            print('Loading dictionaries from: ' + self.config.LOAD_PATH)
            self.subtoken_to_index = pickle.load(file)
            self.index_to_subtoken = pickle.load(file)
            self.subtoken_vocab_size = pickle.load(file)

            self.target_to_index = pickle.load(file)
            self.index_to_target = pickle.load(file)
            self.target_vocab_size = pickle.load(file)

            self.node_to_index = pickle.load(file)
            self.index_to_node = pickle.load(file)
            self.nodes_vocab_size = pickle.load(file)

            self.num_training_examples = pickle.load(file)
            self.epochs_trained = pickle.load(file)
            saved_config = pickle.load(file)
            self.config.take_model_hyperparams_from(saved_config)
            print('Done loading dictionaries')

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None

    def build_gradient_attack_graph(self, input_tensors):
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        # shapes
        # path_source_indices: (batch_size, max_contexts, max_name_parts)


        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            batched_contexts, source_onehot, target_onehot = self.compute_contexts_onehot(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths,
                                                     is_evaluating=False)

            batch_size = tf.shape(target_index)[0]
            outputs, final_states = self.decode_outputs(target_words_vocab=target_words_vocab,
                                                        target_input=target_index, batch_size=batch_size,
                                                        batched_contexts=batched_contexts, valid_mask=valid_mask,
                                                        is_evaluating=False)

            logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
            target_words_nonzero = tf.sequence_mask(target_lengths + 1, maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)

            source_onehot_grad = tf.gradients(loss, source_onehot)[0] # shape (batch, max_contexts, max_name_parts, |self.subtoken_vocab_size|)
            target_onehot_grad = tf.gradients(loss, target_onehot)[0] # shape (batch, max_contexts, max_name_parts, |self.subtoken_vocab_size|)

            source_onehot_grad = tf.reshape(source_onehot_grad, shape=(-1, tf.shape(source_onehot_grad)[1]*tf.shape(source_onehot_grad)[2], tf.shape(source_onehot_grad)[3]))
            target_onehot_grad = tf.reshape(target_onehot_grad, shape=(-1, tf.shape(target_onehot_grad)[1]*tf.shape(target_onehot_grad)[2], tf.shape(target_onehot_grad)[3]))

            path_source_indices = tf.reshape(path_source_indices, shape=(-1, tf.shape(path_source_indices)[1]*tf.shape(path_source_indices)[2]))
            path_target_indices = tf.reshape(path_target_indices, shape=(-1, tf.shape(path_target_indices)[1]*tf.shape(path_target_indices)[2]))

            return source_onehot, target_onehot, source_onehot_grad, target_onehot_grad, path_source_indices, path_target_indices


    def run_gradient_attack(self, replace_tokens, batch_size):
        start_time = time.time()
        if self.queue is None:
            # Hack
            self.config.TEST_BATCH_SIZE = batch_size
            self.config.RNN_DROPOUT_KEEP_PROB = 1.0
            self.config.EMBEDDINGS_DROPOUT_KEEP_PROB = 1.0

            self.queue = reader.Reader(subtoken_to_index=self.subtoken_to_index, node_to_index=self.node_to_index,
                                       target_to_index=self.target_to_index, config=self.config, is_evaluating=True, indexed=True)
            reader_output = self.queue.get_output()
            _, _, self.source_onehot_grad_op, self.target_onehot_grad_op, self.path_source_indices_op, self.path_target_indices_op = self.build_gradient_attack_graph(reader_output)
            self.index_op = reader_output[reader.INDEX_KEY]
            self.path_source_lengths_op = reader_output[reader.PATH_SOURCE_LENGTHS_KEY]
            self.path_target_lengths_op = reader_output[reader.PATH_TARGET_LENGTHS_KEY]
            self.saver = tf.train.Saver(max_to_keep=10)

        if self.config.LOAD_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
        else:
            raise Exception('config.LOAD_PATH not specified')

        self.queue.reset(self.sess)
        start_time = time.time()

        best_replacements = {}

        assert self.config.BEAM_WIDTH==0

        batch_num = 0
        PRINT_FREQ = 25
        print('Start time: ',datetime.datetime.now())
        try:
            while True:
                source_onehot_grad, target_onehot_grad, path_source_indices, path_target_indices, indices = self.sess.run([self.source_onehot_grad_op, 
                                                                                                                            self.target_onehot_grad_op, 
                                                                                                                            self.path_source_indices_op, 
                                                                                                                            self.path_target_indices_op,
                                                                                                                            self.index_op])
                # print(source_onehot_grad.shape, target_onehot_grad.shape, path_source_indices.shape, path_target_indices.shape, indices.shape)
                
                d = get_best_token_replacement(path_source_indices, source_onehot_grad, path_target_indices, target_onehot_grad, indices=indices,
                                                        tok_to_idx=self.subtoken_to_index, idx_to_tok=self.index_to_subtoken,
                                                        replace_tokens=replace_tokens)

                best_replacements.update(d)

                batch_num += 1
                if batch_num%PRINT_FREQ==0:
                    print(datetime.datetime.now(), end=':   ')
                    print('Processed %d batches, %d data points'%(batch_num, batch_size*(batch_num)))

        except tf.errors.OutOfRangeError:
            pass

        elapsed = int(time.time() - start_time)
        print("Time taken: %sh%sm%ss" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        
        return best_replacements


    def run_random_attack(self, replace_tokens, batch_size):
        start_time = time.time()
        if self.queue is None:
            # Hack
            self.config.TEST_BATCH_SIZE = batch_size
            self.config.RNN_DROPOUT_KEEP_PROB = 1.0
            self.config.EMBEDDINGS_DROPOUT_KEEP_PROB = 1.0

            self.queue = reader.Reader(subtoken_to_index=self.subtoken_to_index, node_to_index=self.node_to_index,
                                       target_to_index=self.target_to_index, config=self.config, is_evaluating=True, indexed=True)
            reader_output = self.queue.get_output()
            _, _, self.source_onehot_grad_op, self.target_onehot_grad_op, self.path_source_indices_op, self.path_target_indices_op = self.build_gradient_attack_graph(reader_output)
            self.index_op = reader_output[reader.INDEX_KEY]
            self.path_source_lengths_op = reader_output[reader.PATH_SOURCE_LENGTHS_KEY]
            self.path_target_lengths_op = reader_output[reader.PATH_TARGET_LENGTHS_KEY]
            self.saver = tf.train.Saver(max_to_keep=10)

        if self.config.LOAD_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
        else:
            raise Exception('config.LOAD_PATH not specified')

        self.queue.reset(self.sess)
        start_time = time.time()

        random_replacements = {} if random else None

        assert self.config.BEAM_WIDTH==0

        batch_num = 0
        PRINT_FREQ = 25
        print('Start time: ',datetime.datetime.now())
        try:
            while True:
                path_source_indices, path_target_indices, indices = self.sess.run([
                        self.path_source_indices_op,
                        self.path_target_indices_op,
                        self.index_op])
                
                d = get_random_token_replacement(
                    path_source_indices,
                    path_target_indices,
                    indices=indices,
                    tok_to_idx=self.subtoken_to_index,
                    idx_to_tok=self.index_to_subtoken,
                    replace_tokens=replace_tokens
                )
                random_replacements.update(d)

                batch_num += 1
                if batch_num%PRINT_FREQ==0:
                    print(datetime.datetime.now(), end=':   ')
                    print('Processed %d batches, %d data points'%(batch_num, batch_size*(batch_num)))

        except tf.errors.OutOfRangeError:
            pass

        elapsed = int(time.time() - start_time)
        print("Time taken: %sh%sm%ss" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        
        return random_replacements


#######################################################
# GRADIENT ATTACK FUNCTIONS
#######################################################

import re

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

    if tok=="METHOD_NAME":
        return 'METHOD_NAME'
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



def get_best_token_replacement(source_inputs, source_grads, target_inputs, target_grads, tok_to_idx, idx_to_tok, indices, replace_tokens, distinct=True):
    '''
    *_inputs is numpy array with input vocab indices (batch, max_len)
    *_grads is numpy array (batch, max_len, vocab_size)
    idx_to_tok and tok_to_idx are source vocab maps
    indices is numpy array of size batch
    returns a dict with {index: {"@R_1@":'abc', ...}}
    '''
    def valid_replacement(s, exclude=[]):
        return classify_tok(s)=='WORDS' and s not in exclude
    
    for repl_token in replace_tokens:
        assert repl_token in tok_to_idx, "Replace token not in vocab %s"%repl_token

    assert source_inputs.shape[0]==source_grads.shape[0]
    assert target_inputs.shape[1]==target_grads.shape[1]
    assert source_inputs.shape[0]==source_grads.shape[0]
    assert target_inputs.shape[1]==target_grads.shape[1]
    assert source_inputs.shape==target_inputs.shape
    assert source_grads.shape==target_grads.shape
    assert source_inputs.shape[0]==indices.shape[0]

    best_replacements = {}    
    for i in range(source_inputs.shape[0]):
        src_inp = source_inputs[i]
        src_gradients = source_grads[i]
        tgt_inp = target_inputs[i]
        tgt_gradients = target_grads[i]
        index = str(indices[i])
        
        d = {}              
        for repl_tok in replace_tokens:
            repl_tok_idx = tok_to_idx[repl_tok]
            if repl_tok_idx not in src_inp and repl_tok_idx not in tgt_inp:
                continue
                
            src_mask = src_inp==repl_tok_idx
            tgt_mask = tgt_inp==repl_tok_idx

            src_avg_tok_grads = np.mean(src_gradients[src_mask], axis=0)
            tgt_avg_tok_grads = np.mean(tgt_gradients[src_mask], axis=0)
            
            avg_tok_grads = src_avg_tok_grads + tgt_avg_tok_grads

            exclude = list(d.values()) if distinct else []
            
            max_idx = np.argmax(avg_tok_grads)
            if not valid_replacement(idx_to_tok[max_idx], exclude=exclude):
                idxs = np.argsort(avg_tok_grads)[::-1]
                for idx in idxs:
                    if valid_replacement(idx_to_tok[idx], exclude=exclude):
                        max_idx = idx
                        break
            d[repl_tok] = idx_to_tok[max_idx]

        if len(d)>0:
            best_replacements[index] = d
    
    return best_replacements


def get_random_token_replacement(source_inputs, target_inputs, tok_to_idx, idx_to_tok, indices, replace_tokens, distinct=True):
    '''
    *_inputs is numpy array with input vocab indices (batch, max_len)
    *_grads is numpy array (batch, max_len, vocab_size)
    idx_to_tok and tok_to_idx are source vocab maps
    indices is numpy array of size batch
    returns a dict with {index: {"@R_1@":'abc', ...}}
    '''
    def valid_replacement(s, exclude=[]):
        return classify_tok(s)=='WORDS' and s not in exclude
    
    rand_replacements = {}    
    for i in range(source_inputs.shape[0]):
        src_inp = source_inputs[i]
        tgt_inp = target_inputs[i]
        index = str(indices[i])
        
        d = {}              
        for repl_tok in replace_tokens:
            repl_tok_idx = tok_to_idx[repl_tok]
            if repl_tok_idx not in src_inp and repl_tok_idx not in tgt_inp:
                continue
                
            exclude = list(d.values()) if distinct else []
            
            rand_idx = random.randint(0, len(idx_to_tok)-1)
            while not valid_replacement(idx_to_tok[rand_idx], exclude=exclude):
                rand_idx = random.randint(0, len(idx_to_tok)-1)

            d[repl_tok] = idx_to_tok[rand_idx]

        if len(d)>0:
            rand_replacements[index] = d
    
    return rand_replacements



