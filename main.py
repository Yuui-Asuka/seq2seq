# -*- coding:utf-8 -*-
import numpy as np
import random
import time
import sys
import os
import re
import tensorflow as tf
import math
from dynamic_seq2seq_model import dynamicSeq2seq
from tensorflow.keras.layers import LSTMCell
from attention_model import Transformer
from utils import calc_num_batches
import logging
import jieba
from tqdm import tqdm
from utils import *

class Seq2seq():

    def __init__(self):
        tf.reset_default_graph()
        self.encoder_sege_file = "./tf_data_new/enc.segement"
        self.decoder_sege_file = "./tf_data_new/dec.segement"
        self.encoder_vocabulary = "./tf_data_new/enc.vocab"
        self.decoder_vocabulary = "./tf_data_new/dec.vocab"
        self.eval_enc = "./tf_data_new/eval_enc"
        self.eval_dec = "./tf_data_new/eval_dec"
        self.vocab_file = "./tf_data_new/en_de_vocabs"
        self.batch_size = 20
        self.max_batches = 15000
        self.show_epoch = 10
        self.model_path = './model_2/'
        self.transform_model = Transformer(embedding_size=128,
                                 num_layers=6,
                                 keep_prob_rate=0.2,
                                 learning_rate=0.0001,
                                 learning_decay_rate=0.99,
                                 clip_gradient=True,
                                 is_embedding_scale=True,
                                 multihead_num=8,
                                 max_gradient_norm=5,
                                 vocab_size=40020,
                                 max_encoder_len=200,
                                 max_decoder_len=200,
                                 share_embedding=True,
                                 pad_index=0,
                                 learning_decay_steps=500,
                                 dimension_feedforword=2048,
                                 dimension_model=512,
                                 )
        self.LSTMmodel = dynamicSeq2seq(encoder_cell=LSTMCell(500),
                                          decoder_cell=LSTMCell(500),
                                          encoder_vocab_size=70824,
                                          decoder_vocab_size=70833,
                                          embedding_size=128,
                                          attention=False,
                                          bidirectional=False,
                                          debug=False,
                                          time_major=True)
        # self.dec_vocab = {}
        # self.enc_vocab = {}
        # self.dec_vecToSeg = {}
        # with open(self.encoder_vocabulary, "r",encoding = 'utf-8') as enc_vocab_file:
        #     for index, word in enumerate(enc_vocab_file.readlines()):
        #         self.enc_vocab[word.strip()] = index
        # with open(self.decoder_vocabulary, "r",encoding = 'utf-8') as dec_vocab_file:
        #     for index, word in enumerate(dec_vocab_file.readlines()):
        #         self.dec_vecToSeg[index] = word.strip()
        #         self.dec_vocab[word.strip()] = index

    def load_vocab(self, vocab_fpath):
        vocab = [line for line in open(vocab_fpath, 'r', encoding='utf-8').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token

    def load_data(self, fpath1, fpath2, maxlen1, maxlen2, mode=None):
        sents1, sents2 = [], []
        with open(fpath1, 'r', encoding='utf-8') as f1, open(fpath2, 'r', encoding='utf-8') as f2:
            for sent1, sent2 in zip(f1, f2):
                if len(sent1.split()) + 1 > maxlen1:
                    continue  # 1: __EOS__
                if len(sent2.split()) + 1 > maxlen2:
                    continue  # 1: __EOS__
                sents1.append(sent1.strip())
                sents2.append(sent2.strip())
        if mode == 'eval':
            random_num = random.randint(1, len(sents1)-101)
            sents1 = sents1[random_num: random_num + 100]
            sents2 = sents2[random_num: random_num + 100]
        return sents1, sents2

    def encode(self, inp, type, dict):
        #inp_str = inp.decode("utf-8")
        if type == "x":
            tokens = inp.split() + ["__EOS__"]
        else:
            tokens = ["__GO__"] + inp.split() + ["__EOS__"]

        x = [dict.get(t, dict["__UNK__"]) for t in tokens]
        return x

    def generator_fn(self, sents1, sents2, vocab_fpath):
        token2idx, _ = self.load_vocab(vocab_fpath)
        for sent1, sent2 in zip(sents1, sents2):
            x = self.encode(sent1, "x", token2idx)
            y = self.encode(sent2, "y", token2idx)
            decoder_input, y = y[:-1], y[1:]

            x_seqlen, y_seqlen = len(x), len(y)
            yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)

    def input_fn(self, sents1, sents2, vocab_fpath, batch_size, shuffle=False):
        shapes = (([None], (), ()),
                  ([None], [None], (), ()))
        types = ((tf.int32, tf.int32, tf.string),
                 (tf.int32, tf.int32, tf.int32, tf.string))
        paddings = ((0, 0, ''),
                    (0, 0, 0, ''))

        dataset = tf.data.Dataset.from_generator(
            self.generator_fn,
            output_shapes=shapes,
            output_types=types,
            args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

        if shuffle:  # for training
            dataset = dataset.shuffle(128 * batch_size)

        dataset = dataset.repeat()  # iterate forever
        dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

        return dataset

    def get_batch(self, fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, mode=None, shuffle=False):
        
        sents1, sents2 = self.load_data(fpath1, fpath2, maxlen1, maxlen2, mode)
        batches = self.input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
        num_batches = calc_num_batches(len(sents1), batch_size)
        return batches, num_batches, len(sents1)

    def train(self):
        train_batches, num_train_batches, num_train_samples = self.get_batch(self.encoder_sege_file,
                                                                        self.decoder_sege_file,
                                                                        100, 100,
                                                                        self.vocab_file, 32, mode=None,
                                                                        shuffle=True)
        eval_batches, num_eval_batches, num_eval_samples = self.get_batch(self.eval_enc, self.eval_dec,
                                             100000, 100000,
                                             self.vocab_file, 32,
                                             mode=None,
                                             shuffle=False)

        iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
        xs, ys = iter.get_next()

        train_init_op = iter.make_initializer(train_batches)
        eval_init_op = iter.make_initializer(eval_batches)

        logging.info("# Load model")
        model = self.transform_model
        loss, train_op, global_step, train_summaries = model.run(xs, ys)
        y_hat, eval_summaries = model.eval(xs, ys)
        #y_hat = model.infer(xs, ys)

        logging.info("# Session")
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint('model3')
            if ckpt is None:
                logging.info("Initializing from scratch")
                sess.run(tf.global_variables_initializer())
                save_variable_specs(os.path.join('model3', "specs"))
            else:
                saver.restore(sess, ckpt)

            summary_writer = tf.summary.FileWriter('model3', sess.graph)

            sess.run(train_init_op)
            total_steps = 200 * num_train_batches
            _gs = sess.run(global_step)
            for i in range(_gs, total_steps + 1):
                _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
                epoch = math.ceil(_gs / num_train_batches)
                summary_writer.add_summary(_summary, _gs)

                if _gs and _gs % num_train_batches == 0:
                    logging.info("epoch {} is done".format(epoch))
                    _loss = sess.run(loss)  # train loss
                    logging.info("# test evaluation")
                    _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
                    summary_writer.add_summary(_eval_summaries, _gs)

                    logging.info("# get hypotheses")
                    hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, model.idx2token)

                    logging.info("# write results")
                    model_output = "dialog%02dL%.2f" % (epoch, _loss)
                    print('{}   {}'.format(epoch, _loss))
                    if not os.path.exists('eval'): 
                        os.makedirs('eval')
                    translation = os.path.join('eval', model_output)
                    with open(translation, 'w') as fout:
                        fout.write("\n".join(hypotheses))

                    #logging.info("# calc bleu score and append it to translation")
                    #calc_bleu(hp.eval3, translation)

                    logging.info("# save models")
                    ckpt_name = os.path.join('model3', model_output)
                    saver.save(sess, ckpt_name, global_step=_gs)
                    logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

                    logging.info("# fall back to train mode")
                    sess.run(train_init_op)
            summary_writer.close()

    # def data_set(self, file):
    #     _ids = []
    #     with open(file, "r") as fw:
    #         line = fw.readline()
    #         while line:
    #             sequence = [int(i) for i in line.split()]
    #             _ids.append(sequence)
    #             line = fw.readline()
    #     return _ids

    # def data_iter(self, train_src, train_targets, batches, sample_num):
    #     ''' 获取batch
    #         最大长度为每个batch中句子的最大长度
    #         并将数据作转换:
    #         [batch_size, time_steps] -> [time_steps, batch_size]
    #
    #     '''
    #     batch_inputs = []
    #     batch_targets = []
    #     batch_inputs_length = []
    #     batch_targets_length = []
    #     start_tokens = []
    #     # 随机样本
    #     shuffle = np.random.randint(0, sample_num, batches)
    #     en_max_seq_length = max([len(train_src[i]) for i in shuffle])
    #     de_max_seq_length = max([len(train_targets[i]) for i in shuffle])
    #
    #     for index in shuffle:
    #         _en = train_src[index]
    #         inputs_batch_major = np.zeros(
    #             shape=[en_max_seq_length], dtype=np.int32)  # == PAD
    #         for seq in range(len(_en)):
    #             inputs_batch_major[seq] = _en[seq]
    #         batch_inputs.append(inputs_batch_major)
    #         batch_inputs_length.append(len(_en))
    #
    #         _de = train_targets[index]
    #         inputs_batch_major = np.zeros(
    #             shape=[de_max_seq_length], dtype=np.int32)  # == PAD
    #         for seq in range(len(_de)):
    #             inputs_batch_major[seq] = _de[seq]
    #         batch_targets.append(inputs_batch_major)
    #         batch_targets_length.append(len(_de))
    #         start_tokens.append(1)
    #
    #     #batch_inputs = np.array(batch_inputs).swapaxes(0, 1)
    #     #batch_targets = np.array(batch_targets).swapaxes(0, 1)
    #     #batch_size = batch_targets.shape[1]
    #     return {self.model.encoder_inputs: batch_inputs,
    #             self.model.encoder_inputs_length: batch_inputs_length,
    #             self.model.decoder_inputs: batch_targets,
    #             self.model.decoder_targets: batch_targets,
    #             self.model.decoder_targets_length: batch_targets_length,
    #             self.model.batch_size: self.batch_size,
    #             self.model.keep_prob: 0.1
    #             #self.model.start_tokens:start_tokens
    #             }

    # def train(self):
    #     # 获取输入输出
    #     train_src = self.data_set(self.encoder_vec_file)
    #     train_targets = self.data_set(self.decoder_vec_file)
    #
    #     f = open(self.encoder_vec_file)
    #     self.sample_num = len(f.readlines())
    #     f.close()
    #     print("样本数量%s" % self.sample_num)
    #
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #
    #     #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    #     #config = tf.ConfigProto(gpu_options=gpu_options)
    #
    #     with tf.Session(config=config) as sess:
    #         # 初始化变量
    #         writer = tf.summary.FileWriter('logs/', sess.graph)
    #         ckpt = tf.train.get_checkpoint_state(self.model_path)
    #         if ckpt is not None:
    #             print(ckpt.model_checkpoint_path)
    #             self.model.saver.restore(sess, ckpt.model_checkpoint_path)
    #         else:
    #             sess.run(tf.global_variables_initializer())
    #
    #         loss_track = []
    #         total_time = 0
    #         for batch in range(self.max_batches + 1):
    #             # 获取fd [time_steps, batch_size]
    #             start = time.time()
    #             fd = self.data_iter(train_src,
    #                                 train_targets,
    #                                 self.batch_size,
    #                                 self.sample_num)
    #
    #             loss, _  = sess.run([self.model.loss,
    #                                  self.model.train_op
    #                                                                              ], fd)
    #
    #             stop = time.time()
    #             total_time += (stop - start)
    #             loss_track.append(loss)
    #             if batch == 0 or batch % self.show_epoch == 0:
    #
    #                 print("-" * 50)
    #                 print("n_epoch {}".format(sess.run(self.model.global_step)))
    #                 print('  minibatch loss: {}'.format(
    #                     sess.run(self.model.loss, fd)))
    #                 print('  per-time: %s' % (total_time / self.show_epoch))
    #                 checkpoint_path = self.model_path + "nlp_chat.ckpt"
    #                 # 保存模型
    #                 self.model.saver.save(
    #                     sess, checkpoint_path, global_step=self.model.global_step)
    #
    #                 # 清理模型
    #                 self.clearModel()
    #                 total_time = 0
    #                 for i, (e_in, dt_pred) in enumerate(zip(
    #                         fd[self.model.decoder_targets],
    #                         #sess.run(self.model.decoder_prediction_train, fd).T
    #                         sess.run(self.model.decoder_outputs, fd).T
    #                 )):
    #                     print('  sample {}:'.format(i + 1))
    #                     print('    dec targets > {}'.format(e_in))
    #                     print('    dec predict > {}'.format(dt_pred))
    #                     if i >= 0:
    #                         break
    #
    # def add_to_file(self, strs, file):
    #     with open(file, "a") as f:
    #         f.write(strs + "\n")

    # def add_voc(self, word, kind):
    #     if kind == 'enc':
    #         self.add_to_file(word, self.encoder_vocabulary)
    #         index = max(self.enc_vocab.values()) + 1
    #         self.enc_vocab[word] = index
    #     else:
    #         self.add_to_file(word, self.decoder_vocabulary)
    #         index = max(self.dec_vocab.values()) + 1
    #         self.dec_vocab[word] = index
    #         self.dec_vecToSeg[index] = word
    #     return index
    #
    # def segement(self, strs):
    #     return jieba.lcut(strs)

    # def make_inference_fd(self, inputs_seq):
    #     sequence_lengths = [len(seq) for seq in inputs_seq]
    #     max_seq_length = max(sequence_lengths)
    #
    #     inputs_time_major = []
    #     start_tokens = []
    #     for sents in inputs_seq:
    #         inputs_batch_major = np.zeros(
    #             shape=[max_seq_length], dtype=np.int32)  # == PAD
    #         for index in range(len(sents)):
    #             inputs_batch_major[index] = sents[index]
    #         inputs_time_major.append(inputs_batch_major)
    #         start_tokens.append(1)
    #     inputs_time_major = np.asarray(inputs_time_major)
    #     batch_size = inputs_time_major.shape[0]
    #     inputs_time_major = np.array(inputs_time_major).swapaxes(0, 1)
    #     return {self.model.encoder_inputs: inputs_time_major,
    #             self.model.encoder_inputs_length: sequence_lengths,
    #             self.model.batch_size: batch_size,
    #             self.model.start_tokens: start_tokens}

    # def predict(self):
    #     with tf.Session() as sess:
    #         ckpt = tf.train.get_checkpoint_state(self.model_path)
    #         if ckpt is not None:
    #             print(ckpt.model_checkpoint_path)
    #             self.model.saver.restore(sess, ckpt.model_checkpoint_path)
    #         else:
    #             print("没找到模型")
    #
    #         action = False
    #         while True:
    #             if not action:
    #                 inputs_strs = input("me > ")
    #             if not inputs_strs:
    #                 continue
    #             action = False
    #             segements = self.segement(inputs_strs)
    #             # inputs_vec = [enc_vocab.get(i) for i in segements]
    #             inputs_vec = []
    #             for i in segements:
    #                 inputs_vec.append(self.enc_vocab.get(i, self.model.UNK))
    #             fd = self.make_inference_fd([inputs_vec])
    #             inf_out0 = sess.run(self.model.decoder_outputs,fd)
    #             inf_out_beam = sess.run(self.model.decoder_predict_decode_beam,fd)
    #             inf_out = sess.run(self.model.predict, fd)
    #             inf_out = [i[0] for i in inf_out]
    #             outstrs = ''
    #             for vec in inf_out:
    #                 if vec == self.model.EOS:
    #                     break
    #                 outstrs += self.dec_vecToSeg.get(vec, self.model.UNK)
    #             print(outstrs)

    # def clearModel(self, remain=3):
    #     try:
    #         filelists = os.listdir(self.model_path)
    #         re_batch = re.compile(r"nlp_chat.ckpt-(\d+).")
    #         batch = re.findall(re_batch, ",".join(filelists))
    #         batch = [int(i) for i in set(batch)]
    #         if remain == 0:
    #             for file in filelists:
    #                 if "nlp_chat" in file:
    #                     os.remove(self.model_path + file)
    #             os.remove(self.model_path + "checkpoint")
    #             return
    #         if len(batch) > remain:
    #             for bat in sorted(batch)[:-remain]:
    #                 for file in filelists:
    #                     if str(bat) in file and "nlp_chat" in file:
    #                         os.remove(self.model_path + file)
    #     except Exception as e:
    #         return
    #
    # def test(self):
    #     with tf.Session() as sess:
    #
    #         # 初始化变量
    #         sess.run(tf.global_variables_initializer())
    #
    #         # 获取输入输出
    #         train_src = [[2, 3, 5], [7, 8, 2, 4, 7], [9, 2, 1, 2]]
    #         train_targets = [[2, 3], [6, 4, 7], [7, 1, 2]]
    #
    #         loss_track = []
    #
    #         for batch in range(self.max_batches + 1):
    #             # 获取fd [time_steps, batch_size]
    #             fd = self.data_iter(train_src,
    #                                 train_targets,
    #                                 2,
    #                                 3)
    #
    #             _, loss, _, _ = sess.run([self.model.train_op,
    #                                       self.model.loss,
    #                                       self.model.gradient_norms,
    #                                       self.model.updates], fd)
    #             loss_track.append(loss)
    #
    #             if batch == 0 or batch % self.show_epoch == 0:
    #                 print("-" * 50)
    #                 print("epoch {}".format(sess.run(self.model.global_step)))
    #                 print('  minibatch loss: {}'.format(
    #                     sess.run(self.model.loss, fd)))
    #
    #                 for i, (e_in, dt_pred) in enumerate(zip(
    #                         fd[self.model.decoder_targets].T,
    #                         sess.run(self.model.decoder_prediction_train, fd).T
    #                 )):
    #                     print('  sample {}:'.format(i + 1))
    #                     print('    dec targets > {}'.format(e_in))
    #                     print('    dec predict > {}'.format(dt_pred))
    #                     if i >= 3:
    #                         break

if __name__ == '__main__':
    seq_obj = Seq2seq()
    seq_obj.train()
    #if sys.argv[1]:
    #    if sys.argv[1] == 'train':
    #        seq_obj.train()
    #    elif sys.argv[1] == 'infer':
    #        seq_obj.predict()
