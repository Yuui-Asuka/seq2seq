# -*- coding:utf-8 -*-
import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.layers.core import Dense

class dynamicSeq2seq():
    '''
    Dynamic_Rnn_Seq2seq with Tensorflow-1.0.0

        args:
        encoder_cell            encoder结构
        decoder_cell            decoder结构
        encoder_vocab_size      encoder词典大小
        decoder_vocab_size      decoder词典大小
        embedding_size          embedd成的维度
        bidirectional           encoder的结构
                                True:  encoder为双向LSTM
                                False: encoder为一般LSTM
        attention               decoder的结构
                                True:  使用attention模型
                                False: 一般seq2seq模型
        time_major              控制输入数据格式
                                True:  [time_steps, batch_size]
                                False: [batch_size, time_steps]


    '''
    PAD = 0
    EOS = 2
    UNK = 3

    def __init__(self, encoder_cell,
                 decoder_cell,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 embedding_size,
                 bidirectional=False,
                 attention=False,
                 debug=False,
                 time_major=True):

        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size

        self.word_dict = {'PAD':0,'GO':1,'EOS':2,'UNK':3}

        self.embedding_size = embedding_size
        self.rnn_size = 512
        self.num_layers = 3
        self.keep_prob_placeholder = 0.8
        #self.encoder_cell = encoder_cell
        #self.decoder_cell = decoder_cell

        self.global_step = tf.Variable(-1, trainable=False)
        self.max_gradient_norm = 5
        self.time_major = time_major

        # 创建模型
        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            # 添加dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        # 列表中每个元素都是调用single_rnn_cell函数
        cells = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cells


    def _make_graph(self):
        # 创建占位符
        self._init_placeholders()

        # 兼容decoder输出数据
        self._init_decoder_train_connectors()

        # embedding层
        self._init_embeddings()

        # 判断是否为双向LSTM并创建encoder
        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        # 创建decoder，会判断是否使用attention模型
        self._init_decoder()

        # 计算loss及优化
        self._init_optimizer()

    def _init_placeholders(self):

        self.batch_size = tf.placeholder(
            shape=[],
            dtype=tf.int32,
            name='batch_size'
        )
        self.encoder_inputs = tf.placeholder(
            shape=[None, None],
            dtype=tf.int32,
            name='encoder_inputs',
        )
        # self.encoder_inputs = tf.Variable(np.ones((10, 50)).astype(np.int32))
        self.encoder_inputs_length = tf.placeholder(
            shape=[None],
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        self.decoder_targets = tf.placeholder(
            shape=[None, None],
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=[None],
            dtype=tf.int32,
            name='decoder_targets_length',
        )
        self.start_tokens = tf.placeholder(
            shape = [None],
            dtype=tf.int32,
            name = 'start_tokens'

        )


    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(
                tf.shape(self.decoder_targets))
            # batch_size, sequence_size = tf.unstack(tf.shape(self.decoder_targets))
            # self.encoder_inputs = tf.transpose(self.encoder_inputs, [1, 0])

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            # self.decoder_train_inputs = tf.concat(
            #     [EOS_SLICE, self.decoder_targets], axis=0, name = 'concat001')

            self.decoder_train_length = self.decoder_targets_length + 1
            # self.decoder_train_length = self.decoder_targets_length

            decoder_train_targets = tf.concat(
                 [self.decoder_targets, PAD_SLICE], axis=0)

            decoder_train_targets_seq_len, _ = tf.unstack(
                tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(
                decoder_train_targets_eos_mask, [1, 0])

            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            decoder_train_targets = tf.transpose(decoder_train_targets, [1, 0])
            self.max_target_sequence_length = tf.reduce_max(self.decoder_train_length , name='max_target_len')

            self.mask = tf.sequence_mask(self.decoder_train_length ,
                                         self.max_target_sequence_length, dtype=tf.float32,
                                         name='masks')

            #decoder_train_targets = tf.concat([tf.fill([self.batch_size, 1],
            #                                           self.word_dict['GO']), decoder_train_targets], 1)

            #self.decoder_train_length = self.decoder_train_length + 1
            self.decoder_train_targets = tf.transpose(decoder_train_targets) #这是后面加上一行2的

            self.loss_weights = tf.ones([
                self.batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.encoder_embedding_matrix = tf.get_variable(
                name="encoder_embedding_matrix",
                shape=[self.encoder_vocab_size,self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.decoder_embedding_matrix = tf.get_variable(
                name="decoder_embedding_matrix",
                shape=[self.decoder_vocab_size,self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            # encoder的embedd
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.encoder_embedding_matrix, self.encoder_inputs)

            # decoder的embedd
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.decoder_embedding_matrix, self.decoder_train_targets)

    def _init_simple_encoder(self):
        """
        一般的encdoer
        """
        with tf.variable_scope("Encoder") as scope:
            encoder_cell = self._create_rnn_cell()
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell = encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major = self.time_major,
                                  dtype=tf.float32)
            )
        a = 0

    def _init_bidirectional_encoder(self):
        '''
        双向LSTM encoder
        '''
        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=self.time_major,
                                                dtype=tf.float32)
            )

            self.encoder_outputs = tf.concat(
                (encoder_fw_outputs, encoder_bw_outputs), 2,name = 'concat003')

            if isinstance(encoder_fw_state, LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(
                    c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat(
                    (encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

    def _init_decoder(self):
        with tf.variable_scope("Decoder",reuse=tf.AUTO_REUSE) as scope:
            #fully = tf.contrib.layers.fully_connected(outputs, self.decoder_vocab_size, scope=scope)
            # self.decoder_initial_state =
            # self.decoder_cell.zero_state(batch_size=1,dtype=tf.float32).clone(cell_state=encoder_state)
            decoder_cell = self._create_rnn_cell()



            dense_layer = tf.layers.Dense(self.decoder_vocab_size,
                                          kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            encoder_outputs = tf.transpose(self.encoder_outputs,[1,0,2])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size,
                                                                    memory = encoder_outputs,
                                                                    memory_sequence_length = self.encoder_inputs_length)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell = decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size = self.rnn_size,
                                                               name='Attention_Wrapper')
            decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size,
                                                            dtype=tf.float32).clone(cell_state=self.encoder_state)
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs = self.decoder_train_inputs_embedded,
                    time_major = self.time_major,
                    sequence_length = self.decoder_train_length,
                    name='trainhelper')
            decoder_train = seq2seq.BasicDecoder(cell = decoder_cell,
                                                 helper = training_helper,
                                                 initial_state = decoder_initial_state,
                                                 output_layer=dense_layer)
            self.decoder_outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder_train,
                                                                                 output_time_major=self.time_major,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=self.max_target_sequence_length)
            scope.reuse_variables()
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_dict['GO']
            inference_helper = seq2seq.GreedyEmbeddingHelper(embedding = self.decoder_embedding_matrix,
                                                             start_tokens = self.start_tokens,
                                                             end_token = 2
                                                             )
            decoder_inference = seq2seq.BasicDecoder(
                    cell = decoder_cell,
                    helper = inference_helper,
                    initial_state = decoder_initial_state,
                    output_layer = dense_layer
                    )
            self.decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder_inference,
                                                                           impute_finished=True,
                                                                           output_time_major = True,
                                                                           maximum_iterations=20,
                                                                           scope = scope
                                                                           )
            self.decoder_predict_decode = tf.expand_dims(self.decoder_outputs.sample_id, -1)
            #decoder_initial_state = decoder_cell.zero_state(dtype = tf.float32, batch_size=1 * beam_width).clone(cell_state=tiled_encoder_final_state)

            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                     embedding = self.decoder_embedding_matrix,
                                                                     start_tokens = self.start_tokens,
                                                                     end_token = 2,
                                                                     initial_state = decoder_initial_state,
                                                                     beam_width = 1,
                                                                     output_layer = dense_layer)
            decoder_out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = inference_decoder,
                                                                  maximum_iterations = 20,
                                                                  scope = scope)
            self.decoder_predict_decode_beam = decoder_out.predicted_ids
            self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
            #self.decoder_logits_train = output_fn(self.decoder_outputs_train.rnn_output)
            #self.fully_out = tf.contrib.layers.fully_connected(self.decoder_outputs_train.rnn_output,
            #                                                   self.decoder_vocab_size,scope = scope,
            #                                                   reuse = tf.AUTO_REUSE)
            #w = tf.get_variable(name = 'w',initializer=tf.truncated_normal_initializer,
            #                    dtype=tf.float32,shape=[self.decoder_outputs_train.rnn_output.shape[-1],self.decoder_vocab_size])
            #bias = tf.get_variable(name = 'b',initializer=tf.constant_initializer,shape=[self.decoder_vocab_size])
            #self.fully_out = tf.nn.tanh(self.decoder_outputs_train.rnn_output*w+bias)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train,
                                                      axis=-1,
                                                      name='decoder_prediction_train')
            self.predict = tf.argmax(tf.transpose(self.decoder_outputs.rnn_output,[1,0,2]),axis=-1,name = 'predict')
            #self.out = tf.contrib.layers.fully_connected(self.decoder_outputs.rnn_output,
            #                                             self.decoder_vocab_size,
            #                                             scope = scope,reuse = tf.AUTO_REUSE)
            #self.predict = tf.argmax(self.out,-1,name = 'predict')
            # (self.decoder_logits_inference,
            #  self.decoder_state_inference,
            #  self.decoder_context_state_inference) = (
            #     seq2seq.dynamic_decode(decoder = decoder_inference,
            #         output_time_major=self.time_major,
            #         impute_finished=True,
            #         scope=scope
            #
            #     )
            # )
            # self.decoder_prediction_inference = tf.argmax(
            #     self.decoder_logits_inference.rnn_output, axis=-1, name='decoder_prediction_inference')

    def _init_MMI(self, logits, targets):
        sum_mmi = 0
        x_value_list = 1

    def _init_optimizer(self):
        # 整理输出并计算loss
        #logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        #targets = tf.transpose(self.decoder_train_targets, [1, 0])
        #self.logits = tf.transpose(self.fully_out, [1, 0, 2])
        self.targets = tf.transpose(self.decoder_train_targets, [1, 0])
       #self.logits = self.decoder_logits_train
        #self.targets = self.decoder_train_targets
        self.logits = tf.transpose(self.decoder_logits_train,[1, 0, 2])
        #self.targets = tf.transpose(self.decoder_train_targets)
        self.loss = seq2seq.sequence_loss(logits = self.logits, targets = self.targets,
                                          weights=self.mask)

        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss)

        # add
        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []

        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())
