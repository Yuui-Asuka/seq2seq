from data_load import load_vocab
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging
import numpy as np
import tensorflow as tf

class Transformer:
    def __init__(self,
                 embedding_size,
                 num_layers,
                 keep_prob_rate,
                 learning_rate,
                 learning_decay_steps,
                 learning_decay_rate,
                 clip_gradient,
                 is_embedding_scale,
                 multihead_num,
                 max_gradient_norm,
                 vocab_size,
                 max_encoder_len,
                 max_decoder_len,
                 share_embedding,
                 pad_index,
                 dimension_model,
                 dimension_feedforword,
                 sinusoid=False,
                 mode=True):
        self.embeddings = self.get_token_embeddings(vocab_size, dimension_model, zero_pad=True)
        self.num_layers = num_layers
        self.drop_rate = keep_prob_rate
        self.learning_rate = learning_rate
        self.decay_step = learning_decay_steps
        self.decay_rate = learning_decay_rate
        self.clip_gradient = clip_gradient
        self.scale = is_embedding_scale
        self.multhead_num = multihead_num
        self.max_gradient_norm = max_gradient_norm
        self.vocab_size = vocab_size
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.dimension_model = dimension_model
        self.dimension_feedforword = dimension_feedforword
        self.sinusoid = sinusoid
        self.mode = mode
        self.token2idx, self.idx2token = load_vocab("./tf_data_new/en_de_vocabs")

    def run(self, inputs, targets):
        memory, sents1, src_masks = self.encoder(inputs)
        logits, predict, target, sents2 = self.decoder(targets, memory, src_masks)
        y_ = self.label_smoothing(tf.one_hot(target, depth=self.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(target, 0))
        loss = tf.reduce_sum(ce*nonpadding)/(tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        step = tf.cast(global_step+1, dtype=tf.float32)
        lr = self.learning_rate * self.decay_step * tf.minimum(step * self.decay_step ** -1.5
                                                               , step ** -0.5)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('global_step', global_step)

        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        #decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["__GO__"]
        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * 2
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encoder(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(50)):
            logits, y_hat, y, sents2 = self.decoder(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["__PAD__"]:
                break
            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)
        
        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

    @staticmethod
    def get_token_embeddings(vocab_size, num_units, zero_pad=True):

        with tf.variable_scope("shared_weight_matrix"):
            embeddings = tf.get_variable('weight_mat',
                                         dtype=tf.float32,
                                         shape=(vocab_size, num_units),
                                         initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                        embeddings[1:, :]), 0)
        return embeddings

    @staticmethod
    def positional_encoding(inputs,
                            maxlen,
                            masking=True,
                            scope="positional_encoding"):

        E = inputs.get_shape().as_list()[-1]  # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            return tf.to_float(outputs)

    @staticmethod
    def mask(inputs, key_masks=None, type=None):
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            key_masks = tf.to_float(key_masks)
            key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
            key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
            outputs = inputs + key_masks * padding_num
            return outputs

        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

            paddings = tf.ones_like(future_masks) * padding_num
            outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
            return outputs
        else:
            print("Check if you entered type correctly!")

    @staticmethod
    def ln(inputs, epsilon=1e-8, scope="ln"):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** .5)
            outputs = gamma * normalized + beta

        return outputs

    @staticmethod
    def label_smoothing(inputs, epsilon=0.1):
        V = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / V)

    def scaled_dot_product_attention(self, Q, K, V, key_masks,
                                     causality=False, dropout_rate=0.,
                                     training=True,
                                     scope="scaled_dot_product_attention"):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            # key masking
            outputs = self.mask(outputs, key_masks=key_masks, type="key")

            # causality or future blinding masking
            if causality:
                outputs = self.mask(outputs, type="future")

            # softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # # query masking
            # outputs = mask(outputs, Q, K, type="query")

            # dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def multihead_attention(self, queries, keys, values, key_masks,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_attention"):
        d_model = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.ln(outputs)
        return outputs



    def encoder(self, inputs, is_training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            encoder_inputs, seqlen, sents1 = inputs

            src_masks = tf.math.equal(encoder_inputs, 0)

            encoder = tf.nn.embedding_lookup(self.embeddings, encoder_inputs)
            encoder *= self.dimension_model**0.5

            encoder += self.positional_encoding(encoder, self.max_encoder_len)
            encoder = tf.layers.dropout(encoder, self.drop_rate, training=is_training)


            for i in range(self.num_layers):
                with tf.variable_scope('num_blocks_{}'.format(i), reuse=tf.AUTO_REUSE):
                    encoder = self.multihead_attention(queries=encoder,
                                                  keys=encoder,
                                                  values=encoder,
                                                  key_masks=src_masks,
                                                  num_heads=self.multhead_num,
                                                  dropout_rate=self.drop_rate,
                                                  training=is_training,
                                                  causality=False)
                    encoder = self.ff(encoder, num_units=[self.dimension_feedforword, self.dimension_model])
        memory = encoder
        return memory, sents1, src_masks

    def ff(self, inputs, num_units, scope="positionwise_feedforward"):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.ln(outputs)

        return outputs

    def decoder(self, targets, memory, src_masks, is_training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            decoder_inputs, decoder_targets, seqlen, sents2 = targets

            target_masks = tf.math.equal(decoder_inputs, 0)

            decoder = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)
            decoder *= self.dimension_model**0.5

            decoder += self.positional_encoding(decoder, self.max_decoder_len)
            decoder = tf.layers.dropout(decoder, self.drop_rate, training=is_training)

            for i in range(self.num_layers):
                with tf.variable_scope('num_blocks_{}'.format(i), reuse=tf.AUTO_REUSE):
                    dec = self.multihead_attention(queries=decoder,
                                              keys=decoder,
                                              values=decoder,
                                              key_masks=target_masks,
                                              num_heads=self.multhead_num,
                                              dropout_rate=self.drop_rate,
                                              training=is_training,
                                              causality=True,
                                              scope='self_attention'
                                              )

                    dec = self.multihead_attention(queries=decoder,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.multhead_num,
                                              dropout_rate=self.drop_rate,
                                              training=is_training,
                                              causality=False,
                                              scope='vanilla_attention'
                                              )

                    decoder = self.ff(dec, num_units=[self.dimension_feedforword, self.dimension_model])

        weights = tf.transpose(self.embeddings)
        logits = tf.einsum('ntd,dk->ntk', decoder, weights)
        predict = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, predict, decoder_targets, sents2






























