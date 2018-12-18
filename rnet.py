import tensorflow as tf
import util

class RNet(object):

    # constructor
    def __init__(self,
            n_outputs, 
            kernel_size, 
            strides,
            dilation, 
            window, 
            dropout_keep = 0.8, 
            seq_len = 300):
        # parameter
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.window = window
        self.dropout_keep = dropout_keep
        self.window_size = window - 1


    # forward
    def __call__(self, x): 
        # x: [num_sentences, max_sentence_length, emb]
        # x: [batch_size, seq_len, input_size]
        
        seq_len = util.shape(x, 1)
        # batch_size = tf.shape(x)[0]
        n_inputs = util.shape(x, 2)
        
        # initiate masks
        mask_diag = tf.expand_dims(tf.eye(seq_len), 0) # [1, seq_len, seq_len]
        mask_tril = tf.ones([seq_len, seq_len]) # [1, seq_len, seq_len]
        seq = tf.reshape(tf.range(seq_len), [1, -1])
        # take only left and right words in the given window size
        seq_mask = self._seq_mask_maker(seq, self.window_size) # [seq_len, seq_len]
        # cut upper right and lower left corner
        mask_tril = tf.expand_dims((mask_tril - seq_mask), 0) 
        # [1, seq_len, seq_len]

        w = tf.get_variable("w", shape=[self.kernel_size, n_inputs, self.n_outputs * 4])
        b = tf.get_variable("b", shape=[self.n_outputs * 4])
        conv1 = tf.nn.conv1d(x,
                w,
                stride=self.strides,
                padding="SAME",
                # data_format="NCW"
                )
        conv1 = tf.nn.bias_add(conv1, b)
        conv1 = tf.nn.l2_normalize(conv1)
        # get z e o f: [batch_size, seq_len, hidden_size]
        z, o, e, k = tf.split(conv1, num_or_size_splits = 4, axis = 2)
        z = tf.tanh(z) #nn.relu
        o = tf.sigmoid(o)
        e = tf.tanh(e)
        k = tf.tanh(k)
        
        # get edge matrix A and ax
        b = tf.get_variable("graph_bias", [])
        A = self._build_graph_simple(e, k) # weighted cum graph bidirectional
        A = A + b
        A = tf.tanh(A) # nn.relu
        A_normed = self._simple_norm(A, mask_tril, mask_diag) 
        # [batch_size, seq_len, seq_len]
        z = tf.contrib.layers.fully_connected(inputs = z,
                                                num_outputs = self.n_outputs,
                                                activation_fn = tf.tanh) #nn.relu
        ax = tf.concat((tf.matmul(A_normed, z), z), axis = 2)
        ax = tf.contrib.layers.fully_connected(inputs = ax,
                                                num_outputs = self.n_outputs,
                                                activation_fn = tf.tanh) #nn.relu

        # downsample
        self.downsample = tf.layers.conv1d(inputs = x,
                                                filters = self.n_outputs, 
                                                kernel_size = 1) if n_inputs != self.n_outputs else None
        res = x if self.downsample is None else self.downsample
        
        # final process 
        outputs = o * (ax + res)
        outputs = tf.nn.dropout(outputs, self.dropout_keep)
        return outputs


    def _build_graph_cum(self, e, k, mf, mb): # cumulative edge graph based on all previous words
        # cumulate
        hidden_size = util.shape(e, 2) # e: [batch_size, seq_len, hidden_size]
        seq_len = util.shape(e, 1)
        batch_size = util.shape(e, 0)
        cum = tf.tile(tf.expand_dims(e, 2), [1, 1, seq_len, 1]) # [batch_size, seq_len, seq_len(new), hidden_size]
        # mask_cum = tf.expand_dims(tf.range(seq_len) - tf.transpose(tf.range(seq_len)), 0) # [1, seq_len, seq_len]
        # mask_cum = tf.cast(mask_cum >= 0, tf.float32)
        # tril half of the matrix
        mask_cum_tril = tf.expand_dims(tf.range(seq_len) - tf.transpose(tf.range(seq_len)), 0) # [1, seq_len, seq_len]
        mask_cum_tril_f = tf.cast(mask_cum_tril >= 0, tf.float32)
        mask_cum_tril_b = tf.cast(mask_cum_tril < 0, tf.float32)
        mask_cum_tril_f = tf.tile(tf.expand_dims(mask_cum_tril_f, 2), [1, 1, hidden_size])
        mask_cum_tril_f = tf.tile(tf.expand_dims(mask_cum_tril_f, 0), [batch_size, 1, 1, 1]) 
        mask_cum_tril_b = tf.tile(tf.expand_dims(mask_cum_tril_b, 2), [1, 1, hidden_size])
        mask_cum_tril_b = tf.tile(tf.expand_dims(mask_cum_tril_b, 0), [batch_size, 1, 1, 1])
        # [batch_size, seq_len, seq_len, hidden_size]
        mask_cum_f = tf.contrib.layers.fully_connected(inputs = mf, 
                                                        num_outputs = self.n_outputs,
                                                        activation_fn = tf.nn.relu)
        # [batch_size, seq_len, hidden_size]
        mask_cum_b = tf.contrib.layers.fully_connected(inputs = mb,
                                                        num_outputs = self.n_outputs, 
                                                        activation_fn = tf.nn.relu)
        # mask_cum = m
        # [batch_size, seq_len, hidden_size]
        # mask_cum = tf.tile(tf.expand_dims(mask_cum, 2), [1, 1, seq_len, 1])
        # mask_cum = tf.matmul(m, tf.transpose(mask_cum, [0, 2, 1])) # [batch_size, seq_len, seq_len]
        mask_cum_f = tf.matmul(mask_cum_f, tf.transpose(mask_cum_f, [0, 2, 1])) # [batch_size, seq_len, seq_len, hidden_size]
        mask_cum_f = tf.tile(tf.expand_dims(mask_cum_f, 3), [1, 1, 1, hidden_size])
        mask_cum_b = tf.matmul(mask_cum_b, tf.transpose(mask_cum_b, [0, 2, 1]))
        mask_cum_b = tf.tile(tf.expand_dims(mask_cum_b, 3), [1, 1, 1, hidden_size])
        mask_cum_f = mask_cum_f * mask_cum_tril_f
        mask_cum_b = mask_cum_b * mask_cum_tril_b
        # [batch_size, seq_len, seq_len, hidden_size]
        # mask_cum = mask_cum * mask_cum_tril
        # mask_cum = tf.tile(tf.expand_dims(mask_cum, 2), [1, 1, hidden_size]) 
        # mask_cum = tf.tile(tf.expand_dims(mask_cum, 0), [batch_size, 1, 1, 1]) # [batch_size, seq_len, seq_len, hidden_size]
        # cum = cum * mask_cum
        cum_f = cum * mask_cum_f
        cum_b = cum * mask_cum_b
        cum = cum_f + cum_b
        cum = tf.reduce_sum(cum, axis = 2)
        # cum = cum / tf.reduce_sum(mask_cum, axis = 2)
        # cum = tf.transpose(cum, perm = [0, 2, 1])
        # e * k
        # return tf.matmul(e, tf.transpose(k, [0, 2, 1]))
        return tf.matmul(cum, tf.transpose(k, [0, 2, 1]))
    
    def _build_graph_cum_2(self, e, k):
        batch_size = util.shape(e, 0)
        seq_len = util.shape(e, 1)
        hidden_size = util.shape(e, 2)
        mask_cum = tf.contrib.layers.fully_connected(inputs = tf.concat([e, k], 2),
                                                        num_outputs = self.n_outputs,
                                                        activation_fn = tf.nn.relu)
        mask_cum = tf.matmul(mask_cum, tf.transpose(mask_cum, [0, 2, 1]))
        mask_cum = tf.tile(tf.expand_dims(mask_cum, 3), [1, 1, 1, hidden_size])
        cum = tf.tile(tf.expand_dims(e, 2), [1, 1, seq_len, 1])
        # [batch_size, seq_len, seq_len, hidden_size]
        cum = cum * mask_cum
        cum = tf.reduce_sum(cum, axis = 2)
        return tf.matmul(cum, tf.transpose(k, [0, 2, 1]))
        


    def _build_graph_simple(self, e, k):
        return tf.matmul(e, tf.transpose(k, [0, 2, 1]))


    def _build_graph_naive(self, e, z, seq_len):
        e_tiled = tf.tile(tf.expand_dims(e, 1), [1, seq_len, 1, 1])
        # [batch_size, seq_len, seq_len, hidden_size] 
        z_tiled = tf.transpose(tf.tile(tf.expand_dims(z, 1), [1, seq_len, 1, 1]), 
                                perm = [0, 2, 1, 3])
        # [batch_size, seq_len, seq_len, hidden_size]
        b = tf.get_variable("graph_bias", [self.n_outputs])
        dot = e_tiled * z_tiled + b
        # [batch_size, seq_len, seq_len, hidden_size]
        return tf.reduce_sum(dot, axis = 3)


    def _simple_norm(self, A, mask_tril, mask_diag):
        A_tril = A * mask_tril
        A_normed = A_tril / (tf.reduce_sum(A_tril, axis=2, keepdims=True) + 0.001)
        return A_normed * (mask_tril - mask_diag)


    def _seq_mask_maker(self, seq, window_size): # seq: [seq_len, seq_len]
        dif = seq - tf.transpose(seq)
        left_mask = tf.cast(dif < -window_size, tf.float32)
        right_mask = tf.cast(dif > window_size, tf.float32)
        seq_mask = left_mask + right_mask
        return seq_mask

        
