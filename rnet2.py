import tensorflow as tf
import util

class RNet2(object):
    # constructor
    def __init__(self,
            n_outputs,
            kernel_size,
            strides,
            dilation,
            window,
            dropout_keep = 0.8,
            seq_len = 300):
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.dropout_keep = dropout_keep
        self.window_size = window - 1


    # forward
    def __call__(self, x):
        # x: [num_sentences, max_sentence_length, emb]
        # x: [batch_size, seq_len, input_size]
        seq_len = util.shape(x, 1)
        n_inputs = util.shape(x, 2)
        batch_size = util.shape(x, 0)

        mask_diag = tf.expand_dims(tf.eye(seq_len), 0) # [1, seq_len, seq_len]
        mask_tril = tf.ones([seq_len, seq_len]) # [seq_len, seq_len]
        seq = tf.reshape(tf.range(seq_len), [1, -1]) #[1. seq_len]
        seq_mask = self._seq_mask_maker(seq, self.window_size) # [seq_len, seq_len]
        mask_tril = tf.expand_dims((mask_tril - seq_mask), 0) # [1, seq_len, seq_len]

        w = tf.get_variable("w", shape = [self.kernel_size, n_inputs, self.n_outputs * 4])
        b = tf.get_variable("b", shape = [self.n_outputs * 4])
        conv1 = tf.nn.conv1d(x, 
                w, 
                stride = self.strides,
                padding = 'SAME')
        conv1 = tf.nn.bias_add(conv1, b)
        conv1 = tf.nn.l2_normalize(conv1)

        z, e, k, o = tf.split(conv1, num_or_size_splits = 4, axis = 2)
        # [batch_size, seq_len, hidden_size]
        z = tf.nn.relu(z)
        e = tf.tanh(e)
        k = tf.tanh(k)
        o = tf.sigmoid(o)

        A = self._build_graph_simple(e, k)
        A += tf.get_variable("graph_bias", [])
        A = tf.nn.relu(A)
        A_normed = self._simple_norm(A, mask_tril, mask_diag)
        # [batch_size, seq_len, seq_len]
        
        z_att = tf.matmul(A_normed, z)
        r_net_inp = tf.concat([z_att, z], 2) 
        r_net = tf.contrib.layers.fully_connected(inputs = r_net_inp,
                                                    num_outputs = self.n_outputs,
                                                    activation_fn = tf.sigmoid)
        # i, r = tf.split(ir_net, num_or_size_splits = 2, axis = 2)
        r = r_net
        
        h_net_inp = tf.concat([z_att, r * z], 2)
        h_net = tf.contrib.layers.fully_connected(inputs = h_net_inp, 
                                                    num_outputs = self.n_outputs, 
                                                    activation_fn = tf.nn.relu)
        #ax = tf.contrib.layers.fully_connected(inputs = ax,
        #                                            num_outputs = self.n_outputs,
        #                                            activation_fn = tf.nn.relu)
        ax = h_net # [batch_size, seq_len, hidden_size]
        i_net_inp = tf.concat([x, ax], 2)
        i_net = tf.contrib.layers.fully_connected(inputs = i_net_inp, 
                                                    num_outputs = self.n_outputs,
                                                    activation_fn = tf.sigmoid)
        i = i_net
        
        self.downsample = tf.layers.conv1d(inputs = x,
                                            filters = self.n_outputs,
                                            kernel_size = 1) if n_inputs != self.n_outputs else None
        res = x if self.downsample is None else self.downsample
        # [batch_size, seq_len, hidden_size]
        h = (1 - i) * res + i * ax
        #h = i * res + r * ax
        outputs = (o * h)
        outputs = tf.nn.dropout(outputs, self.dropout_keep)
        
        return outputs


    def _build_graph_simple(self, e, k):
        return tf.matmul(e, tf.transpose(k, [0, 2, 1]))


    def _simple_norm(self, A, mask_tril, mask_diag):
        A_tril = A * (mask_tril - mask_diag)
        A_normed = A_tril / (tf.reduce_sum(A_tril, axis = 2, keepdims = True) + 1e-6)
        return A_normed


    def _seq_mask_maker(self, seq, window_size):
        dif = seq - tf.transpose(seq)
        left_mask = tf.cast(dif < -window_size, tf.float32)
        right_mask = tf.cast(dif > window_size, tf.float32)
        seq_mask = left_mask + right_mask
        return seq_mask


    def _build_graph_cum(self, e, k):
        batch_size = util.shape(e, 0)
        seq_len = util.shape(e, 1)
        hidden_size = util.shape(e, 2)
        mask_cum = tf.contrib.layers.fully_connected(inputs = tf.concat([e, k], 2),
                                                        num_outputs = self.n_outputs,
                                                        activation_fn = tf.nn.relu)
        mask_cum = tf.matmul(mask_cum, tf.transpose(mask_cum, [0, 2, 1]))
        # [batch_size, seq_len, seq_len]
        mask_cum = tf.tile(tf.expand_dims(mask_cum, 3), [1, 1, 1, hidden_size])
        # [batch_size, seq_len, seq_len, hidden_size]
        cum = tf.tile(tf.expand_dims(e, 2), [1, 1, seq_len, 1])
        # [batch_size, seq_len, seq_len, hidden_size]
        cum = cum * mask_cum
        cum = tf.reduce_sum(cum, axis = 2) # [batch_size, seq_len, hidden_size]
        return tf.matmul(cum, tf.transpose(k, [0, 2, 1]))

        