from gbrbm import GBRBM
from data_handler import DataReader
from utilsnn import tf_xavier_init
import os
import tensorflow as tf
import numpy as np


class AEFFNN(object):
    def __init__(self, input_size, n_class, layer_sizes, layer_names, FFNN_layer):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layer_names = layer_names
        self.n_class = n_class

        self.encoding_matrices = []
        self.encoding_biases = []
        self.FFNN_layer = FFNN_layer
        self.x = tf.placeholder("float", [None, input_size])
        self.y = tf.placeholder('float', [None, n_class])
        self.encoded_x = None
        assert len(layer_sizes) == len(layer_names)

    def pretrain(self, trX, epoches):
        # TODO 更灵活的添加隐层
        n_hidden1 = 40
        n_hidden2 = 4
        rbm1 = GBRBM(self.input_size, n_hidden1, learning_rate=0.01, momentum=0.95, err_function='mse',
                     use_tqdm=False, sample_visible=False, sigma=1)
        rbm2 = GBRBM(n_hidden1, n_hidden2, learning_rate=0.01, momentum=0.95, err_function='mse',
                     use_tqdm=False, sample_visible=False, sigma=1)
        rbm1.fit(trX, n_epoches=epoches, batch_size=32, shuffle=True, verbose=True)
        out1 = rbm1.transform(trX)
        rbm2.fit(out1, n_epoches=epoches, batch_size=32, shuffle=True, verbose=True)
        out2 = rbm2.transform(out1)
        if not os.path.isdir('out'):
            os.mkdir('out')
        rbm1.save_weights('./out/rbm1.chp', 'rbm1')
        rbm2.save_weights('./out/rbm2.chp', 'rbm2')

    def fulltrain(self, trX, trY, lr):
        self.def_network(lr)
        self.load_weight()
        # todo batch
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: trX, self.y: trY})
        return cost

    def def_network(self, lr):
        next_layer_input = self.x
        for i in range(len(self.layer_sizes)):
            dim = self.layer_sizes[i]
            input_dim = int(next_layer_input.get_shape()[1])
            W = tf.Variable(tf_xavier_init(input_dim, dim, const=1.0), dtype=tf.float32)
            b = tf.Variable(tf.zeros([dim]), dtype=tf.float32)
            self.encoding_matrices.append(W)
            self.encoding_biases.append(b)
            output = tf.nn.sigmoid(tf.matmul(next_layer_input, W) + b)
            next_layer_input = output
        self.encoded_x = next_layer_input
        # FFNN
        self.W = tf.Variable(tf.zeros([self.layer_sizes[-1], self.FFNN_layer], np.float32), name='Weight_FFNN')
        self.b = tf.Variable(tf.zeros([self.FFNN_layer], np.float32), name='bias_FFNN')
        self.W_out = tf.Variable(tf.zeros([self.FFNN_layer, self.n_class], np.float32), name='Weight_output')
        self.b_out = tf.Variable(tf.zeros([self.n_class], np.float32), name='bias_output')
        # compute cost
        # self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
        # self.cost = tf.sqrt(tf.reduce_mean(tf.square((tf.matmul(self.x, self.W) + self.b) - self.y)))

        self.y_ = tf.matmul(self.encoded_x, self.W) + self.b
        self.y_logits = tf.matmul(self.y_, self.W_out) + self.b_out
        tf.add_to_collection('pred_network', self.y_logits)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits, labels=self.y,
                                                                           name='cross_entropy'))
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def load_weight(self):
        pass


if __name__ == '__main__':
    n_visible = 33
    n_class = 2
    train = DataReader('./CSI500/train/data.xlsx')
    test = DataReader('./CSI500/test/data.xlsx')
    trX, trY, teX, teY = train.rtn, train.label, test.rtn, test.label
    model = AEFFNN(n_visible, n_class, [40, 4], ['rbm1', 'rbm2'], 0.1)
    model.pretrain(teX, 40)
