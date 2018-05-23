import tensorflow as tf
from utilsnn import xavier_init
import numpy as np


class AutoEncoder(object):
    def __init__(self, input_size, n_class, layer_sizes, layer_names, FFNN_layer, tied_weights=False, optimizer=tf.train.AdamOptimizer(0.1),
                 transfer_function=tf.nn.sigmoid):

        self.layer_names = layer_names
        self.tied_weights = tied_weights

        # Build the encoding layers
        self.x = tf.placeholder("float", [None, input_size])
        self.y = tf.placeholder('float', [None, n_class])
        next_layer_input = self.x

        assert len(layer_sizes) == len(layer_names)

        self.encoding_matrices = []
        self.encoding_biases = []
        for i in range(len(layer_sizes)):
            dim = layer_sizes[i]
            input_dim = int(next_layer_input.get_shape()[1])

            # Initialize W using xavier initialization
            W = tf.Variable(xavier_init(input_dim, dim, transfer_function), name=layer_names[i][0])

            # Initialize b to zero
            b = tf.Variable(tf.zeros([dim]), name=layer_names[i][1])

            # We are going to use tied-weights so store the W matrix for later reference.
            self.encoding_matrices.append(W)
            self.encoding_biases.append(b)

            output = transfer_function(tf.matmul(next_layer_input, W) + b)

            # the input into the next layer is the output of this layer
            next_layer_input = output

        # The fully encoded x value is now stored in the next_layer_input
        self.encoded_x = next_layer_input

        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        self.encoding_matrices.reverse()

        self.decoding_matrices = []
        self.decoding_biases = []

        for i, dim in enumerate(layer_sizes[1:] + [int(self.x.get_shape()[1])]):
            W = None
            # if we are using tied weights, so just lookup the encoding matrix for this step and transpose it
            if tied_weights:
                W = tf.identity(tf.transpose(self.encoding_matrices[i]))
            else:
                W = tf.Variable(xavier_init(self.encoding_matrices[i].get_shape()[1].value,self.encoding_matrices[i].get_shape()[0].value, transfer_function))
            b = tf.Variable(tf.zeros([dim]))
            self.decoding_matrices.append(W)
            self.decoding_biases.append(b)

            output = transfer_function(tf.matmul(next_layer_input, W) + b)
            next_layer_input = output

        # i need to reverse the encoding matrices back for loading weights
        self.encoding_matrices.reverse()
        self.decoding_matrices.reverse()

        # the fully encoded and reconstructed value of x is here:
        self.reconstructed_x = next_layer_input
        self.W = tf.Variable(tf.zeros([input_size, FFNN_layer], np.float32), name='Weight_FFNN')
        self.b = tf.Variable(tf.zeros([FFNN_layer], np.float32), name='bias_FFNN')
        self.W_out = tf.Variable(tf.zeros([FFNN_layer, n_class], np.float32), name='Weight_output')
        self.b_out = tf.Variable(tf.zeros([n_class], np.float32), name='bias_output')
        # compute cost
        # self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
        # self.cost = tf.sqrt(tf.reduce_mean(tf.square((tf.matmul(self.x, self.W) + self.b) - self.y)))
        
        self.y_ = tf.matmul(self.reconstructed_x, self.W) + self.b
        self.y_logits = tf.matmul(self.y_, self.W_out) + self.b_out
        tf.add_to_collection('pred_network', self.y_logits)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits, labels=self.y,
                                                                           name='cross_entropy'))
        self.optimizer = optimizer.minimize(self.cost)

        # initalize variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def transform(self, X):
        return self.sess.run(self.encoded_x, {self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.reconstructed_x, feed_dict={self.x: X})

    def load_rbm_weights(self, path, layer_names, layer):
        saver = tf.train.Saver({layer_names[0]: self.encoding_matrices[layer]},
                               {layer_names[1]: self.encoding_biases[layer]})
        saver.restore(self.sess, path)

        if not self.tied_weights:
            self.sess.run(self.decoding_matrices[layer].assign(tf.transpose(self.encoding_matrices[layer])))

    def print_weights(self):
        print('Matrices')
        for i in range(len(self.encoding_matrices)):
            print('Matrice', i)
            print(self.encoding_matrices[i].eval(self.sess).shape)
            print(self.encoding_matrices[i].eval(self.sess))
            if not self.tied_weights:
                print(self.decoding_matrices[i].eval(self.sess).shape)
                print(self.decoding_matrices[i].eval(self.sess))

    def load_weights(self, path):
        dict_w = self.get_dict_layer_names() 
        saver = tf.train.Saver(dict_w)
        saver.restore(self.sess, path)

    def save_weights(self, path):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        saver.save(self.sess, path)

    def get_dict_layer_names(self):
        dict_w = {}
        for i in range(len(self.layer_names)):
            dict_w[self.layer_names[i][0]] = self.encoding_matrices[i]
            dict_w[self.layer_names[i][1]] = self.encoding_biases[i]
            if not self.tied_weights:
                dict_w[self.layer_names[i][0]+'d'] = self.decoding_matrices[i]
                dict_w[self.layer_names[i][1]+'d'] = self.decoding_biases[i]
        return dict_w

    def partial_fit(self, X, y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: y})
        return cost

    def validation(self, X, y):
        y_logits = self.sess.run(self.y_logits, feed_dict={self.x: X})
        print(self.sess.run(tf.argmax(y_logits, 1)))
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y,
                                                                               name='val_cross_entropy'))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        cost, accuracy = self.sess.run((cross_entropy, accuracy))
        return cost, accuracy

    def predict(self, X):
        y_logits = self.sess.run(self.y_logits, feed_dict={self.x: X})
        return y_logits
