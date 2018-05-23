from gbrbm import GBRBM
from data_handler import DataReader
from util import tf_xavier_init
from sklearn.metrics import roc_auc_score
import datetime
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
        self.y = tf.placeholder('float', [None, 1])
        self.encoded_x = None
        self.Train_history = {'error': [], 'roc': []}
        self.Val_history = {'error': [], 'roc': []}
        assert len(layer_sizes) == len(layer_names)

    def pretrain(self, trX, epoches):
        # todo 更灵活的添加隐层
        n_hidden1 = 40
        n_hidden2 = 4
        rbm1 = GBRBM(self.input_size, n_hidden1, learning_rate=0.01, momentum=0.95, err_function='mse',
                     use_tqdm=False, sample_visible=False, sigma=1)
        rbm2 = GBRBM(n_hidden1, n_hidden2, learning_rate=0.01, momentum=0.95, err_function='mse',
                     use_tqdm=False, sample_visible=False, sigma=1)
        err1 = rbm1.fit(trX, n_epoches=epoches, batch_size=32, shuffle=True, verbose=True)
        out1 = rbm1.transform(trX)
        err2 = rbm2.fit(out1, n_epoches=epoches, batch_size=32, shuffle=True, verbose=True)
        out2 = rbm2.transform(out1)
        if not os.path.isdir('out'):
            os.mkdir('out/rbm1')
            os.mkdir('out/rbm2')
        rbm1.save_weights('./out/rbm1/rbm1', 'rbm1')
        rbm2.save_weights('./out/rbm2/rbm2', 'rbm2')
        return err1, err2

    def fulltrain(self, trX, trY, teX, teY, lr, epoches):
        self.def_network(lr)
        self.load_weight_rbms('./out/rbm1/rbm1', 'rbm1', 0)
        self.load_weight_rbms('./out/rbm2/rbm2', 'rbm2', 1)
        self.fit(trX, trY, teX, teY, n_epoches=epoches)
        self.save_model()
        self.plot_training_history()

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
        self.W = tf.Variable(tf.truncated_normal([self.layer_sizes[-1], self.FFNN_layer]), name='Weight_FFNN', dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.FFNN_layer], np.float32), name='bias_FFNN', dtype=tf.float32)
        self.W_out = tf.Variable(tf.truncated_normal([self.FFNN_layer, 1]), name='Weight_output', dtype=tf.float32)
        self.b_out = tf.Variable(tf.zeros([1], np.float32), name='bias_output', dtype=tf.float32)
        # compute cost
        # self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
        # self.cost = tf.sqrt(tf.reduce_mean(tf.square((tf.matmul(self.x, self.W) + self.b) - self.y)))

        self.y_ = tf.matmul(self.encoded_x, self.W) + self.b
        self.y_logits = tf.matmul(self.y_, self.W_out) + self.b_out
        tf.add_to_collection('pred_network', self.y_logits)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_logits, labels=self.y, 
                                                                           name='cross_entropy'))
        self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.cost)
        self.predict_op = tf.nn.sigmoid(self.y_logits)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def save_model(self):
        now = datetime.datetime.now()
        time_str = now.strftime('%Y-%m-%d %H%M%S')
        dir_str = 'out/model/'+time_str
        if not os.path.isdir('out/model'):
            os.mkdir('out/model')
            os.mkdir(dir_str)
        saver = tf.train.Saver()
        saver.save(self.sess, dir_str + '/AEFFNN') 

    def load_weight_rbms(self, filename, name, layer):
        saver = tf.train.Saver({name + '_w': self.encoding_matrices[layer],
                                name + '_h': self.encoding_biases[layer]})
        saver.restore(self.sess, filename)
    
    def fit(self,
            data_x,
            data_y,
            teX, 
            teY,
            n_epoches=10,
            batch_size=32,
            shuffle=True,
            verbose=True):
        assert n_epoches > 0
        assert len(data_x) == len(data_y) 

        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            data_y_cpy = data_y.copy()
            data = np.c_[data_x_cpy, data_y_cpy]
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x.copy()
            data_y_cpy = data_y.copy()
            data = np.c_[data_x_cpy, data_y_cpy]

        errs = []

        for e in range(n_epoches):
            if verbose:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0
            epoch_roc = []

            if shuffle:
                np.random.shuffle(inds)
                data = data[inds]

            r_batches = range(n_batches)

            for b in r_batches:
                batch_x = data[b * batch_size:(b + 1) * batch_size, :-1]
                batch_y = data[b * batch_size:(b + 1) * batch_size, -1]
                # batch_y = self.sess.run(tf.one_hot(batch_y, depth=2))
                # print(batch_x, batch_y)
                batch_cost, batch_roc = self.partial_fit(batch_x, batch_y)
                epoch_errs[epoch_errs_ptr] = batch_cost
                epoch_errs_ptr += 1
                epoch_roc.append(batch_roc)
            
            # validation
            val_cost, val_roc = self.validation(teX, teY)

            if verbose:
                err_mean = epoch_errs.mean()
                roc_mean = np.mean(epoch_roc)
                '''
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                '''
                print('Train error: {:.4f}, roc: {:.4f}'.format(err_mean, roc_mean))
                print('Val error: {:.4f}, roc: {:.4f}'.format(val_cost, val_roc))
                print('')
                sys.stdout.flush()

            self.Train_history['error'].append(epoch_errs.mean())
            self.Train_history['roc'].append(np.mean(epoch_roc))
            self.Val_history['error'].append(val_cost)
            self.Val_history['roc'].append(val_roc)
            errs = np.hstack([errs, epoch_errs])

        return errs
    
    def partial_fit(self, batch_x, batch_y):
        '''
        y_logits = self.sess.run(self.y_logits, feed_dict={self.x: batch_x})
        #print(y_logits)
        #print(self.sess.run(tf.argmax(y_logits, 1)))
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(batch_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = self.sess.run(accuracy)
        '''
        
        # print(accuracy)
        batch_y = batch_y.reshape(-1, 1)
        cost, opt, predicts = self.sess.run((self.cost, self.optimizer, self.predict_op),
                                            feed_dict={self.x: batch_x, self.y: batch_y})
        try:
            roc = roc_auc_score(batch_y, predicts)
        except ValueError:
            roc = 0.5
        return cost, roc
    
    def validation(self, teX, teY):
        teY = teY.reshape(-1, 1)
        cost, predicts = self.sess.run((self.cost, self.predict_op), feed_dict={self.x: teX, self.y: teY})
        roc = roc_auc_score(teY, predicts)
        return cost, roc

    def plot_training_history(self):
        plt.figure(1)
        plt.plot(range(len(self.Train_history['error'])), self.Train_history['error'], label='train_cost')
        plt.plot(range(len(self.Val_history['error'])), self.Val_history['error'], label='validation_cost')
        plt.xlabel('training epochs')
        plt.ylabel('cost')
        plt.legend()
        plt.title('cost history')
        plt.savefig('out/cost.png')

        plt.figure(2)
        plt.plot(range(len(self.Train_history['roc'])), self.Train_history['roc'], label='train_roc')
        plt.plot(range(len(self.Val_history['roc'])), self.Val_history['roc'], label='validation_roc')
        plt.xlabel('training epochs')
        plt.ylabel('roc')
        plt.legend()
        plt.title('roc history')
        plt.savefig('out/roc.png')
        plt.show()


if __name__ == '__main__':
    n_visible = 32
    n_class = 2
    train = DataReader('./CSI500/train/data.xlsx')
    test = DataReader('./CSI500/test/data.xlsx')
    trX, trY, teX, teY = train.rtn, train.label, test.rtn, test.label
    model = AEFFNN(n_visible, n_class, [40, 4], ['rbm1', 'rbm2'], 50)
    err1, err2 = model.pretrain(teX, 40)
    model.fulltrain(trX, trY, teX, teY, lr=0.1, epoches=10000)
