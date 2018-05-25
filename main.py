from model import AEFFNN
from data_handler import DataReader
import pandas as pd


# todo 保存&载入模型
n_visible = 32
n_class = 2
train1 = pd.read_hdf('D:/alpha data/20120101_20130101.h5')
train2 = pd.read_hdf('D:/alpha data/20130101_20140101.h5')
train3 = pd.read_hdf('D:/alpha data/20140101_20150101.h5')
train4 = pd.read_hdf('D:/alpha data/20150101_20160101.h5')
train5 = pd.read_hdf('D:/alpha data/20160101_20170101.h5')
train = pd.concat([train1, train2, train3, train4, train5])
test = pd.read_hdf('20170101_20180101.h5')
trX, trY, teX, teY = train.rtn, train.label, test.rtn, test.label
# model.pretrain(teX, 40)
# todo 验证集设为可选参数
model = AEFFNN(n_visible, n_class, layer_sizes=[16, 8], layer_names=['rbm1', 'rbm2'], FFNN_layer=32)
model.pretrain(trX, 500)
# model.fulltrain(trX, trY, teX, teY, lr=0.1, epoches=10000)

# test
'''
test = DataReader('./CSI500/test/data.xlsx')
teX = test.rtn
predicts = model.sess.run(model.predict_op, feed_dict={model.x: teX})
predicts = pd.Series(predicts.reshape(1, -1)[0])
predicts.to_excel('predicts.xlsx')
'''