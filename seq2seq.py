import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import rnn
from mxnet.gluon import nn
import numpy as np
from mxnet.ndarray import softmax
import pickle

ctx = mx.gpu(1)
batch_size = 100
max_len = 55
prin_or_not = 0# int(input("should we print or not: "))
vocab_size = 60001
"""
implement a demo without padding and bucketing and no need mask
"""

def gen_data(batch_size,max_len):
    x = []
    test_y = []
    train_y = []
    for sample in range(batch_size):
        temp = []
        for index in range(max_len):
            if((sample%2==0)&(index==0)):
                temp.append(np.random.randint(1,10))
            else:
                temp.append(np.random.randint(1,10))
        x.append(temp)
        y_temp = sorted(temp)
        test_y.append(y_temp)
        y_temp = [0] + y_temp
        y_temp = y_temp[0:max_len]
        train_y.append(y_temp)
    if(prin_or_not==1):
        print("test_y: " + str(test_y[0]))

    x = nd.array(x,ctx=ctx)
    test_y = nd.array(test_y,ctx=ctx)

    train_y = nd.array(train_y,ctx=ctx)
    return x,train_y,test_y


class seq2seq(nn.Block):
    def __init__(self,**kwargs):
        super(seq2seq,self).__init__(**kwargs)
        with self.name_scope():
            self.Embed = nn.Embedding(vocab_size,200)
            self.gru_1 = rnn.GRU(hidden_size=1024,layout='NTC',num_layers=4)
            self.gru_2 = rnn.GRU(hidden_size=1024,layout='NTC',num_layers=4)
            self.mlp   = nn.Dense(units=vocab_size,flatten = False)

    def forward(self, x,y):

        input = x
        truth = y

        input = self.Embed(input)
        truth = self.Embed(truth)

        h_0 = nd.zeros((4,batch_size,1024),ctx=ctx)
        all_state,final_state = self.gru_1(input,h_0)

        state,_ = self.gru_2(truth,final_state)
        result  = self.mlp(state)
        return result

    def test(self,x):
        input = x
        input = self.Embed(input)
        h_0 = nd.zeros((1,batch_size,50),ctx=ctx)
        all_state,final_state = self.gru_1(input,h_0)

        return final_state

seq2seq = seq2seq()
seq2seq.initialize(ctx=ctx)

max_epoch = 50000
max_epoch = 20
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(seq2seq.collect_params(), 'adam', {'learning_rate': 0.001})

X = np.load('train_pos.npy')
y = np.load('train_res.npy')

T = y

X = nd.array(X,ctx=ctx)
y = nd.array(y,ctx=ctx)



train_dataset = gluon.data.ArrayDataset(X, y)
train_data_iter = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)

def gen_sentence(array,sequence):
    tt =[]
    for index in sequence:
        word = array[int(index)]
        tt.append(word)
    print(tt)

pkl_file = open('list.pkl', 'rb')
train_list = pickle.load(pkl_file)



for epoch in range(max_epoch):
    train_loss = 0.
    train_acc = 0.
    step = 0
    for x,test_y in train_data_iter:
        step = step + 1
        temp_y = test_y[:,0:max_len-1]
        train_y = nd.concat(nd.zeros((batch_size,1),ctx=ctx),temp_y,dim=1)
        print(train_y.shape)
        with autograd.record():
            output = seq2seq(x,train_y)
            output = output.reshape((batch_size*max_len,-1))
            test_y = test_y.reshape((batch_size*max_len,-1))
            loss = softmax_cross_entropy(output,test_y)
        loss.backward()
        if(prin_or_not==1):
            res = output.asnumpy()
            res = res.reshape((batch_size,max_len,-1))
            res = res[0]
            a = []
            for line in res:
                a.append(np.where(line==np.max(line))[0][0])
            print("pri: "+str(a))
        trainer.step(batch_size)
        print("epoch: " + str(epoch) +" step: "+str(step))
        print("loss:")
        print(nd.mean(loss).asscalar())
        """
        test
        """
        if(step%1000 == 1):
            print("exce test")
            res = output.asnumpy()
            res = res.reshape((batch_size, max_len, -1))
            res = res[0]
            a = []
            for line in res:
                a.append(np.where(line == np.max(line))[0][0])
            seq2seq.save_params("no_focal_loss.params")
            gen_sentence(train_list,a)
    seq2seq.save_params("no_focal_loss" + str(epoch%10) + ".params")
