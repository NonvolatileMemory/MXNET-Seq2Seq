import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import rnn
from mxnet.gluon import nn
import numpy as np
from mxnet.ndarray import softmax
import pickle

ctx = mx.cpu()
batch_size = 2
vocab_size = 60001
max_len = 55
prin_or_not = int(input("should we print or not: "))

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
        #print("x: "+str(x))
        #print("train_y: "+ str(train_y))
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
            self.gru_1 = rnn.GRU(hidden_size=1024,layout='NTC',num_layers = 4)
            self.gru_2 = rnn.GRU(hidden_size=1024,layout='NTC',num_layers = 4)
            self.mlp   = nn.Dense(units=vocab_size,flatten = False)

    def forward(self, x,y):

        input = x
        truth = y

        input = self.Embed(input)
        truth = self.Embed(truth)

        h_0 = nd.zeros((1,batch_size,50),ctx=ctx)
        all_state,final_state = self.gru_1(input,h_0)

        state,_ = self.gru_2(truth,final_state)
        result  = self.mlp(state)
        return result

    def test(self,x):
        decode_len = 10
        input = x
        input = self.Embed(input)
        res = np.zeros((batch_size,decode_len),dtype='int')
        h_0 = nd.zeros((4,batch_size,1024),ctx=ctx)
        pred = nd.zeros((batch_size,1,1))
        pred = self.Embed(pred)
        all_state,f_state = self.gru_1(input,h_0)
        #state,f_state = self.gru_2(pred,final_state)
        #pred = self.mlp(state)
        #print(pred.shape)
        #pred = nd.argmax(pred,axis = 2)
        #pred = self.Embed(pred)
        #print(pred.shape)
        #res.append(pred.asnumpy())
        for i in range(decode_len):
            state,f_state = self.gru_2(pred,f_state)
            pred = nd.argmax(self.mlp(state),axis = 2)
            pred_t = pred.asnumpy().reshape((batch_size,))
            res[:,i] = pred_t
            pred = self.Embed(pred)

        return res

seq2seq = seq2seq()
seq2seq.initialize(ctx=ctx)

max_epoch = 50000

seq2seq.load_params('seq2seq.params', ctx)  

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(seq2seq.collect_params(), 'adam', {'learning_rate': 0.001})


X = np.zeros((700000,342))

vocab = pickle.load(open("vocab.pkl",'rb'))

with open('post_test','r') as f:
    count = 0
    for line in f:
        word_count = 0
        for word in line.split(" "):
            X[count][word_count] = vocab.get(word,20001)
            word_count = word_count + 1
        count = count + 1


X = X[0:2200]

#X = np.load('train_pos.npy')
y = X

# x shape (24847, 342)

print(X.shape)

X = nd.array(X,ctx=ctx)
y = nd.array(y,ctx=ctx)

train_dataset = gluon.data.ArrayDataset(X, y)
train_data_iter = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)

def gen_sentence(array,sequence):
    tt =[]
    for index in sequence:
        word = array[int(index)]
        tt.append(word)
    return tt

pkl_file = open('list.pkl', 'rb')
train_list = pickle.load(pkl_file)

f = open("result",'w')

for epoch in range(1):
    train_loss = 0.
    train_acc = 0.
    step = 0
    res_f = []
    for x,test_y in train_data_iter:
        
        output = seq2seq.test(x)
        #output shape:batch_size*max_len*vocab_size

        for i in range(batch_size):
            res = output[i]
            tt = gen_sentence(train_list,res)
            print(res)
            f.write(" ".join(tt))
            f.write("\n")
            print(" ".join(tt))
    f.close()
