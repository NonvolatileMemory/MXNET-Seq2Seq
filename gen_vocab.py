import matplotlib.pyplot as plt
test_file_path = input('filename:')
count = 0
vocab_hat = {}
test_file = open(test_file_path,'r',encoding='utf-8')
for line in test_file:
    oldword = '_'
    for word in line.split(' '):
        print(line[0])
        if(line[0] == 'r'):
            if((word!= 'response') & (word!= ':')):
                if word in vocab_hat:
                    vocab_hat[word] = vocab_hat[word] + 1
                else:
                    vocab_hat[word] = 1
                oldword = word
                break;
test_file.close()

vocab_hat_vocab = vocab_hat
print(len(vocab_hat))
import numpy as np
vocab_temp = sorted(vocab_hat.items(), key = lambda d:d[1] ,reverse=True)
np_sum_test  = np.sum(np.array(list(map(lambda y: y[1],vocab_temp))))
for word,_ in vocab_hat.items():
    vocab_hat[word] = vocab_hat[word]/np_sum_test
vocab_hat = sorted(vocab_hat.items(), key = lambda d:d[1] ,reverse=True)

print(len(vocab_hat))

truth_file_path = input("trtuh_File")
count = 0
vocab_truth = {}
truth_file = open(truth_file_path,'r',encoding='utf-8')
cc = 0
index = int(input("di ji ge ci:"))

for line in truth_file:
    if(line.split(" ")[0] != " "):
 #   for word in line.split(' '):
  #      if(cc%2 == 1):
        print(len(line.split(' ')))
        if(len(line.split(' '))<index):
            word = '<pad>'
        else:
            word = line.split(' ')[index-1]
        if word in vocab_truth:
            vocab_truth[word] = vocab_truth[word] + 1
        else:
            vocab_truth[word] = 1
        oldword = word
#        break;
    cc = cc + 1
truth_file.close()

truth_temp = sorted(vocab_truth.items(), key = lambda d:d[1] ,reverse=True)
print("-------------------------")
truth_len = len(truth_temp)
a = int(input("first step: "))
b = int(input("seconde step: "))
label = list(map(lambda x: x[0],truth_temp[a:b]))
print(len(truth_temp))
np_sum_truth  = np.sum(np.array(list(map(lambda y: y[1],truth_temp))))

while(False):
    x = int(input("小于词频统计: "))
    cipin_count = 0
    for word,_ in vocab_truth.items():
        if(vocab_truth[word] < x):
            cipin_count = cipin_count + 1

    print("小于给定词频的个数 有 ： "+ str(cipin_count))

for word,_ in vocab_truth.items():
    vocab_truth[word] = vocab_truth[word]/np_sum_truth
value_truth = []
for s in label:
    print(s)
    value_truth.append(vocab_truth.get(str(s), 0))

value_test = []
for ss in label:
    value_test.append(vocab_hat_vocab.get(str(ss),0))

print("truth length is " +str(truth_len))

#plt.bar(range(len(value_truth)),value_test,alpha=0.4)
#plt.bar(range(len(value_truth)),value_truth,alpha=0.7)
plt.bar(range(len(value_truth)),value_test,tick_label = label,alpha=0.4)
plt.bar(range(len(value_truth)),value_truth,tick_label = label,alpha=0.7)
plt.show()