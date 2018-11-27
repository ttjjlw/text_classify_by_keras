# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
import random
import pandas as pd
import pickle
import numpy as np
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.layers.merge import concatenate
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten,merge,Input,Embedding,GRU,GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

def padding_sentence(data_num,padding_token=0,padding_sentence_length=None):
    max_sentence_length=padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in data_num])
    for sentence in data_num:
        if len(sentence) > max_sentence_length:
            while len(sentence)>padding_sentence_length:
                sentence.pop()
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return data_num, max_sentence_length

def shuffle_data(train_set,label_set,seed):
    zip_train_label=list(zip(train_set,label_set))
    random.Random(seed).shuffle(zip_train_label)
    train_input,label_input=zip(*zip_train_label)
    return train_input,label_input
index=0  #index=0不能丢
def generate_batch(data_num,labels,batch_size):
    global index
    x_train=[0 for j in range(batch_size)]
    y_label=[0 for k in range(batch_size)]
    length=len(data_num)
    for i in range(batch_size):
        x_train[i]=data_num[index]
        y_label[i]=labels[index]
        index+=1
        index%=length
    x_train,y_label=shuffle_data(x_train,y_label,seed=None)
    x_train=np.array(x_train)  #array[[10,21,31,41...],[11,22,333,4444,...]...]shape=batch_size*max_sentence_length
    y_label=np.array(y_label)
    return x_train,y_label


#没有embed_matrix 初始化词向量矩阵
def creat_model(sentence_length,vocal_size,embeding_dim):
    model = Sequential()
    model.add(Embedding(vocal_size+1, embeding_dim, input_length=sentence_length))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(19, activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl','rb') as f:
    data_num=pickle.load(f)
    # data_num=data_num[:10000]
with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl','rb') as f:
    dictionary=pickle.load(f)
    print('dic_len:',len(dictionary))

df=pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
label=df['class']-1
# label=label[:10000]
label=to_categorical(label,num_classes=19)
#-----------------------------------------------------数据处理部分-----------------------------------------
X_train_word_ids,X_test_word_ids,train_label,test_label=train_test_split(data_num,label,test_size=0.02,
                                                             random_state=1)

#这两代码的目的是选择部分训练集，部分长度，组建n-gram词（选的太多，n-gram会太大，后面的词向量矩阵会太大发生oom错误）
select_train=X_train_word_ids[:20000]
X_train_padding,_=padding_sentence(data_num=select_train,padding_sentence_length=50)


# 模型结构：词嵌入(n-gram)-最大化池化-全连接
# 生成n-gram组合的词(以3为例)
ngram = 2
# 将n-gram词加入到词表
def create_ngram(sent, ngram_value):
    return set(zip(*[sent[i:] for i in range(ngram_value)]))

ngram_set = set()
for sentence in X_train_padding:
    for i in range(2, ngram + 1):
        set_of_ngram = create_ngram(sentence, i)
    ngram_set.update(set_of_ngram)
# 给n-gram词汇编码
start_index = len(dictionary) + 1  #要保证dictionary除了不含0外，其他编号都不缺，即从1到max连续
token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}  # 给n-gram词汇编码
indice_token = {token_indice[k]: k for k in token_indice}
max_features = np.max(list(indice_token.keys())) + 1


# 将n-gram词加入到输入文本的末端
def add_ngram(sequences, token_indice, ngram_range):
    new_sequences = []
    for sent in sequences:
        new_list = sent[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

x_train = add_ngram(X_train_word_ids, token_indice, ngram)
x_test = add_ngram(X_test_word_ids, token_indice, ngram)
x_train,max_setence_length = padding_sentence(x_train, padding_sentence_length=1500)
x_test,_ = padding_sentence(x_test,padding_sentence_length=1500)
#-----------------------------------------------------数据处理部分----------------------------------------

model=creat_model(sentence_length=max_setence_length,vocal_size=max_features,embeding_dim=100)
for i in range(10000):
    batch_xs, batch_ys = generate_batch(data_num=x_train, labels=train_label, batch_size=64)
    cost=model.train_on_batch(batch_xs,batch_ys)
    if i%10==0:
        print('cost:',cost)
    if i%100==0 and i!=0:
        cost,accuracy=model.evaluate(np.array(x_test),np.array(test_label),batch_size=64)
        print('test:',accuracy)
model.save('fastText_model.h5')



