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
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten,merge,Input,Embedding,GRU,BatchNormalization,Bidirectional
from sklearn.model_selection import train_test_split

BATCH_SIZE=64
FILTER_SIZE=3
PADDING_SENTENCE_LENGTH=1500
NUM_CLASSES=19
DROUPOUT_RATE=0.5
#-----------------------------------------------函数定义-------------------------------------
def padding_sentence(data_num,padding_token=0,padding_sentence_length=None):
    max_sentence_length=padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in data_num])
    for sentence in data_num:
        if len(sentence) > max_sentence_length:
            while len(sentence)>padding_sentence_length:
                sentence.pop()
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return data_num, max_sentence_length

def shuffle_data(train_set,label_set,seed=None):
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

#embed_matrix 初始化词向量矩阵
def creat_model(filter_size,sentence_length,vocal_size,embeding_dim,embed_matrix):
    # 模型结构：词嵌入-卷积池化-全连接 ---拼接-全连接
    #                -双向GRU-全连接
    main_input = Input(shape=(sentence_length,), dtype='float64')
    embed = Embedding(vocal_size+ 1, embeding_dim, input_length=sentence_length,weights=[embed_matrix])(main_input)
    cnn = Conv1D(256, filter_size, padding='same', strides=1, activation='relu')(embed)
    normal_cnn=BatchNormalization()(cnn)
    max_pool = MaxPooling1D(pool_size=sentence_length-filter_size+1)(normal_cnn)
    flatten = Flatten()(max_pool)
    cnn_out = Dense(256)(flatten)

    rnn = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(embed)
    normal_rnn=BatchNormalization()(rnn)
    rnn_out = Dense(256)(normal_rnn)
    con = concatenate([cnn_out, rnn_out], axis=-1)
    main_output = Dense(19, activation='softmax')(con)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def train_model(train_data,train_label,test_data,test_label,pre_embed_matrix,batch_size):
    model=creat_model(filter_size=FILTER_SIZE,sentence_length=PADDING_SENTENCE_LENGTH,vocal_size=len(embed_matrix)-1,
                      embeding_dim=len(pre_embed_matrix[0]),embed_matrix=pre_embed_matrix)
    max_acc=0
    for i in range(10000):
        #train_data是list or array or tuple类型，如[[1,2,3,4,5,21,31]#一句话长度为截断或pading到149,[2,3,4,5,2,3,3]#同左]
        #train_label是list or array or tuple类型，且是one-hot表示如:[[1,0],[0,1]]
        batch_xs, batch_ys = generate_batch(data_num=train_data, labels=train_label, batch_size=batch_size)
        # batch_xs是array类型，如array([[1,2,3,4,5,21,31]#一句话长度为截断或pading到149,[2,3,4,5,2,3,3]#同左])
        # batch_ys是array类型，且是one-hot表示如array[[1,0],[0,1]]
        cost=model.train_on_batch(batch_xs,batch_ys)
        if i%10==0:
            print('cost:',cost)
        if i%(len(train_data)//batch_size)==0 and i!=0:
            cost,accuracy=model.evaluate(np.array(test_data),np.array(test_label),batch_size=64)
            print('test:',accuracy)
            if max_acc < accuracy:
                max_acc = accuracy
                model.save('bin_CNN+RNN_best_model.h5')
        if i%(len(train_data)//batch_size)==0 and i!=0:
            train_data,train_label=shuffle_data(train_data,train_label,seed=None)

#-----------------------------------------------函数定义-------------------------------------

print('导入仅仅数字编码化后的数据.......')
with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl','rb') as f:
    data_num=pickle.load(f)
# with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl','rb') as f:
#     dictionary=pickle.load(f)
#     print('dic_len:',len(dictionary))
print('padding 数字编码化后的数据........')
padding_data_num,max_sentence_length=padding_sentence(data_num,padding_sentence_length=PADDING_SENTENCE_LENGTH)
df=pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
label=df['class']-1
label=to_categorical(label,num_classes=NUM_CLASSES)

print('切分训练数据、label和验证集数据、label......')
train_data,test_data,train_label,test_label=train_test_split(padding_data_num,label,test_size=0.02,
                                                             random_state=1)

print('导入预训练的词向量........')
with open(r'D:\localE\code\DaGuang\final_set\embeddings300_3000001.pkl','rb') as f:
    embed_matrix=pickle.load(f)

print('训练模型阶段：')
train_model(train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label,
            batch_size=BATCH_SIZE,pre_embed_matrix=embed_matrix)

