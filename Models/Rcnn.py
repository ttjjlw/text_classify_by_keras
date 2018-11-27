# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
import random
import pandas as pd
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model,model_from_json
from keras.utils import to_categorical
from keras.layers.merge import concatenate
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten,merge,Input,Embedding,GRU,LSTM,Lambda,TimeDistributed,BatchNormalization
from keras import backend
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
#全局变量
BATCH_SIZE=32
EPOCHS=8
NUM_CLASSES=19
PADDING_SENTENCE_LENGTH=1500
DROPOUT_RATE=0.5
##--------------------------------------函数定义---------------------------------------------------------
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

# 模型结构：词嵌入*3-LSTM*2-拼接-全连接-最大化池化-全连接
#embed_matrix 初始化词向量矩阵
def creat_model(sentence_length,vocal_size,embeding_dim,embed_matrix):
    # 模型共有三个输入，分别是左词，右词和中心词
    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    # 构建词向量
    embedder = Embedding(vocal_size + 1,output_dim=embeding_dim , input_length=sentence_length,weights=[embed_matrix])
    doc_embedding = embedder(document)
    l_embedding = embedder(left_context)
    r_embedding = embedder(right_context)

    # 分别对应文中的公式(1)-(7)
    forward = LSTM(256, return_sequences=True)(l_embedding)  # 等式(1)
    # 等式(2)
    backward = LSTM(256, return_sequences=True, go_backwards=True)(r_embedding)
    together = concatenate([forward, doc_embedding, backward], axis=2)  # 等式(3)
    normal_together=BatchNormalization()(together)
    semantic = TimeDistributed(Dense(128, activation="tanh"))(normal_together)  # 等式(4)
    # 等式(5)
    pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(128,))(semantic)
    normal_pool=BatchNormalization()(pool_rnn)
    drop=Dropout(rate=DROPOUT_RATE)(normal_pool)
    output = Dense(NUM_CLASSES, activation="softmax")(drop)  # 等式(6)和(7)
    model = Model(inputs=[document, left_context, right_context], outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def transform_data(X_train_word_ids,X_test_word_ids,dictionary):
    left_train_word_ids = [[len(dictionary)] + x[:-1] for x in X_train_word_ids]
    left_test_word_ids = [[len(dictionary)] + x[:-1] for x in X_test_word_ids]
    right_train_word_ids = [x[1:] + [len(dictionary)] for x in X_train_word_ids]
    right_test_word_ids = [x[1:] + [len(dictionary)] for x in X_test_word_ids]

    # 分别对左边和右边的词进行编码
    X_train_padded_seqs=pad_sequences(X_train_word_ids, maxlen=PADDING_SENTENCE_LENGTH)
    X_test_padded_seqs=pad_sequences(X_test_word_ids,maxlen=PADDING_SENTENCE_LENGTH)
    left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=PADDING_SENTENCE_LENGTH)
    left_test_padded_seqs= pad_sequences(left_test_word_ids, maxlen=PADDING_SENTENCE_LENGTH)
    right_train_padded_seqs= pad_sequences(right_train_word_ids, maxlen=PADDING_SENTENCE_LENGTH)
    right_test_padded_seqs= pad_sequences(right_test_word_ids, maxlen=PADDING_SENTENCE_LENGTH)
    return X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs
def train_model(X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs,pre_embed_matrix):
    model=creat_model(sentence_length=PADDING_SENTENCE_LENGTH,vocal_size=len(embed_matrix)-1,
                      embeding_dim=len(pre_embed_matrix[0]),embed_matrix=pre_embed_matrix)
    model_json = model.to_json()
    with open('RCNNmodel_frame.json', 'w') as f:
        f.write(model_json)
    # 这种命名方式只会保存最好的模型
    weights_path = 'RCNNweights.best.h5'
    # 这种命名方式，每当检测到模型性能提高时，就进行保存
    # weights_path='weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath=weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]
    model.fit([X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs],train_label,
               batch_size=BATCH_SIZE,
               epochs=EPOCHS,
              callbacks=callback_list,
               validation_data=([X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs],test_label))
def load_model(weights_path,model_frame_path):
    with open(model_frame_path,'r') as f:
        model_json=f.read()
    model=model_from_json(model_json)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def load_model_predict(X_test_data,left_test_data,right_test_data,test_label):
    model=load_model(weights_path='RCNNweights.best.h5',model_frame_path='RCNNmodel_frame.json')
    cost,acc=model.evaluate([X_test_data,left_test_data,right_test_data],test_label,batch_size=BATCH_SIZE)
    return cost,acc
##--------------------------------------函数定义---------------------------------------------------------

print('导入仅仅数字编码化后的数据和字典.......')
with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl','rb') as f:
    data_num=pickle.load(f)
with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl','rb') as f:
    dictionary=pickle.load(f)
    # print('dic_len:',len(dictionary))
df=pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
label=df['class']-1
label=to_categorical(label,num_classes=NUM_CLASSES)

print('导入预训练的词向量........')
with open(r'D:\localE\code\DaGuang\final_set\embeddings300_3000001.pkl','rb') as f:
    embed_matrix=pickle.load(f)
    print(len(embed_matrix))
print('切分训练数据、label和验证集数据、label......')
#我们需要重新整理数据集(只需要把数据预处理成data_num即[[sequence1],[sequence2]...]),label处理成one-hot 形式，代入以下即可
X_train_word_ids,X_test_word_ids,train_label,test_label=train_test_split(data_num,label,test_size=0.02,random_state=1)
print('针对RCNN模型在data_num基础上再一步进行数据预处理.........')
X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs=transform_data(X_train_word_ids,X_test_word_ids,dictionary=dictionary)

print('训练模型阶段：')
train_model(X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs,pre_embed_matrix=embed_matrix)
print('加载模型预测阶段：')
cost,acc=load_model_predict(X_test_data=X_test_padded_seqs,left_test_data=left_test_padded_seqs,
                            right_test_data=right_test_padded_seqs,test_label=test_label)
print(cost,'\n',acc)




