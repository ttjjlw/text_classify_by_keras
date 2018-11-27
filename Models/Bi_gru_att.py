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
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten,merge,Input,Embedding,GRU,Bidirectional,Permute,multiply,BatchNormalization
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
PADING_SENTENCE_LENGTH=1500
NUM_CLASSES=19
BATCH_SIZE=32
EPOCHS=10
DROPOUT_RATE=0.5
TIME_STEPS=1500#也就是padding_sentence_length

WEIGHTS_PATH='att_bi_gru_weights.best.h5'
MODEL_FRAME_PATH='att_bi_gru_model_frame.json'
#定义attention第一种方法
# first way attention
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul
#定义attention第二种方法
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

##------------------------------------------------函数定义--------------------------------------------
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


#embed_matrix 初始化词向量矩阵
def creat_model(sentence_length,vocal_size,embeding_dim,embed_matrix):
    # 模型结构：词嵌入-双向GRU-Attention-全连接
    inputs = Input(shape=(sentence_length,), dtype='float64')
    embed = Embedding(vocal_size + 1, embeding_dim, input_length=sentence_length,weights=[embed_matrix])(inputs)
    gru = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(embed)
    #attention = attention_3d_block(gru)
    # attention_flatten = Flatten()(attention)
    # drop = Dropout(0.3)(attention_flatten)
    # output = Dense(19, activation='softmax')(drop)
    #or//
    attention = AttentionLayer()(gru)
    drop=Dropout(DROPOUT_RATE)(attention)
    output=Dense(NUM_CLASSES,activation='softmax')(drop)
    #//
    model = Model(inputs, output)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def train_model(train_data,train_label,test_data,test_label,pre_embed_matrix,batch_size,epochs):
    model=creat_model(sentence_length=PADING_SENTENCE_LENGTH,vocal_size=len(pre_embed_matrix)-1,
                      embeding_dim=len(pre_embed_matrix[0]),embed_matrix=pre_embed_matrix)
    model_json=model.to_json()
    with open(MODEL_FRAME_PATH,'w') as f:
        f.write(model_json)
    #这种命名方式只会保存最好的模型
    weights_path=WEIGHTS_PATH
    #这种命名方式，每当检测到模型性能提高时，就进行保存
    #weights_path='weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint=ModelCheckpoint(filepath=weights_path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callback_list=[checkpoint]
    model.fit(x=np.array(train_data),y=np.array(train_label),batch_size=batch_size,epochs=epochs,callbacks=callback_list,
              validation_data=(np.array(test_data),np.array(test_label)))
    model.evaluate(np.array(test_data),np.array(test_label),batch_size=batch_size)
def load_model(weights_path,model_frame_path):
    with open(model_frame_path,'r') as f:
        model_json=f.read()
    model=model_from_json(model_json)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def load_model_predict(test_data,test_label):
    model=load_model(weights_path=WEIGHTS_PATH,model_frame_path=MODEL_FRAME_PATH)
    cost,acc=model.evaluate(np.array(test_data),np.array(test_label),batch_size=64)
    return cost,acc


##------------------------------------------------函数定义--------------------------------------------

print('导入仅仅数字编码化后的数据.......')
with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl','rb') as f:
    data_num=pickle.load(f)

# with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl','rb') as f:
#     dictionary=pickle.load(f)
#     print('dic_len:',len(dictionary))
print('padding 数字编码化后的数据........')

padding_data_num=pad_sequences(data_num,maxlen=PADING_SENTENCE_LENGTH)
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
            pre_embed_matrix=embed_matrix,batch_size=BATCH_SIZE,epochs=EPOCHS)

print('加载模型预测阶段：')
cost,acc=load_model_predict(test_data=test_data,test_label=test_label)
print(cost,'\n',acc)

