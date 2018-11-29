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
from keras import backend as K
import sklearn as sk
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import time
#训练策略：
#1/预训练词向量先保持trainable=false,Truncat=pre 训练2-3个epoch,2/再post 训练2-3个epoch
#3/trainable=true ,前向截断训练1-2个epoch，4/后向截断训练1-2epoch
#再shuffle每篇文章，训练1-2个epoch
#再reverse每篇,训练1-2个epoch
def Gettime():
    t=time.asctime(time.localtime(time.time()))
    t=t.split()[2:4]
    d=[t[0]]+t[1].split(':')[:-1]
    d='_'.join(d)
    return d
T=Gettime()
#全局变量
BATCH_SIZE=32
EPOCHS=1
NUM_CLASSES=19
PADDING_SENTENCE_LENGTH=150
DROPOUT_RATE=0.5

model_name='Rcnn/'
WEIGHTS_PATH="save_modelRcnn/29_19_50_False_orderly_pre_weights_best.h5"#载入模型时用，按需修改
Truncat='pre'   #句子是前向截断还是后向截断
SHUFFLE='orderly'   #每行样本进行[shuffe,reverse],为其他时就不处理
IS_LOAD=True if WEIGHTS_PATH else False     #是否载入模型，分为载入模型训练和载入模型预测
TRAINABLE=False   #embeddings是否可以训练
frame_dir='frame/'+model_name
MODEL_FRAME_PATH=frame_dir+str(TRAINABLE)+'_model_frame.json'
model_dir='save_model/'+model_name
#模型保存位置
SAVE_WEIGHTS_PATH=model_dir+T+'_'+str(TRAINABLE)+'_'+str(SHUFFLE)+'_'+Truncat+'_'+'weights_best.h5'
MODE='train'    #训练模式还是测试测试集模型
##--------------------------------------函数定义---------------------------------------------------------
def filter_tail(data_num,label):
    assert (len(data_num)==len(label))

    for line in data_num:  # [44, 47, 41, 18, 14, 45]
        if len(line) >= 6:
            if line[-6:] == [44, 47, 41, 18, 14, 45]:
                for i in range(6):
                    line.pop()
    index=0
    remove=[]
    for line in data_num:
        if len(line)<6:
            if line==[word_num for word_num in line if word_num in [44, 47, 41, 18, 14, 45]]:
                remove.append(index)
        index+=1
    filter_data=[]
    filter_label=[]
    size=len(data_num)
    print(len(data_num)==100370)
    for i in range(size):
            if i not in remove:
                filter_data.append(data_num[i])
                filter_label.append(label[i])
    assert (len(filter_data)==len(filter_label))
    return filter_data,filter_label
def rows0to0(embed_matrix):
    embeddings = np.delete(embed_matrix, 0, axis=0)
    zero = np.zeros(len(embed_matrix[0]), dtype=np.int32)
    embeddings = np.row_stack((zero, embeddings))
    return embeddings
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
#自定义squeeze层
def squeeze(input):
    return Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(input)
def creat_model(sentence_length,vocal_size,embeding_dim,embed_matrix):
    # 模型共有三个输入，分别是左词，右词和中心词
    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    # 构建词向量
    embedder = Embedding(vocal_size + 1,output_dim=embeding_dim , input_length=sentence_length,weights=[embed_matrix],trainable=TRAINABLE)
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
    pool_rnn = squeeze(semantic)
    normal_pool=BatchNormalization()(pool_rnn)
    drop=Dropout(rate=DROPOUT_RATE)(normal_pool)
    output = Dense(NUM_CLASSES, activation="softmax")(drop)  # 等式(6)和(7)
    model = Model(inputs=[document, left_context, right_context], outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def transform_data(X_train_word_ids,X_test_word_ids,dictionary,truncating='pre'):
    left_train_word_ids = [[len(dictionary)] + x[:-1] for x in X_train_word_ids]
    right_train_word_ids = [x[1:] + [len(dictionary)] for x in X_train_word_ids]
    if X_test_word_ids:
        left_test_word_ids = [[len(dictionary)] + x[:-1] for x in X_test_word_ids]
        right_test_word_ids = [x[1:] + [len(dictionary)] for x in X_test_word_ids]

    # 分别对左边和右边的词进行编码
    X_train_padded_seqs=pad_sequences(X_train_word_ids, maxlen=PADDING_SENTENCE_LENGTH,truncating=truncating)
    left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=PADDING_SENTENCE_LENGTH,truncating=truncating)
    right_train_padded_seqs= pad_sequences(right_train_word_ids, maxlen=PADDING_SENTENCE_LENGTH,truncating=truncating)
    if X_test_word_ids:
        X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=PADDING_SENTENCE_LENGTH,truncating=truncating)
        left_test_padded_seqs = pad_sequences(left_test_word_ids, maxlen=PADDING_SENTENCE_LENGTH,truncating=truncating)
        right_test_padded_seqs= pad_sequences(right_test_word_ids, maxlen=PADDING_SENTENCE_LENGTH,truncating=truncating)
        return X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs
    return X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs
def train_model(X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,train_label,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs,test_label,pre_embed_matrix):

    if IS_LOAD:
        if not os.path.exists(MODEL_FRAME_PATH):
            print('请先设置is_load=False,Trainable=True,来保存{}'.format(MODEL_FRAME_PATH))
            exit()
        model=load_model(weights_path=WEIGHTS_PATH,model_frame_path=MODEL_FRAME_PATH)
    else:
        model=creat_model(sentence_length=PADDING_SENTENCE_LENGTH,vocal_size=len(embed_matrix)-1,
                          embeding_dim=len(pre_embed_matrix[0]),embed_matrix=pre_embed_matrix)
        model_json = model.to_json()
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        with open(MODEL_FRAME_PATH, 'w') as f:
            f.write(model_json)
        print('保存框架的位置：',MODEL_FRAME_PATH)
        if TRAINABLE:
            print('当is_load=false，Trainable=True时，只用来保存{}'.format(MODEL_FRAME_PATH))
            exit()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('model保存路径：', SAVE_WEIGHTS_PATH)
    # 这种命名方式只会保存最好的模型
    weights_path = SAVE_WEIGHTS_PATH
    # 这种命名方式，每当检测到模型性能提高时，就进行保存
    # weights_path='weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath=weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]
    model.fit([X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs],train_label,
               batch_size=BATCH_SIZE,
               epochs=EPOCHS,
              callbacks=callback_list,
              verbose=2,
               validation_data=([X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs],test_label))
    y_pred = model.predict([X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs])
    y_pred = np.argmax(y_pred, axis=1)
    test_label = np.argmax(test_label, axis=1)
    print("f1_score", sk.metrics.f1_score(test_label, y_pred, average='macro'))  # 各类F1_score调和平均数
    f1_score = sk.metrics.f1_score(test_label, y_pred, average=None)  # 以数组的形式展开各类的F1_score
    print("f1_lis:", f1_score)
    print('min:', np.argmin(f1_score), 'max:', np.argmax(f1_score))  # f1_score.index(min(f1_score))
def load_model(weights_path,model_frame_path):
    print('载入预训练模型:', weights_path)
    print('载入预训练框架：', model_frame_path)
    with open(model_frame_path,'r') as f:
        model_json=f.read()
    model=model_from_json(model_json,custom_objects={'squeeze': squeeze})
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
# def load_model_predict(X_test_data,left_test_data,right_test_data):
#     model=load_model(weights_path=WEIGHTS_PATH,model_frame_path=MODEL_FRAME_PATH)
#     logits=model.predict([X_test_data,left_test_data,right_test_data])
#     return logits
##--------------------------------------函数定义---------------------------------------------------------
if MODE=='train':
    print('导入仅仅数字编码化后的数据和字典.......')
    with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl','rb') as f:
        data_num=pickle.load(f)
        data_num=data_num[:1000]
    df=pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
    label=df['class']-1
    label=label[:1000]
    data_num,label=filter_tail(data_num=data_num,label=label)
    print('去除无效数据之后还剩：',len(data_num))
    label=to_categorical(label,num_classes=NUM_CLASSES)
    if SHUFFLE=='shuffle':
        print('样本shuffle中....')
        for line in data_num:
            random.Random(1).shuffle(line)
    if SHUFFLE=='reverse':
        print('样本reverse中....')
        for line in data_num:
            line.reverse()
    with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl','rb') as f:
        dictionary=pickle.load(f)
        # print('dic_len:',len(dictionary))



    print('导入预训练的词向量........')
    with open(r'D:\localE\code\DaGuang\final_set\embeddings300_3000001.pkl','rb') as f:
        embed_matrix=pickle.load(f)
        print(len(embed_matrix))
    embed_matrix=rows0to0(embed_matrix)
    print('切分训练数据、label和验证集数据、label......')
    #我们需要重新整理数据集(只需要把数据预处理成data_num即[[sequence1],[sequence2]...]),label处理成one-hot 形式，代入以下即可
    X_train_word_ids,X_test_word_ids,train_label,test_label=train_test_split(data_num,label,test_size=0.02,random_state=1)
    print('针对RCNN模型在data_num基础上再一步进行数据预处理.........')
    X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs=transform_data(X_train_word_ids,X_test_word_ids,dictionary=dictionary,truncating=Truncat)

    print('训练模型阶段：')
    train_model(X_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs,train_label,X_test_padded_seqs, left_test_padded_seqs,right_test_padded_seqs,test_label,pre_embed_matrix=embed_matrix)
    # print('加载模型预测阶段：')
    # cost,acc=load_model_predict(X_test_data=X_test_padded_seqs,left_test_data=left_test_padded_seqs,
    #                             right_test_data=right_test_padded_seqs,test_label=test_label)
    # print(cost,'\n',acc)

if MODE=='predict':
    # 'predict'阶段一直出现name 'backend' is not defined错误，未解决
    #解决办法，把Lambda层自定义函数，仅需再导入模型时，使model=model_from_json(model_json,custom_objects={'squeeze': squeeze})
    from keras import backend
    with open(r'D:\localE\code\DaGuang\300dim_03\test_filter_num_line03.pkl', 'rb') as f:
        test_set = pickle.load(f)
    with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl','rb') as f:
        dictionary=pickle.load(f)
    model = load_model(weights_path=WEIGHTS_PATH, model_frame_path=MODEL_FRAME_PATH)
    method_set = ['orderly']# 'shuffle', 'reversed']
    truncat = ['pre']#, 'post']
    result = []
    for method in method_set:
        if method == 'shuffle':
            for line in test_set:
                random.Random(1).shuffle(line)
        if method == 'reverse':
            for line in test_set:
                line.reverse()
        for tru in truncat:
            X_test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs=transform_data(test_set,X_test_word_ids=None,dictionary=dictionary,truncating=tru)
            print('predict.....')
            logits=model.predict([X_test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs])
            result.append(logits)
    predict = sum(result)
    label = np.argmax(predict, axis=1)
    print(type(label))
    label = list(label + 1)
    id=np.array([i+1 for i in range(len(label))])
    assert (len(label) == len(test_set))
    dic={'id':id,'class':label}
    df = pd.DataFrame(dic)

    #防止出错而未保存结果
    try:
        save_dir='result/'+model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path=save_dir+T+'Rcnn.csv'
        df.to_csv(save_path,index=False)
        print('结果保存路径:{}'.format(save_path))
    except:
        df.to_csv(r'result\Rcnn.csv')
    print('finish!')


