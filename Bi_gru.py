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
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten,merge,Input,Embedding,GRU,BatchNormalization,Bidirectional
from sklearn.model_selection import train_test_split
import sklearn as sk
import os
import time
# 训练策略：
# 1/预训练词向量先保持shuffle=orderly,trainable=false,Truncat=pre 训练2-3个epoch,2/再post 训练2-3个epoch
# 3/trainable=true ,前向截断训练1-2个epoch，4/后向截断训练1-2epoch
# 再shuffle每篇文章，训练1-2个epoch
# 再reverse每篇,训练1-2个epoch

# 注意保存每次训练完的最好模型，不能重名，否则会覆盖。
# save_model/Textcnn/True_shuffle_post_weights_best.h5 0.752741,卷积层忘了加relu函数
def Gettime():
    t = time.asctime(time.localtime(time.time()))
    t = t.split()[2:4]
    d = [t[0]] + t[1].split(':')[:-1]
    d = '_'.join(d)
    return d


T = Gettime()
ONLY_TEST = False  # 当True时直接载入模型进行验证集上验证
EMBED_DROP_RATE = 0.15  # 未用
DROUPOUT_RATE = 0.5
NUM_CLASSES = 19
PADDING_SENTENCE_LENGTH = 150
EPOCHS = 2
BATCH_SIZE = 32
LEARN_RATE = 0.001
FILTER_SIZE=3
model_name='Bi_gru/'
WEIGHTS_PATH = "save_model/Bi_gru/29_19_08_False_orderly_post_weights_best.h5"  # 载入模型时用，注意每次手动更新加载最优模型
Truncat = 'post'  # 句子是前向截断还是后向截断
SHUFFLE = 'orderly'  # 每行样本进行[shuffle,reverse],为空时就不处理
IS_LOAD = True if WEIGHTS_PATH else False  # 是否载入模型，分为载入模型训练和载入模型预测
TRAINABLE = False  # embeddings是否可以训练
frame_dir = 'frame/'+model_name
MODEL_FRAME_PATH = frame_dir + str(TRAINABLE) + '_model_frame.json'
model_dir = 'save_model/'+model_name
# 模型保存位置
SAVE_WEIGHTS_PATH = model_dir + T + '_' + str(TRAINABLE) + '_' + str(SHUFFLE) + '_' + Truncat + '_' + 'weights_best.h5'
MODE = 'predict'  # 训练模式还是测试测试集模型[train or predict]
#-----------------------------------------------函数定义-------------------------------------
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
def creat_model(sentence_length,vocal_size,embeding_dim,embed_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=vocal_size + 1,output_dim=embeding_dim, input_length=sentence_length
                        ,weights=[embed_matrix]))
    model.add(Bidirectional(GRU(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(GRU(256, dropout=0.5, recurrent_dropout=0.5)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROUPOUT_RATE))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def load_model(weights_path, model_frame_path):
    print('载入预训练模型:', weights_path)
    print('载入预训练框架：', model_frame_path)
    with open(model_frame_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def train_model(train_data,train_label,test_data,test_label,pre_embed_matrix,batch_size):
    if IS_LOAD:
        if not os.path.exists(MODEL_FRAME_PATH):
            print('请先设置is_load=False,Trainable=True,来保存{}'.format(MODEL_FRAME_PATH))
            exit()
        model=load_model(weights_path=WEIGHTS_PATH,model_frame_path=MODEL_FRAME_PATH)
        if ONLY_TEST:
            y_pred = model.predict(np.array(test_data))
            y_pred = np.argmax(y_pred, axis=1)
            test_label=np.argmax(test_label,axis=1)
            print("f1_score", sk.metrics.f1_score(test_label, y_pred, average='macro'))  # 各类F1_score调和平均数
            f1_score = sk.metrics.f1_score(test_label, y_pred, average=None)  # 以数组的形式展开各类的F1_score
            print("f1_lis:", f1_score)
            print('min:', np.argmin(f1_score), 'max:', np.argmax(f1_score))  # f1_score.index(min(f1_score))
            print('ONLY_TEST=True仅仅用来测试')
            exit()
    else:
        model=creat_model(sentence_length=PADDING_SENTENCE_LENGTH,vocal_size=len(embed_matrix)-1,
                          embeding_dim=len(pre_embed_matrix[0]),embed_matrix=pre_embed_matrix)
        # 保存两套框架，一套embedding
        # 层trainable, 一套不可train
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        model_json = model.to_json()
        with open(MODEL_FRAME_PATH, 'w') as f:
            f.write(model_json)
        print('model_frame保存路径：', MODEL_FRAME_PATH)
        if TRAINABLE:
            print('当is_load=false，Trainable=True时，只用来保存{}'.format(MODEL_FRAME_PATH))
            exit()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('model保存路径：',SAVE_WEIGHTS_PATH)
    # 这种命名方式只会保存最好的模型
    weights_path=SAVE_WEIGHTS_PATH
    # 这种命名方式，每当检测到模型性能提高时，就进行保存
    #weights_path='weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint=ModelCheckpoint(filepath=weights_path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callback_list=[checkpoint]
    history=model.fit(x=np.array(train_data),y=np.array(train_label),batch_size=batch_size,epochs=EPOCHS,callbacks=callback_list,
              verbose=2,validation_data=(np.array(test_data),np.array(test_label)))
    # _,acc=model.evaluate(np.array(test_data),np.array(test_label),batch_size=batch_size)
    y_pred=model.predict(np.array(test_data))
    y_pred=np.argmax(y_pred,axis=1)
    test_label=np.argmax(test_label,axis=1)
    print("f1_score", sk.metrics.f1_score(test_label, y_pred, average='macro'))  # 各类F1_score调和平均数
    f1_score = sk.metrics.f1_score(test_label, y_pred, average=None)  # 以数组的形式展开各类的F1_score
    print("f1_lis:", f1_score)
    print('min:', np.argmin(f1_score), 'max:', np.argmax(f1_score))  # f1_score.index(min(f1_score))

#-----------------------------------------------函数定义-------------------------------------
if MODE=='train':
    print('导入仅仅数字编码化后的数据.......')
    with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl','rb') as f:
        data_num=pickle.load(f)
        data_num=data_num[:1000]
    df = pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
    label = df['class'] - 1
    label=label[:1000]
    data_num, label = filter_tail(data_num=data_num, label=label)
    print('去除无效数据之后还剩：', len(data_num))

    if SHUFFLE == 'shuffle':
        print('样本shuffle中....')
        for line in data_num:
            random.Random(1).shuffle(line)
    if SHUFFLE == 'reverse':
        print('样本reverse中....')
        for line in data_num:
            line.reverse()

    label = to_categorical(label, num_classes=NUM_CLASSES)

    print('padding 数字编码化后的数据........')
    padding_data_num=pad_sequences(data_num,maxlen=PADDING_SENTENCE_LENGTH,truncating=Truncat)
    print('切分训练数据、label和验证集数据、label......')
    train_data,test_data,train_label,test_label=train_test_split(padding_data_num,label,test_size=0.02,
                                                                 random_state=1)

    print('导入预训练的词向量........')
    with open(r'D:\localE\code\DaGuang\final_set\embeddings300_3000001.pkl','rb') as f:
        embed_matrix=pickle.load(f)
    embed_matrix=rows0to0(embed_matrix)

    print('训练模型阶段：')
    train_model(train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label,
                batch_size=BATCH_SIZE,pre_embed_matrix=embed_matrix)
else:
    with open(r'D:\localE\code\DaGuang\300dim_03\test_filter_num_line03.pkl','rb') as f:
        test_set=pickle.load(f)
        test_set=test_set[:100]
    model = load_model(weights_path=WEIGHTS_PATH,
                       model_frame_path=MODEL_FRAME_PATH)  # samples,label_sizes
    method_set=['orderly']
    truncat=['pre','post']
    result=[]
    for method in method_set:
        if method=='shuffle':
            for line in test_set:
                random.Random(1).shuffle(line)
        if method=='reverse':
            for line in test_set:
                line.reverse()
        for tru in truncat:
            test_set_padding=pad_sequences(maxlen=PADDING_SENTENCE_LENGTH,sequences=test_set,truncating=tru)
            print('predict.....')
            logits=model.predict(test_set_padding)
            result.append(logits)
    predict=sum(result)
    label=np.argmax(predict,axis=1)
    print(type(label))
    label=list(label+1)
    id=np.array([i+1 for i in range(len(label))])
    dic={'id':id,'class':label}
    assert(len(label)==len(test_set))
    df=pd.DataFrame(dic)
    save_dir='result/'+model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        save_path=save_dir+T+'result.csv'
        df.to_csv(save_path,index=None)
        print('结果保存路径:{}'.format(save_path))
    except:
        df.to_csv('result/28_15_09_result.csv',index=None)
    print('finish!')
