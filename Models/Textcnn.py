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
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten,merge,Input,Embedding,BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
from keras.optimizers import Adam,SGD
from matplotlib import pyplot as plt
import sklearn as sk
#训练策略：
#预训练词向量先保持trainable=false,训练1-2个epoch
#trainable=true ,前向截断训练2-3个epoch，后向截断训练2-3epoch
#再shuffle每篇文章，训练1-2个epoch
#再reverse每篇,训练1-2个epoch

#注意保存每次训练完的最好模型，不能重名，否则会覆盖。


ONLY_TEST=True #当True时直接载入模型进行验证集上验证
FILTER_SIZE_LIST=[2,3,4]
FILTERS_NUM=120
DROUPOUT_RATE=0.4
NUM_CLASSES=19
PADDING_SENTENCE_LENGTH=1500
EPOCHS=2
BATCH_SIZE=32
WEIGHTS_PATH='shuffle_weights.best.h5'  #预测时载入的模型
MODEL_FRAME_PATH='model_frame.json'
LEARN_RATE=0.001
SHUFFLE='shuffle0'   #每行样本进行[shuffe,reverse],为空时就不处理
IS_LOAD=True       #是否载入模型，分为载入模型训练和载入模型预测
TRAINABLE=False   #embeddings是否可以训练
MODE='train'    #训练模式还是测试测试集模型
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
def creat_model(filter_size_list,sentence_length,learn_rate,vocal_size,embeding_dim,embed_matrix):
    seq=Input(shape=(sentence_length,))
    #'Embedding' object
    embeddings=Embedding(input_dim=vocal_size+1,output_dim=embeding_dim,input_length=sentence_length,
                         weights=[embed_matrix],trainable=TRAINABLE)

    embed=embeddings(seq)
    embed=Dropout(rate=0.15)(embed)
    pooled_out=[]
    for filter_size in filter_size_list:
        cov1=Conv1D(filters=FILTERS_NUM,kernel_size=filter_size,strides=1,padding='valid')(embed)
        normal_cov1=BatchNormalization()(cov1)
        # cov2=Conv1D(FILTERS_NUM,kernel_size=filter_size,strides=1,padding='same')(normal_cov1)
        # normal_cov2=BatchNormalization()(cov2)
        pool=MaxPooling1D(pool_size=sentence_length-filter_size+1)(normal_cov1)
        pool_flatten=Flatten()(pool)
        pooled_out.append(pool_flatten)
    merge=concatenate(pooled_out,axis=1)
    out=Dropout(DROUPOUT_RATE)(merge)
    output=Dense(NUM_CLASSES,activation='softmax',kernel_constraint=maxnorm(4))(out)
    model=Model([seq],output)
    adam=Adam(lr=learn_rate,decay=0.01)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def load_model(weights_path,model_frame_path):
    with open(model_frame_path,'r') as f:
        model_json=f.read()
    model=model_from_json(model_json)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


def train_model(train_data,train_label,test_data,test_label,pre_embed_matrix,batch_size=64,epochs=3,load=False,learn_rate=0.001):
    if load:
        print('载入预训练模型:',WEIGHTS_PATH)
        model=load_model(weights_path=WEIGHTS_PATH,model_frame_path=MODEL_FRAME_PATH)
        if ONLY_TEST:
            y_pred = model.predict(np.array(test_data))
            y_pred = np.argmax(y_pred, axis=1)
            test_label=np.argmax(test_label,axis=1)
            print("f1_score", sk.metrics.f1_score(test_label, y_pred, average='macro'))  # 各类F1_score调和平均数
            f1_score = sk.metrics.f1_score(test_label, y_pred, average=None)  # 以数组的形式展开各类的F1_score
            print("f1_lis:", f1_score)
            print('min:', np.argmin(f1_score), 'max:', np.argmax(f1_score))  # f1_score.index(min(f1_score))
            exit()
    else:
        model=creat_model(filter_size_list=FILTER_SIZE_LIST,sentence_length=PADDING_SENTENCE_LENGTH,
                          learn_rate=learn_rate,vocal_size=len(pre_embed_matrix)-1,embeding_dim=len(pre_embed_matrix[0]),embed_matrix=pre_embed_matrix)
        if TRAINABLE:
            model_json=model.to_json()
            with open('model_frame.json','w') as f:
                f.write(model_json)
        else:
            model_json = model.to_json()
            with open('model_frame_false.json', 'w') as f:
                f.write(model_json)

    #这种命名方式只会保存最好的模型
    weights_path=WEIGHTS_PATH
    if SHUFFLE=='shuffle':
        print('保存shuffle后的weights')
        weights_path='shuffle_weights.best.h5'
    if SHUFFLE=='reverse':
        print('保存reverse后的weights')
        weights_path = 'reverse_weights.best.h5'
    #这种命名方式，每当检测到模型性能提高时，就进行保存
    #weights_path='weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint=ModelCheckpoint(filepath=weights_path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callback_list=[checkpoint]
    history=model.fit(x=np.array(train_data),y=np.array(train_label),batch_size=batch_size,epochs=epochs,callbacks=callback_list,
              verbose=2,validation_data=(np.array(test_data),np.array(test_label)))
    # _,acc=model.evaluate(np.array(test_data),np.array(test_label),batch_size=batch_size)
    y_pred=model.predict(np.array(test_data))
    y_pred=np.argmax(y_pred,axis=1)
    test_label=np.argmax(test_label,axis=1)
    print("f1_score", sk.metrics.f1_score(test_label, y_pred, average='macro'))  # 各类F1_score调和平均数
    f1_score = sk.metrics.f1_score(test_label, y_pred, average=None)  # 以数组的形式展开各类的F1_score
    print("f1_lis:", f1_score)
    print('min:', np.argmin(f1_score), 'max:', np.argmax(f1_score))  # f1_score.index(min(f1_score))
    print(history.history.keys())
    print(history.history['acc'])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

# for i in range(10000):
#     #train_data是list or array or tuple类型，如[[1,2,3,4,5,21,31]#一句话长度为截断或pading到149,[2,3,4,5,2,3,3]#同左]
#     #train_label是list or array or tuple类型，且是one-hot表示如:[[1,0],[0,1]]
#     batch_xs, batch_ys = generate_batch(data_num=train_data, labels=train_label, batch_size=64)
#     # batch_xs是array类型，如array([[1,2,3,4,5,21,31]#一句话长度为截断或pading到149,[2,3,4,5,2,3,3]#同左])
#     # batch_ys是array类型，且是one-hot表示如array[[1,0],[0,1]]
#     cost=model.train_on_batch(batch_xs,batch_ys)
#     if i%10==0:
#         print('cost:',cost)
#     if i%100==0 and i!=0:
#         cost,accuracy=model.evaluate(np.array(test_data),np.array(test_label),batch_size=256)
#         print('test:',accuracy)
#     if i%(100000//64)==0:
#         train_data,train_label=shuffle_data(train_data,train_label,seed=1)
#
# model.save('TextCNN_model.h5')

def load_model_predict(test_data,test_label):
    model=load_model(weights_path=WEIGHTS_PATH,model_frame_path=MODEL_FRAME_PATH)
    cost,acc=model.evaluate(np.array(test_data),np.array(test_label),batch_size=64)
    return cost,acc
def predict_test_set(test_set,weights_path,model_frame_path):
    model=load_model(weights_path=weights_path,model_frame_path=model_frame_path)
    predict=model.predict(test_set)
    return predict
##--------------------------------------函数定义---------------------------------------------------------

if MODE=='train':
    print('导入仅仅数字编码化后的数据.......')
    with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl','rb') as f:
        data_num=pickle.load(f)
    j=0
    remove=[]
    for line in data_num:#[44, 47, 41, 18, 14, 45]
        if len(line)>=6:
            if line[-6:]==[44, 47, 41, 18, 14, 45]:
                for i in range(6):
                    line.pop()
                if len(line)==0:
                    remove.append(j)
        else:#24564,32019,58468
            remove.append(j)
        j+=1

    if SHUFFLE=='shuffle':
        print('样本shuffle中....')
        for line in data_num:
            random.Random(1).shuffle(line)
    if SHUFFLE=='reverse':
        print('样本reverse中....')
        for line in data_num:
            line.reverse()

    df=pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
    label=df['class']-1

    for j in remove:
        data_num.pop(j)
        label.pop(j)
    assert(len(data_num)==len(label))
    print('去除无效数据之后还剩：',len(data_num))
    label=to_categorical(label,num_classes=NUM_CLASSES)


    # with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl','rb') as f:
    #     dictionary=pickle.load(f)
    #     print('dic_len:',len(dictionary))
    print('padding 数字编码化后的数据........')
    padding_data_num=pad_sequences(data_num,maxlen=PADDING_SENTENCE_LENGTH,truncating='pre')


    print('切分训练数据、label和验证集数据、label......')
    train_data,test_data,train_label,test_label=train_test_split(padding_data_num,label,test_size=0.02,
                                                                 random_state=1)
    print('导入预训练的词向量........')
    with open(r'D:\localE\code\DaGuang\final_set\embeddings300_3000001.pkl','rb') as f:
        embed_matrix=pickle.load(f)
        print(len(embed_matrix))

    print('训练模型阶段：')

    valid_acc=train_model(train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label,
                    pre_embed_matrix=embed_matrix,batch_size=BATCH_SIZE,epochs=EPOCHS,load=IS_LOAD,learn_rate=LEARN_RATE)

if MODE=='predict':
    with open(r'D:\localE\code\DaGuang\300dim_03\test_filter_num_line03.pkl','rb') as f:
        test_set=pickle.load(f)
    j = 0
    remove = []
    for line in test_set:  # [44, 47, 41, 18, 14, 45]
        if len(line) >= 6:
            if line[-6:] == [44, 47, 41, 18, 14, 45]:
                for i in range(6):
                    line.pop()
                if len(line) == 0:
                    remove.append(j)
        else:  # 24564,32019,58468
            remove.append(j)
        j += 1
    method_set=['shuffle','reversed']
    result=[]
    for method in method_set:
        if method=='shuffle':
            for line in test_set:
                random.Random(1).shuffle(line)
        if method=='reverse':
            for line in test_set:
                line.reverse()
        test_set_padding=pad_sequences(maxlen=1500,sequences=test_set)
        print('predict.....')
        logits=predict_test_set(test_set_padding,weights_path='shuffle_weights.best.h5',
                                model_frame_path=MODEL_FRAME_PATH)
        result.append(logits)
    predict=result[0]+result[1]
    label=np.argmax(predict,axis=1)
    print(type(label))
    label=list(label+1)
    assert(len(label)==len(test_set))
    df=pd.DataFrame(data=label,columns=['class'])
    df.to_csv(r'result\textcnn.csv')
    print('finish!')
