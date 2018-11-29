# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
import random
import pandas as pd
import pickle
import numpy as np
from keras.models import Sequential, Model, model_from_json
from keras.utils import to_categorical
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, merge, Input, Embedding, GRU, \
    GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import os
import sklearn as sk
import time


# 训练策略：
# 1/预训练词向量先保持trainable=false,Truncat=pre 训练2-3个epoch,2/再post 训练2-3个epoch
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
NGRAM = 2
SEED = 1
model_name='Fasttext/'
WEIGHTS_PATH = 'save_model/Fasttext/28_22_15_False_orderly_post_weights_best.h5'  # 载入模型时用，注意每次手动更新加载最优模型
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


# ---------------------------------------------函数定义————————————————————————————
def keep(data_num, rate=0.5):
    # random delete some text
    data = []
    for line_num in data_num:
        length = len(line_num)
        if length == 0:
            print(line_num)
            continue
        line_num = np.random.choice(line_num, size=int(length * rate), replace=False)
        data.append(list(line_num))
    return data


def padding_sentence(data_num, padding_token=0, padding_sentence_length=None):
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in data_num])
    for sentence in data_num:
        if len(sentence) > max_sentence_length:
            while len(sentence) > padding_sentence_length:
                sentence.pop()
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return data_num, max_sentence_length


def shuffle_data(train_set, label_set, seed):
    zip_train_label = list(zip(train_set, label_set))
    random.Random(seed).shuffle(zip_train_label)
    train_input, label_input = zip(*zip_train_label)
    return train_input, label_input


index = 0  # index=0不能丢


def generate_batch(data_num, labels, batch_size):
    global index
    x_train = [0 for j in range(batch_size)]
    y_label = [0 for k in range(batch_size)]
    length = len(data_num)
    for i in range(batch_size):
        x_train[i] = data_num[index]
        y_label[i] = labels[index]
        index += 1
        index %= length
    x_train, y_label = shuffle_data(x_train, y_label, seed=None)
    x_train = np.array(x_train)  # array[[10,21,31,41...],[11,22,333,4444,...]...]shape=batch_size*max_sentence_length
    y_label = np.array(y_label)
    return x_train, y_label


# 没有embed_matrix 初始化词向量矩阵
def creat_model(sentence_length, vocal_size, embeding_dim):
    model = Sequential()
    model.add(Embedding(vocal_size + 1, embeding_dim, input_length=sentence_length))  # 不能使用预训练词向量
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(DROUPOUT_RATE))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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


# ------------------------------------------函数定义---------------------------------------------------
if MODE == 'train':
    with open(r'D:\localE\code\DaGuang\300dim_03\train_filter_num_line03.pkl', 'rb') as f:
        data_num = pickle.load(f)
        data_num = data_num
    with open(r'D:\localE\code\DaGuang\300dim_03\dictionary03.pkl', 'rb') as f:
        dictionary = pickle.load(f)
        print('dic_len:', len(dictionary))

    df = pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
    label = df['class'] - 1
    label = label
    label = to_categorical(label, num_classes=NUM_CLASSES)
    X_train_word_ids, X_test_word_ids, train_label, test_label = train_test_split(data_num, label, test_size=0.02,
                                                                                  random_state=SEED)
    # -----------------------------------------------------针对Fasttext,数据处理部分-----------------------------------------
    # 这两代码的目的是选择部分训练集，部分长度，组建n-gram词（选的太多，n-gram会太大，后面的词向量矩阵会太大发生oom错误）
    select_train = X_train_word_ids[:20000]
    X_train_padding, _ = padding_sentence(data_num=select_train, padding_sentence_length=50)  # 截断前50个词

    # 模型结构：词嵌入(n-gram)-最大化池化-全连接
    # 生成n-gram组合的词(以3为例)
    ngram = NGRAM


    # 将n-gram词加入到词表
    def create_ngram(sent, ngram_value):
        return set(zip(*[sent[i:] for i in range(ngram_value)]))


    ngram_set = set()
    for sentence in X_train_padding:  # 利用X_train_padding样本生成ngram词
        for i in range(2, ngram + 1):  # 2不能改成ngram
            set_of_ngram = create_ngram(sentence, i)
        ngram_set.update(set_of_ngram)
    # 给n-gram词汇编码
    start_index = len(dictionary) + 1  # 要保证dictionary除了不含0外，其他编号都不缺，即从1到max连续
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}  # 给n-gram词汇编码
    indice_token = {token_indice[k]: k for k in token_indice}
    max_features = np.max(list(indice_token.keys())) + 1


    # 将n-gram词加入到输入文本的末端
    def add_ngram(sequences, token_indice, ngram_range):
        new_sequences = []
        for sent in sequences:
            new_list = sent[:]
            for i in range(len(new_list) - ngram_range + 1):
                for ngram_value in range(2, ngram_range + 1):  # 同上
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:  # ngram 在字典上则加到文本后面
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)
        return new_sequences


    x_train = add_ngram(X_train_word_ids, token_indice, ngram)
    x_test = add_ngram(X_test_word_ids, token_indice, ngram)
    x_train = pad_sequences(x_train, maxlen=PADDING_SENTENCE_LENGTH)
    x_test = pad_sequences(x_test, maxlen=PADDING_SENTENCE_LENGTH)
    # -----------------------------------------------------针对Fasttext,数据处理部分----------------------------------------

    if IS_LOAD:
        if not os.path.exists(MODEL_FRAME_PATH):
            print('请先设置is_load=False,Trainable=True,来保存{}'.format(MODEL_FRAME_PATH))
            exit()
        model = load_model(weights_path=WEIGHTS_PATH, model_frame_path=MODEL_FRAME_PATH)
        if ONLY_TEST:
            y_pred = model.predict(np.array(x_test))
            y_pred = np.argmax(y_pred, axis=1)
            test_label = np.argmax(test_label, axis=1)
            print("f1_score", sk.metrics.f1_score(test_label, y_pred, average='macro'))  # 各类F1_score调和平均数
            f1_score = sk.metrics.f1_score(test_label, y_pred, average=None)  # 以数组的形式展开各类的F1_score
            print("f1_lis:", f1_score)
            print('min:', np.argmin(f1_score), 'max:', np.argmax(f1_score))  # f1_score.index(min(f1_score))
            print('ONLY_TEST=True仅仅用来测试')
            exit()
    else:
        model = creat_model(sentence_length=PADDING_SENTENCE_LENGTH, vocal_size=max_features, embeding_dim=100)
        # 保存两套框架，一套embedding 层trainable,一套不可train
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
    print('model保存路径：', SAVE_WEIGHTS_PATH)
    # 这种命名方式只会保存最好的模型
    weights_path = SAVE_WEIGHTS_PATH
    # 这种命名方式，每当检测到模型性能提高时，就进行保存
    # weights_path='weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath=weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]
    model.fit(x_train, y=train_label, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
              validation_data=(x_test, test_label), callbacks=callback_list)
    y_pred = model.predict(np.array(x_test))
    y_pred = np.argmax(y_pred, axis=1)
    test_label = np.argmax(test_label, axis=1)
    print("f1_score", sk.metrics.f1_score(test_label, y_pred, average='macro'))  # 各类F1_score调和平均数
    f1_score = sk.metrics.f1_score(test_label, y_pred, average=None)  # 以数组的形式展开各类的F1_score
    print("f1_lis:", f1_score)
    print('min:', np.argmin(f1_score), 'max:', np.argmax(f1_score))  # f1_score.index(min(f1_score))

if MODE == 'predict':
    with open(r'D:\localE\code\DaGuang\300dim_03\test_filter_num_line03.pkl', 'rb') as f:
        test_set = pickle.load(f)
        test_set = test_set

    model = load_model(weights_path=WEIGHTS_PATH, model_frame_path=MODEL_FRAME_PATH)
    result = []
    for rate in [0.5]:  # ,0.7,0.9]:
        test_set = keep(data_num=test_set, rate=rate)
        for tru in ['pre']:  # ,'post']:
            test_set_pad = pad_sequences(test_set, maxlen=PADDING_SENTENCE_LENGTH, truncating=tru)
            logits = model.predict(test_set_pad)
            result.append(logits)
    predict = sum(result)
    label = np.argmax(predict, axis=1)
    print(type(label))
    label = list(label + 1)
    id = np.array([i + 1 for i in range(len(label))])
    assert (len(label) == len(test_set))
    dic = {'id': id, 'class': label}
    df = pd.DataFrame(dic)
    save_dir='result/'+model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        save_path = save_dir + T + 'fasttext.csv'
        df.to_csv(save_path,index=None)
        print('结果保存路径：{}'.format(save_path))
    except:
        df.to_csv('result/28_15_09_fasttext.csv',index=None)
    print('finish!')

# for i in range(10000):
#     batch_xs, batch_ys = generate_batch(data_num=x_train, labels=train_label, batch_size=64)
#     cost=model.train_on_batch(batch_xs,batch_ys)
#     if i%10==0:
#         print('cost:',cost)
#     if i%100==0 and i!=0:
#         cost,accuracy=model.evaluate(np.array(x_test),np.array(test_label),batch_size=64)
#         print('test:',accuracy)
# model.save('fastText_model.h5')
