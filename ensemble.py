#ensemble.py
from CNN import load_revise_data,get_word2vec_dictionaries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, Input,Conv1D,MaxPooling1D,Dropout,concatenate,LSTM,Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model,load_model,Sequential
from keras.preprocessing.text import one_hot
from keras import backend as K
import os
from gensim.models import Word2Vec
from matplotlib import pyplot

def Ensemble_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embeddings_matrix):
    #first model
    first_input = Input(shape=(80,), dtype='int32')
    embedder = Embedding(len(embeddings_matrix), 30, input_length=80, weights=[embeddings_matrix], trainable=False)(
        first_input)
    cnn = []
    sizes = [3, 5, 5]
    for size in sizes:
        c = Conv1D(30, kernel_size=size, padding='same', strides=1, activation='relu')(embedder)
        p = MaxPooling1D(pool_size=30)(c)
        cnn.append(p)

    # 合并三个模型的输出向量
    cnn = concatenate([c for c in cnn], axis=-1)
    flat = Flatten()(cnn)
    dense = Dense(16, activation='sigmoid')(flat)
    #drop = Dropout(0.5)(flat)
    reshape= Reshape((16,1))(dense)
    #first_output = Dense(2, activation='sigmoid')(drop)
    #second model
    #x_train_padded_seqs = np.reshape(x_train_padded_seqs, (x_train_padded_seqs.shape[0], x_train_padded_seqs[1], 1))
    #second_input=Input((x_train_padded_seqs.shape[1],1,),dtype='float32')
    #l1=LSTM(4,return_sequences=True)(reshape)
    #l2=LSTM(4,return_sequences=True)(l1)
    l3=LSTM(1, activation = "tanh",recurrent_activation = "sigmoid", use_bias = True,kernel_regularizer=keras.regularizers.l2(0.01))(reshape)
    drop1=Dropout(0.1)(l3)
    second_output=Dense(2,activation='sigmoid')(drop1)
    #merger=concatenate([first_output,second_output],axis=1)
    #main_output=Dense(2,activation='sigmoid')(merger)

    #Ensemble two models
    model = Model(inputs=first_input,outputs=second_output)
    '''
    model2=Sequential()
    model2.add(Embedding(length,32,input_length=80))
    model2.add(LSTM(32,recurrent_activation='hard_sigmoid'))
    model2.add(Dropout(0.5))
    model2.add(Dense(2,activation='sigmoid'))
    '''
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])
    one_hot_train_labels = keras.utils.to_categorical(y_train, num_classes=2)  # 将标签转换为one-hot编码
    one_hot_test_labels = keras.utils.to_categorical(y_test, num_classes=2)
    history=model.fit(x_train_padded_seqs, one_hot_train_labels, batch_size=64,
              validation_data=(x_test_padded_seqs, one_hot_test_labels), shuffle=True, epochs=10)
    model.save('cnn地震_new_ensemble1.h5')

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('train loss vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train','validation'])
    pyplot.show()

    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=-1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
    print('召回率：', metrics.recall_score(y_test, y_predict, pos_label='1'))
    print('精确率：', metrics.precision_score(y_test, y_predict, pos_label='1'))

trainDF=load_revise_data()
print(len(trainDF['text']))
texts,embeddings_matrix=get_word2vec_dictionaries(trainDF['text'])
x_train_padded,x_test_padded,y_train,y_test=train_test_split(texts,trainDF['label'],test_size=0.2,stratify=trainDF['label'])
Ensemble_model(x_train_padded,y_train,x_test_padded,y_test,embeddings_matrix)