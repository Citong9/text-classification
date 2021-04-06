#CNN.py
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import jieba,keras,numpy
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, Input,Conv1D,MaxPooling1D,Dropout,concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model,load_model
from keras.preprocessing.text import one_hot
from keras import backend as K
import os
from gensim.models import Word2Vec,KeyedVectors

def load_data():
  file=open("D:/垃圾分类/地震/train_data.txt","r",encoding="utf-8")
  stopwords=open("D:/垃圾分类/停用词表.txt","r",encoding="utf-8").read()
  keyword=open("D:/垃圾分类/停用词表2.txt","r",encoding="utf-8").readlines()
  keywords=[]
  for word in keyword:
    word=word.replace('\n','')
    keywords.append(word)
  texts=[]
  label=[]
  while True:
    txt=file.readline()
    if len(txt)==0:
      break
    for ch in {"<SEP>",'\n'}:
      txt=txt.replace(ch,"")
    words=jieba.lcut(txt)
    string=''
    value=0
    if '地震' in words:
      '''
      for word in keywords:
        if word in words:
          value=1
      if value==0:
        words[-1]='0'
        label.append('0')
      else:
        words[-1]='1'
        label.append('1')
      '''
      if words[-1]=='0':
        label.append('0')
      else:
        label.append('1')
      for word in words:
        if word in stopwords:
          continue
        else:
          string=string+word+' '
      texts.append(string[:-1])
  #创建一个dataframe
  df=pd.DataFrame()
  df['text']=texts
  df['label']=label
  return(df)

def load_sample_data():
  file=open("D:/垃圾分类/地震/traindata.txt","r",encoding="utf-8")
  stopwords=open("D:/垃圾分类/停用词表.txt","r",encoding="utf-8").read()
  keyword=open("D:/垃圾分类/停用词表2.txt","r",encoding="utf-8").readlines()
  keywords=[]
  for word in keyword:
    word=word.replace('\n','')
    keywords.append(word)
  texts=[]
  label=[]
  while True:
    txt=file.readline()
    if len(txt)==0:
      break
    for ch in {"<SEP>",'\n'}:
      txt=txt.replace(ch,"")
    words=jieba.lcut(txt)
    string=''
    value=0
    if '地震' in words:
     for word in keywords:
       if word in words:
          value=1
     if value==0:
          words[-1]='0'
          label.append(0)
     else:
          words[-1]='1'
          label.append(1)

     for word in words:
         if word in stopwords:
           continue
         else:
           string=string+word+' '
     texts.append(string[:-1])
  #创建一个dataframe
  trainDF=pd.DataFrame()
  trainDF['text']=texts
  trainDF['label']=label
  #预处理
  x=trainDF.loc[:,trainDF.columns!='label']
  y=trainDF.loc[:,trainDF.columns=='label']
  positive_number=len(y[y.label==1])
  negative_number=len(y[y.label==0])
  positive_indices=np.array(y[y.label==1].index)
  negative_indices=np.array(y[y.label==0].index)
  #下采样
  random_positive_indices=np.random.choice(positive_indices,negative_number-1000,replace=False)
  random_positive_indices=np.array(random_positive_indices)
  under_sample_indices=np.concatenate([negative_indices,random_positive_indices])
  under_sample_data=trainDF.iloc[under_sample_indices,:]
  x_sample=under_sample_data.loc[:,under_sample_data.columns!='label']
  y_sample=under_sample_data.loc[:,under_sample_data.columns=='label']
  list1=[str(i) for i in x_sample.text]
  list2=[str(i) for i in y_sample.label]
  df=pd.DataFrame()
  df['text']=list1
  df['label']=list2
  return(df)

def load_revise_data():
  texts=[]
  label=[]
  f=open('D:/垃圾分类/xinlang_new_train.txt','r',encoding="utf-8")
  lines=f.readlines()
  for line in lines:
    texts.append(line[:-4])
    label.append(line[-3])
  #创建一个dataframe
  trainDF=pd.DataFrame()
  trainDF['text']=texts
  trainDF['label']=label
  #预处理
  x=trainDF.loc[:,trainDF.columns!='label']
  y=trainDF.loc[:,trainDF.columns=='label']
  positive_number=len(y[y.label=='1'])
  negative_number=len(y[y.label=='0'])
  positive_indices=np.array(y[y.label=='1'].index)
  negative_indices=np.array(y[y.label=='0'].index)
  #下采样
  random_positive_indices=np.random.choice(positive_indices,negative_number+200,replace=False)
  random_positive_indices=np.array(random_positive_indices)
  under_sample_indices=np.concatenate([negative_indices,random_positive_indices])
  under_sample_data=trainDF.iloc[under_sample_indices,:]
  x_sample=under_sample_data.loc[:,under_sample_data.columns!='label']
  y_sample=under_sample_data.loc[:,under_sample_data.columns=='label']
  list1=[str(i) for i in x_sample.text]
  list2=[str(i) for i in y_sample.label]
  df=pd.DataFrame()
  df['text']=list1
  df['label']=list2
  return(df)

def get_word2vec_dictionaries(texts):
  def get_word2vec_model(texts=None):
    if os.path.exists('word2vec_model2'):
      model=KeyedVectors.load('word2vec_model2')
      return model
    else:
      model=Word2Vec(texts,size=50,window=5,min_count=5,workers=5)
      model.save('word2vec_model2')
      return model

  word2vec_model=get_word2vec_model(texts)
  vocab_list=[word for word,Vocab in word2vec_model.wv.vocab.items()]
  word_index={" ":0}
  word_vector={}
  embeddings_matrix=np.zeros((len(vocab_list)+1,word2vec_model.vector_size))
  for i in range(len(vocab_list)):
    word=vocab_list[i]
    word_index[word]=i+1
    word_vector[word]=word2vec_model.wv[word]
    embeddings_matrix[i+1]=word2vec_model.wv[word]
  data=[]
  for sentence in texts:
    new_txt=[]
    for word in sentence:
      try:
        new_txt.append(word_index[word])
      except:
        new_txt.append(0)
    data.append(new_txt)
  texts=pad_sequences(data,maxlen=80)
  return texts,embeddings_matrix

'''
#if __name__=='__main__':
tokenizer=Tokenizer()
tokenizer.fit_on_texts(trainDF['text'])
vocab=tokenizer.word_index#得到每个词的编号
x_train,x_test,y_train,y_test=train_test_split(trainDF['text'],trainDF['label'],test_size=0.2,stratify=trainDF['label'])
#转化为数字列表
x_train_word_ids=tokenizer.texts_to_sequences(x_train)
x_test_word_ids=tokenizer.texts_to_sequences(x_test)
#每行样本长度设置为固定值
x_train_padded=pad_sequences(x_train_word_ids,maxlen=50)
x_test_padded=pad_sequences(x_test_word_ids,maxlen=50)

def mycrossentropy(y_true, y_pred):
    return -K.mean(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))
    #return K.mean(0*K.binary_crossentropy(y_pred,y_true) + 1*K.binary_crossentropy(y_pred, K.ones_like(y_pred)/2))
'''
def TextCNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embeddings_matrix):
    main_input = Input(shape=(80,), dtype='int32')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(embeddings_matrix), 30, input_length=80,weights=[embeddings_matrix],trainable=False)(main_input)
    cnn=[]
    sizes=[3,5,5]
    for size in sizes:
       c=Conv1D(30, kernel_size=size, padding='same', strides=1, activation='relu')(embedder)
       p=MaxPooling1D(pool_size=30)(c)
       cnn.append(p) 

    # 合并三个模型的输出向量
    cnn = concatenate([c for c in cnn],axis=-1)
    flat = Flatten()(cnn)
    dense = Dense(16, activation='relu')(flat)
    drop = Dropout(0.5)(dense)
    main_output=Dense(2,activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
        
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','binary_crossentropy'])
    one_hot_train_labels = keras.utils.to_categorical(y_train, num_classes=2)  # 将标签转换为one-hot编码
    one_hot_test_labels = keras.utils.to_categorical(y_test, num_classes=2)
    model.fit(x_train_padded_seqs, one_hot_train_labels, batch_size=16,validation_data=(x_test_padded_seqs,one_hot_test_labels),shuffle=True,epochs=20)
    model.save('cnn地震_new1.h5')

    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=-1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
    print('召回率：',metrics.recall_score(y_test,y_predict,pos_label='1'))
    print('精确率：',metrics.precision_score(y_test,y_predict,pos_label='1'))
    #f1为精确率和召回率的调和均值
    '''

def TextCNN_model(x_train_padded, y_train, x_test_padded, y_test):
    model = keras.Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=100)) #使用Embeeding层将每个词编码转换为词向量
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 4, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=2)  # 将标签转换为one-hot编码
    model.fit(x_train_padded, one_hot_labels,epochs=5, batch_size=800)
    model.save('cnn台风.h5')
    
    y_predict = model.predict_classes(x_test_padded)  # 预测的是类别，结果就是类别号
    y_predict = list(map(str, y_predict))
    #result = model.predict(x_test_padded)  # 预测样本属于每个类别的概率
    #result_labels = (result> 0.5).astype("int32")  # 获得最大概率对应的标签
    #y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
    '''
'''
trainDF=load_revise_data()
texts,embeddings_matrix=get_word2vec_dictionaries(trainDF['text'])
x_train_padded,x_test_padded,y_train,y_test=train_test_split(texts,trainDF['label'],test_size=0.2,stratify=trainDF['label'])
TextCNN_model(x_train_padded,y_train,x_test_padded,y_test,embeddings_matrix)
'''
