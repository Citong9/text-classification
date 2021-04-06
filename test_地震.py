#test_地震
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import jieba,numpy,pandas,re,csv
#import sys
#from CNN import mycrossentropy
from gensim.models import Word2Vec,KeyedVectors
from datatextCNN import  ReLabel

def init_data(model_number,word2vec_number,test_number):
  stopwords=open("D:/垃圾分类/停用词表.txt", "r", encoding="utf-8").read()
  #model=load_model('cnn地震.h5',custom_objects={'mycrossentropy':mycrossentropy})
  if model_number==1:
    model=load_model('cnn地震4.h5')
  if model_number==2:
    model=load_model('cnn地震.h5')
  if model_number==3:
    model=load_model('cnn地震_new1.h5')
  if model_number==4:
    model=load_model('cnn地震_new_ensemble2.h5')
  if word2vec_number==1:
    word2vec_model=Word2Vec.load('word2vec_model1')
  if word2vec_number==2:
    word2vec_model=Word2Vec.load('word2vec_model2')
  vocab_list=[word for word,Vocab in word2vec_model.wv.vocab.items()]
  word_index={" ":0}
  for i in range(len(vocab_list)):
    word=vocab_list[i]
    word_index[word]=i+1
  if test_number==1:
    test=open('D:/垃圾分类/weibo_long_test.txt','r',encoding="utf-8").readlines()
  if test_number==2:
    test=[]
    with open('D:/垃圾分类/xinlang_new_test.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
          test.append(row[1]+row[2])
  return test,word_index,model,stopwords

#while True:
def textCNNFilter(test,word_index,model,stopwords):
  pattern = re.compile(r'[^\u4e00-\u9fa5]')
  for text in test:
   text = re.sub(pattern, '', text)
   text=text.replace("\n","")
   #text=input("text:")
   if text==' ':
      break
   else:
    words=jieba.lcut(text)
    texts=[]
    string=''
    for word in words:
      if word in stopwords:
        continue
      else:
        string=string+word+' '
    texts.append(string[:-1])
    trainDF=pandas.DataFrame()
    trainDF['text']=texts
    '''
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(trainDF['text'])
    x_word_ids=tokenizer.texts_to_sequences(trainDF['text'])
    x_padded=pad_sequences(x_word_ids,maxlen=50)
    print(x_padded)
    '''
    data=[]
    for sentence in trainDF['text']:
      new_txt=[]
      for word in sentence:
        try:
          new_txt.append(word_index[word])
        except:
          new_txt.append(0)
      data.append(new_txt)
    x_padded=pad_sequences(data,maxlen=80)
    
    result = model.predict(x_padded)  # 预测样本属于每个类别的概率
    result_labels = numpy.argmax(result, axis=-1)  # 获得最大概率对应的标签
    if (texts==['转发 微博'] or '政治' in string[:-1]):
        result_labels=1-result_labels
    if ('车险' in string[:-1]):
        result_labels=0
    #if model==model1:
    if int(result_labels)==0:
       result_labels=ReLabel(string[:-1])
    print('{}  {}'.format(text[:70],int(result_labels)))

test,word_index,model,stopwords=init_data(2,2,1)
textCNNFilter(test,word_index,model,stopwords)

