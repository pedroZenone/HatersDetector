# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:05:24 2018

Este código entrena un modelo de clasificación binario.
Pipeline:
    Preprocesamiento --Text preproc--> Bag of Words/Tfidf/LSA --Vector of words--> Modelo ----> Hater/Non-Hater

La clase target (1) se encuentra altamente desbalanceada (10/90%), es por ello que se debe 
entrenar maximizando area sobre la curva PR. Una vez elejido el modelo que menor error tenga
se elije el umbral de corte en base al F1score (ya que se busca tener buena precision y buen
recall)

@author: pedzenon
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_curve, auc

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
import emoji
from nltk.corpus import stopwords
from string import punctuation  
import re
from nltk import word_tokenize  
from unicodedata import normalize
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

# %%       Definitions

# =============================================================================
# Preprocessing class:
# =============================================================================

class NLP_preproc:    
    
    def __init__(self,dataSource,lstopWords = [],verbose = 0, withStopwords = True, withLemma = True):
        
        self.data = dataSource
        self.verbose = verbose           
        
        # levanto las stop words
        if(withStopwords == True):
            self.my_stopwords = stopwords.words('english') + stopwords.words('spanish')  + lstopWords + ['RT', 'rt']
        else:
            self.my_stopwords = lstopWords + ['RT', 'rt']
            
        # Creo el diccionario del lematizer!
        self.lemmaDict = {}
        
        if(withLemma == True):
            with open('lemmatization-es_v2.txt', 'rb') as f:
               data = f.read().decode('utf8').replace(u'\r', u'').split(u'\n')
               data = [a.split(u'\t') for a in data]
            
            with open('lemmatization-es_add_v2 .txt', 'rb') as f:
               data_2 = f.read().decode('utf8').replace(u'\r', u'').split(u'\n')
               data_2 = [a.split(u'\t') for a in data_2]
            
            data = data+data_2  # uno los dos diccionarios y cargo las keys con valor
               
            for a in data:
               if len(a) >1:
                  self.lemmaDict[a[1]] = a[0]
                  
            if(verbose > 0):
                    print("Lemma dict Uploaded")  

    def my_lemmatizer(self,word):
       return self.lemmaDict.get(word, word)
       #return word
       
    
    def repetidos(self,x):
        y = x
        abc = [x for x in 'abcdfghjkpqvwxyzuieosntm']   # saco la ll y rr
        for letra in abc:
            y = re.sub(r''+letra+'{2,}',letra,y)
        y = re.sub(r'l{3,}','ll',y)
        y = re.sub(r'r{3,}','rr',y)
        return y
    
    def delete_containedWord(self,y):        
        indexes = [i for i,x in enumerate(self.texts) if(not re.search(r'\b'+y+r'\b',x))]
        self.texts = [self.texts[x] for x in indexes]
        self.token = [word_tokenize(x) for x in self.texts]
        self.tweets = [self.tweets[x] for x in indexes] 
        self.data = self.data.iloc[indexes,:]
    
    def delete_MentionedAuthor(self,x):
        self.data = self.data.reset_index(drop = True)
        self.data = self.data.loc[self.data["Mentioned Authors"] != x]
        indexes = self.data.index.tolist()
        self.texts = [self.texts[x] for x in indexes]
        self.token = [self.token[x] for x in indexes]
        self.tweets = [self.tweets[x] for x in indexes]         
        
    def acentos_out(self,s):
        x = re.sub(
            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
            normalize( "NFD", s), 0, re.I )
        return x 
        
    def tokenize(self,text):  
        
        if(self.include_emoji == True):
            emojis = re.findall(r'[^\x00-\xFF]', text)   # me fijo los potenciales emjis
            emojis = [y for y in emojis if(y in emoji.UNICODE_EMOJI)]   # los termino de filtrar, esta estapa ya es mucho mas liviana
        else:
            emojis = []
            
        signos = list(re.findall(r'[?¿!¡]|\.{3}',text)) #agrego los signos de pregunta / excalamacion
        emojis = emojis + signos
        
        text=text.lower()
        text = re.sub(r'https?(.*?)(\s|$)',' linkpagina ',text)    # las paginas de internet las transformo a un nombre que sea link
        text = re.sub(r'pic\.twitter(.*?)(\s|$)',' ',text)    # las paginas de internet las transformo a un nombre que sea link
        text = re.sub(r'\B@\S*\s?','',text)  # le saco el @algo
        text = re.sub(r'\B#\S*\s?','',text)  # le saco el #algo ya que me interesa el contenido en si, capaz que pone un #pelotudo el pibe..
        text = self.acentos_out(text)        
        text = ''.join(re.findall(r'[a-z\s]',text)) # le saco los caracteres que no sean words ni numeros. 
        text = self.repetidos(text)
        text = re.sub(r'(^|\s|\w*)(jaj|kak|jsj|kaj|ksk|jsk)(.*?)(\s|$)',' jaja ',text)  # estandarizo el jaja
        tokens =  word_tokenize(text)
        tokens = [x for x in tokens if(x not in self.my_stopwords)] 
        tokens = [self.my_lemmatizer(x) for x in tokens]   
        tokens = tokens + emojis   # le agrego los emojis
        return tokens 

    def preprocessing(self,include_emoji = True):
        
        self.include_emoji = include_emoji
        
        self.tweets = self.data["Full Text"].as_matrix().tolist()
        self.token = [self.tokenize(x) for x in self.tweets]
        self.texts = [' '.join(x) for x in self.token]      
        
        if(self.verbose > 0):
            print("Data Preprocesada. Para obtener los tweets crudos: get_rawTweets()")
            print("Para obtener los tweets procesados en Tokens: get_procTokenTweets()")
            print("Para obtener los tweets procesados en text : get_procTextTweets()")
            
    def get_rawTweets(self):
        return self.tweets
    
    def get_procTokenTweets(self):
        return self.token
    
    def get_procTextTweets(self):
        return self.texts
    
    def get_Data(self):
        return self.data
    
    # Te grafica la cantidad de tweets en el tiempo
    def exploratoryPlot(self):
        analysis = pd.to_datetime(self.data["Date"], format='%Y-%m-%d', errors='coerce')
        analysis = analysis.apply(lambda x: str(x.year) + '-' + str(x.month).zfill(2) + '-' + str(x.day).zfill(2))
        GB=pd.DataFrame(analysis.groupby(analysis).count())
        GB.columns = ["Count"]
        GB = GB.reset_index()
        ax = sns.barplot(x = "Count",y = "Date",data = GB)
        ax.set( ylabel = "Date",xlabel="Count(Tweets/Instagram)")
        plt.show()
    
    # En caso de querer recargar las stopwords y no tner que reprocesar todo devuelta!    
    def update_StopWords(self,lStopWords):

        self.token = [[x for x in y if(x not in lStopWords)] for y in self.token]
        self.texts = [' '.join(x) for x in self.token]
   

    def get_potencialLotWords(self,hist=0,tresh = 15):    
        
        self.tresh = tresh
        
        def do_nothing(tokens):
            return tokens
        
        vectorizer = CountVectorizer(tokenizer=do_nothing,
                                 preprocessor=None,
                                 lowercase=False)
        
        vectorizer.fit_transform(self.token)  # a sparse matrix
        vocab = vectorizer.get_feature_names()  # a list
        
        if(hist == 1):
            lplt = [len(x) for x in vocab]  # me armo una lista para graficar la distribucion de largos
            plt.hist(lplt, bins = np.arange(min(lplt),max(lplt),1))
            
        return [x for x in vocab if(len(x) >= tresh)]
           
    
    def truncator(self,x,pattern,lPattern):
        
        if(pattern.search(x)):
            token = word_tokenize(x)
            
            aux = [[ k  for k in lPattern if((k in y) & (len(y) >= self.tresh))] for y in token]   # las palabras long las va a poner con sus posibles combinaciones
            auxx= [x if(len(x) > 0) else [token[i]] for i,x in enumerate(aux)] # las que no son plabras long quedaban vacias, entonces las relleno con esta sentencia
            flatten = list(itertools.chain.from_iterable(auxx)) # hago flat la lista 
            
            return ' '.join([self.my_lemmatizer(x) for x in flatten] ) # la vuelvo a pasar por el lemmatizer y la transformo en texto
        
        return x
    
    
    def truncateLongWords(self,lLong):
        
        pattern = re.compile(''.join(['|'+ y for y in lLong])[1:])
        self.texts = [self.truncator(x,pattern,lLong) for x in self.texts]
        self.token = [word_tokenize(x) for x in self.texts]
        
    def countVectorizer(self):
        def do_nothing(tokens):
            return tokens
    
        vectorizer = CountVectorizer(tokenizer=do_nothing,
                                 preprocessor=None,
                                 lowercase=False)
        
        dtm = vectorizer.fit_transform(self.token)  # a sparse matrix
        vocab = vectorizer.get_feature_names()  # a list
        words_freq = np.asarray(dtm.sum(axis=0)).T
        
        DataFinal = pd.DataFrame([],columns = ["word","frequency"])
        DataFinal["word"] = vocab
        DataFinal["frequency"] = words_freq
        
        return DataFinal
    
    def inspeccion(self,find):
        self.data = self.data.reset_index(drop= True)
        indexes = [i for i,x in enumerate(self.texts) if(len(re.findall(find,x)) > 0)]
        resu = pd.DataFrame([],columns = ['indiceRaw','mensage'])
        resu['mensage'] = [self.tweets[i] for i in indexes]
        resu['indiceRaw'] = indexes
        urls = self.data.Url.values.tolist()
        resu["url"] =  [urls[i] for i in indexes]
        return resu


# =============================================================================
# ~   Funciones de entrenamiento
# =============================================================================

#   Esto es necesario para TfiDF, este stage te devuelve una matriz sparse pero
#   como los modelos trabajan con Matrices densas

class Denser(BaseEstimator, TransformerMixin):
    """
    To dense
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()

resu = []

# Esta función arma el pipeline en base a las stages que le pasas por parametro
# y entrena el modelo con los hiperparametros que le pasas en params. Devuelve el }
# pipeline ganador

def trainMe(stage1,stage2,params,X_train,y_train):
    pipe = Pipeline([                
                    ('feature_gen',stage1 ),
                    ('model',stage2)
                ])
    
    clf = RandomizedSearchCV(pipe, param_distributions= params, cv=5,scoring='recall',n_jobs = -1,verbose = 10)
    clf.fit(X_train,y_train)
    return {'pipe': clf,'score':clf.best_score_}


# Esta bunción entrena segun el pipeline: stage1->Modelo
# Los modelos con los que prueba son: SVM, Random Forest, Logistic regression, Naive Bayes
def SVC_RF_LR(stage1,balance,X_train,y_train):

    resu = []
    # SVC
    stage2 = SVC(class_weight=balance, probability=True)
        
    params =   {'model__C': [1, 10, 100, 1000], 'model__gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'model__kernel': ['rbf','linear']}
#                {'model__C': [1, 10, 100, 1000], 'model__kernel': ['linear']},
             
    
    resu.append(trainMe(stage1,stage2,params,X_train,y_train))
    
    # RF
    stage2 = RandomForestClassifier(class_weight=balance )
    
    params = { 
                'model__n_estimators': [100,200, 500,800, 1200,1800],
                'model__max_features': ['auto', 'sqrt', 'log2'],
                'model__max_depth' : [4,5,6,7,8],
                'model__criterion' :['gini', 'entropy']
                }
         
    
    resu.append(trainMe(stage1,stage2,params,X_train,y_train))
    
    # LR
    
    stage2 = LogisticRegression(class_weight=balance,solver = 'liblinear')
    
    pipeLr = Pipeline([                
                        ('feature_gen',stage1 ),
                        ('model',stage2)
                    ])       
    
    scores = cross_val_score(pipeLr, X_train, y_train, scoring="roc_auc", cv=5)  
    pipeLr.fit(X_train, y_train)
    resu.append({'pipe': pipeLr,'score':scores.mean()})
    
    # NB
    
    stage2 = GaussianNB()

    pipeNB = Pipeline([                
                    ('feature_gen',stage1 ),
                    ('2Dense',Denser()),
                    ('model',stage2)
                ])       

    scores = cross_val_score(pipeNB, X_train, y_train, scoring="roc_auc", cv=5)   
    pipeNB.fit(X_train, y_train)
    resu.append({'pipe': pipeNB,'score':scores.mean()})
    
    return resu

# %%    Main
    
fileDir = os.path.dirname(os.path.realpath('__file__'))
fileData = os.path.join(fileDir,"data")

# Levanto la data
data = pd.read_excel(os.path.join(fileData,"trainingData.xlsx"))
data["permalink"] = "nolink"
data = data[['author','destino','permalink','reply','target']]
data.columns = ['author','mentions','permalink','Full Text','hater']
data1 = pd.read_excel(os.path.join(fileData,"trainingData2.xlsx"))
data1["permalink"] = "nolink"
data1 = data1[['author','destino','permalink','reply','target']]
data1.columns = ['author','mentions','permalink','Full Text','hater']
data2 = pd.read_excel(os.path.join(fileData,"bardos1_retrain.xlsx"))
data2 = data2[['author','mentions','permalink','Full Text','hater']]
data3 = pd.read_excel(os.path.join(fileData,"tageame2.xlsx"))
data3 = data3[['author','mentions','permalink','Full Text','hater']]

data = data.append(data1,ignore_index = True).append(data2,ignore_index = True)
data = data.append(data3,ignore_index = True)

data = pd.read_excel("./data/trainme_fullv1.xlsx")
data = data.dropna(subset = ["hater","Full Text"])

# adapto y corro el preproc

#data.columns = ["Full Text" if (x == "reply") else x for x in data.columns]
#data["Full Text"] = data["Full Text"].astype(str)
data.loc[data["hater"] == 'º',"hater"] = 1
data["hater"] = data["hater"].astype(int)
prepoc = NLP_preproc(data,withLemma = False,withStopwords = False)
prepoc.preprocessing(False)

other_stopwords = pd.read_excel("stopwords.xlsx")["stopwords"].tolist()
single_words = [x for x in 'abcdefghijklmnopqrstuvwxyz'] 
prepoc.update_StopWords(other_stopwords+single_words)

counter = prepoc.countVectorizer()
# https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

# Borro los tweets que contengan palabras prohibidas
prepoc.delete_containedWord('cfk')
prepoc.delete_containedWord('cristina')
prepoc.delete_containedWord('macri')
prepoc.delete_containedWord('macrismo')
prepoc.delete_containedWord('peron')
prepoc.delete_containedWord('peronista')
prepoc.delete_containedWord('lareta')
prepoc.delete_containedWord('peroncho')
prepoc.delete_containedWord('river')
prepoc.delete_containedWord('boca')

data = prepoc.get_Data()
data["Text"] = prepoc.get_procTextTweets()
data = data.loc[data["Text"] != '']
#data.loc[data["target"] == 'º',"target"] = 0  

# Arranco a setear y entrenar:
X_train, X_test, y_train, y_test = train_test_split(data["Text"], data["hater"], test_size=0.33, random_state=42)

##   TF-idf:
stage1 = TfidfVectorizer( max_features=10000,analyzer='word', max_df = .8, min_df=.001, token_pattern=r'(\S+)')

asd = SVC_RF_LR(stage1,"balanced",X_train,y_train)
models = ['SVC','RF','LR','NB']
models_ = [x + "_tfidf" for x in models]
resus = pd.DataFrame(asd)
resus["model"] = models_

## CountVectorizer:
stage1 = CountVectorizer( analyzer='word', token_pattern=r'(\S+)')
asd = SVC_RF_LR(stage1,"balanced",X_train,y_train)

aux = pd.DataFrame(asd)
models_ = [x + "_Counter" for x in models]
aux["model"] = models_
resus = resus.append(aux,ignore_index = True)

# LSA:

stage1 = TfidfVectorizer( max_features=10000,analyzer='word', max_df = .8, min_df=.001, token_pattern=r'(\S+)')
x_idf = stage1.fit_transform(X_train)
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(x_idf, y_train)


pipeLr = Pipeline([                
                    ('svd',TruncatedSVD() ),
                    ('norm', Normalizer(copy=False)),
                    ('model',LogisticRegression(solver = 'liblinear'))
                ]) 

param_grid = {'svd__n_components':list(range(100,700,20))}

search = RandomizedSearchCV(pipeLr,param_distributions=param_grid,scoring="average_precision", cv=5,verbose = 20)
search.fit(X_sm,y_sm)

resus = resus.append([{'pipe': search, 'score':search.best_score_, 'model': 'LSA'}],ignore_index = True)
in2 = stage1.transform(X_test)
probas_test = search.predict_proba(in2)
y_pred = search.predict(in2)

from sklearn.metrics import recall_score

recall_score(y_test,y_pred)

# Wining model:

stage2 = GaussianNB()
stage1 = CountVectorizer( analyzer='word', token_pattern=r'(\S+)')

stage2 = LogisticRegression(class_weight='balanced',solver = 'liblinear')
    
pipeLr = Pipeline([                
                ('feature_gen',stage1 ),
                ('model',stage2)
            ])   

pipeLr.fit(X_train, y_train)
from sklearn.metrics import precision_recall_curve

# Me armo una tablita con F1 score, Precision, Recall en base al umbral. El umbral
# ganador es el que mas F1 tenga.

y_pred = pipeLr.predict_proba(X_test)
precision,recall,tresh = precision_recall_curve(y_test,y_pred[:,1])
f1 = 2*(precision*recall)/(precision + recall)
tresh = np.append(tresh,1)
scores = pd.DataFrame([],columns = ["tresh","precision","recall","f1"])
scores["tresh"] = tresh
scores["precision"] = precision
scores["recall"] = recall
scores["f1"] = f1

tresh = 0.367775 # Recall = 0.77; precision = 0.56

import joblib
# guardo modelo
fileModel = os.path.join(fileDir,"model")
joblib.dump(pipeLr, os.path.join(fileModel,'model_PR1.pkl'))

# =============================================================================
# Ahora armo el modelo final...
# =============================================================================

best_model = resus.loc[resus.model == "LR_Counter"]["pipe"].values[0]
probas_test = best_model.predict_proba(X_test)

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test,probas_test[:,1])
print("Area bajo la curva: ", auc(fpr, tpr))

import scikitplot as skplt
skplt.metrics.plot_roc_curve(y_test, probas_test)
plt.show()

# Seleccionando Cut off:
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
optimal_tresh = list(roc.ix[(roc.tf-0).abs().argsort()[:1]]["thresholds"])[0]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])

# Optimal tresh: 0.302

fileDir = os.path.dirname(os.path.realpath('__file__'))
fileModel = os.path.join(fileDir,"model")

import joblib
# guardo modelo
joblib.dump(best_model, os.path.join(fileModel,'model.pkl'))

# =============================================================================
#                           Uso el modelo  
# =============================================================================

fileDir = os.path.dirname(os.path.realpath('__file__'))
fileModel = os.path.join(fileDir,"model")

import joblib
      
# Levanto los datos

data = pd.read_excel("./data/MoreInfluencersAzu.xlsx")

# Preproc text

data.columns = ["Full Text" if (x == "text") else x for x in data.columns]
data["Full Text"] = data["Full Text"].astype(str)
prepoc = NLP_preproc(data)
prepoc.preprocessing(True)

single_words = [x for x in 'abcdefghijklmnopqrstuvwxyz'] 
prepoc.update_StopWords(single_words + ['pedro','pedrito','sol','perez','flor','gime','jimena','jime','bimbo','angela'])

data["Text"] = prepoc.get_procTextTweets()
data = data.loc[data["Text"] != '']

# cargo modelo
model = joblib.load(os.path.join(fileModel,'model.pkl'))
 
optimal_tresh = 0.302
cutter = np.vectorize(lambda x: 1 if x > optimal_tresh else 0)
data["hater"] = cutter(model.predict_proba(data["Text"])[:,1])

bardeos = data.loc[data["hater"] == 1]
haters = list(set(bardeos["author"]))

# =============================================================================
#               Scrapeo los haters para ver a quien le hablaron
# =============================================================================

       
# =============================================================================
#               Levanto todos los usuarios me fijo quien bardeo
# =============================================================================
    
import threading
import signal
t = threading.Thread(target=foo, name="test", args=(1000,))
t.start()

t.is_alive()
t.ident
signal.pthread_kill(t.ident, signal.SIGTERM)

p = multiprocessing.Process(target=tweeter_scrap, name="Scraper", args=(hater,))
p.start()

if(p.is_alive()):
    print("hola")

tweeter_scrap(hater)

import multiprocessing

# Your foo function
def foo(n):
    for i in range(n):
        print ("Tick",n)
        time.sleep(1)

p = [1,2,3,4]
if __name__ == '__main__':
    
    for i in range(4):
    # Start foo as a process
        p[i] = multiprocessing.Process(target=foo, name="Foo", args=(i,))
        p[i].start()
    
        # Wait 10 seconds for foo
#        time.sleep(5)
    
        # Terminate foo
#        p[i].terminate()
    
        # Cleanup
        p[i].join()
        print("paso")
 
import multiprocessing

TIMEOUT = 60

def foo(n):
    for i in range(n):
        print ("Tick",n)
        time.sleep(1)

process = multiprocessing.Process(target=foo, name="Foo", args=(1,))
process.daemon = True
process.start()

process.join(10)
if process.is_alive():
    print("Function is hanging!")
    process.terminate()
    print("Kidding, just terminated!")


excels = os.listdir('./data/other_influencers')  
excels = [x for x in excels if('.xlsx' in x)]      
first = pd.read_excel('./data/other_influencers/' + excels[0])

for excel in excels[1:]:
    first = first.append(pd.read_excel('./data/other_influencers/' + excel), ignore_index = True)
    
    
df = first.sample(frac=1).reset_index(drop=True)
df = df.dropna()
df.to_excel('MoreInfluencersAzu.xlsx')