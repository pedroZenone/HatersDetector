#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:23:14 2018

@Description: Clase de preprocesamiento + entrenamiento de modelo para detectar usuarios Hater vs Non Haters

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
    
    def __init__(self,dataSource,lstopWords = [],verbose = 0):
        
        self.data = dataSource
        self.verbose = verbose           
        
        # levanto las stop words
        self.my_stopwords = stopwords.words('spanish') + stopwords.words('english') + lstopWords + ['RT', 'rt']
        
        # Creo el diccionario del lematizer!
        self.lemmaDict = {}
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
       
    
    def repetidos(self,x):
        y = x
        abc = [x for x in 'abcdfghjkpqvwxyzui']   # saco la ll y rr
        for letra in abc:
            y = re.sub(r''+letra+'{2,}',letra,y)
        y = re.sub(r'l{3,}','ll',y)
        y = re.sub(r'r{3,}','rr',y)
        y = re.sub(r'e{3,}','ee',y)
        y = re.sub(r'o{3,}','oo',y)
        y = re.sub(r's{3,}','ss',y)
        y = re.sub(r'n{3,}','nn',y)
        y = re.sub(r't{3,}','tt',y)
        y = re.sub(r'm{3,}','mm',y)
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

class Denser(BaseEstimator, TransformerMixin):
    """
    To dense
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()

resu = []

def trainMe(stage1,stage2,params,X_train,y_train):
    pipe = Pipeline([                
                    ('feature_gen',stage1 ),
                    ('model',stage2)
                ])
    
    clf = RandomizedSearchCV(pipe, param_distributions= params, cv=5,scoring='roc_auc',n_jobs = -1,verbose = 10)
    clf.fit(X_train,y_train)
    return {'pipe': clf,'score':clf.best_score_}


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
    
    scores = cross_val_score(pipeLr, X_train, y_train, scoring="roc_auc", cv=10)  
    pipeLr.fit(X_train, y_train)
    resu.append({'pipe': pipeLr,'score':scores.mean()})
    
    # NB
    
    stage2 = GaussianNB()

    pipeNB = Pipeline([                
                    ('feature_gen',stage1 ),
                    ('2Dense',Denser()),
                    ('model',stage2)
                ])       

    scores = cross_val_score(pipeNB, X_train, y_train, scoring="roc_auc", cv=10)   
    pipeNB.fit(X_train, y_train)
    resu.append({'pipe': pipeNB,'score':scores.mean()})
    
    return resu
