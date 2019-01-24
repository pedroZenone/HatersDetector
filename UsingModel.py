#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:26:44 2018

Este codigo leventa el modelo entrenado y aplica a datos no vistos.
Pipeline de trabajo:
    Raw Data ----> Preprocesing --Text proc--> Modelo ----> Hater/Non-Hater 

@author: pedzenon
"""

from NLP_preproc import NLP_preproc  # Codigo que arme para preprocesar el textp
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

class Denser(BaseEstimator, TransformerMixin):
    """
    To dense
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()


# seteo directrorios...
fileDir = os.path.dirname(os.path.realpath('__file__'))
fileData = os.path.join(fileDir,"data")
fileModel = os.path.join(fileDir,"model")

# levanto la data

data = pd.read_excel(os.path.join(fileData,"combate_users_data.xlsx"))

# Preproceso los datos
data.columns = ["Full Text" if (x == "text") else x for x in data.columns]
data["Full Text"] = data["Full Text"].astype(str)
prepoc = NLP_preproc(data)
prepoc.preprocessing(False)  # start lemmatizing

other_stopwords = pd.read_excel("stopwords.xlsx")["stopwords"].tolist()
single_words = [x for x in 'abcdefghijklmnopqrstuvwxyz'] 
prepoc.update_StopWords(other_stopwords+single_words)

data["Text"] = prepoc.get_procTextTweets()
data = data.loc[data["Text"] != '']

size = 100000
chunk = int(data.shape[0]/size)

# cargo modelo y predict
model = joblib.load(os.path.join(fileModel,'model_PR1.pkl'))

optimal_tresh = 0.367775
cutter = np.vectorize(lambda x: 1 if x > optimal_tresh else 0)

# Como es mucha data, parto el dataset en pedazos de 100000 y hago el predict, luego me quedo con la clase de interes
for i in range(chunk):
    if(i == 0):
        chunk_df = data.iloc[0:size,:]
        chunk_df["hater"] = cutter(model.predict_proba(chunk_df["Text"])[:,1])
        chunk_df = chunk_df.loc[chunk_df.hater == 1]
    else:
        chunk_aux = data.iloc[(i*size) + 1:(i+1)*size,:]
        chunk_aux["hater"] = cutter(model.predict_proba(chunk_aux["Text"])[:,1])
        chunk_aux = chunk_aux.loc[chunk_aux.hater == 1]
        chunk_df = chunk_df.append(chunk_aux,ignore_index = True)
    print((i*size) + 1 ," ", (i+1)*size )

if(data.shape[0] > chunk_df.shape[0]):
    chunk_aux = data.iloc[(chunk*size) + 1:,:]
    chunk_aux["hater"] = cutter(model.predict_proba(chunk_aux["Text"])[:,1])
    chunk_aux = chunk_aux.loc[chunk_aux.hater == 1]
    chunk_df = chunk_df.append(chunk_aux,ignore_index = True)


chunk_df.to_excel("combate_bardeos.xlsx")   # sprite tiene 700000 comentarios, 63k bardeos.  Combate tiene 200k comentarios, 21k bardeos


# =============================================================================
#  Una vez que tengo todos los bardeos de todas las clases los levanto y saco 
#  los usuarios unicos para rankearlos.
# =============================================================================

# Levanto todos los bardeos y rankeo
bardos = pd.read_excel("./data/bardeos/bardos3_PR.xlsx").append(
        pd.read_excel("./data/bardeos/bardos1_PR.xlsx"),ignore_index = True).append(
                pd.read_excel("./data/bardeos/bardos2_PR.xlsx"),ignore_index = True).append(
                        pd.read_excel("./data/bardeos/47st_bardeos.xlsx"),ignore_index = True).append(
                                pd.read_excel("./data/bardeos/combate_bardeos.xlsx"),ignore_index = True).append(
                                        pd.read_excel("./data/bardeos/sprite_bardeos.xlsx"),ignore_index = True)

bardos = bardos.drop_duplicates(subset = ["author","mentions","Full Text"])

def ranking(bardos_aux):
    hateados = list(bardos_aux["mentions"].values)
    df = pd.DataFrame([],columns = ["menciones"])
    df["menciones"] = re.findall(r'\S+',' '.join(hateados))
    rank = df[["menciones"]].groupby(["menciones"]).size().reset_index()
    rank_unique = bardos_aux[["mentions"]].groupby(["mentions"]).size().reset_index()
    rank = pd.merge(rank,rank_unique, left_on=["menciones"],right_on = ["mentions"],how= "left")
    rank = rank.drop(["mentions"],axis = 1)
    rank.columns = ["menciones","count_all","count_unique"]
    return rank

rank = ranking(bardos)
rank.loc[rank.count_unique >5].to_excel("rankingSPR_47st_comb_others.xlsx")

# Si queres analizar algun usuario en particular y ver sus menciones:
bardeada = bardos.loc[bardos["mentions"] == "yazschnan"]

# Aca hice uso de la API de twitter para agregarle datos adicionales del usuario
bardeos = pd.read_excel("./data/bardeos/rankingSPR_47st_comb_others.xlsx")
bardeos_trunc = pd.merge(test,UserData,left_on = ["menciones"],right_on = ["user_name"],how = "left").drop(["user_name"],axis = 1)
bardeos_truncc = bardeos_trunc.loc[(bardeos_trunc["followers"] > 1000) & (bardeos_trunc["followers"] < 40000)]

def my_filter(x):
    
    x = x.lower()
    if(re.search(r'periodis|macri|cristina|peron|kirchner|nestor',x) == None):
        return True
    else:
        return False

# Me genero una tabla con el ranking de usuarios que son potenciales no influencers    
bardeos_trunccc = bardeos_truncc[bardeos_truncc["description"].apply(lambda x: my_filter(x))]
bardeos_trunccc.to_excel("usuariosNoInfluencers_Ranking.xlsx")
# Me genero una base con los insultos con los usuarios de arriba
bardos_toanalize = bardos[bardos["mentions"].isin(bardeos_trunccc["menciones"])]
bardos_toanalize.to_excel("usuariosNoInfluencers_Tweets.xlsx")

# =============================================================================
#  Uso el modelo de LDA para ver cuantos usuarios tengo en cada topico!
# =============================================================================



"""
    CONCLUSION:
        Arrancamos con +3millones de tweets y terminamos en 340mil tweets de bardeos y 100k de usuarios barderos
        de 100k de usuarios, solo 6500 usuarios recibieron mas de 5 hateos 
        Encontramos 5 clusters, (partiendo de los 100k usuarios) de los cuales:
"""

"""
Cosas importantes:
    ./bardeos/rankingSPR_47st_comb_others.xlsx -> tiene el rankeo de todos los usuarios
    ./bardeos/... hay varios xlsx con los bardeos de combate,sprite,47st y others que scrapie
    ./modelo/modelo_PR  este es el modelo que esta andando 10 puntos!
    
"""

