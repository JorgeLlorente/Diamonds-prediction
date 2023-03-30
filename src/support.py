import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import LabelEncoder 

import pickle

import sys
sys.path.append("../")

import src.support as sp



def detectar_outliers(lista_columnas, dataframe):

    dicc_indices = {} # creamos un diccionario donde almacenaremos índices de los outliers

    # iteramos por la lista de las columnas numéricas de nuestro dataframe
    for col in lista_columnas:

        #calculamos los cuartiles Q1 y Q3
        Q1 = np.nanpercentile(dataframe[col], 25)
        Q3 = np.nanpercentile(dataframe[col], 75)

        # calculamos el rango intercuartil
        IQR = Q3 - Q1

        # calculamos los límites
        outlier_step = 1.5 * IQR

        # filtramos nuestro dataframe para indentificar los outliers
        outliers_data = dataframe[(dataframe[col] < Q1 - outlier_step) | (dataframe[col] > Q3 + outlier_step)]


        if outliers_data.shape[0] > 0: # chequeamos si nuestro dataframe tiene alguna fila. 

            dicc_indices[col] = (list(outliers_data.index)) # si tiene fila es que hay outliers y por lo tanto lo añadimos a nuestro diccionario


    return dicc_indices



def label_encoder(df, columnas):
    le = LabelEncoder()
    for col in df[columnas].columns:
        nuevo_nombre = col + "_encoded"
        df[nuevo_nombre] = le.fit_transform(df[col])
        with open(f'../data/encoding{col}.pkl', 'wb') as s:
            pickle.dump(le, s)
    return df