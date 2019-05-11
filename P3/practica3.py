# -*- coding: utf-8 -*-
"""
PRÁCTICA 3
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA               # Reducir dimensiones
from sklearn.preprocessing import StandardScaler    # Escalar los datos
from sklearn.model_selection import StratifiedKFold # Particionar las muestras

def read_data_values(in_file, separator=None):
    """
    Funcion para leer los datos de un archivo
    
    :param in_file Archivo de entrada
    :param separator Separador que se utiliza en el archivo (por defecto None)
    
    :return Devuelve los datos leidos del archivo
    """
    
    # Cargar los datos en un DataFrame
    # Se indica que la primera columna no es el header
    df = pd.read_csv(in_file, sep=separator, header=None)
    
    # Obtener los valores del DataFrame
    data = df.values
    
    return data


def divide_data_labels(sample):
    """
    Funcion que divide una muestra en los datos y las etiquetas
    
    :param sample Conjunto de valores que se quieren separar
    
    :return Devuelve los datos y las etiquetas
    """
    
    data = sample[:, :-1]
    labels = sample [:, -1]
    
    return data, labels


def stratify_k_fold(k, X, y):
    """
    Funcion para dividir la muestra en K conjuntos de test y entrenamiento
    conservando el 80% de los datos en train y el 20% en test de forma
    proporcional (conservando las clases)
    
    :param k Numero de folds (particiones) a crear
    :param X Conjunto de datos
    :param y Conjunto de etiquetas
    
    :return Devuelve cada vez un conjunto de X e y de entrenamiento y de test
    """    
    
    # Crear un nuevo StratifiedKFold que mezcle los datos
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)
    
    # Generar las particiones y devolverlas una por una
    for train_indx, test_indx in skf.split(X, y):
        X_train, y_train = X[train_indx], y[train_indx]
        X_test, y_test = X[test_indx], y[test_indx]
        
        yield X_train, y_train, X_test, y_test
        

def scale_data(data, scaler=None):
    """
    Funcion para normalizar el conjunto de datos restando la media y dividiendo
    entre la desviacion tipica utilizando un objeto de escalado ya creado o uno
    nuevo
    
    :param data Conjunto de datos a transformar
    :param scaler Objeto de escalado utilizado (por defecto None)
    
    :return Devuelve el escalado aplicado y los datos transformados
    """
    
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(data)
    
    transformed_data = scaler.transform(data)
    
    return scaler, transformed_data
    

def pca_dimensionality_reduction(data, pca=None):
    
    if pca == None:
        

# Establecer la semilla aleatoria
np.random.seed(1)

# Lectura de los datos
sample = read_data_values('datos/optdigits.tra', separator=None)

# Obtener datos y etiquetas
data, labels = divide_data_labels(sample)
print(sample)

print(data.shape)
print(labels.shape)

# Escalar los datos (restar la media de cada caracteristica y dividir entre desviacion tipica)
scaler, data = scale_data(data)


# Reducir la dimensionalidad de los datos
pca = PCA(n_components=.90)
pca.fit(data)
data_trans = pca.transform(data)

print('Ratios de varianza explicada: ', pca.explained_variance_ratio_.sum())
print('Componentes ', pca.components_.shape)

print(data_trans.shape)
print(data_trans)