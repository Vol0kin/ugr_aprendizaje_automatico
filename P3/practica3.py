# -*- coding: utf-8 -*-
"""
PRÁCTICA 3
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA               # Reducir dimensiones
from sklearn.preprocessing import StandardScaler    # Escalar los datos

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



# Lectura de los datos
sample = read_data_values('datos/airfoil_self_noise.dat', separator='\t')

# Obtener datos y etiquetas
data = sample[:, :-1]
labels = sample[:, -1]

print(sample)

print(data.shape)
print(labels.shape)

# Escalar los datos (restar la media de cada caracteristica y dividir entre desviacion tipica)
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)



pca = PCA(n_components=.90)
pca.fit(data)
data_trans = pca.transform(data)

print('Ratios de varianza explicada: ', pca.explained_variance_ratio_.sum())
print('Componentes ', pca.components_.shape)

print(data_trans.shape)
print(data_trans)