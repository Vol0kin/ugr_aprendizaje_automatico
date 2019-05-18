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
import seaborn as sns

def read_data_values(in_file, separator=None):
    """
    Funcion para leer los datos de un archivo
    
    :param in_file Archivo de entrada
    :param separator Separador que se utiliza en el archivo (por defecto None)
    
    :return Devuelve los datos leidos del archivo en un DataFrame
    """
    
    # Cargar los datos en un DataFrame
    # Se indica que la primera columna no es el header
    df = pd.read_csv(in_file, sep=separator, header=None)

    return df

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
        

# Establecer la semilla aleatoria
np.random.seed(1)

# Lectura de los datos
df = read_data_values('datos/optdigits.tra', separator=None)
sample = df.values

# Obtener datos y etiquetas
data, labels = divide_data_labels(sample)
print(sample)

print(data.shape)
print(labels.shape)

# Escalar los datos (restar la media de cada caracteristica y dividir entre desviacion tipica)
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)


# Reducir la dimensionalidad de los datos
pca = PCA(n_components=.90)
pca.fit(data)
data_trans = pca.transform(data)

print('Ratios de varianza explicada: ', pca.explained_variance_ratio_.sum())
print('Componentes ', pca.components_.shape)

print(data_trans.shape)
print(data_trans)

df = read_data_values('datos/airfoil_self_noise.dat', separator='\t')
# Asignamos nombres a las columnas (según los atributos)
column_names = ['Frequency', 'Angle of attack', 'Chord length',
              'Free-stream velocity', 'SSD thickness', 'Sound Pressure']
df.columns = column_names

# Informacion sobre numero de filas, columnas y valores nulos
print('Num. de valores del conjunto de datos y valores faltantes:')
print(df.info())

# Obtener número de valores únicos por atributo
print('\nNumero de valores distintos por atributo')

for atribute in column_names:
    print(atribute + ': ' + str(df[atribute].unique().shape[0]))

# Mostrar los valores mínimos de cada atributo
print('\nValores minimos de cada atributo:')
print(df.min())

# Mostrar los valores máximos de cada atributo
print('\nValores maximos de cada atributo:')
print(df.max())

sns.pairplot(df)
print(df.head())