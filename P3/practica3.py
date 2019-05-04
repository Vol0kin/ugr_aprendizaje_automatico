# -*- coding: utf-8 -*-
"""
PRÁCTICA 3
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA  # Reducir dimensiones
from sklearn import preprocessing      # Escalar los datos

sample = pd.read_csv('datos/optdigits.tra', header=None).values
print(sample)

# Obtener datos y etiquetas
data = sample[:, :-1]
labels = sample[:, -1]

# Escalar los datos (restar la media de cada caracteristica y dividir entre desviacion tipica)
data = preprocessing.scale(data)

print(data)
print(labels)

pca = PCA(n_components=10)
pca.fit(data)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print('Componentes ', pca.components_.shape)