# -*- coding: utf-8 -*-

# Práctica 0
# Autor: Vladislav Nikolov Vasilev

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# #############################################################################
# Parte 1

# Cargar base de datos iris
iris = datasets.load_iris()

# Cargar características (x) y clase (y)
x = iris.data
y = iris.target

# Obtener dos últimas columnas de x
x_last_cols = x[:, -2:]

# Visualizacion

# Diccionario de colores K = numero_grupo, V = color
color_dict = {0: 'red', 1: 'green', 2: 'blue'}

# Recorrer los grupos unicos y pintarlos de un color cada uno
for sub_group in np.unique(y):
    index = np.where(y == sub_group)
    plt.scatter(x_last_cols[index, 0], x_last_cols[index, 1], c = color_dict[sub_group], label = 'Group {}'.format(sub_group))

plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()

plt.show()

# #############################################################################
# Parte 2

# #############################################################################
# Parte 3

# Rango de valores [0, 2 * pi]
x_axis = np.linspace(0, 2 * np.pi, 100)

# Valores de las funciones
sin_x = np.sin(x_axis)
cos_x = np.cos(x_axis)
sin_cos_x = sin_x + cos_x

# Visualización

# Limpiar la figura
plt.clf()

plt.plot(x_axis, sin_x, 'k--', label = 'sin(x)')
plt.plot(x_axis, cos_x, 'b--', label = 'cos(x)')
plt.plot(x_axis, sin_cos_x, 'r--', label = 'sin(x) + cos(x)')

plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.title('Trigonometric functions')
plt.legend()
plt.grid()

plt.show()