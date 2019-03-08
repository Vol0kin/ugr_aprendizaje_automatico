# -*- coding: utf-8 -*-

"""
Práctica 1.
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import matplotlib.pyplot as plt

# Función E(u,v)
def E(u, v):
    return (u**2 * np.exp(v) - 2 * v**2 * np.exp(-u))**2


# Derivada parcial de E respecto a u
def diff_Eu(u, v):
    return 2 * (u**2 * np.exp(v) - 2 * v**2 * np.exp(-u)) * (2 * u * np.exp(v) + 2 * v**2 * np.exp(-u))

# Derivada parcial de E respecto a v
def diff_Ev(u, v):
    return 2 * (u**2 * np.exp(v) - 2 * v**2 * np.exp(-u)) * (u**2 * np.exp(v) - 4 * v * np.exp(-u))

# Gradiente de E
def gradient_E(u, v):
    return np.array([diff_Eu(u, v), diff_Ev(u, v)])

def descent_gradient(initial_w, eta, error, max_iter):
    w = initial_w.copy()
    iter = 0

    while E(*w) > error and iter < max_iter:
        iter += 1
        print("w: {}, gradiente: {}".format(w, np.array([diff_Eu(*w), diff_Ev(*w)])))
        w = w - eta * np.array([diff_Eu(*w), diff_Ev(*w)])



    return w, iter

print('EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS')
print('Ejercicio 1')

# Fijamos los parámetros que se van a usar en el cómputo de la gradiente descendente
eta = 0.01
max_iter = 10000000000
error = 1e-14
initial_w = np.array([1.0, 1.0])

w, it = descent_gradient(initial_w, eta, error, max_iter)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

input("\n--- Pulsar tecla para continuar ---\n")
