# -*- coding: utf-8 -*-

"""
Práctica 1.
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas   # Tablas

###############################################################################
# Funciones necesarias

#################################################################
#################################################################
#################################################################
# EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS

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

# Funcion f(x, y)
def f(x, y):
    return x**2 + 2 * y**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Derivada parcial de f respecto a x
def diff_fx(x, y):
    return 2 * x + 4 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Derivada parcial de f respecto a y
def diff_fy(x, y):
    return 4 * y + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

# Gradiente de f
def gradient_f(x, y):
    return np.array([diff_fx(x, y), diff_fy(x, y)])


def descent_gradient(initial_w, function, gradient, eta=0.01, threshold=None, iterations=100):
    """
    Función para el cálculo de la gradiente descendente
    
    :param initial_w: Pesos iniciales
    :param function: Función a evaluar
    :param gradient: Función gradiente a utilizar
    :param eta: Valor de la tasa de aprendizaje (por defecto 0.01)
    :param threshold: Valor umbral con el que parar (por defecto None)
    :param iterations: Número máximo de iteraciones que tiene que hacer el bucle
                       (por defecto 100)
    
    :returns: Devuelve el peso final (w), el número de iteraciones que ha llevado
              conseguir llegar hasta éste, un array con todos los w y un array con
              los valores de w evaluados en function
    """
    
    w = np.copy(initial_w)                  # Se copia initial_w para evitar modificarlo
    iter = 0                                # Se inicializan las iteraciones a 0
    w_list = []                             # Se inicializa una lista vacía con los valors de w
    func_values_list = []                   # Se inicializa una lista vacía con los valors de la función
    
    w_list.append(w)                        # Añadir valor inicial de w
    func_values_list.append(function(*w))   # Añadir valor inicial de w evaluado en function

    # Se realiza el cálculo de la gradiente descendente mientras no se superen
    # el número máximo de iteraciones.
    while iter < iterations:
        iter += 1
        w = w - eta * gradient(*w)              # Actualización de w con los nuevos valores (wj = wj - eta * gradient(wj))
        
        w_list.append(w)                        # Añadir nuevo w
        func_values_list.append(function(*w))   # Añadir nueva evaluación de w en function
        
        # Si se ha especificado un umbral en el que parar y se ha pasado
        # se sale del bucle
        if threshold and function(*w) < threshold:
            break

    return w, iter, np.array(w_list), np.array(func_values_list)

#################################################################
#################################################################
#################################################################
#################################################################
# EJERCICIO SOBRE REGRESIÓN LINEAL

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return 

# Gradiente Descendente Estocastico
def sgd(?):
    #
    return w

# Pseudoinversa	
def pseudoinverse(?):
    #
    return w

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))


###############################################################################
###############################################################################
###############################################################################
###############################################################################

print('EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS')
print('Ejercicio 1')

# Se fijan los parámetros que se van a usar en el cómputo de la gradiente descendente
# (w inicial, num. iteraciones, valor mínimo)
initial_w = np.array([1.0, 1.0])
max_iter = 10000000000
error = 1e-14

w, it, w_array, func_val = descent_gradient(initial_w, E, gradient_E, threshold=error, iterations=max_iter)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
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

print('Ejercicio 3 a')

# Se fijan los parámetros que se van a usar en el cómputo de la gradiente descendente
# en los dos casos
# w inicial, eta del segundo caso a estudiar y número máximo de iteraciones
initial_w = np.array([0.1, 0.1])
eta = 0.1
max_iter = 50

# Primer caso: w_inicial = (0.1, 0.1), eta 0.01, iteraciones = 50
w_1, it_1, w_array_1, func_val_1 = descent_gradient(initial_w, f, gradient_f, iterations=max_iter)

# Segundo caso: w_inicial = (0.1, 0.1), eta 0.1, iteraciones = 50 
w_2, it_2, w_array_2, func_val_2 = descent_gradient(initial_w, f, gradient_f, eta=eta, iterations=max_iter)

# Mostrar por pantalla los resultados obtenidos
print('eta = 0.01') 
print('Coordenadas obtenidas = ({}, {})'.format(w_1[0], w_1[1]))
print('Valor de la función = {}\n'.format(func_val_1[-1]))

print('eta = 0.1') 
print('Coordenadas obtenidas = ({}, {})'.format(w_2[0], w_2[1]))
print('Valor de la función = {}'.format(func_val_2[-1]))

# Dividir el rango [0, 50] en tantos puntos como valores evaluados de la función se tengan
x_axis = np.linspace(0, max_iter, func_val_1.shape[0])

# VISUALIZACIÓN

# Limpiar ventana
plt.clf()

# Dibujar cada función
plt.plot(x_axis, func_val_1, 'r', label='$\eta = 0.01$')
plt.plot(x_axis, func_val_2, 'g', label='$\eta = 0.1$')

# Poner etiquetas y leyenda
plt.xlabel('Iterations')
plt.ylabel('Function value')

plt.title('Comparation between different learning ratios')
plt.legend()

# Mostrar
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('Ejercicio 3 b\n')

# Se fijan los parámetros que se van a usar en el cómputo de la gradiente descendente
# en los dos casos
# w inicial de cada caso y número máximo de iteraciones
initial_w_1 = np.array([0.1, 0.1])
initial_w_2 = np.array([1.0, 1.0])
initial_w_3 = np.array([-0.5, -0.5])
initial_w_4 = np.array([-1.0, -1.0])
max_iter = 50

# Cálculo del gradiente descendente para cada caso
w_1, it_1, w_array_1, func_val_1 = descent_gradient(initial_w_1, f, gradient_f, iterations=max_iter)
w_2, it_2, w_array_2, func_val_2 = descent_gradient(initial_w_2, f, gradient_f, iterations=max_iter)
w_3, it_3, w_array_3, func_val_3 = descent_gradient(initial_w_3, f, gradient_f, iterations=max_iter)
w_4, it_4, w_array_4, func_val_4 = descent_gradient(initial_w_4, f, gradient_f, iterations=max_iter)

# Mostrar por pantalla los resultados obtenidos usando pandas
# Crear una lista con los nombres de las columnas
column_header = ['x_0', 'y_0', 'x_f', 'y_f', 'Valor punto final']
row_header = ['Punto 1', 'Punto 2', 'Punto 3', 'Punto 4']

# Crear un array con los valores de cada fila
rows = np.array([[initial_w_1[0], initial_w_1[1], w_array_1[-1, 0], w_array_1[-1, 1], func_val_1[-1]],
                [initial_w_2[0], initial_w_2[1], w_array_2[-1, 0], w_array_2[-1, 1], func_val_2[-1]],
                [initial_w_3[0], initial_w_3[1], w_array_3[-1, 0], w_array_3[-1, 1], func_val_3[-1]],
                [initial_w_4[0], initial_w_4[1], w_array_4[-1, 0], w_array_4[-1, 1], func_val_4[-1]]])

# Crear un nuevo DataFrame
df = pandas.DataFrame(rows, index=row_header, columns=column_header)

# Mostrarlo por pantalla
print(df)

print('\n\n\n')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

# Etiquetas que se asignarán a los datos
label5 = 1
label1 = -1

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')


#Seguir haciendo el ejercicio...
