# -*- coding: utf-8 -*-

"""
Práctica 1.
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas                                 # Mostrar los datos como tablas

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
    Función para el cálculo del gradiente descendente
    
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
        w = w - eta * gradient(*w)              # Actualización de w con los nuevos valores
        
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
def Err(x, y, w):
    error = np.square(x.dot(w) - y.reshape(-1, 1))        # Calcular el error cuadrático para cada vector de características
    error = error.mean()                                  # Calcular la media de los errors cuadráticos (matriz con una columna)
    
    return error

# Derivada de la función del error
def diff_Err(x,y,w):
    d_error = x.dot(w) - y.reshape(-1, 1)           # Calcular producto vectorial de x*w y restarle y
    d_error =  2 * np.mean(x * d_error, axis=0)     # Realizar la media del producto escalar de x*error y la media en el eje 0
    
    d_error = d_error.reshape(-1, 1)                # Cambiar la forma para que tenga 3 filas y 1 columna
    
    return d_error

# Gradiente Descendente Estocastico
def sgd(X, y, eta, M=64, iterations=200):
    """
    Función para calcular el Gradiente Descendente Estocástico.
    Selecciona minibatches aleatorios de tamaño M de la muestra original
    y ajusta en un número de iteraciones los pesos.
    
    :param X: Muestra de entrenamiento
    :param y: Vector de etiquetas
    :param eta: Ratio de aprendizaje
    :param M: Tamaño de un minibatch (64 por defecto)
    :param iterations: Número máximo de iteraciones
    
    :return w: Pesos ajustados
    """
    
    # Crear un nuevo vector de pesos inicializado a 0, establecer el número de iteraciones
    # inicial y obtener el número de elementos (N)
    w = np.zeros((3, 1), np.float64)
    N = X.shape[0]
    iter = 0
    
    # Mientras el número de iteraciones sea menor al máximo, obtener un minibatch
    # de tamaño M con valores aleatorios de X y ajustar los pesos con estos valores
    while iter < iterations:
        iter += 1
        
        # Escoger valores aleatorios de índices sin repeticiones y obtener los elementos
        index = np.random.choice(N, M, replace=False)
        minibatch_x = X[index]
        minibatch_y = y[index]
        
        # Actualizar w
        w = w - eta * diff_Err(minibatch_x, minibatch_y, w)
    
    return w

# Pseudoinversa	
def pseudoinverse(X, y):
    """
    Función para el cálculo de pesos mediante el algoritmo de la pseudoderivada
    
    :param X: Matriz que contiene las caracterísiticas
    :param y: Matriz que contiene las etiquetas relacionadas a las características
    
    :returns w: Pesos calculados mediante ecuaciones normales
    """
    
    X_transpose = X.transpose()                     # Guardamos la transpuesta de X
    y_transpose = y.reshape(-1, 1)                  # Convertimos y en una matriz columna (1 fila con n columnas)
    
    # Aplicamos el algoritmo para calcular la pseudoinversa
    w = np.linalg.inv(X_transpose.dot(X))
    w = w.dot(X_transpose)
    
    # Hacemos el producto de matrices de la pseudoinversa y la matriz columna y
    w = w.dot(y_transpose)
    
    return w

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Crea una matriz de etiquetas a partir de las entradas
def sign_labels(X):
    return np.sign(np.square(X[:, 0] - 0.2) + np.square(X[:, 1]) - 0.6)

# Inserta ruido en un porcentaje de las etiquetas de la población
def insert_noise(labels, ratio=0.1):
    # Calcular el número de elementos con ruido según el ratio
    n = labels.shape[0]
    noisy_elements = int(n * ratio)
    
    # Obtener una muestra aleatoria entre [0, n) de n * ratio elementos sin repeticiones
    index = np.random.choice(np.arange(n), noisy_elements, replace=False)
    
    # Cambiar el signo del porcentaje de etiquetas
    labels[index] = -labels[index]


#################################################################
#################################################################
#################################################################
#################################################################
# EJERCICIO BONUS

# Derivada segunda de f respecto a x
def ddiff_fx(x, y):
    return 2 - 8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Derivada segunda de f respecto a y
def ddiff_fy(x, y):
    return 4 - 8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Derivada segunda de f respecto a x, y (equivale a derivada segunda resepcto a y, x)
def ddiff_fxy(x, y):
    return 8 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

# Matriz Hessiana
def hessian_f(x, y):
    return np.array([[ddiff_fx(x, y), ddiff_fxy(x, y)], [ddiff_fxy(x, y), ddiff_fy(x, y)]])

# Método de Newton
def newtons_method(initial_w, iterations=50):
    """
    Función para el cálculo de pesos mediante el Método de Newton
    Es una modificación del algoritmo de Gradiente Descendente usando
    la invertida de la matriz Hessiana como ratio de aprendizaje
    
    :param initial_w: Valor de w inicial
    :param iterations: Número máximo de iteraciones (por defecto 50) 
    
    :return Devuelve los pesos ajustados (w), el número de iteraciones
            para llegar a esos pesos (iter), un array con los valores
            de w (w_list) y un array con los valores de la función
            (f_list)
    """
    
    w = np.copy(initial_w)        # Copiar los w iniciales para no modificarlos
    iter = 0                      # Iniciar las iteraciones a 0
    w_list = []                   # Se inicializa una lista vacía con los valors de w
    f_list = []                   # Se inicializa una lista vacía con los valors de la función
    
    w_list.append(w)              # Añadir valor inicial de w
    f_list.append(f(*w))          # Añadir valor inicial de w evaluado en function
    
    # Mientras el número de iteraciones no supere el máximo, calcular
    # la hessiana, invertirla, calcular el gradiente y ajustar w
    # Añadir además a las listas correspondientes los valores de w y de w evaluado en f
    while iter < iterations:
        iter += 1
        
        # Calcular la Hessiana, invertirla (pseudoinversa) y calcular el gradiente
        hessian = hessian_f(*w)
        hessian = np.linalg.inv(hessian)
        gradient = gradient_f(*w)        
        
        # Calcular theta (producto vectorial del Hessiana invertida y el gradiente)
        theta = hessian.dot(gradient.reshape(-1, 1))
        theta = theta.reshape(-1,)                      # Hacer que theta sea un vector fila
       
        # Actualizar w
        w = w - theta
        
        # Añadir w y su valor a las listas correspondientes
        w_list.append(w)
        f_list.append(f(*w))
        
    
    return w, iter, np.array(w_list), np.array(f_list)

# Función para pintar las comparaciones entre el Método de Newton y el GD
def plot_comparison(x_axis, newton_vals, gd_vals ,point):
    plt.clf()

    # Dibujar rectas
    plt.plot(x_axis, newton_vals, 'r-', label="Newton's Method")
    plt.plot(x_axis, gd_vals, 'g-', label='Gradient Descent')
    
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    
    plt.title("Comparison between Newton's Method and GD for ${}$".format(point))
    plt.legend()
    
    plt.show()
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################

print('EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS')

#############################################################

print('Ejercicio 1\n')

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

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################

print('Ejercicio 3 a\n')

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

plt.title('Comparison between different learning ratios')
plt.legend()

# Mostrar
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################

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

input("\n--- Pulsar tecla para continuar ---\n")
print('\n\n\n')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESIÓN LINEAL\n')
print('Ejercicio 1\n')

# Etiquetas que se asignarán a los datos
label5 = 1
label1 = -1

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

# Diccionario de colores y tupla de etiquetas
color_dict = {label1: 'blue', label5: 'red'}
labels = (label1, label5)
label_values = {label1: 1, label5: 5}

# Cálculo de w mediante SGD
w = sgd(x, y, 0.05)

# VISUALIZACIÓN DEL AJUSTE
plt.clf()           # Limpiar ventana de visualización

# Recorrer el conjunto de puntos y pintarlos de un color según su etiqueta
for l in labels:
    index = np.where(y == l)
    plt.scatter(x[index, 1], x[index, 2], c=color_dict[l], label='{}'.format(label_values[l]))

# Pintar recta de ajuste (y = w0 + w1 * x1 + w2 * x2)
# En cada caso obtener valor de x2 a partir de x1 y las predicciones, igualando a 0
plt.plot([0, 1], [-w[0]/w[2], (-w[0] - w[1])/w[2]], 'k-')

# Poner etiquetas a los ejes, título y leyenda
plt.xlabel('Average Intensity')
plt.ylabel('Symmetry')

plt.title('Linear Regression using $SGD$')
plt.legend()

plt.show()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")


# Cálculo de w mediante la pseudoinversa
w = pseudoinverse(x, y)

# VISUALIZACIÓN DEL AJUSTE
plt.clf()           # Limpiar ventana de visualización

# Recorrer el conjunto de puntos y pintarlos de un color según su etiqueta
for l in labels:
    index = np.where(y == l)
    plt.scatter(x[index, 1], x[index, 2], c=color_dict[l], label='{}'.format(label_values[l]))

# Pintar recta de ajuste (y = w0 + w1 * x1 + w2 * x2)
# En cada caso obtener valor de x2 a partir de x1 y las predicciones, igualando a 0
plt.plot([0, 1], [-w[0]/w[2], (-w[0] - w[1])/w[2]], 'k-')

# Poner etiquetas a los ejes, título y leyenda
plt.xlabel('Average Intensity')
plt.ylabel('Symmetry')

plt.title('Linear Regression using Pseudoinverse')
plt.legend()

plt.show()

print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################

print('Ejercicio 2\n')

#############################################################

print('Apartado a\n')

# Obtener una muestra de tamaño 1000 en el cuadrado [-1, 1] x [-1, 1]
# Preparar parámetros y obtener muestra
N = 1000
dim = 2
size = 1

train_x = simula_unif(N, dim, size)

# VISUALIZACIÓN
plt.clf()

# Crear nube de puntos
plt.scatter(train_x[:, 0], train_x[:, 1])

# Poner título y etiquetas en los ejes
plt.xlabel('$x_1$ axis')
plt.ylabel('$x_2$ axis')

plt.title(r'Random points in $[-1, 1] \times [-1, 1]$ square')

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################

print('Apartado b\n')

# Generar las etiquetas
train_y = sign_labels(train_x)

# Preparar datos de visualizacion: colores y conjunto de etiquetas
color_dict = {1.0: 'red', -1.0: 'green'}
label_set = np.unique(train_y)

# VISUALIZACIÓN DE LOS DATOS ANTES DE INSERTAR RUIDO EN LA MUESTRA
plt.clf()

# Recorrer el conjunto de puntos y pintarlos de un color según su etiqueta
for label in label_set:
    index = np.where(train_y == label)
    plt.scatter(train_x[index, 0], train_x[index, 1], c=color_dict[label], label='Group {}'.format(label))

# Poner etiquetas a los ejes, título y leyenda
plt.xlabel('$x_1$ axis')
plt.ylabel('$x_2$ axis')

plt.title(r'Random points in $[-1, 1] \times [-1, 1]$ square before inserting noise in the sample')
plt.legend()

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Insertamos ruido en la muestra
insert_noise(train_y)

# VISUALIZACIÓN DE LOS DATOS DESPUÉS DE INSERTAR RUIDO EN LA MUESTRA
plt.clf()

# Recorrer el conjunto de puntos y pintarlos de un color según su etiqueta
for label in label_set:
    index = np.where(train_y == label)
    plt.scatter(train_x[index, 0], train_x[index, 1], c=color_dict[label], label='Group {}'.format(label))

# Poner etiquetas a los ejes, título y leyenda
plt.xlabel('$x_1$ axis')
plt.ylabel('$x_2$ axis')

plt.title(r'Random points in $[-1, 1] \times [-1, 1]$ square after inserting noise in the sample')
plt.legend()

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################

print ('Apartado c\n')

# Crear vector columna de 1's 
ones = np.ones((N, 1), dtype=np.float64)

# Obtener x juntando el vector de 1's con train_x
train_x = np.c_[ones, train_x]

w = sgd(train_x, train_y, 0.05)

# VISUALIZACIÓN DE LOS DATOS DESPUÉS DE INSERTAR RUIDO EN LA MUESTRA
plt.clf()

# Recorrer el conjunto de puntos y pintarlos de un color según su etiqueta
for label in label_set:
    index = np.where(train_y == label)
    plt.scatter(train_x[index, 1], train_x[index, 2], c=color_dict[label], label='Group {}'.format(label))

# Pintar recta de ajuste (y = w0 + w1 * x1 + w2 * x2)
# En cada caso obtener valor de x2 a partir de x1 y las predicciones, igualando a 0
plt.plot([-1, 1], [(-w[0] + w[1])/w[2], (-w[0] - w[1])/w[2]], 'k-')

# Poner etiquetas a los ejes, título, leyenda y poner el rango del eje Y en [-1.2, 1.2]
plt.xlabel('$x_1$ axis')
plt.ylabel('$x_2$ axis')

plt.ylim(-1.2, 1.2)

plt.title(r'Linear Regression in $[-1, 1] \times [-1, 1]$ square using $SGD$')
plt.legend()

plt.show()

# Mostrar bondad del ajuste
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(train_x ,train_y, w))

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################

print ('Apartado d\n')

# Crear listas de errores
e_in_l = []
e_out_l = []

# Realizar 1000 experimentos
for _ in range(0, N):
    # Generar datos de entrenamiento
    train_x = simula_unif(N, dim, size)
    train_y = sign_labels(train_x)
    
    train_x = np.c_[ones, train_x]
    insert_noise(train_y)
    
    # Generar datos de prueba
    test_x = simula_unif(N, dim, size)
    test_y = sign_labels(test_x)
    test_x = np.c_[ones, test_x]
    
    w = sgd(train_x, train_y, 0.05)
    
    # Calcular los errores dentro y fuera de la muestra y añadirlos
    e_in_l.append(Err(train_x, train_y, w))
    e_out_l.append(Err(test_x, test_y, w))

# Convertir listas de errores a arrays
e_in = np.array(e_in_l)
e_out = np.array(e_out_l)

# Mostrar por pantalla el resultado de los errores medios
print ('Valor medio de la bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", e_in.mean())
print ("Eout: ", e_out.mean())

input("\n--- Pulsar tecla para continuar ---\n")

print('\n\n\n')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('BONUS\n')
print('Método de Newton\n')

# Definición de puntos iniciales
initial_w_1 = np.array([0.1, 0.1])
initial_w_2 = np.array([1.0, 1.0])
initial_w_3 = np.array([-0.5, -0.5])
initial_w_4 = np.array([-1.0, -1.0])

# Cálculo del gradiente descendente para cada caso
w_1, it_1, w_array_1, func_val_1_n = newtons_method(initial_w_1)
w_2, it_2, w_array_2, func_val_2_n = newtons_method(initial_w_2)
w_3, it_3, w_array_3, func_val_3_n = newtons_method(initial_w_3)
w_4, it_4, w_array_4, func_val_4_n = newtons_method(initial_w_4)

# Mostrar por pantalla los resultados obtenidos usando pandas
# Crear una lista con los nombres de las columnas
column_header = ['x_0', 'y_0', 'x_f', 'y_f', 'Valor punto final']
row_header = ['Punto 1', 'Punto 2', 'Punto 3', 'Punto 4']

# Crear un array con los valores de cada fila
rows = np.array([[initial_w_1[0], initial_w_1[1], w_array_1[-1, 0], w_array_1[-1, 1], func_val_1_n[-1]],
                [initial_w_2[0], initial_w_2[1], w_array_2[-1, 0], w_array_2[-1, 1], func_val_2_n[-1]],
                [initial_w_3[0], initial_w_3[1], w_array_3[-1, 0], w_array_3[-1, 1], func_val_3_n[-1]],
                [initial_w_4[0], initial_w_4[1], w_array_4[-1, 0], w_array_4[-1, 1], func_val_4_n[-1]]])

# Crear un nuevo DataFrame
df = pandas.DataFrame(rows, index=row_header, columns=column_header)

# Mostrarlo por pantalla
print(df)

# Crear un eje X para la visualización
x_axis = np.linspace(0, 50, 51)

# VISUALIZACIÓN DEL MÉTODO DE NEWTON PARA TODOS LOS PUNTOS

plt.clf()

# Dibujar rectas
plt.plot(x_axis, func_val_1_n, 'r-', label='Point 1')
plt.plot(x_axis, func_val_2_n, 'g-', label='Point 2')
plt.plot(x_axis, func_val_3_n, 'b-', label='Point 3')
plt.plot(x_axis, func_val_4_n, 'y-', label='Point 4')

plt.xlabel('Iterations')
plt.ylabel('Function value')

plt.title("Newton's Method to compute Gradient Descent")
plt.legend()

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# VISUALIZACIÓN DE COMPARACIÓN ENTRE EL MÉTODO DE NEWTON Y EL GRADIENTE DESCENDENTE
# Llamada a la función para comparar
plot_comparison(x_axis, func_val_1_n, func_val_1, '(0.1, 0.1)')

input("\n--- Pulsar tecla para continuar ---\n")

plot_comparison(x_axis, func_val_2_n, func_val_2, '(1.0, 1.0)')

input("\n--- Pulsar tecla para continuar ---\n")

plot_comparison(x_axis, func_val_3_n, func_val_3, '(-0.5, -0.5)')

input("\n--- Pulsar tecla para continuar ---\n")

plot_comparison(x_axis, func_val_4_n, func_val_4, '(-1.0, -1.0)')

input("\n--- Pulsar tecla para continuar ---\n")

