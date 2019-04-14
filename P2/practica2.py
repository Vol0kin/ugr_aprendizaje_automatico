# -*- coding: utf-8 -*-
"""
PRÁCTICA 2
Nombre Estudiante: Vladislav Nikolov Vasilev
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)

###############################################################################
###############################################################################
#                               FUNCIONES
###############################################################################
###############################################################################

###############################################################################
# Ejericicio 1.1


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


###############################################################################
# Ejercicio 1.2

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def error_rate(x, y, a, b):
    """
    Funcion para calcular los ratios de acierto y error entre los valores
    reales de las etiquetas y los predichos.
    
    :param x: Array de vectores de caracteristicas
    :param y: Array de etiquetas
    :param a: Pendiente de la recta
    :param b: Valor independiente de la recta
    
    :return Devuelve el ratio de aciertos y el ratio de errores
    """
    
    # Crear lista de y predichas
    predicted_y = []
    
    # Predecir cada valor
    for value in x:
        predicted_y.append(f(value[0], value[1], a, b))
    
    # Convertir a array
    predicted_y = np.array(predicted_y)
    
    return np.mean(y == predicted_y), np.mean(y != predicted_y)

def insert_noise(y, ratio=0.1):
    
    # Obtener número de elementos sobre los que aplicar ruido
    # (redondear)
    noisy_pos = round(np.where(y == 1)[0].shape[0] * ratio)
    noisy_neg = round(np.where(y == -1)[0].shape[0] * ratio)
    
    pos_index = np.random.choice(np.where(y == 1)[0], noisy_pos, replace=False)
    neg_index = np.random.choice(np.where(y == -1)[0], noisy_neg, replace=False)
    
    y[pos_index] = -1
    y[neg_index] = 1

###############################################################################
# Ejercicio 1.3

# Función del primer apartado
def f1(X):
    return (X[:, 0] - 10) ** 2 + (X[:, 1] - 20) ** 2 - 400

# Función del segundo apartado
def f2(X):
    return 0.5 * (X[:, 0] + 10) ** 2 + (X[:, 1] - 20) ** 2 - 400

# Función del tercer apartado
def f3(X):
    return 0.5 * (X[:, 0] - 10) ** 2 - (X[:, 1] + 20) ** 2 - 400

# Función del cuarto apartado
def f4(X):
    return X[:, 1] - 20 * X[:, 0] ** 2  - 5 * X[:, 0]  + 3

# Funcion proporcionada para mostrar las graficas
def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()

def error_rate_func(x, y, func):
    """
    Función para calcular los ratios de acierto y error al predecir un conjunto
    de puntos x mediante una función func, con respecto de los valores reales y
    
    :param x: Puntos que se usarán para predecir
    :param y: Etiquetas reales de los puntos
    :param func: Función con la que se predecirán los puntos
    
    :return Devuelve el ratio de puntos predichos correctamente y el ratio de
            puntos predichos erróneamente
    """
    
    # Predecir los puntos
    predicted_y = func(x)
    
    # Hacer que los valores predichos estén en el rango (-1, 1)
    predicted_y = np.clip(predicted_y, -1, 1)
    
    return np.mean(predicted_y == y), np.mean(predicted_y != y) 

###############################################################################
# Ejercicio 2.1

# Función para ajustar un clasificador basado en el algoritmo PLA
def adjust_PLA(X, y, max_iter, initial_values):
    return None

###############################################################################
###############################################################################
#                               EJERCICIOS
###############################################################################
###############################################################################


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x = simula_unif(50, 2, [-50,50])

# Mostrar nube de puntos y asignar leyendas
plt.scatter(x[:, 0], x[:, 1])

plt.title(r'Uniform values generated in $[-50, 50] \times [-50, 50]$ square')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

x = simula_gaus(50, 2, np.array([5,7]))

# Mostrar nube de puntos y asignar leyendas
plt.scatter(x[:, 0], x[:, 1])

plt.title('Normal values generated using $\mu = 0$ and $\sigma = 5$ on $x_1$ and $\sigma = 7$ on $x_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# Simular 50 puntos uniformemente distribuidos en el cuadrado [-50, 50] x [-50, 50]
x = simula_unif(50, 2, [-50, 50])

# Simular una recta en el rango [-50, 50] y calcular sus coeficientes
a, b = simula_recta([-50, 50])

# Obtener los valores de la etiqueta
# Guardarlos primero en una lista y convertirlos en un array
y = []

for value in x:
    y.append(f(value[0], value[1], a, b))

y = np.array(y)

# Crear diccionario de colores y etiquetas unicas
color_dict = {1: 'red', -1: 'blue'}
labels = np.unique(y)

# Pintar puntos
plt.clf()

for l in labels:
    index = np.where(y == l)
    plt.scatter(x[index, 0], x[index, 1], c=color_dict[l], label='Group {}'.format(l))

# Pintar recta
plt.plot([-50, 50], [-50 * a + b, 50 * a + b], 'k-')

# Poner leyenda, etiquetas a los ejes y título
plt.legend()
plt.xlabel('$x$ axis')
plt.ylabel('$y$ axis')
plt.title(r'Uniform values in $[-50, 50] \times [-50, 50]$ square and classification line')
plt.show()

# Obtener ratios de acierto y error
accuracy, error = error_rate(x, y, a, b)

print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, 
#        junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

insert_noise(y)

# Pintar puntos
plt.clf()

for l in labels:
    index = np.where(y == l)
    plt.scatter(x[index, 0], x[index, 1], c=color_dict[l], label='Group {}'.format(l))

# Pintar recta
plt.plot([-50, 50], [-50 * a + b, 50 * a + b], 'k-')

# Poner leyenda, etiquetas a los ejes y título
plt.legend()
plt.xlabel('$x$ axis')
plt.ylabel('$y$ axis')
plt.title(r'Uniform values in $[-50, 50] \times [-50, 50]$ square with noise')
plt.show()

# Obtener ratios de acierto y error
accuracy, error = error_rate(x, y, a, b)

print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))

#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera
# de clasificación de los puntos de la muestra en lugar de una recta

# Mostrar para f1 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y, f1)
plot_datos_cuad(x, y, f1)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar para f2 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y, f2)
plot_datos_cuad(x, y, f2)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar para f3 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y, f3)
plot_datos_cuad(x, y, f3)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar para f4 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y, f4)
plot_datos_cuad(x, y, f4)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    
    return None  

#CODIGO DEL ESTUDIANTE

# Random initializations
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    print()    

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL():
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
