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
    """
    Función para insertar ruido en una muestra de forma proporcional en cada
    clase.
    
    :param y: Etiquetas sobre las que insertar ruido
    :param ratio: Ratio de elementos de cada clase que modificar
    """
    # Obtener número de elementos sobre los que aplicar ruido
    # (redondear)
    noisy_pos = round(np.where(y == 1)[0].shape[0] * ratio)
    noisy_neg = round(np.where(y == -1)[0].shape[0] * ratio)
    
    # Obtener las posiciones de forma aleatoria
    pos_index = np.random.choice(np.where(y == 1)[0], noisy_pos, replace=False)
    neg_index = np.random.choice(np.where(y == -1)[0], noisy_neg, replace=False)
    
    # Cambiar los valores
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
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),
                         np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
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
    
    # Predecir las etiquetas
    predicted_y = func(x)
    
    # Hacer que los valores predichos estén en el rango (-1, 1)
    predicted_y = np.clip(predicted_y, -1, 1)
    
    return np.mean(predicted_y == y), np.mean(predicted_y != y) 

###############################################################################
# Ejercicio 2.1

# Función para ajustar un clasificador basado en el algoritmo PLA
def adjust_PLA(data, label, max_iter, initial_values):
    """
    Implementación del PLA para ajustar una serie de pesos
    para un perceptrón
    
    :param data: Conjunto de datos con los que entrenar el perceptrón
    :param label: Conjunto de etiquetas, una por cada grupo de características
    :param max_iter: Número máximo de iteraciones (épocas) que hace el PLA
                     (limitar el número de iteraciones en caso de que no converja)
    :param initial_values: Valores iniciales de w
    
    :return Devuelve los pesos obtenidos (w) junto con el número de épocas
            que ha tardado en converger (epoch)
    """
    
    # Copiar valores iniciales de w
    w = np.copy(initial_values)
    
    # Inicializar la convergencia a falso
    convergence = False
    
    # Inicializar el número de épocas realizadas a 0
    epoch = 0
    
    # Mientras no se haya convergido, ajustar el Perceptron
    while not convergence:
        # Incrementar el número de épocas y decir que se ha convergido
        convergence = True
        epoch += 1
        
        # Recorrer cada elemento de los datos con su correspondiente etiqueta
        # Si se ve que el valor predicho no se corresponde con el real
        # se dice que no se ha convergido en esta época
        for x, y in zip(data, label):
            # Calculor valor predicho (función signo)
            predicted_y = signo(w.dot(x.reshape(-1, 1)))
            
            # Comprobar si el valor predicho es igual al real
            if predicted_y != y:
                w += y * x
                convergence = False  
        
        # Si se ha alcanzado el máximo de épocas, terminar
        if epoch == max_iter:
            break            
    
    return w, epoch

###############################################################################
# Ejercicio 2.2

def error_func(data, labels, w):
    """
    Función para calcular el error en un conjunto de datos.
    
    :param data: Conjunto de datos
    :param labels: Conjunto de etiquetas
    :param w: Vector de pesos
    
    :return Devuelve el error
    """
    
    # Obtener número de elementos e inicializar error inicial
    N = data.shape[0]
    error = 0.0
    
    # Recorrer cada elemento e ir incrementando el error con
    # la función del ERM
    for x, y in zip(data, labels):
        error += np.log(1 + np.exp(-y * w.dot(x.reshape(-1, 1))))
        
    return error[0] / N

# Función gradiente del sigmoide
def gradient_sigmoid(x, y, w):
    return -(y * x)/(1 + np.exp(y * w.dot(x.reshape(-1,))))

# Función para ajustar un clasificador basado en regresión lineal mediante
# el algoritmo SGD
def sgdRL(data, labels, initial_w, threshold=0.01, lr=0.01):
    """
    Función que calcula unos pesos para la Regresión Logística mediante
    el Gradiente Descendente Estocástico
    
    :param data: Conjunto de datos
    :param labels: Conjunto de etiquetas
    :param initial_w: Valores iniciales de w
    :param threshold: Límite de las diferencias entre los w de dos épocas con
                      el que parar
    :param lr: Ratio de aprendizaje
    
    :return Devuelve un vector de pesos (w)
    """
    # Copiar el w inicial, los datos y las etiquetas
    w = np.copy(initial_w)
    x_data = np.copy(data)
    y_data = np.copy(labels)
    
    # Obtener número de elementos
    N = x_data.shape[0]
    
    # Establecer una diferencia inicial entre w_(t-1) y w_t
    delta = np.inf
    
    # Mientras la diferencia sea superior al umbral,
    # generar una nueva época e iterar sobre los datos
    while delta > threshold:
        # Crear una nueva permutación y aplicarla a los datos
        # para generar una nueva época
        indexes = np.random.permutation(N)
        x_data = x_data[indexes, :]
        y_data = y_data[indexes]
        
        # Guardar w_(t-1)
        prev_w = np.copy(w)
        
        # Actualizar w con la nueva época
        for x, y in zip(x_data, y_data):
            w = w - lr * gradient_sigmoid(x, y, w)
        
        # Comprobar el nuevo delta
        delta = np.linalg.norm(prev_w - w)
    
    return w.reshape(-1,)


###############################################################################
###############################################################################
#                               EJERCICIOS
###############################################################################
###############################################################################


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
print('Ejercicio 1.1\n')

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
print('Ejercicio 1.2a\n')

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

print('Ejercicio 1.2b\n')

# Array con 10% de indices aleatorios para introducir ruido
y_noisy = np.copy(y)

insert_noise(y_noisy)

# Pintar puntos
plt.clf()

for l in labels:
    index = np.where(y_noisy == l)
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
accuracy, error = error_rate(x, y_noisy, a, b)

print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))

#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera
# de clasificación de los puntos de la muestra en lugar de una recta

print('Ejercicio 1.3\n')


# Mostrar para f1 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y_noisy, f1)
plot_datos_cuad(x, y_noisy, f1)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar para f2 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y_noisy, f2)
plot_datos_cuad(x, y_noisy, f2)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar para f3 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y_noisy, f3)
plot_datos_cuad(x, y_noisy, f3)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar para f4 el ratio de puntos bien/mal clasificados junto con la
# representación gráfica de la función y los puntos
accuracy, error = error_rate_func(x, y_noisy, f4)
plot_datos_cuad(x, y_noisy, f4)
print('Ratio de aciertos: {}'.format(accuracy))
print('Ratio de error: {}'.format(error))
input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON
print('Ejercicio 2.1a\n')

# Crear el conjunto de datos añadiendo una columna con unos a los x
data = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]

# Crear array de zeros
zeros = np.array([0.0, 0.0, 0.0])

# Inicializar la lista que contendrá las iteraciones
iterations = []

print('Algoritmo PLA con w_0 = [0.0, 0.0, 0.0]\n')

# Lanzar el algoritmo PLA con w = [0, 0, 0] y guardar las iteraciones
for i in range(0,10):
    w, iter = adjust_PLA(data, y, 10000, zeros)
    iterations.append(iter)
    print('Valor w: {} \tNum. iteraciones: {}'.format(w, iter))    

# Mostrar media de iteraiones
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Inicializar la lista que contendrá las iteraciones
iterations = []

print('Algoritmo PLA con w_0 aleatorio\n')

# Lanzar el algoritmo PLA con w aleatorio y guardar las iteraciones
for i in range(0,10):
    # Generar w_0
    initial_w = simula_unif(3, 1, [0.0, 1.0]).reshape(-1,)
    
    w, iter = adjust_PLA(data, y, 10000, initial_w)
    iterations.append(iter)
    print('w_0 = {}'.format(initial_w))
    print('Valor w: {} \tNum. iteraciones: {}\n'.format(w, iter))    

# Mostrar media de iteraiones
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
print('Ejercicio 2.1b\n')

# Inicializar la lista que contendrá las iteraciones
iterations = []

print('Algoritmo PLA con w_0 = [0.0, 0.0, 0.0] y datos con ruido\n')

# Lanzar el algoritmo PLA con w = [0, 0, 0] y guardar las iteraciones
# Ahora con los datos del ejercicio 1.2.b
for i in range(0,10):
    w, iter = adjust_PLA(data, y_noisy, 10000, zeros)
    iterations.append(iter)
    print('Valor w: {} \tNum. iteraciones: {}'.format(w, iter))    

# Mostrar media de iteraiones
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Inicializar la lista que contendrá las iteraciones
iterations = []

print('Algoritmo PLA con w_0 aleatorio y datos con ruido\n')

# Lanzar el algoritmo PLA con w aleatorio y guardar las iteraciones
# Ahora con los datos del ejercicio 1.2.b
for i in range(0,10):
    # Generar w_0
    initial_w = simula_unif(3, 1, [0.0, 1.0]).reshape(-1,)
    
    w, iter = adjust_PLA(data, y_noisy, 10000, initial_w)
    iterations.append(iter)
    print('w_0 = {}'.format(initial_w))
    print('Valor w: {} \tNum. iteraciones: {}\n'.format(w, iter))    

# Mostrar media de iteraiones
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
print('Ejercicio 2.2\n')

# Fijamos la semilla
# Se cambia la semilla porque con la primera se obtiene una recta muy mala
np.random.seed(2)

# Simular 100 puntos 2D de forma uniforme en el rango [0, 2]
x_train = simula_unif(100, 2, [0.0, 2.0])

# Simular una recta en el rango [0, 2] y calcular sus coeficientes
a, b = simula_recta([0.0, 2.0])

# Inicializar las etiquetas a una nueva lista
y_train = []

# Recorrer los valores de x_train y generar los valores de las etiquetas
# utilizando la recta de clasificación
for value in x_train:
    y_train.append(f(value[0], value[1], a, b))

# Convertir la lista de etiquetas a array
y_train = np.array(y_train)

# Obtener los valores únicos de las etiquetas
labels = np.unique(y_train)

# Visualización de los datos generados para el problema

# Limpiar la ventana
plt.clf()

# Pintar los puntos según su clase
for l in labels:
    index = np.where(y_train == l)
    plt.scatter(x_train[index, 0], x_train[index, 1], c=color_dict[l], label='Group {}'.format(l))

# Pintar recta
plt.plot([0.0, 2.0], [b, 2.0 * a + b], 'k-')

# Añadir leyendas, títuloy nombres a los ejes
plt.title(r'Uniform values in $[0, 2] \times [0, 2]$ square with classification line')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

# Mostrar la gráfica
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Crear los datos añadiendo una columna de unos al principio
x_train = np.c_[np.ones((x_train.shape[0], 1), dtype=np.float64), x_train]

# Calcular el w con error = 0.01 y eta = 0.01
w = sgdRL(x_train, y_train, zeros)

# Visualización de los resultados obtenidos

# Limpiar la ventana
plt.clf()

# Pintar los puntos según su clase
for l in labels:
    index = np.where(y_train == l)
    plt.scatter(x_train[index, 1], x_train[index, 2], c=color_dict[l], label='Group {}'.format(l))

# Pintar recta
plt.plot([0.0, 2.0], [-w[0] / w[2], (-w[0] - 2.0 * w[1]) / w[2]], 'k-')

# Añadir leyendas, títuloy nombres a los ejes
plt.title(r'Uniform values in $[0, 2] \times [0, 2]$ square with logistic regression line')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

# Mostrar la gráfica
plt.show()

# Mostrar información sobre Ein
e_in = error_func(x_train, y_train, w)
print('Ein = {}'.format(e_in))

###############################################################################
input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

# Generar 2500 puntos de test en 2 dimensiones en el rango [0, 2]
n_sample = 2500
x_test = simula_unif(n_sample, 2, [0.0, 2.0])

# Añadir columna de unos a x_test
x_test = np.c_[np.ones((x_test.shape[0], 1), dtype=np.float64), x_test]

# Inicializar las etiquetas de test a una nueva lista
y_test = []

# Recorrer los valores de x_train y generar los valores de las etiquetas
# utilizando la recta de clasificación
for value in x_test:
    y_test.append(f(value[1], value[2], a, b))

# Visualización de los resultados obtenidos

# Limpiar la ventana
plt.clf()

# Pintar los puntos según su clase
for l in labels:
    index = np.where(y_test == l)
    plt.scatter(x_test[index, 1], x_test[index, 2], c=color_dict[l], label='Group {}'.format(l))

# Pintar recta
plt.plot([0.0, 2.0], [-w[0] / w[2], (-w[0] - 2.0 * w[1]) / w[2]], 'k-')

# Añadir leyendas, títuloy nombres a los ejes
plt.title(r'{} uniform values in $[0, 2] \times [0, 2]$ square with logistic regression line'.format(n_sample))
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

# Mostrar la gráfica
plt.show()

# Mostrar información sobre Eout
e_out = error_func(x_test, y_test, w)
print('Eout = {}'.format(e_out))

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
