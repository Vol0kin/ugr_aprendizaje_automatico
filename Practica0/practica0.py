# -*- coding: utf-8 -*-

# Práctica 0
# Autor: Vladislav Nikolov Vasilev

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Dividir la muestra proporcionalmente según el ratio
def stratify_sample(in_vec, labels, ratio=0.8):
    sample = np.c_[in_vec, labels]               # Juntar por columnas vec. características y etiquetas en una matriz 
    group_set = np.unique(y)                     # Grupos únicos que existen
    np.random.shuffle(sample)                    # Mezclar la muestra para distribuirla aleatoriamente
    
    # Listas donde se guardarán las selecciones
    train_list = []
    test_list = []
    
    # De la muestra mezclada, escoger los n primeros elementos de cada grupo
    # y juntarlos en la lista de entrenamiento, el resto en la de test
    # Cada grupo de elementos  escogidos es una lista, por tanto
    # se tienen que combinar luego
    # n = num_elementos_grupo * ratio
    for group in group_set:
        elem_group = sample[sample[:, -1] == group]
        n_elem = elem_group.shape[0]
        n_selected_elem = int(n_elem * ratio)
        
        train_list.append(elem_group[:n_selected_elem, :])
        test_list.append(elem_group[n_selected_elem:, :])
    
    # Juntar las sub-listas en una única matriz
    training = np.concatenate(train_list)
    test = np.concatenate(test_list)
    
    # Volver a mezclar las muestras para distribuirlas aleatoriamente
    np.random.shuffle(training)
    np.random.shuffle(test)
    
    return training, test
        

# #############################################################################
# Parte 1

# Cargar base de datos iris
iris = datasets.load_iris()

# Cargar características (x) y clase (y)
x = iris.data
y = iris.target

# Obtener dos últimas columnas de x
x_last_cols = x[:, -2:]

# Visualización
color_dict = {0: 'red', 1: 'green', 2: 'blue'}      # Diccionario de colores K = numero_grupo, V = color
group_set = np.unique(y)

# Recorrer los grupos únicos y pintarlos de un color cada uno
for group in group_set:
    index = np.where(y == group)
    plt.scatter(x_last_cols[index, 0], x_last_cols[index, 1], c = color_dict[group], label = 'Group {}'.format(group))

plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.title('Iris dataset classification')
plt.legend()

plt.show()

# #############################################################################
# Parte 2

training, test = stratify_sample(x_last_cols, y)

print("Number of Group 0 elements in training sample: {}".format(np.count_nonzero(training[:, -1] == 0)))
print("Number of Group 1 elements in training sample: {}".format(np.count_nonzero(training[:, -1] == 1)))
print("Number of Group 2 elements in training sample: {}".format(np.count_nonzero(training[:, -1] == 2)))
print("Size of training sample: {}".format(training.shape))
print("Training sample:\n{}".format(training))

print("Number of Group 0 elements in test sample: {}".format(np.count_nonzero(test[:, -1] == 0)))
print("Number of Group 1 elements in test sample: {}".format(np.count_nonzero(test[:, -1] == 1)))
print("Number of Group 2 elements in test sample: {}".format(np.count_nonzero(test[:, -1] == 2)))
print("Size of test sample: {}".format(test.shape))
print("Test sample:\n{}".format(test))


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