# -*- coding: utf-8 -*-
"""
PRÁCTICA 3
Autor: Vladislav Nikolov Vasilev
"""

import numpy as np
import pandas as pd

# Módulo avanzado para dibujar gráficas
import seaborn as sns
import matplotlib.pyplot as plt

# Función para dividir los datos en train y test
from sklearn.model_selection import train_test_split

# Importar funcionalidades para crear pipelines
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# Importar escalado de datos que se va a aplicar
from sklearn.preprocessing import StandardScaler

# Reduccion PCA
from sklearn.decomposition import PCA

# Importar modelos que se van a utilizar
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Metricas
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Importar funcionalidad para probar pipelines
from sklearn.model_selection import cross_val_score

# k-fold con proporcion de clases
from sklearn.model_selection import StratifiedKFold

# Ignorar warnings de conversion
import warnings
warnings.filterwarnings('ignore')

# Establecer la semilla que vamos a utilizar
np.random.seed(1)

###############################################################################
# Funciones utiles para los problemas
###############################################################################

##################################################################
# Lectura y division de los datos

def read_data_values(in_file, separator=None):
    """
    Funcion para leer los datos de un archivo
    
    :param in_file Archivo de entrada
    :param separator Separador que se utiliza en el archivo (por defecto
                     None)
    
    :return Devuelve los datos leidos del archivo en un DataFrame
    """
    
    # Cargar los datos en un DataFrame
    # Se indica que la primera columna no es el header
    if separator == None:
        df = pd.read_csv(in_file, header=None)
    else:
        df = pd.read_csv(in_file, sep=separator, header=None)
    
    return df


def divide_data_labels(input_data):
    """
    Funcion que divide una muestra en los datos y las etiquetas
    
    :param input_data Conjunto de valores que se quieren separar
                      juntados en un DataFrame
    
    :return Devuelve los datos y las etiquetas
    """
    
    #Obtener los valores
    values = input_data.values
    
    # Obtener datos y etiquetas
    X = values[:, :-1]
    y = values[:, -1]
    
    return X, y


def create_mlr_pipeline(c_list):
    """
    Funcion para crear una lista de pipelines con el
    modelo de Regresion Logistica Multinomial dados
    unos valores de C, aplicando antes un escalado
    y PCA con 95% de varianza explicada
    
    :param c_list: Lista de valores C. Un valor por
                   cada RLM del pipeline
    
    :return Devuelve una lista con los pipelines
    """
    
    # Crear lista de pipelines
    pipelines = []
    
    # Insertar nuevo pipeline
    for c in c_list:
        pipelines.append(
            make_pipeline(StandardScaler(), PCA(n_components=0.95),
                          LogisticRegression(multi_class='multinomial',
                                             solver='newton-cg',
                                             C=c, random_state=1)))
    
    return pipelines


def create_svmc_pipeline(c_list):
    """
    Funcion para crear una lista de pipelines con el
    modelo de SVM dados unos valores de C, aplicando 
    antes un escalado y PCA con 95% de varianza explicada
    
    :param c_list: Lista de valores C. Un valor por
                   cada SVM del pipeline
    
    :return Devuelve una lista con los pipelines
    """
    
    # Crear lista de pipelines
    pipelines = []
    
    # Insertar nuevo pipeline
    for c in c_list:
        pipelines.append(
            make_pipeline(StandardScaler(), PCA(n_components=0.95),
                          LinearSVC(C=c, random_state=1, loss='hinge')))
    
    return pipelines


##################################################################
# Creacion de pipelines

def create_ridge_pipelines(alpha):
    """
    Funcion para crear una lista de pipelines para Ridge
    Regression con un valor de alfa para cada uno
    
    :param alpha: Valor alfa asociado a la ponderacion de
                  la regularizacion
    
    :return Devuelve una lista de pipelines, cada uno con
            su propio valor de alfa
    """
    
    # Creamos una nueva lista de pipelines
    pipelines = []
    
    # Insertamos un nuevo pipeline que utiliza StandardScaler
    # y Ridge Regression con un valor de alpha dado
    for a in alpha:
        pipelines.append(make_pipeline(StandardScaler(), Ridge(alpha=a)))
    
    return pipelines


def create_svmr_pipelines(epsilon, c_list):
    """
    Funcion para crear una lista de pipelines para SVM
    Regression con una pareja de epsilon y C para cada
    uno.
    Cada elemento contiene un StandardScaler y un LinearSVR,
    que utilizara la norma l2 para el error, 
    
    :param epsilon: Lista de valores epsilon asociados
                    a la amplitud del margen
    :param c_list: Lista de valores C que ponderan el error
    
    :return Devuelve una lista de pipelines, cada uno con
            sus propios epsilon y c
    """
    
    # Crear una nueva lista de pipelines
    pipelines = []
    
    # Insertar un nuevo pipeline que utiliza StandardScaler
    # y LinearSVR con los valors de epsilon y C dados en las listas
    for e in epsilon:
        for c in c_list:
            pipelines.append(make_pipeline(StandardScaler(),
                LinearSVR(epsilon=e, C=c,
                          loss='squared_epsilon_insensitive', random_state=1)))
    
    return pipelines


##################################################################
# Evaluacion de modelos

def evaluate_models(models, X, y, cv=10, metric='neg_mean_absolute_error'):
    """
    Funcion para evaluar un conjunto de modelos con un conjunto
    de caracteristicas y etiquetas.
    
    
    :param models: Lista que contiene pipelines o modelos que
                   se quieren evaluar
    :param X: Conjunto de datos con los que evaluar
    :param y: Conjunto de etiquetas
    :param cv: Numero de k-folds que realizar (por defecto 10)
    :param metric: Metrica de evaluacion (por defecto es la norma-l1,
                   neg_mean_absolute_error)
    
    :return Devuelve una lista con los valores medios y una lista
            con las desviaciones tipicas
    """
    
    # Crear listas de medias y desviacions
    means = []
    deviations = []
    
    # Para cada modelo, obtener los resultados de
    # evaluar el modelo con todas las particiones
    # Guardar los resultados en las listas correspondientes
    for model in models:
        results = cross_val_score(model, X, y, scoring=metric, cv=cv)
        
        # Guardar valor medio de los errores
        # Se guarda el valor absoluto porque son valores negativos
        means.append(abs(results.mean()))
        
        # Guardar desviaciones
        deviations.append(np.std(results))
    
    return means, deviations


##################################################################
# Visualizar resultados de eval. y graficas de clases

def print_evaluation_results(models, means, deviations, metric):
    """
    Funcion para mostrar por pantalla los resultados
    de la evaluacion
    
    :param models: Nombres de los modelos evaluados
    :param means: Lista con los valores medios de las
                  evaluaciones
    :param deviations: Lista con los valores de las
                       desv. tipicas de las evaluaciones
    :param metric: Metrica a evaluar
    """
    
    print('Evaluation results for each model')
    
    # Crear un DataFrame con el formato de salida
    out_df = pd.DataFrame(index=models, columns=[metric, 'Standard Deviation'],
                         data=[[mean, dev] for mean, dev in zip(means, deviations)])
    print(out_df)


def visualize_distribution(classes, values, columns):
    """
    Funcion para visualizar en un grafico de barras las
    clases con el numero de valores de cada una
    
    :param classes: Clases que se quieren representar
    :param values: Numero de elementos por clase
    :param columns: Nombres que dar a las clases y a los
                    numeros de elementos al pintar el grafico
    """
    
    # Crear DataFrame para mostrar el grafico
    df_plot = pd.DataFrame(columns=columns,
            data=[[v, l] for v, l in zip(classes, values)])
    
    # Crear grafico, poner titulo y mostrar
    sns.barplot(x=columns[0], y=columns[1], data=df_plot)
    plt.title('Number of samples per digit class')
    plt.show()


# Clase del modelo de regresion medio
class MeanRegression:
    def __init__(self):
        self.predict_value = 0
        
    def fit(self, y):
        # Calcular valor medio
        self.predict_value = y.mean()
    
    def predict(self, y):
        # Crear array de predicted con el mismo tamaño que
        # las etiquetas que se quieren predecir
        predicted = np.full_like(y, self.predict_value)
        
        return predicted

###############################################################################
# Problema de Regresión

#########################################################
# Lectura y previsualizacion inicial de los datos

print('Reading data...')
df = read_data_values('datos/airfoil_self_noise.dat', separator='\t')
print('Data read!')

# Asignamos nombres a las columnas (según los atributos)
column_names = ['Frequency', 'Angle of attack', 'Chord length',
                'Free-stream velocity', 'SSD thickness', 'Sound Pressure']
df.columns = column_names

# Mostrar primeros valores de los datos
print('Sample examples for regression problem')
print(df.head())

input('---Press any key to continue---\n\n')

#########################################################
# Dividir en train y test y mostrar estadisticos

# Obtener valores X, Y
X, y = divide_data_labels(df)

print('Spliting data between training and test...')

# Dividir los datos en training y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1, shuffle=True)

print('Data split!')

# Crear DataFrame con los datos de training
train_df = pd.DataFrame(columns=column_names, data=np.c_[X_train, y_train])

# Crear DataFrame con los datos de test
test_df = pd.DataFrame(columns=column_names, data=np.c_[X_test, y_test])

# Obtener tabla de correlación de Pearson
corr = train_df.corr()
print('Pearson correlation values between variables')
print(corr)

input('---Press any key to continue---\n\n')

#########################################################
# Mostrar graficas de crecimiento de pares de variables

# Crear pares de plots para cada 2 atributos
# También se incluyen las etiquetas
sns.set_style("whitegrid")
sns.pairplot(train_df)

print('Graphics showing evolution between each pair of variables')

# Mostrar el plot
plt.show()

input('---Press any key to continue---\n\n')

#########################################################
# Evaluar modelos

# Crear listas con hiperparametros para Ridge
ridge_alpha = [0.2, 0.5, 1.0]

# Crear listas con hiperparametros para SVR
svmr_epsilon = [0.2, 1.5]
svmr_c = [0.1, 1.0]

# Crear los nombres de los modelos
model_names = ['Linear Regression', 'Ridge alpha=0.2',
               'Ridge alpha=0.5', 'Ridge alpha=1.0',
               'SVMR e=0.2, c=0.1', 'SVMR e=0.2, c=1.0',
               'SVMR e=1.0, c=0.1', 'SVMR e=1.0, c=1.0']

# Crear pipelines para cada modelo
reg_pipe = [make_pipeline(StandardScaler(), LinearRegression())]
ridge_pipe = create_ridge_pipelines(ridge_alpha)
svmr_pipe = create_svmr_pipelines(svmr_epsilon, svmr_c)

# Juntar todos los pipelines en una lista con los modelos
models = reg_pipe + ridge_pipe + svmr_pipe

print('Evaluating models...')
# Obtener valores medios y desviaciones de las evaluaciones
means, deviations = evaluate_models(models, X_train, y_train)

# Mostrar valores por pantalla
print_evaluation_results(model_names, means, deviations, 'Mean MAE')

input('---Press any key to continue---\n\n')

#########################################################
# Ajustar modelo

print('Linear Regression fit process')

# Creamos el objeto que vamos a usar para escalar
scaler = StandardScaler()

# Ajustar el scaler y transformamos los datos
scaler.fit(X_train)

print('Scaling training data...')
X_train = scaler.transform(X_train)
print('Data scaled!')

# Creamos el modelo que vamos a ajusat
reg = LinearRegression()

# Ajustar el modelo
print('Fitting Linear Regression model...')
reg.fit(X_train, y_train)
print('Linear Regression model fitted!')

input('---Press any key to continue---\n\n')

#########################################################
# Comparar modelo con modelo medio

print('Testing Linear Regression with test sample')

# Crear modelo de regresion media y ajustarlo
mean_reg = MeanRegression()
mean_reg.fit(y_train)

# Escalar datos de test
X_test_transf = scaler.transform(X_test)

# Predecir valores con cada modelo
predict_reg = reg.predict(X_test_transf)
predict_mean_reg = mean_reg.predict(y_test)

reg_test_err = mean_absolute_error(y_test, predict_reg)
mean_reg_test_err = mean_absolute_error(y_test, predict_mean_reg)

print('Comparing Linear Regression to mean linear model')
print('Linear Regression E_test = ', reg_test_err)
print('Mean Linear Regression E_test = ', mean_reg_test_err)
print('Error proportion MER / LR: ', mean_reg_test_err / reg_test_err)

input('---Press any key to continue---\n\n')

#########################################################
# Mostrar grafica de valores predichos vs reales

# Visualizar grafica comparativa de valores
# Se comparan los valores reales y los predichos
sns.scatterplot(x=y_test, y=predict_reg)
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.title('Difference between real and predicted values')
plt.show()

input('---Press any key to continue---\n\n')

###############################################################################
# Problema de Clasificación

#########################################################
# Lectura y previsualizacion inicial de los datos

# Leer los datos de training y test
print('Reading data...')
train_df = read_data_values('datos/optdigits.tra')
test_df = read_data_values('datos/optdigits.tes')
print('Data read!')

# Separar las caracteristicas y las etiquetas
X_train, y_train = divide_data_labels(train_df)
X_test, y_test = divide_data_labels(test_df)

# Mostrar primeros valores del conjunto de training
# Mostrar primeros valores de los datos
print('Training sample examples for classification problem')
print(train_df.head())

input('---Press any key to continue---\n\n')

#########################################################
# Mostrar información de los datos

print('Showing data information...')

# Determinar si faltan valores para los dos conjunts
print('Missing values in train? ', train_df.isnull().values.any())
print('Missing values in test? ', test_df.isnull().values.any())

# Determinar numero de muestras por conjunto
print('Training sample size: ', train_df.shape[0])
print('Test sample size: ', test_df.shape[0])

input('---Press any key to continue---\n\n')

#########################################################
# Mostrar información de distribucion de clases

print("Plotting information about classes' distribution...")

# Obtener las clases
classes = np.unique(train_df.values[:, -1])

# Obtener el numero de muestras por clase
num_elements = []

for i in classes:
    num_elements.append(np.where(y_train == i)[0].shape[0])

# Establecer nombres de los ejes [x, y]
axis = ['digits', 'num_samples']

# Visualizar grafico
visualize_distribution(classes, np.asarray(num_elements), axis)

input('---Press any key to continue---\n\n')

#########################################################
# Evaluar modelos

# Crear lista de valores de C para los dos modelos
c_list = [0.01, 0.1, 1.0, 10.0]

# Crear los nombres de los modelos
model_names = ['MLR c=0.01', 'MLR c=0.1',
               'MLR c=1.0', 'MLR c=10.0',
               'SVMC c=0.01', 'SVMC c=0.1',
               'SVMC c=1.0', 'SVMC c=10.0']

# Crear 10-fold que conserva la proporcion
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Crear pipelines para cada modelo
mlr_pipe = create_mlr_pipeline(c_list)
svmc_pipe = create_svmc_pipeline(c_list)

# Juntar todos los pipelines en una lista con los modelos
models = mlr_pipe + svmc_pipe

# Obtener valores medios y desviaciones de las evaluaciones
print('Evaluating models...')

means, deviations = evaluate_models(models, X_train, y_train,
                                    cv=cv, metric='accuracy')

# Mostrar valores por pantalla
print_evaluation_results(model_names, means, deviations, 'Mean Accuracy')

input('---Press any key to continue---\n\n')

#########################################################
# Ajuste del modelo

print('Multinomial Logistic Regression fit process')

# Creamos un nuevo scaler
scaler = StandardScaler()

# Entrenamos el scaler y transformamos los datos
scaler.fit(X_train)

print('Scaling training data...')
X_train = scaler.transform(X_train)
print('Training data scaled!')

# Creamos un nuevo objeto de PCA
pca = PCA(n_components=0.95)

print('Dimensions before PCA: ', X_train.shape[1])

# Ajustamos PCA y transformamos
pca.fit(X_train)
X_train = pca.transform(X_train)

print('Dimensions after PCA: ', X_train.shape[1])

# Crear modelo
mlr = LogisticRegression(multi_class='multinomial',
                    solver='newton-cg', random_state=1)

# Ajustar el modelo
print('Fitting Multinomial Regression Model...')
mlr.fit(X_train, y_train)
print('Multinomial Regression Model fitted!')

input('---Press any key to continue---\n\n')

#########################################################
# Predecir nuevas etiquetas y mostrar estadisticos

print('Testing Multinomial Linear Regression with test sample...')

# Escalar y reducir dimensiones en test
X_test_scal = scaler.transform(X_test)
X_test_red = pca.transform(X_test_scal)

# Predecir las etiquetas
y_pred = mlr.predict(X_test_red)

# Obtener la precision
accuracy = accuracy_score(y_test, y_pred)

# Mostrar valor de la precision
print('Accuracy score: ', accuracy)

# Mostar matriz de confusion
print('Confusion matrix')
print(confusion_matrix(y_test, y_pred))

input('---Press any key to continue---\n\n')