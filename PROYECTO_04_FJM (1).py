#!/usr/bin/env python
# coding: utf-8

# # Entrega 04

# In[13]:


import pandas
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt


# In[2]:


dataframe= pandas.read_csv("inmuebles.csv", delim_whitespace= False, header = 0, delimiter=';')


# In[3]:


dataset = dataframe.values


# ### Se importó la base de la entrega 01 con las modificaciones realizadas, y para poder realizar el analisis se transforman los datos de tipo float a tipo int de forma manual. Luego se separan las columnas para volver a tener la base de datos original pero con el tipo de datos int, que requiere tensorflow.

# In[4]:


dataframe.head()


# In[5]:


dataframe.dtypes


# ### El modelo escogido es neural network. Este modelo lo que realiza es predicciones basandas en informacion existente.
# ### 
# ### La red neuronal consiste de diversas partes, entre las que se encuentran:
# ### - imput layers: son las capas que toman inputs basadas en informacion previa existente
# ### - Hidden layers: Son las capaz que usan retropropagacion para optimizar los pesos de las variables input, con el objetivo de mejorar el poder predictivo del modelo
# ### - Output layers: Son las predicciones que se obtienen basandas en informacion del input y hiden layers.
# ### Como se muestra mas adelante, la cantidad de epochs es la cantidad de veces que se entreno el modelo en el sentido adelante- atras, en el que se crea un ciclo a traves de todo el set de entrenamiento. Con el objetivo de que las perdidas decrezcan y el nivel del modelo mejore por cada pasada. De esta manera el modelo va a predecir el valor de Y de forma mas acertada por cada pasada. En este modelo particular, la cantidad de pasadas que se realizan son 20. Como se observa a medida que corren los epochs, el error disminuye hasta llegar al epoch 19, que comienza a subir nuevamente. 
# ### La funcion matematica que se utiliza para procesar la informacion en el hidden layer es el metodo de activacion RELU (REctified Linear Unit), que determina si la informacion obtenida en el imput layer va a pasar a la proxima etapa llamada output layer.

# In[6]:


X = dataframe[["rooms", "bedrooms", "bathrooms"]]
Y = dataframe["price"]


# In[7]:


X = np.asarray(X).astype(np.float32)
Y=np.asarray(Y).astype(np.float32)


# In[8]:


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.33, random_state=42)


# In[9]:


model = keras.Sequential([
    keras.layers.Dense(512, input_dim = train_x.shape[1], kernel_initializer = "normal", activation="relu"),
    keras.layers.Dense(512, kernel_initializer = "normal", activation="relu"),
    keras.layers.Dense(1, kernel_initializer = "normal", activation = "linear")
])


# In[10]:


model.compile(loss="mean_absolute_error", optimizer="adam", metrics="mean_absolute_error")


# In[18]:


history = model.fit(train_x, train_y, epochs=20, validation_split = 0.3)


# ### Lo que se obtiene a medida que se realizan las 20 iteraciones (epochs), es la perdida (loss) arrojada por el set de entrenamiento y la perdida (val_loss) aplicada por el set de prueba. A media que se realizan iteraciones esta perdida disminuye, asi como tambien lo hace el MAE.

# In[14]:


print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ### El proceso para crear el modelo neural network, es en primer lugar, instalar las librerias necesarias. 
# ### Dentro de las librerias que utilizamos se encuentra keras, que es una libreria de codigo abierta que se utiliza para redes neuronales, que funciona como interfaz de la libreria tensorflow. Tensorflow por su parte es la mayor plataforma de aprendizaje automatico y la elegida para este modelo. El resto de los recursos son los mismos que en las entregas anteriores, es decir sklearn, numpy, python, matplotlib, etc.
# ### El nuevo modelo lo que hace es utilizar la misma base que la empleada en la entrega 01, de properatia, solo que con las modificaciones finales, en las cuales se eliminan los valores faltantes, se completan, y se dejan las columnas relevantes para el modelo. Luego, se realiza un test train split de los datos, se crean las capas de input, hidden y output, y se realiza el fit del modelo. 
# ### El modelo de red neuronal, lo que va a hacer es intentar minimizar el valor de perdida o "loss", que representa la diferencia entre lo que el modelo predce y la realidad.  Esto se realiza mediante las iteraciones. En este caso, por ser de regresion el valor que va a intentar disminuir es el valor de MAE. 
# ### Como se puede ver en el grafico, la diferencia o la perdida va disminuyendo por lo que el modelo va realizando predicciones mas acertadas a lo que realmente sucede.
# ### Si bien la cantidad de epochs se detiene en 20 por el poder de procesador de la maquina, esta podria continuar para obtener un menor error absoluto medio (MAE), y un modelo mas preciso.-

# 

# In[ ]:




