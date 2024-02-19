import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt

from pandas import read_csv

from PIL import Image

from pathlib import Path

import os

import shutil

'''
#Para ver que directorios o archivos hay en el directorio actual
import os

# get current working directory
cwd = os.getcwd()

#get files in directory
files = os.listdir(cwd) 

print(files)
'''

train=read_csv('oracle_CV/train.csv')
test=read_csv("oracle_CV/test.csv")

#print(train) #-> Las estiquetas van entre 0 y 7

train_images=[]
test_images=[]


# Separamos en carpetas separadas imágenes de train y test
# TODO:Cambiar este bucle solo queremos cargar las imagenes de test
for i in range(len(test['path_img'])):
    test_images.append(tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(f"oracle_CV/{test['path_img'][i]}"),channels=3),(128,128))/255)

'''# Para separar las imágenes en dos carpetas diferentes
contenidos=f"oracle_CV/{train['path_img']}"
for i in range(len(train['path_img'])):
    shutil.copy(f"oracle_CV/{train['path_img'][i]}","oracle_CV/train_imgs")
    if i<len(test['path_img']):
        shutil.copy(f"oracle_CV/{test['path_img'][i]}","oracle_CV/test_imgs")
'''

'''# Separar imágenes en diferentes clases por carpetas
for i in range(len(train['path_img'])):
    shutil.copy(f"oracle_CV/{train['path_img'][i]}",f"oracle_CV/train_imgs/{train['label'][i]}")
'''

# ESTA MAL
'''for i in range(len(test['path_img'])):
    shutil.copy(f"oracle_CV/train_imgs/{train['path_img'][i]}",f"oracle_CV/train_imgs/{test['label'][i]}")'''


# Define la ruta de la carpeta de entrada y salida
train_input_folder = "oracle_CV/train_imgs"
test_input_folder="oracle_CV/test_imgs"
output_folder = "ruta/a/carpeta/de/salida"

# Define el tamaño de la imagen de salida deseado
size = [128, 128]

'''#Creamos una función necesario para cargarydimensionar las imágenes
def cargarydim(ruta):
    img=tf.io.read_file(ruta)
    tensor=tf.io.decode_image(img,channels=3,dtype=tf.dtypes.float32)
    tensor=tf.image.resize(tensor,size)
    return tensor'''

# Crea un objeto ImageDataGenerator para redimensionar imágenes
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
validation_split=0.2,
rotation_range=20,
zoom_range=0.2,
width_shift_range=0.1,
height_shift_range=0.1,
horizontal_flip=True,
vertical_flip=False,
preprocessing_function=tf.keras.applications.imagenet_utils.preprocess_input
)
input_data = data_generator.flow_from_directory(directory=train_input_folder,
target_size=size,
batch_size=32,
class_mode='categorical',
classes=["0","1","2","3","4","5","6","7"],
subset='training')

'''# Genera un iterador de flujo de datos para leer imágenes de validación
val_data = data_generator.flow_from_directory(directory=test_input_folder,
target_size=size,
batch_size=32,
subset='validation')'''



# Define el modelo de red neuronal convolucional
'''model = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], 3)),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(8, activation='softmax')
])'''

'''
# No funciona no se porque
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), input_shape=(size[0],size[1],3),activation="relu", padding='same'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                 input_shape=(size[0],size[1],3), activation="relu",padding='same'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                 input_shape=(size[0],size[1],3), activation="relu", padding='same'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3),
                 input_shape=(size[0],size[1],3), activation="relu",padding='same'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(8,activation="softmax"))
'''

'''nb_train_samples=7837
model=keras.Sequential()
inputShape=(128,128,3)

model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=inputShape))
model.add(tf.keras.layers.Conv2D(32,(3,3)))
model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dense(8,activation="softmax"))'''



'''model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128,3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])'''

'''# Compila el modelo
model.compile(optimizer='adam',
loss='mse',
metrics=['accuracy'])

# Entrena el modelo
model.fit(x=input_data, epochs=10, verbose=1)'''

'''
# Compila el modelo
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
'''

'''model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], 3)),
    keras.layers.Flatten(input_shape=(size[0], size[1],3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(8, activation='softmax')
])'''

'''# Compila el modelo
model.compile(optimizer='adadelta',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])'''

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(128,128,3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten(input_shape=(128,128,3)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(), #adam?
              metrics=['accuracy'])

model_history=model.fit(
    input_data,
    epochs=25,
    steps_per_epoch=7837//32
)

'''# Entrena el modelo
model.fit(input_data,
epochs=10)'''


test2 = tf.convert_to_tensor(test_images)

#print(test2)
#img = np.array(test2[8])
#print(img)
#img = img.reshape(128,128,3)

predictions=model.predict(tf.keras.applications.imagenet_utils.preprocess_input(test2))

# Aqui vemos imágenes de test junto con su predicción

# 0: hamburguesas, 1:japo, 2: bollitos, 3: patatas fritas 4:perritos calientes, 5:pizza, 6:sandwich 7: tostadas con queso
comidas=["hamburguesas", "japo", "bollitos", "patatas", "perritos", "pizza", "sandwich", "tostadas con queso"]
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(comidas[np.argmax(predictions[i])])
plt.show()


'''
# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos modelo
model.fit(input_data, epochs=10)'''

'''for i in range(10):
    print(train_images[i].shape)'''