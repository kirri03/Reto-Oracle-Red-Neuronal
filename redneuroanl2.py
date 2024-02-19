import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt

from pandas import read_csv

from PIL import Image

from pathlib import Path

import numpy as np

import json



train=read_csv('oracle_CV/train.csv')
test=read_csv("oracle_CV/test.csv")

X = np.zeros((len(test), 128, 128, 3), dtype=np.float32)

for i, path in enumerate(test['path_img']):
    img = tf.keras.preprocessing.image.load_img(f"oracle_CV/{path}", target_size=(128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
    X[i] = img

X = tf.keras.applications.imagenet_utils.preprocess_input(X)

#print(train) #-> Las estiquetas van entre 0 y 7

train_images=[]
val_images=[]
test_images=[]


# Define la ruta de la carpeta de entrada y salida
train_input_folder = "oracle_CV/train_imgs"
val_input_folder="oracle_CV/val_images"
test_input_folder="oracle_CV/test_imgs"
output_folder = "ruta/a/carpeta/de/salida"

# Define el tamaño de la imagen de salida deseado
size = [128, 128]



# Crea un objeto ImageDataGenerator para redimensionar imágenes

# Se podría hacer un bucle sobre esto para que en cada iteración del bucle cambien los valores
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
validation_split=0.2,
rotation_range=15,
zoom_range=0.2,
shear_range=0.2,
width_shift_range=0.1,
height_shift_range=0.1,
horizontal_flip=True,
vertical_flip=False,
fill_mode='nearest',
preprocessing_function=tf.keras.applications.imagenet_utils.preprocess_input
)
input_data = data_generator.flow_from_directory(directory=train_input_folder,
target_size=size,
batch_size=32,
class_mode='categorical',
classes=["0","1","2","3","4","5","6","7"],
subset='training')

# Genera un iterador de flujo de datos para leer imágenes de validación
val_data = data_generator.flow_from_directory(directory=val_input_folder,
target_size=size,
batch_size=32,
class_mode='categorical',
classes=["0","1","2","3","4","5","6","7"],
subset='validation')



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

'''model = tf.keras.Sequential()
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
    validation_data=val_data,
    epochs=25
)'''

# Define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(128,128,3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten(input_shape=(128,128,3)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8, activation='softmax'))

# Define optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

# Train model
model_history=model.fit(
    input_data,
    epochs=50
)

'''# Entrena el modelo
model.fit(input_data,
epochs=10)'''



#print(test2)
#img = np.array(test2[8])
#print(img)
#img = img.reshape(128,128,3)

predictions=model.predict(X)


# Aqui vemos imágenes de test junto con su predicción

# 0: hamburguesas, 1:japo, 2: bollitos, 3: patatas fritas 4:perritos calientes, 5:pizza, 6:sandwich 7: tostadas con queso
comidas=["hamburguesas", "japo", "bollitos", "patatas", "perritos", "pizza", "sandwich", "tostadas con queso"]
plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i], cmap=plt.cm.binary)
    plt.xlabel(comidas[np.argmax(predictions[i])])
plt.show()

resultados = {}

# Recorrer la lista test_idx y obtener los valores a partir de np.argmax
for i, id in enumerate(test['idx_test']):
    resultados[id] = int(np.argmax(predictions[i]))

# Convertir el diccionario a formato JSON
json_resultados = json.dumps(resultados)
print(json_resultados)

'''
# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos modelo
model.fit(input_data, epochs=10)'''

'''for i in range(10):
    print(train_images[i].shape)'''

