# Importamos las librerías necesarias para el procesamiento y análisis de datos
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples, list_datasets, extract_test_samples

# Extraemos las imágenes y etiquetas de entrenamiento del conjunto de datos EMNIST
train_images, train_labels = extract_training_samples('byclass')
# Imprimimos las dimensiones para verificar el tamaño de los datos
print(train_images.shape)
print(train_labels.shape)

# Extraemos las imágenes y etiquetas de prueba del conjunto de datos EMNIST
test_images, test_labels = extract_test_samples('byclass')
# Imprimimos las dimensiones para verificar el tamaño de los datos
print(test_images.shape)
print(test_labels.shape)

# Normalizamos las imágenes de entrenamiento y prueba para que sus valores estén entre 0 y 1
train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

# Añadimos una dimensión extra a las imágenes para que tengan el formato correcto
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Visualizamos una imagen de entrenamiento con su etiqueta
i = 8
plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.xlabel(train_labels[i])
plt.show()

# Configuramos parámetros para el aumento de datos
rotation_range_val = 15
width_shift_val = 0.10
height_shift_val = 0.10

# Creamos un generador de imágenes para el entrenamiento con los parámetros de aumento
train_datagen = ImageDataGenerator(rotation_range=rotation_range_val,
                                   width_shift_range=width_shift_val,
                                   height_shift_range=height_shift_val)

# Adaptamos el generador a las imágenes de entrenamiento
train_datagen.fit(train_images.reshape(train_images.shape[0], 28, 28, 1))

# Definimos el número de filas y columnas para visualizar las imágenes
num_row = 4
num_col = 8
num = num_row * num_col

# Visualizamos imágenes de entrenamiento antes del aumento de datos
print('BEFORE:\n')
fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
for i in range(num):
    ax = axes1[i//num_col, i % num_col]
    ax.imshow(train_images[i], cmap='gray_r')
    ax.set_title('Label: {}'.format(train_labels[i]))
plt.tight_layout()
plt.show()

# Visualizamos imágenes de entrenamiento después del aumento de datos
print('AFTER:\n')
fig2, axes2 = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
for X, Y in train_datagen.flow(train_images.reshape(train_images.shape[0], 28, 28, 1), train_labels.reshape(train_labels.shape[0], 1), batch_size=num, shuffle=False):
    for i in range(0, num):
        ax = axes2[i//num_col, i % num_col]
        ax.imshow(X[i].reshape(28, 28), cmap='gray_r')
        ax.set_title('Label: {}'.format(int(Y[i])))
    break
plt.tight_layout()
plt.show()

# Creamos un generador de imágenes para el conjunto de validación sin aumento de datos
val_datagen = ImageDataGenerator()
val_datagen.fit(test_images.reshape(test_images.shape[0], 28, 28, 1))

# Definimos la arquitectura de la red neuronal convolucional
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(62, activation='softmax')
])

# Mostramos un resumen del modelo
model.summary()

# Compilamos el modelo con un optimizador, una función de pérdida y métricas
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Entrenamos el modelo con el generador de imágenes de entrenamiento y validación
history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=1024),
                    validation_data=val_datagen.flow(test_images, test_labels, batch_size=32), epochs=20)

# Extraemos y almacenamos datos de precisión y pérdida del entrenamiento
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)
history_df = pd.DataFrame({
    'Epoch': epochs,
    'Accuracy': accuracy,
    'Validation Accuracy': val_accuracy,
    'Loss': loss,
    'Validation Loss': val_loss
})
history_df.to_csv('model_history.csv', index=False)  # Guardamos los datos en un archivo CSV

# Evaluamos el modelo con el conjunto de prueba y mostramos la precisión
scores = model.evaluate(test_images, test_labels)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Graficamos la precisión y la pérdida del modelo durante el entrenamiento
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

# Guardamos el modelo para uso futuro
model.save("model.json")

