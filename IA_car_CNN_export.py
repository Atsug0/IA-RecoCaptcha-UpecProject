import numpy as np
import keras
import time
import os
import cv2
import math
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
#from keras.layers import Dense, Flatten, Dropout #type de couches
from keras.layers import Dense, Flatten, Dropout #type de couches
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import BatchNormalization
from keras.layers import BatchNormalization
#from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from keras.metrics import binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


train_path = 'data/train/'
valid_path = 'data/valid/'

batch_size = 32

# Création des générateurs avec augmentation de données pour l'entraînement

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # Normalisation
    rotation_range=20,        # Rotation aléatoire
    width_shift_range=0.2,    # Décalage horizontal
    height_shift_range=0.2,   # Décalage vertical
    shear_range=0.2,          # Transformation en cisaillement
    zoom_range=0.2,           # Zoom aléatoire
    horizontal_flip=True      # Inversion horizontale
)


# Générateur de validation avec uniquement une normalisation
valid_datagen = ImageDataGenerator(
    rescale=1.0/255.0  # Normalisation sans augmentation de données
)

# Chargement des images augmentées pour l'entraînement
train_batches = train_datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='binary'  # Binaire : car / not_car
)

# Chargement des images de validation
valid_batches = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='binary'
)

steps_train = train_batches.samples // train_batches.batch_size

steps_valid = valid_batches.samples // valid_batches.batch_size
num_classes = train_batches.num_classes

cls_train = train_batches.classes
cls_test =  valid_batches.classes
class_names = list(train_batches.class_indices.keys())



# Définition d'un modèle CNN simple
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sortie binaire
])

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Résumé du modèle
model.summary()



# Callbacks

start = time.time()

import matplotlib.pyplot as plt

history = model.fit(train_batches, validation_data=valid_batches, epochs=100, verbose=1)
model.save('car_not_car_model.h5')

end = time.time()

print(f"Le modèle a été entraîné en {end - start:.2f} secondes.")

