import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Charger votre modèle IA
model = load_model('car_not_car_model.h5')  # Remplacez par le chemin vers votre modèle

def predict_car(img_path):
    """
    Prédit si une image contient une voiture ou non.
    Args:
        image_path (str): Chemin de l'image à analyser.
    Returns:
        bool: True si une voiture est détectée, False sinon.
    """
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (100, 100))
    img_normalized = img_resized / 255.0  # Normalisation
    img_input = np.reshape(img_normalized, [1, 100, 100, 3])  # Ajouter la dimension batch

    # Prédire avec le modèle
    predic = model.predict(img_input)
    if 1 - predic[0][0] < 0.7:
        return False
    else :
        return True
   
