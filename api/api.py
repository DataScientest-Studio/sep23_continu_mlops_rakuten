import pickle
import numpy as np
from fastapi import FastAPI
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import urllib.request
from PIL import Image

# model_link = "https://drive.google.com/file/d/1F8WjoOqhF2QLkceAQVFSJ2I9eI58bZJe/view?usp=drive_link"
# model = load_model("/home/alex/Downloads/bimodal.h5")
model = load_model("../models/bimodal.h5")
# filename_model, headers_model = urllib.request.urlretrieve("https://drive.google.com/uc?id=1F8WjoOqhF2QLkceAQVFSJ2I9eI58bZJe")
# model = load_model(filename_model)
# model = tf.keras.models.model_from_json("/home/alex/Downloads/bimodal.json/model.json")

# image_path="./images/images/crop_300_train/image_664491_product_184739.jpg"
# image_path="/home/alex/Downloads/image_664491_product_184739.jpg"
filename_image, headers_image = urllib.request.urlretrieve("https://drive.google.com/uc?export=view&id=16o4Pnph638fJ17-b4ugbqNY6lEGmjEg9")
# img = Image.open(filename_image)
image_path=filename_image
text_input='Jeremy Mc Grath Vs Pastrana'

with open("../models/label_encoder.pickle", "rb") as handle:
    le = pickle.load(handle)

with open("../models/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

def preprocess_image(image_path, resize=(200, 200)):
    im = tf.io.read_file(image_path)
    im = tf.image.decode_png(im, channels=3)
    im = tf.image.resize(im, resize, method='nearest')
    return im

def preprocess_text(text, tokenizer, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence[0]  # Retirer np.expand_dims dans la partie de prédiction

categories = {
    2583: "Accessoires de piscine",
    1560: "Mobilier et décoration pour maison",
    1300: "Modélisme",
    2060: "Accessoires et décorations pour maison",
    2522: "Papeterie",
    1280: "Jouets et peluches enfant",
    2403: "Livres / Magazines en lots",
    2280: "Livres documentaires",
    1920: "Linge de maison",
    1160: "Jeu de cartes à collectionner",
    1320: "Puériculture",
    10: "Littérature étrangère",
    2705: "Littérature historique",
    1140: "Figurines",
    2582: "Accessoires et décorations pour jardin",
    40: "Jeux vidéo neufs",
    2585: "Accessoires et outils de jardinage",
    1302: "Jeux de plein air",
    50: "Accessoires de gaming",
    2462: "Jeux vidéos d’occasion",
    2905: "Jeux vidéo à télécharger",
    60: "Consoles de jeux vidéo",
    2220: "Animalerie",
    1301: "Articles pour nouveaux-nés et bébés",
    1940: "Épicerie",
    1180: "Figurines à peindre et à monter",
    1281: "Jouets et peluches enfant"
}

# Prétraitement de l'image et du texte
image = preprocess_image(image_path)
text = preprocess_text(text_input, tokenizer)

# Ajustement de la forme des données pour correspondre à l'entrée du modèle
image = np.expand_dims(image, axis=0)  # Ajout d'une dimension pour le batch
text = np.expand_dims(text, axis=0)    # Ajout d'une dimension pour le batch

# API
api = FastAPI()

@api.get('/')
def get_index():
   return 'hello world'

@api.get('/prediction')
def get_prediction():
    prediction = model.predict([image, text])
    
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = le.inverse_transform(predicted_class)
    
    if predicted_label[0] in categories:
        return categories[predicted_label[0]]
    else:
        return "Catégorie non trouvée"
