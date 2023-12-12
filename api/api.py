import pickle
import numpy as np
import pandas as pd
import gdown
import os.path
from fastapi import FastAPI, HTTPException
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import urllib.request


# --------
# données
# --------

# données produits
df = pd.read_csv('https://drive.google.com/uc?id=1g3QDdr55vGlCviXBzRTIiNpCSXxSUw-X')
df = df[["productid", "path", "Titre_annonce", "Description"]]

# images
google_drive_details = pd.read_csv('https://drive.google.com/uc?id=1gNsTA2Xb3CjLj193tACefqS_KMXNyBic')
gdrive_images = google_drive_details[["id", "name"]]



# -------
# modele
# -------

# google drive link : https://drive.google.com/file/d/1F8WjoOqhF2QLkceAQVFSJ2I9eI58bZJe/view?usp=drive_link
file_id="1F8WjoOqhF2QLkceAQVFSJ2I9eI58bZJe"
url = f'https://drive.google.com/uc?id={file_id}'
output = '../../model.h5'

if os.path.isfile(output):
    print('model file already exist')
else:
    gdown.download(url, output, quiet=False)

model = load_model(output)


with open("../data/label_encoder.pickle", "rb") as handle:
    le = pickle.load(handle)

with open("../data/tokenizer.pickle", "rb") as handle:
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

# liste désignations des catégories
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


# ----
# API
# ----
api = FastAPI(
    title="API Rakuten",
    description="API de prédiction de catégorisation",
    version="1.0")


@api.get('/status', name="Status")
async def get_status():
    """
    Renvoie 1 si l'API est fonctionnelle
    """
    return {
        'status': 1
    }


@api.get('/predictcategory/{productid:int}', name="Prédiction")
async def get_prediction(productid):
    """
    Description :
    renvoie la prédiction de la catégorie du produit

    Argument :
    l'ID du produit

    Renvoie : Les différentes informations de l'article ainsi que la prédiction de la catégorie
    - product id : l'identifiant
    - product designation : la désignation
    - product description : la description s'il y en a une (sinon 'nodata')
    - product image : l'URL de l'image sur Google Drive
    - predicted category : la prédiction de la catégorie

    Erreurs:
    - erreur 404 : produit inconnu si l'id n'est pas dans la liste.
    """
    df_product = df[df['productid']==productid]

    if not df_product.empty:

        image_file = str(df_product["path"].iloc[0]).split('/')[-1]
        image_file_google_id = gdrive_images[gdrive_images['name']==image_file]
        image_path, headers = urllib.request.urlretrieve("https://drive.google.com/uc?export=view&id=" + image_file_google_id['id'].iloc[0])

        texts_product = df_product.loc[:,("Titre_annonce", "Description")]
        if texts_product['Description'].iloc[0] == "nodata":
            texts_product['text_input'] = texts_product['Titre_annonce']
        else:
            texts_product['text_input'] = texts_product[['Titre_annonce', 'Description']].agg(' '.join, axis=1)

        text_input = texts_product['text_input'].iloc[0]


        # Prétraitement de l'image et du texte
        image = preprocess_image(image_path)
        text = preprocess_text(text_input, tokenizer)

        # Ajustement de la forme des données pour correspondre à l'entrée du modèle
        image = np.expand_dims(image, axis=0)  # Ajout d'une dimension pour le batch
        text = np.expand_dims(text, axis=0)    # Ajout d'une dimension pour le batch

        prediction = model.predict([image, text])
        
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = le.inverse_transform(predicted_class)
        
        if predicted_label[0] in categories:
            return {
                'product id': productid,
                'product designation' : texts_product['Titre_annonce'].iloc[0],
                'product description' : texts_product['Description'].iloc[0],
                'product image' : "https://drive.google.com/uc?export=view&id=" + image_file_google_id['id'].iloc[0],
                'predicted category': categories[predicted_label[0]]
                }
        else:
            return "Catégorie non trouvée"
    
    else:
        raise HTTPException(status_code=404, detail='Produit inconnu')
