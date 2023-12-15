from datetime import datetime, timedelta
import os.path
import re
import shutil
import urllib.request
import pickle
import gdown
import jwt
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Optional

# NLTK et traitement de texte
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode


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
    
def preprocess_image_jpg(image_path, resize=(200, 200)):
    im = tf.io.read_file(image_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.resize(im, resize, method='nearest')
    return im


stop_words = set(stopwords.words('french'))
stop_words.update(stopwords.words('english'))
stop_words.update(stopwords.words('german'))

def remove_accents(text):
    return unidecode(text) if text else text

def preprocess_text(text, tokenizer, max_len=100):
    text = remove_accents(text)
    text = text.lower()
    text = re.sub('[0-9]', '', text)
    text = re.sub('[^\w\s]', '', text)
    nltk_tokenizer = RegexpTokenizer(r'\b[a-z]{2,}\b')
    tokens = nltk_tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned = [word for word in lemmatized if word not in stop_words]
    sequence = tokenizer.texts_to_sequences([' '.join(cleaned)])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence[0]  

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


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configuration pour le JWT
SECRET_KEY = "secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRATION = 30

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

users_db = {

    "luffy": {
        "username": "luffy",
        "name": "luffy",
        "email": "luffy@chapeau.com",
        "hashed_password": pwd_context.hash('123456789'),
        "resource" : "rakuten",
    },
    
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = users_db.get(username, None)
    if user is None:
        raise credentials_exception
    return user

@api.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Description:
    Cette route permet à un utilisateur de s'authentifier en fournissant un nom d'utilisateur et un mot de passe. Si l'authentification est réussie, elle renvoie un jeton d'accès JWT.

    Args:
    - form_data (OAuth2PasswordRequestForm, dépendance): Les données de formulaire contenant le nom d'utilisateur et le mot de passe.

    Returns:
    - Token: Un modèle de jeton d'accès JWT.

    Raises:
    - HTTPException(400, detail="Incorrect username or password"): Si l'authentification échoue en raison d'un nom d'utilisateur ou d'un mot de passe incorrect, une exception HTTP 400 Bad Request est levée.
    """

    user = users_db.get(form_data.username)
    hashed_password = user.get("hashed_password")
    if not user or not verify_password(form_data.password, hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRATION)
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}

@api.get('/status', name="Status")
async def get_status():
    """
    Renvoie 1 si l'API est fonctionnelle
    """
    return {
        'status': 1
    }

@api.get("/secured")
def read_private_data(current_user: str = Depends(get_current_user)):
    """
    Description:
    Cette route renvoie un message uniquement si l'utilisateur est authentifié.

    Args:
    - current_user (str, dépendance): Le nom d'utilisateur de l'utilisateur actuellement authentifié.

    Returns:
    - JSON: Renvoie un JSON contenant un message de salutation sécurisé si l'utilisateur est authentifié, sinon une réponse non autorisée.

    Raises:
    - HTTPException(401, detail="Unauthorized"): Si l'utilisateur n'est pas authentifié, une exception HTTP 401 Unauthorized est levée.
    """

    return {"message": "go pour planter l api!"}

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



@api.post("/get_prediction_input")
async def get_prediction_input(titre: str = Form(...), 
                               description: Optional[str] = Form(None), 
                               image: UploadFile = File(...)):
    """
    Renvoie la prédiction de la catégorie du produit basée sur le titre, 
    la description (optionnelle) et l'image fournis.
    """

    # Sauvegarde temporaire de l'image
    image_path = f"temp_{image.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Prétraitement de l'image
    image_processed = preprocess_image_jpg(image_path)

    # Construction du texte pour le prétraitement
    text_input = titre if description is None else f"{titre} {description}"
    text_processed = preprocess_text(text_input, tokenizer)

    # Suppression de l'image temporaire
    os.remove(image_path)

    # Ajustement de la forme des données pour correspondre à l'entrée du modèle
    image_processed = np.expand_dims(image_processed, axis=0)  # Ajout d'une dimension pour le batch
    text_processed = np.expand_dims(text_processed, axis=0)    # Ajout d'une dimension pour le batch

    # Prédiction
    prediction = model.predict([image_processed, text_processed])
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = le.inverse_transform(predicted_class)

    if predicted_label[0] in categories:
        return {
            'predicted category': categories[predicted_label[0]]
        }
    else:
        raise HTTPException(status_code=401, detail='Prediction de categorie invalide')
    
class Item(BaseModel):
    productID:str
    categoryPredicted:str
    categoryChangedByUser:str
    categoryID:str

@api.post('/Feedback')
async def add_data(product : Item):
    '''
    Description :
    -   Ajout du feedback utilisateur sur la performance du modèle à la base de données 
    -   Pour l'intstant il ajoute seulement les données fournis par l'utilisateur dans le fichier ProductUSerFeedback.csv

    Renvoie : Les différentes informations de l'article qui ont été ajouté à la base

    '''
    data_to_add = pd.DataFrame.from_dict(product, orient='columns')
    data_to_add = data_to_add.T
    data_to_add.columns = data_to_add.iloc[0]
    data_to_add = data_to_add[1:] # enlever la première ligne

    #df = pd.read_csv('./ProductUserFeedback.csv')
    data_to_add.to_csv('../data/ProductUserFeedback.csv', mode ='a', index = False, header = False)

    return product        