import pandas as pd
import numpy as np
import re
import math
import glob
import pickle
import joblib
from datetime import datetime
import os
from sklearn.metrics import accuracy_score

import mlflow

from unidecode import unidecode
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D, Dense, GRU, Flatten, 
                                     Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, 
                                     Input, LSTM, Concatenate, Attention)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import EfficientNetB1

from tensorflow.keras import layers

from tensorflow.keras import Model
from tensorflow.keras import callbacks

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer, snowball
from nltk.corpus import stopwords


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

###import des variables environnement donnees ds docker-compose

path_data=  r'/app/drive/data/'
path_images_train=  r'/app/drive/images/image_train' 
path_images_test=  r'/app/drive/images/image_test'  
path_model_prod= r'/app/drive/models/bimodal.h5'
path_model=  r'/app/drive/models_entrainement/'
path_mlflow=r'/app/drive/MLflow/'


##demarrer mlflow

mlflow.set_tracking_uri("http://entrainement:5000")


def get_or_create_experiment(name):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(name)
    
experiment_id = get_or_create_experiment('training_rakuten')

with mlflow.start_run(experiment_id =experiment_id):

    ###Les fichiers csv proviennent du challenge Rakuten
    ### Une copie sur ce google drive : https://drive.google.com/drive/folders/1PltQt7eFWu5lkGf4jRdqqRIZaykBlta8?usp=drive_link

    X_train_path=os.path.join(path_data,'X_train_update.csv')
    y_train_path=os.path.join(path_data,'Y_train.csv')

    X_train = pd.read_csv(X_train_path, index_col=0)
    y_train = pd.read_csv(y_train_path, index_col=0)


    ##################################################################################################
    #model texte
    ##################################################################################################

    def remove_accents(text):
        return unidecode(text) if pd.notnull(text) else text

    X_train['designation'] = X_train['designation'].apply(remove_accents)
    X_train['description'] = X_train['description'].apply(remove_accents)


    # Convertir les majuscules en minuscules
    X_train['designation'] = X_train['designation'].str.lower()
    X_train['description'] = X_train['description'].str.lower()

    # Supprimer les nombres 
    X_train['designation'] = X_train['designation'].str.replace('[0-9]', '', regex=True)
    X_train['designation'] = X_train['designation'].str.replace('[^\w\s]', '', regex=True)

    # et les caractères spéciaux/ponctuation
    X_train['description'] = X_train['description'].str.replace('[0-9]', '', regex=True)
    X_train['description'] = X_train['description'].str.replace('[^\w\s]', '', regex=True)


    # Preprocessing stopword francais/anglais/allemand

    stop_words = set(stopwords.words('french'))

    # On va créer une liste comportant tous les stop words français, anglais et allemand
    ang = set(stopwords.words('english'))
    ang = list(ang)
    ger = set(stopwords.words('german'))
    ger = list(ger)

    # Création d'une fonction pour ajouter des mots stop words
    def ajout(list):
        for i in list:
            stop_words.add(i)
            
    ajout(ang)
    ajout(ger)


    def process_text(text):
        # Tokenize and remove single characters
        tokenizer = RegexpTokenizer(r'\b[a-z]{2,}\b')
        tokens = tokenizer.tokenize(text.lower())

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

        # Remove stopwords
        cleaned = [word for word in lemmatized if word not in stop_words]

        return ' '.join(cleaned)

    # Apply the function to the DataFrame
    for col in ['designation', 'description']:
        X_train[col] = X_train[col].fillna('').apply(process_text)


    # Fill missing values with a placeholder
    X_train.fillna('nodat', inplace=True)

    # Concatenate the 'designation' and 'description' columns
    # This avoids the need to use word_tokenize at this stage

    X_train['combined'] = X_train['designation'] + ' ' + X_train['description']

    X_train_final = X_train['combined']


    y_train_final =y_train.prdtypecode

    
    print("Dimensions de y_train_final:", y_train_final.shape)
    


    ### encodage identique de y pour les 2 models text et image

    label_output_path=os.path.join(path_data,'label_encoder.joblib')


    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_final)

    joblib.dump(le, label_output_path)

    ####One-hot

    y_train_onehot=to_categorical(y_train_encoded, num_classes=27)


    ###sauver le labelencoder

    label_output_path_pickle=os.path.join(path_data,'label_encoder.pickle')

    with open(label_output_path_pickle, "wb") as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

    # Création de nos ensemble d'apprentissage et test

    X_trainn, X_testt, y_trainn, y_testt = train_test_split(X_train_final, y_train_onehot, test_size=0.2, random_state = 42)


    train_size = int(X_trainn.shape[0])
    train_posts = X_trainn[:train_size]
    train_tags = y_trainn[:train_size]

    train_size = int(X_testt.shape[0])
    test_posts = X_testt[:train_size]
    test_tags = y_testt[:train_size]

    # Create a tokenizer and fit it on the training posts
    tokenizer = Tokenizer(num_words=None, char_level=False)
    tokenizer.fit_on_texts(train_posts)

    # Convert texts to sequences for both training and testing sets
    x_train = tokenizer.texts_to_sequences(train_posts)
    x_test = tokenizer.texts_to_sequences(test_posts)

    vocab_size = len(tokenizer.word_index) + 1

    # Padding sequences
    max_len = 100
    x_train_text = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test_text = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


    ##sauver tokenizer

    label_output_tokenizer=os.path.join(path_data,'tokenizer.pickle')

    with open(label_output_tokenizer, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Dataset de notre jeu de données

    text_train_set =tf.data.Dataset.from_tensor_slices((x_train_text, y_trainn))
    text_test_set = tf.data.Dataset.from_tensor_slices((x_test_text, y_testt))



    # Ajouter la fonction load_image dans le pipeline des opérations. Séparer le résultat en lot de taille 32.
    text_train_set = text_train_set.map(lambda text, y: [text, y]).batch(32).repeat(-1)
    text_test_set = text_test_set.map(lambda text, y: [text, y]).batch(32).repeat(-1)


    

    ##################################################################################################
    #model image
    ##################################################################################################



    ##images d'origine telechargez sur le challenge Rakuten 500*500 et avec bord blanc non viré

    train_images_path = os.path.join(path_images_train, '*.jpg')
    test_images_path = os.path.join(path_images_test, '*.jpg')

    #liste_train = glob.glob('/app/data/image_train/*.jpg')
    #liste_test = glob.glob('/app/data/image_test/*.jpg')

    liste_train = glob.glob(train_images_path)
    liste_test = glob.glob(test_images_path)

    ###pour enlever le probleme de \ /
    liste_train_clean=[]
    for i in liste_train :
        liste_train_clean.append(i.replace("\\","/"))



    #path_train=r'/app/data/X_train_update.csv'
    X_train=pd.read_csv(X_train_path, index_col=0)


    ### extraction des noms d images pour pouvoir faire le merge des 2 df
    import re
    r=re.compile(r"image_[0-9]+")
    r2=re.compile(r"[0-9]+")
    liste_image=[]             

    for path in liste_train :
        image=r.findall(path)[0]
        liste_image.append(r2.findall(image)[0])


    df=pd.DataFrame({'imageid':liste_image, 'path':liste_train_clean})

    ### split 80/20
    X_train_image, X_test_image, y_train, y_test = train_test_split(df.path, y_train_onehot, test_size=0.2, random_state=42)


    np.array_equal(y_test, y_testt)


    @tf.function
    def load_image(filepath, resize=(200, 200)):
        im = tf.io.read_file(filepath)
        im = tf.image.decode_png(im, channels=3)
        return tf.image.resize(im, resize, method='nearest')


    # Chargement des données
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train_image, y_train))
    dataset_train = dataset_train.map(lambda x, y : [load_image(x), y], num_parallel_calls=-1).batch(32).repeat(-1)

    dataset_test =tf.data.Dataset.from_tensor_slices((X_test_image, y_test))
    dataset_test = dataset_test.map(lambda x, y : [load_image(x), y], num_parallel_calls=-1).batch(32).repeat(-1)


    

    ##################################################################################################
    #concatenation des 2 models => bimodal
    ##################################################################################################

    # Définition d'un générateur python
    def generator(image_set, treated_text_set):
        iter_image = iter(image_set)
        iter_text_treated = iter(treated_text_set)
        while True:
            X_im, y = next(iter_image)
            X_text_treated, y_text = next(iter_text_treated) 
            yield [X_im, X_text_treated], y_text



    # Définition du générateur final.
    gen_train = generator(dataset_train, text_train_set)
    gen_test = generator(dataset_test, text_test_set)



    model = load_model(path_model_prod)

    ### on freeze toutes les couches presentes avant le merge 
    for layer in model.layers[:12]:
        layer.trainable = False

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



    # Savegarde automatique des poids
    checkpoint = callbacks.ModelCheckpoint(filepath='checkpoint', 
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='min',
                                        save_freq='epoch')

    # Réduction automatique du taux d'apprentissage
    lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            patience=1,
                                            factor=0.1,
                                            verbose=2,
                                            mode='min')

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                            patience=1,
                                            mode='min',
                                            restore_best_weights=True)


    class MlflowLoggerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            mlflow.log_metrics(logs, step=epoch)


##### Pour accelerer l entrainement on va diminuer drastiquement train_steps afin que chaque epoch soit rapide. Mais on va augmenter le nombre d epochs a 20
##### Les patiences sont également fixé à 1 afin d'avoir un entrainement rapide pour la démo            
            
                
    train_steps = math.ceil(len(y_train)/32) /32  ### /32 pour diminuer temps par epoch
    validation_steps = math.ceil(len(y_test)/32) /32  ### /32 pour diminuer temps par epoch


    model.fit(
        x=gen_train,
        steps_per_epoch = train_steps,
        validation_data = gen_test,
        validation_steps = validation_steps,
        verbose=1,
        epochs=20,
        callbacks=[lr_plateau, early_stopping,MlflowLoggerCallback()]
        #callbacks=[lr_plateau, early_stopping]
    )
    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    filename = f"classification_report_{formatted_datetime}.txt"
    evaluate_path=os.path.join(path_model,filename)
    artifact_path=os.path.join(path_mlflow,filename)

    def evaluate_model(model, gen_test, validation_steps, path):
        y_true = []
        y_pred = []

        for _ in range(validation_steps):
            (X_im, X_text_treated), y_text = next(gen_test)
            y_true.extend(y_text.numpy())
            y_pred_batch = model.predict([X_im, X_text_treated], verbose=0)
            y_pred.extend(y_pred_batch)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)

        #print(classification_report(y_true_labels, y_pred_labels))
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
        report_str = "Classification Report\n" + "\n".join([f"{k}: {v}" for k, v in report.items()])
        
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        mlflow.log_param("accuracy", accuracy)    

        with open(path, "w") as file:
            file.write(report_str)

        mlflow.log_artifact(path)

    # Utilisez la fonction
    gen_test = generator(dataset_test, text_test_set)
    evaluate_model(model, gen_test, validation_steps, evaluate_path)


    model_path=os.path.join(path_model,'bimodal.h5')
    model_artifact_path=os.path.join(artifact_path,'bimodal.h5')
    
    model.save(model_path, include_optimizer=False)

    mlflow.log_artifact(model_path)

    mlflow.end_run()

