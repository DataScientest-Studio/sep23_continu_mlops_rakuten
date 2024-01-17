# Utilisation d'une image Ubuntu comme base
FROM tensorflow/tensorflow:latest-gpu


# Définition du répertoire de travail
WORKDIR /app

# Copie du script de test dans le conteneur
COPY requirements.txt /app
COPY bimodel_training.py /app

# Installation des dépendances nécessaires
RUN pip3 install --upgrade pip && pip3 install requests
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt
RUN pip3 install mlflow --ignore-installed blinker

# Téléchargement des stopwords NLTK
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet

# Commande à exécuter lorsque le conteneur est lancé
CMD ["python3", "bimodel_training.py"]
