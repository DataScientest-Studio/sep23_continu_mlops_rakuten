# Utilisez une image de base Ubuntu
FROM ubuntu:20.04

# Mettre à jour les paquets et installer Python
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python

# Définition du répertoire de travail
WORKDIR /app

# Mettez à jour pip
RUN pip3 install --upgrade pip

# Installez MLflow
RUN pip3 install mlflow[extras]

# Exposez le port utilisé par le serveur MLflow
EXPOSE 5000

# Définissez la commande pour lancer le serveur MLflow
CMD ["mlflow", "ui", "--host", "0.0.0.0"]
