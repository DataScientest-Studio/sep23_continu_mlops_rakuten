# Utilisez une image de base Python
FROM python:3.8

# Définition du répertoire de travail
WORKDIR /app

COPY script_path.py /app

# Mettez à jour pip
RUN pip install --upgrade pip


# Installez MLflow
RUN pip install mlflow[extras]

# Configuration des variables d'environnement pour MLflow
ENV MLFLOW_TRACKING_URI=/app/MLflow
ENV MLFLOW_ARTIFACT_URI=/app/MLflow

#RUN python3 script_path.py

# Exposez le port utilisé par le serveur MLflow
EXPOSE 5000

# Définissez la commande pour lancer le serveur MLflow
CMD ["mlflow", "ui", "--host", "0.0.0.0"]
