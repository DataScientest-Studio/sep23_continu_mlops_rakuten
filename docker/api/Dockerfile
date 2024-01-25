FROM ubuntu:20.04

# maj paquets et installer python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copiez les dépendances
COPY ./requirements.txt /app/requirements.txt
COPY ./api.py /app

RUN mkdir /app/data

# Installez les dépendances
RUN pip3 install -r requirements.txt && \
    pip install typing-extensions --upgrade

# Exposez le port utilisé par l'API
EXPOSE 8000

# Commande pour exécuter l'API
CMD ["uvicorn", "api:api", "--host", "0.0.0.0", "--reload","--port", "8000"]