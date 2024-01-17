# Utilisez l'image officielle Apache Airflow comme base
FROM apache/airflow:2.1.1

USER root

# Mettez à jour les paquets
RUN apt-get update

# Installez docker.io (client Docker)
RUN apt-get install -y docker.io

# Installez Docker Compose via pip
RUN pip install --upgrade pip && \
    pip install docker-compose

ENTRYPOINT ["/entrypoint"]
CMD ["airflow", "webserver"]
