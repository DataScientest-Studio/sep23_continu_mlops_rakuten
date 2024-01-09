#!/bin/bash

# Démarrer l'initialisation d'Airflow
docker-compose up airflow-init

# Une fois l'initialisation terminée, démarrer les autres services
docker-compose up -d
