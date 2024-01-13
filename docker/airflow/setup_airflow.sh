#!/bin/bash

# Démarrer l'initialisation d'Airflow
docker-compose up airflow-init

##il y aura des erreurs lors des premieres tentatives.
##changer le owner des dossiers genere chown -R <owner>:<owner> ./
##donner les droits d ecriture et lecture

# Une fois l'initialisation terminée, démarrer les autres services
docker-compose up -d
