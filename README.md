
## Contexte du projet

Ce projet est issu d'un challenge Rakuten qui avait pour objectif la prédiction d'une classe de produits Rakuten à partir de sa description texte et image (27 catégories).
Sa réalisation a eu lieu dans le cadre de la formation MLops de Datascientest (https://datascientest.com/).
L'objectif de ce projet était de développer et de déployer en production l'algorithme de prédiction de catégorie en suivant une approche CI/CD (Intégration Continue/Déploiement Continu).

## Principe de fonctionnement de l'application

La mise en production de l'application est basée sur:

    - 1 docker contenant une API générant les prédictions.
    - 1 docker contenant le script de ré-entrainement et l'application MLFlow pour un suivi des ré-entrainements.
    - 1 docker contenant airflow afin de vérifier le bon fonctionnement de l'API et de gérer le déclenchement des ré-entrainements.
    - 1 docker rclone permettant le téléchargement des données images et textes depuis google drive.

## Déploiement

    1 - Cloner ce dépot Git
    2 - Se déplacer dans le dossier docker  et générer un fichier .env :
        Dans votre shell :
        echo 'API_USERNAME = luffy' >.env
        echo 'API_PASSWORD=<PWD de l API>' >>.env
        echo -e "AIRFLOW_UID=$(id -u)" >>.env
    3 - Dans le dossier docker, rentrer dans votre shell :
        docker compose up

Le 1er lancement de "docker compose up" dure au moins 10 min (création des images docker), téléchargement des fichiers de google drive.
Lors du 1er lancement, il se peut que l'image d'entrainement se lance avant la fin du téléchargement des fichiers de google drive, ce qui provoquera une erreur. Dans ce cas, attendre quelques minutes et refaire un "docker compose up".

    4 - L'api est disponible et accessible localement à http://localhost:8080
        La documentation de l'api se trouve ici : http://localhost:8080/docs
    5 - MLflow est accessible ici : http://localhost:5000
    6 - Airflow est accessible ici : http://localhost:8080
        3 DAGS disponibles :
            - test : Test du bon fonctionnement de l'API. Tests d'authentification et de prédiction.
            - training : Déclenche le ré-entrainement et l'incorporation de nouvelles données image et texte. 
            - feedback : traitement des feedbacks utilisateur 

## Streamlit

Une application Streamlit est disponible pour simuler l'interface utilisateur sur le site Rakuten.

Pour lancer l'application :

```shell
cd streamlit_app
streamlit run main.py
```

L'application est accessible à l'adresse : [localhost:8501](http://localhost:8501)

---

Le projet a été développé par :

- Alexandre Laroche ([GitHub](https://github.com/Alex-Laroche) / [LinkedIn](https://www.linkedin.com/in/alexandre-laroche-a96360263/))
- Meryem Nedjwa Maameri ([GitHub](https://github.com/MeryemMAAMERI) / [LinkedIn](https://www.linkedin.com/in/meryem-maameri/))
- Alexandre Luneau ([GitHub](https://github.com/alexluneau) / [LinkedIn](https://www.linkedin.com/in/alexandre-luneau-6b45a51))
