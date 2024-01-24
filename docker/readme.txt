#Generer un fichier .env dans le dossier docker contenant : 

API_USERNAME=luffy
API_PASSWORD=<PWD de l API correspondat>

# Effectuer la commande ci-dessous afin de gerer les acces de airflow:
 echo -e "AIRFLOW_UID=$(id -u)" >>.env

# Lors du 1er docker compose up, si les fichiers ne sont pas entierement telechargé dans l image rclone, il se peut que l image entrainement echoue a etre lancee.
Attendre quelques minutes et refaire un docker compose up

