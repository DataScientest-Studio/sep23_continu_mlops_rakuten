#!/bin/sh

# Copier les fichiers initiaux depuis le drive
rclone --config /rclone.conf copy "Gdrive_rakuten2:/" /drive --exclude "images/image_train/**" --exclude "images/image_test/**"  
chmod -R 777 /drive/

###Shell est séquentiel du coup, tant que le transfer rclone n est pas termine, rien ne ce passe mise a part le message d attente
while [ ! -f "/drive/donnees_entrainement/data.tar" ]; do
    echo "En attente du fichier /drive/donnes_entrainement/data.tar..."
    sleep 60  # Attendre 60 secondes avant de vérifier à nouveau
done

if [ -f "/drive/donnees_entrainement/data.tar" ]; then
    echo "Le fichier images.tar a été trouvé. Decompression en cours sans écraser les fichiers existants"

    tar -xf "/drive/donnees_entrainement/data.tar" -k -C "/drive/donnees_entrainement"
    echo "Le fichier tar a été décompressé dans /drive/donnees_entrainement/"
    
    chmod -R 777 /drive/donnees_entrainement/
else
    echo "Le fichier data.tar n'existe pas dans /drive/donnees_entrainement."
fi

sync_to_drive() {
    echo "Copying files to drive"
    rclone --config /rclone.conf copy /drive Gdrive_rakuten2:/ --exclude "images/image_train/**" --exclude "images/image_test/**" --exclude "donnees_entrainement/**"
}

sync_from_drive() {
    echo "De google drive à local"
    rclone --config /rclone.conf copy Gdrive_rakuten2:/ /drive --exclude "images/image_train/**" --exclude "images/image_test/**" --exclude "donnees_entrainement/**"
}


# Surveillance du dossier /drive pour les modifications. On essaie d eviter trop d echanges entre google drive et les donnees local
while true ; do
    if inotifywait -r -e modify,create,delete -t 20 /drive; then   ### attendre 15s par evenement
        sync_to_drive   
        sync_from_drive
    fi
    sleep 15 ###attenddre 15s avant prochaine boucle
done