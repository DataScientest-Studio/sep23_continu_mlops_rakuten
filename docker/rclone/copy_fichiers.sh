#!/bin/sh

# Copier les fichiers initiaux depuis le drive
rclone --config /rclone.conf copy "Gdrive_rakuten2:/" /drive --exclude "/images/image_train/**" --exclude "/images/image_test/**"
chmod -R 777 /drive/

###Shell est séquentiel du coup, tant que le transfer rclone n est pas termine, rien ne ce passe mise a part le message d attente
while [ ! -f "/drive/images.tar" ]; do
    echo "En attente du fichier /drive/images.tar..."
    sleep 60  # Attendre 60 secondes avant de vérifier à nouveau
done

if [ ! -f "/drive/images/images.tar" ]; then
    # Créer le répertoire de destination s'il n'existe pas
        # Décompresser le fichier tar dans le répertoire de destination
    tar -xf "/drive/images.tar" -C "/drive/images"
    echo "Le fichier tar a été décompressé dans /drive/images/"
else
    echo "Le fichier images.tar existe déjà "
fi

sync_to_drive() {
    echo "Copying files to drive"
    rclone --config /rclone.conf copy /drive Gdrive_rakuten2:/ --exclude "/images/image_train/**" --exclude "/images/image_test/**"
}

sync_from_drive() {
    echo "De google drive à local"
    rclone --config /rclone.conf copy Gdrive_rakuten2:/ /drive --exclude "/images/image_train/**" --exclude "/images/image_test/**"
}


# Surveillance du dossier /drive pour les modifications. On essaie d eviter trop d echanges entre google drive et les donnees local
while true ; do
    if inotifywait -r -e modify,create,delete -t 30 /drive; then   ### attendre 30s par evenement
        sync_to_drive   
        sync_from_drive
    fi
    sleep 30 ###attenddre 30s avant prochaine boucle
done