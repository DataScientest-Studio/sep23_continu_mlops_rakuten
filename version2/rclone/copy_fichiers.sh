#!/bin/sh

# Copier les fichiers initiaux depuis le drive
rclone --config /rclone.conf copy "Gdrive_rakuten2:/" /drive
chmod -R 777 /drive/

sync_to_drive() {
    echo "Copying files to drive"
    rclone --config /rclone.conf copy /drive Gdrive_rakuten2:/
}

sync_from_drive() {
    echo "De google drive à local"
    rclone --config /rclone.conf copy Gdrive_rakuten2:/ /drive
}


# Surveillance du dossier /drive pour les modifications
while inotifywait -r -e modify,create,delete /drive; do
    sync_to_drive
    sync_from_drive
done