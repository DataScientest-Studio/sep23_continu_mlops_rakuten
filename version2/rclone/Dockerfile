FROM alpine:latest

# Installer rclone et inotify-tools
RUN apk --no-cache add rclone inotify-tools

# Créer le répertoire de configuration pour rclone
RUN mkdir -p /root/.config/rclone

# Copier les fichiers de configuration et le script dans le container
COPY rclone.conf /rclone.conf
COPY copy_fichiers.sh /copy_fichiers.sh


# Rendre le script exécutable
RUN chmod +x /copy_fichiers.sh

# Définir les volumes
VOLUME /drive

