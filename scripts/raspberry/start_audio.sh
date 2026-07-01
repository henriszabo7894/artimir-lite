#!/bin/bash
# Récepteur audio Artimir — écoute le flux UDP AAC/MPEGTS sur le port 1235
# Le PC de mapping envoie l'audio de narration ici quand une narration est déclenchée
# À lancer SUR le Raspberry Pi, de préférence avant (ou en même temps que) l'affichage

LISTEN_PORT="1235"

echo "[AUDIO] Récepteur audio démarré, écoute UDP sur le port ${LISTEN_PORT}."
echo "[AUDIO] En attente des narrations depuis le PC de mapping..."

mpv \
    --no-video \
    --demuxer-lavf-format=mpegts \
    "udp://@:${LISTEN_PORT}"

echo "[AUDIO] mpv audio terminé."
