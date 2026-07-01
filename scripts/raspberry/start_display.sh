#!/bin/bash
# Affichage Artimir — lance mpv en plein écran sur la vidéo du PC de mapping
# À lancer SUR le Raspberry Pi, après avoir démarré mapping_server.py sur le PC

PC_IP="192.168.0.10"
PC_PORT="8000"
VIDEO_URL="http://${PC_IP}:${PC_PORT}/video_feed"

echo "[DISPLAY] Attente du serveur Artimir (${PC_IP}:${PC_PORT})..."

until curl -s --max-time 2 "http://${PC_IP}:${PC_PORT}/status" > /dev/null 2>&1; do
    echo "[DISPLAY] Serveur pas encore prêt, nouvel essai dans 2s..."
    sleep 2
done

echo "[DISPLAY] Serveur OK — lancement de l'affichage plein écran."

mpv \
    --fs \
    --profile=low-latency \
    --untimed \
    --no-cache \
    --video-unscaled=no \
    --panscan=1.0 \
    "$VIDEO_URL"

echo "[DISPLAY] mpv terminé."
