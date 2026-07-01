#!/bin/bash
PI_IP="192.168.0.20"
PI_PORT="1235"
AUDIO_FILE="$1"

if [ -z "$AUDIO_FILE" ]; then
    echo "Usage: bash send_audio.sh /chemin/vers/audio.mp3.mpeg"
    exit 1
fi

echo "[AUDIO] Envoi de $AUDIO_FILE vers $PI_IP:$PI_PORT"

ffmpeg -re -i "$AUDIO_FILE" \
    -f mpegts \
    -acodec mp2 \
    "udp://$PI_IP:$PI_PORT"
