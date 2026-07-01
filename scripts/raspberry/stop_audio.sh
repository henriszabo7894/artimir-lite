#!/bin/bash
# Arrêt propre du récepteur audio UDP Artimir

if pkill -f "mpv.*udp"; then
    echo "[AUDIO] Récepteur audio arrêté."
else
    echo "[AUDIO] Aucun processus audio mpv trouvé."
fi
