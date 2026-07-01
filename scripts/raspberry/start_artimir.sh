#!/bin/bash
# Lance l'affichage ET l'audio Artimir en parallèle dans des sessions tmux séparées
# Utilisation : bash start_artimir.sh
# Pour arrêter : tmux kill-session -t artimir_display && tmux kill-session -t artimir_audio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v tmux &>/dev/null; then
    echo "[ERREUR] tmux n'est pas installé. Lance start_display.sh et start_audio.sh dans deux terminaux séparés."
    exit 1
fi

echo "[ARTIMIR] Démarrage audio en arrière-plan (session tmux : artimir_audio)..."
tmux new-session -d -s artimir_audio "bash '${SCRIPT_DIR}/start_audio.sh'"

echo "[ARTIMIR] Démarrage affichage vidéo (session tmux : artimir_display)..."
echo "[ARTIMIR] Ferme cette fenêtre quand tu veux arrêter l'affichage."
tmux new-session -d -s artimir_display "bash '${SCRIPT_DIR}/start_display.sh'"

echo ""
echo "[ARTIMIR] Sessions actives :"
tmux ls
echo ""
echo "Pour voir les logs :"
echo "  tmux attach -t artimir_display"
echo "  tmux attach -t artimir_audio"
echo ""
echo "Pour tout arrêter :"
echo "  tmux kill-session -t artimir_display"
echo "  tmux kill-session -t artimir_audio"
echo "  (ou : tmux kill-server)"
