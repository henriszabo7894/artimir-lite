#!/usr/bin/env bash
# Setup complet d'Artimir 2.0 sur une machine neuve (GPU NVIDIA requis).
# Le repo est volontairement leger : poids, sources vendored, assets et
# modeles LLM sont tous recuperes par ce script plutot que commites.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== 1. Environnement Python 3.12 ==="
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "=== 2. Dependances standard ==="
pip install -r requirements.txt

echo "=== 3. PyTorch nightly cu128 (GPU Blackwell : RTX 50xx) ==="
# A part du index-url specifique, torch/torchvision ne peuvent pas passer par
# requirements.txt : la RTX 5080/5060 (architecture Blackwell) n'est
# supportee que par les builds nightly cu128 pour le moment, pas par les
# roues stables classiques.
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

echo "=== 4. Poids LivePortrait + source vendored + assets projet ==="
python download_weights.py
python setup_liveportrait.py
python download_assets.py

echo "=== 5. Ollama + modele Mistral ==="
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
ollama pull mistral:7b-instruct-v0.3-q4_K_M

echo "=== 6. Boucles idle (vangogh/joconde/courbet) ==="
python generate_idle_loops.py

echo "=== Setup termine ==="
echo "Lancer le serveur avec : source venv/bin/activate && python mapping_server.py"
