# Artimir 2.0

Installation de face mapping en temps réel qui anime des portraits peints (La
Joconde, l'autoportrait de Van Gogh, Le Désespéré de Courbet) à partir d'un
flux webcam, via LivePortrait. Voir `CLAUDE.md` pour l'architecture détaillée.

Le repo est volontairement léger : poids du modèle, source LivePortrait
vendored, images sources, narration vidéo et modèle LLM sont tous
git-ignorés et récupérés à la demande plutôt que commités.

## Setup sur une machine neuve

```bash
./setup.sh
```

Ou étape par étape :

```bash
# 1. Environnement Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# 2. Dépendances standard
pip install -r requirements.txt

# 3. PyTorch nightly cu128 - séparé de requirements.txt car l'index-url est
#    spécifique et incompatible avec un `pip install -r` classique. Nightly
#    cu128 nécessaire pour les GPU Blackwell (RTX 50xx) ; sur une carte plus
#    ancienne, une roue stable cu121 classique suffit (voir CLAUDE.md).
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Poids LivePortrait (~500 Mo), source vendored, assets du projet
python download_weights.py
python setup_liveportrait.py
python download_assets.py

# 5. Ollama + modèle Mistral pour le dialogue visiteur
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct-v0.3-q4_K_M

# 6. Boucles vidéo idle par œuvre
python generate_idle_loops.py
```

## Lancer

```bash
source venv/bin/activate
python mapping_server.py   # serveur complet, écoute sur :8000
python main.py              # démo simple sans serveur HTTP, joconde uniquement
```

## Note CUDA Toolkit

Le CUDA Toolkit système n'est **pas nécessaire** pour l'usage courant : les
roues PyTorch embarquent leur propre runtime CUDA, et le dialogue LLM passe
par Ollama (qui gère lui-même son accès GPU). Un CUDA Toolkit installé
séparément (`nvcc`, en-têtes de dev) ne redevient utile que si on retente un
jour l'approche `llama-cpp-python` abandonnée au profit d'Ollama, qui compile
son propre binding contre le toolkit local.
