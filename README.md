# Artimir — backend IA / mapping (dépôt FaceMappin)

Artimir est une installation muséale interactive qui **anime des tableaux peints
en temps réel** à partir du flux d'une webcam, ajoute un **dialogue vocal** avec
l'œuvre (LLM local), génère une **photo souvenir** du visiteur peinte dans le
style du tableau, et pilote un **réglage de hauteur motorisé** pour s'adapter à
la taille du visiteur. Le projet est porté sous la marque **Kinophos** dans un
cadre B2B à destination des musées.

Résultat notable : **Olympiades des Sciences de l'Ingénieur (OSI) 2026 — 1er au
niveau régional, 2e au niveau national.**

Ce dépôt (`FaceMappin`) contient **la partie IA / backend / GPU** : le serveur de
mapping, le moteur d'animation, le générateur de souvenir et les scripts de test.
L'interface visiteur (site web) et une partie du contrôle physique vivent dans
d'autres dépôts / sur d'autres machines (voir [Dépôts et machines liés](#8-dépôts-et-machines-liés)).

---

## Table des matières

1. [Présentation](#1-présentation)
2. [Architecture générale](#2-architecture-générale)
3. [Pipeline de génération souvenir](#3-pipeline-de-génération-souvenir)
4. [Setup sur une machine neuve](#4-setup-sur-une-machine-neuve)
5. [Utilisation et test](#5-utilisation-et-test)
6. [Autres composants](#6-autres-composants)
7. [Roadmap / travaux en cours](#7-roadmap--travaux-en-cours)
8. [Dépôts et machines liés](#8-dépôts-et-machines-liés)

---

## 1. Présentation

Trois œuvres sont actuellement supportées :

| Œuvre | Tableau |
|-------|---------|
| `joconde` | La Joconde (Léonard de Vinci) |
| `vangogh` | Autoportrait de Vincent Van Gogh |
| `courbet` | Le Désespéré (Gustave Courbet) |

Fonctionnalités par œuvre :

- **Mapping temps réel** — la webcam pilote le visage du tableau via le modèle
  LivePortrait (le tableau reproduit les mouvements de tête et expressions du
  visiteur).
- **Dialogue LLM** — le visiteur parle à l'œuvre, qui répond en restant dans son
  personnage (Ollama + Mistral 7B en local, voir §6).
- **Photo souvenir** — le visage du visiteur est repeint dans le style du tableau
  et exposé pour récupération par le site (voir §3).
- **Bio audio adaptée à l'âge** — narration pré-enregistrée en trois variantes
  (`enfant` / `ado` / `adulte`), streamée en audio vers le Raspberry Pi.
- **Réglage de hauteur motorisé** — un vérin ajuste la hauteur de l'écran/cadre à
  la taille du visiteur (voir §6).
- **Effet parallax** — profondeur simulée à partir de cartes de profondeur
  pré-calculées (`{artwork}_depth.npy`), pilotée par le yaw/pitch estimés.

---

## 2. Architecture générale

Trois ensembles logiciels coopèrent sur un réseau Wi-Fi local à IPs statiques :

```
                         Wi-Fi local (IPs statiques)
   ┌─────────────────────────┐        ┌──────────────────────────────────────┐
   │      Raspberry Pi        │        │            PC Linux (GPU)            │
   │  192.168.0.20            │        │  192.168.0.10                        │
   │                          │        │                                      │
   │  app.py (Flask, GPIO)    │◀──────▶│  mapping_server.py (FastAPI :8000)   │
   │  - affiche la vidéo      │  HTTP  │  - moteur LivePortrait (GPU)         │
   │    (mpv sur /video_feed) │        │  - flux vidéo MJPEG + UDP MPEGTS     │
   │  - joue l'audio (mpv)    │◀──────▶│  - narration audio (ffmpeg UDP)     │
   │  - GPIO / contrôle       │  vidéo │  - pipeline souvenir (SDXL)         │
   │    physique              │  audio │  - dialogue LLM (Ollama)            │
   └─────────────┬────────────┘        └───────────────────┬──────────────────┘
                 │ série (USB)                              │
        ┌────────▼─────────┐                    ┌───────────▼───────────┐
        │ Arduino Mega     │                    │  Site web Artimir-Web │
        │ + driver IBT-2   │                    │  (Node / React,       │
        │ → vérin VEVOR    │                    │   dépôt séparé)       │
        └──────────────────┘                    │  exposé via           │
                                                │  Cloudflare Tunnel    │
                                                │  → artimir.org        │
                                                └───────────────────────┘
```

- **PC Linux (GPU)** — exécute ce dépôt. `mapping_server.py` sert un flux MJPEG
  sur `GET /video_feed`, diffuse aussi les frames en UDP MPEGTS (port 1234) et la
  narration audio en AAC/MPEGTS (UDP port 1235) vers le Pi.
- **Raspberry Pi** — affiche la vidéo et joue l'audio (via `mpv`), et pilote le
  contrôle physique (GPIO + liaison série vers l'Arduino). Son code (`app.py`,
  Flask/GPIO, contrôle du vérin) **ne fait pas partie de ce dépôt**.
- **Site web (Artimir-Web)** — interface visiteur (choix de l'œuvre, âge, dialogue,
  récupération du souvenir). Dépôt séparé (Node/React), exposé publiquement sur
  `artimir.org` via un **Cloudflare Tunnel**.

### Hardware clé

- GPU **NVIDIA RTX 5060 8 Go** (architecture Blackwell) — contrainte VRAM
  structurante pour le pipeline souvenir (voir §3).
- **Webcam Logitech C920** (capture visiteur).
- Vérin **VEVOR** piloté par **Arduino Mega + driver moteur IBT-2**.
- **Haut-parleur USB** (sortie audio narration/dialogue).

---

## 3. Pipeline de génération souvenir

C'est la partie la plus complexe du projet. Tout le code vit dans
`souvenir_generator.py` ; l'intégration au flux webcam vit dans `mapping_server.py`.
Objectif : à partir de quelques frames webcam du visiteur, produire une image du
visiteur **repeint dans le style du tableau**, sans jamais faire planter le serveur.

### Approche actuelle : Differential Diffusion (une passe)

Plutôt qu'un composite dur (masque + collage du visage), l'approche nominale
re-diffuse **tout le canevas** en une seule passe SDXL, guidée par une **carte de
force continue** (Differential Diffusion) plus forte au centre du visage et
décroissante vers les cheveux / le cou :

- **SDXL Inpainting** comme modèle de base (FP16), généré sur un canevas 1024².
- **IP-Adapter** (`ip-adapter-plus-face`, conditionnement identité via CLIP
  vision, **pas** FaceID) pour transférer les traits du visiteur.
- **ControlNet depth** (variante « mid », optionnel) pour contraindre la structure
  du tableau. Réutilise les cartes de profondeur du projet (`{artwork}_depth.npy`),
  sinon fallback MidasDetector.
- Prompts « rendu peint » par œuvre : sfumato / impasto / clair-obscur selon le
  tableau, avec un **negative prompt** qui rejette explicitement la texture
  photographique et les accessoires modernes (bijoux, reflets de peau,
  éclairage studio, pores, aspect CGI). Attention : le budget CLIP est de 77
  tokens et coupe la fin du prompt — tout ajout doit être vérifié au tokenizer.

Le ControlNet peut être désactivé pour rester dans le budget VRAM :

```bash
SOUVENIR_DIFFDIFF_NO_CONTROLNET=1   # pipeline diffdiff sans ControlNet
```

### Chaîne de fallback (jamais d'exception qui remonte)

`generate_souvenir()` ne lève **jamais** : chaque étage est protégé et dégrade
proprement vers le suivant, en loguant la méthode en échec et la cause.

```
diffdiff  ──échec──▶  ip_adapter  ──échec──▶  legacy (warp + composite)
(1 passe SDXL      (inpaint masqué        (triangulation Delaunay +
 + IP-Adapter       du visage seul)        SDXL inpaint / AdaIN /
 + ControlNet opt.)                        sans style ; produit
                                           toujours une image)
```

La méthode ayant produit l'image finale est logguée (`méthode=...`) et exposée via
`SouvenirGenerator.last_method`. En dernier recours, le tableau original avec cadre
est retourné plutôt qu'une erreur.

### Gestion du genre

Le tableau lui-même biaise le rendu (la Joconde tire vers le féminin). Pour
contrer ce biais, le genre estimé injecte des mots-clés dans les prompts :

- **Heuristique géométrique** : ratio `distance inter-yeux / largeur mâchoire`.
  Seuils : `≤ 0.76 → female`, `≥ 0.84 → male`, entre les deux → `neutral`.
- **Bande neutre** volontairement large : sur un cas ambigu (âge, calvitie,
  morphologie atypique) on **n'injecte aucun mot de genre** et on renforce plutôt
  la fidélité structurelle au visage de référence (scale IP-Adapter relevé).
- **Override manuel** : `--gender male|female|neutral` (voir §5) force le genre et
  ignore l'heuristique.

**Limitations connues** : ce n'est pas un vrai classifieur mais une heuristique de
proportions. Les cas limites tombent en `neutral` par conception (choix
volontaire : ne pas trancher avec confiance sur un cas incertain). Le genre sera à
terme collecté côté site plutôt que via la CLI (voir §7).

### Gestion VRAM sur 8 Go

Le budget 8 Go est la contrainte centrale. Le générateur :

- charge SDXL en **FP16**, avec `vae slicing` / `vae tiling` pour réduire le pic au
  décodage 1024² ;
- bascule automatiquement en `enable_model_cpu_offload()` quand la VRAM libre
  passe sous ~7 Go, et retente en `sequential_cpu_offload` après un OOM ;
- **purge la VRAM entre chaque étage de fallback** (`gc.collect()` →
  `torch.cuda.synchronize()` → `empty_cache()`, avec log allocated/reserved avant
  et après), et détache les hooks d'offload accelerate + supprime les gros modules
  (UNet, VAE, ControlNet, encodeurs) avant de charger l'étage suivant.

> Note : lors d'une génération, les 5 modèles LivePortrait sont toujours résidents
> en VRAM. `mapping_server.py` fige le moteur pendant la génération pour libérer le
> GPU, et le cpu_offload absorbe le reste. Si la méthode retenue tombe
> systématiquement sur `legacy`, c'est le signe que le plafond VRAM est atteint
> (le déchargement de LivePortrait avant génération est une piste, hors périmètre
> actuel).

---

## 4. Setup sur une machine neuve

Prérequis système : **GPU NVIDIA**, **Python 3.12** (via le PPA deadsnakes sur
Ubuntu), pilote **CUDA 12.8** côté driver.

Tout en un :

```bash
./setup.sh
```

Ou étape par étape :

```bash
# 1. Environnement Python 3.12
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. Dépendances standard
pip install -r requirements.txt

# 3. PyTorch — installé SÉPARÉMENT de requirements.txt (index-url spécifique,
#    incompatible avec un `pip install -r` classique).
#    GPU Blackwell (RTX 50xx, dont la RTX 5060) : nightly cu128 obligatoire,
#    les roues stables ne supportent pas encore Blackwell.
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
#    Sur une carte plus ancienne (pré-Blackwell), la roue stable cu121 suffit :
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Poids LivePortrait (~500 Mo), source vendored, images + narration
python download_weights.py         # poids LivePortrait -> weights/
python setup_liveportrait.py       # source LivePortrait -> liveportrait_src/
python download_assets.py          # joconde/vangogh/courbet + driving_narration.mp4

# 5. Poids du générateur de souvenir
python download_ip_adapter.py      # IP-Adapter + encodeur CLIP vision
python download_sdxl.py            # SDXL Inpainting
python download_controlnet.py      # ControlNet depth SDXL (optionnel)
python download_adain_weights.py   # AdaIN + FaceLandmarker (fallback / scoring)

# 6. Ollama + modèle Mistral pour le dialogue visiteur
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct-v0.3-q4_K_M

# 7. Boucles vidéo idle par œuvre
python generate_idle_loops.py
```

### Ce qui N'EST PAS dans le dépôt (et pourquoi)

Le dépôt est **volontairement léger** : tout ce qui est lourd ou regénérable est
git-ignoré et récupéré à la demande par les scripts ci-dessus. Absents du dépôt :

- poids des modèles (`weights/`, `models/sdxl_inpainting/`, `models/ip_adapter/`,
  `models/controlnet_depth_sdxl/`, `models/adain/`, `models/face_landmarker.task`),
- source LivePortrait vendored (`liveportrait_src/`),
- images sources des tableaux (`joconde.jpg`, `vangogh.jpg`, `courbet.jpg`) et
  vidéo de narration (`driving_narration.mp4`),
- modèle LLM (`models/*.gguf`, récupéré via `ollama pull`),
- cartes de profondeur / calibration (`*_depth.npy`, `*_depth.png`) et boucles
  idle (`idle_loops/*.mp4`), regénérables,
- photos de test et sorties (`test_frames/`, `output/`, `audio/`).

### Note CUDA Toolkit

Le CUDA Toolkit système n'est **pas nécessaire** en usage courant : les roues
PyTorch embarquent leur propre runtime CUDA, et Ollama gère lui-même son accès
GPU. Un toolkit séparé (`nvcc`, en-têtes de dev) ne redevient utile que pour
recompiler un binding natif (approche `llama-cpp-python`, abandonnée au profit
d'Ollama).

---

## 5. Utilisation et test

### Lancer le serveur

```bash
source venv/bin/activate
python mapping_server.py   # serveur complet FastAPI + pipeline webcam, :8000
python main.py             # démo simple sans serveur HTTP (joconde uniquement, overlay FPS)
```

### Tester le pipeline souvenir sur des images fixes

`test_souvenir.py` prend un dossier d'images (visages simulés) et produit une
photo souvenir :

```bash
python test_souvenir.py --artwork joconde --frames_dir test_frames/ --gender neutral
```

Arguments : `--artwork {joconde|vangogh|courbet}`, `--frames_dir <dossier>`,
`--gender {male|female|neutral}` (défaut : estimation automatique), `--output
<chemin>`. Le script logue le score du meilleur frame, la méthode finale, le temps
total et le pic VRAM.

### Tester le buffer webcam → souvenir (headless)

`test_session_souvenir.py` **simule le flux C920** en poussant de vraies frames
dans le buffer de session, puis déroule tout le chemin (buffer → meilleur frame →
`generate_souvenir` → fichier → endpoint → purge) sans webcam ni boucle de rendu :

```bash
python test_session_souvenir.py --frames_dir test_frames/ --artwork joconde --feed_seconds 5
```

Arguments : `--frames_dir`, `--artwork`, `--feed_seconds` (durée d'alimentation du
buffer avant la capture).

### Déclencher une session en conditions réelles (via l'API)

Une **session** bufferise les frames du visiteur pendant qu'il est là, sélectionne
le meilleur candidat (scoring EAR + frontalité, top-K en mémoire) et génère le
souvenir à la fin (à `SESSION_TIME`, 300 s par défaut, ou sur capture explicite).
Le buffer est purgé au démarrage et en fin de session — aucun visage ne fuite d'un
visiteur au suivant.

```bash
# Démarrer une session (optionnellement en fixant l'œuvre)
curl -X POST "http://192.168.0.10:8000/command/session/start?artwork=joconde"

# Déclencher la capture immédiatement (sinon auto après SESSION_TIME)
curl -X POST "http://192.168.0.10:8000/command/session/capture"

# Suivre l'état : active → generating → resultReady
curl "http://192.168.0.10:8000/command/session/status"

# Récupérer l'image générée
curl -o souvenir.jpg "http://192.168.0.10:8000/souvenir/<session_id>"
```

Les autres commandes du serveur (bascule d'œuvre, freeze/resume, âge, narration
start/pause/resume/stop, capture d'un frame animé) suivent la même convention
`POST /command/...`.

---

## 6. Autres composants

### Dialogue LLM (`test_llm_dialogue.py`)

Backend **Ollama** en local (`http://localhost:11434`), modèle **Mistral 7B**
(`mistral:7b-instruct-v0.3-q4_K_M`). Chaque œuvre a un **system prompt** qui la
fait rester en permanence dans son personnage. Particularités du prompt :

- réponses **très courtes** (1–2 phrases, ≤ 40 mots) car dites à voix haute devant
  un visiteur pressé, pas lues sur écran ;
- **décline toute question hors sujet** (actualité, politique, météo, questions
  personnelles…) même si la réponse est connue, puis recentre sur l'art/l'artiste ;
- ne révèle jamais être une IA, et **refuse de changer de voix/ton/rôle** demandé
  par le visiteur (pirate, robot, imitation…) ;
- instruction d'âge adaptée (`enfant` / `ado` / `adulte`).

Le script est une **boucle de test terminal** pour valider les réponses avant
intégration à `mapping_server.py` (`/switch <œuvre>` pour changer d'œuvre).

### STT — reconnaissance vocale (`test_stt_server.py`)

Serveur FastAPI autonome (port **8001**) avec **faster-whisper** (`small`, CUDA
FP16). Le visiteur ouvre l'URL affichée **depuis son téléphone** (même réseau
local), appuie pour parler ; l'audio est envoyé à `POST /transcribe` et transcrit
en français.

Pourquoi un serveur dédié et le micro du téléphone ? **Le micro physique de la
C920 a été retiré** (contrainte d'installation) : on utilise donc le micro du
téléphone du visiteur comme entrée vocale, découplé du reste du pipeline.

### Contrôle du vérin (réglage de hauteur)

Le vérin **VEVOR** est piloté par un **Arduino Mega + driver IBT-2**, commandé en
série depuis le Raspberry Pi. Le code de contrôle (`verin_controller.py`, firmware
Arduino) **ne fait pas partie de ce dépôt** — il vit côté Raspberry / contrôle
physique.

**Limitation connue** : le positionnement est **temporel** (durée de commande du
moteur), sans capteur physique de position — la hauteur atteinte est estimée par
le temps de course, pas mesurée.

---

## 7. Roadmap / travaux en cours

Souvenir :

- **Alignement guidé** du visage (développé par Pierre) — intégration prévue via
  un fichier séparé, pour améliorer la fidélité de projection.
- **Endpoint email sécurisé** pour envoyer le souvenir au visiteur.
- **Consentement RGPD** avant capture/envoi.
- **Cadres décoratifs** finalisés autour de l'image souvenir.
- **Genre collecté côté site** (formulaire visiteur) plutôt que via `--gender`.

Répartition des rôles :

- **IA / backend / GPU** (ce dépôt, mapping + souvenir + LLM).
- **Web / physique / communication** (Pierre) — site Artimir-Web, contrôle
  physique, aspects communication/marque.

---

## 8. Dépôts et machines liés

| Élément | Où | Rôle |
|---------|----|----|
| **FaceMappin** (ce dépôt) | PC Linux GPU (`192.168.0.10`) | Backend IA, mapping temps réel, pipeline souvenir, LLM/STT |
| **Artimir-Web** (`abdulalfayad78/Artimir-V2`) | Serveur web / Cloudflare Tunnel → `artimir.org` | Interface visiteur (Node / React) |
| Contrôle Raspberry Pi (`app.py`, `verin_controller.py`, firmware Arduino) | Raspberry Pi (`192.168.0.20`) | Affichage, audio, GPIO, contrôle du vérin |
