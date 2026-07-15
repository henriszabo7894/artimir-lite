"""
Télécharge les poids nécessaires au générateur de souvenir (souvenir_generator.py).

Deux jeux de poids :

1. AdaIN (transfert de style, optionnel) -> models/adain/
   - vgg_normalised.pth  (encodeur VGG19 "normalised")
   - decoder.pth         (décodeur AdaIN)
   Architecture attendue : celle de naoto0804/pytorch-AdaIN. Sans ces poids, le
   pipeline fonctionne quand même : l'étape de style est simplement ignorée.

2. FaceLandmarker de MediaPipe Tasks (REQUIS) -> models/face_landmarker.task
   Bundle officiel Google. Sans lui, aucune détection de visage n'est possible.
   Récupéré avec l'option --with-face-model.

⚠️  Sécurité : les .pth AdaIN sont chargés via torch.load. PyTorch >= 2.6 utilise
    weights_only=True par défaut (désérialisation restreinte aux tenseurs), ce qui
    limite le risque. Vérifiez néanmoins la source ci-dessous avant de l'utiliser.
    Le dépôt par défaut a été choisi car il expose exactement les deux fichiers
    attendus ; vous pouvez le remplacer par --repo / la source officielle
    naoto0804/pytorch-AdaIN (poids distribués via Google Drive).

Usage :
    python download_adain_weights.py                    # AdaIN seul
    python download_adain_weights.py --with-face-model  # AdaIN + FaceLandmarker
    python download_adain_weights.py --face-model-only  # FaceLandmarker seul
    python download_adain_weights.py --repo <hf_repo> --vgg <file> --decoder <file>
"""

import argparse
import os
import shutil
import urllib.request

from huggingface_hub import hf_hub_download

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAIN_DIR = os.path.join(_BASE_DIR, "models", "adain")
MODELS_DIR = os.path.join(_BASE_DIR, "models")

# Dépôt HF exposant les deux fichiers attendus (noms distants -> noms locaux).
DEFAULT_REPO = "BhupatiNadar/adain-style-transfer-models"
DEFAULT_VGG_REMOTE = "vgg_normalized.pth"
DEFAULT_DECODER_REMOTE = "decoder_final.pth"

# Modèle FaceLandmarker officiel (bucket Google MediaPipe).
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _download_hf(repo, remote_name, local_path):
    if os.path.exists(local_path):
        print(f"[+] {local_path} déjà présent, on passe.")
        return
    print(f"[*] Téléchargement {repo}:{remote_name} ...")
    cached = hf_hub_download(repo_id=repo, filename=remote_name)
    shutil.copy(cached, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"    [+] Sauvegardé : {local_path} ({size_mb:.1f} Mo)")


def download_adain(repo, vgg_remote, decoder_remote):
    os.makedirs(ADAIN_DIR, exist_ok=True)
    _download_hf(repo, vgg_remote, os.path.join(ADAIN_DIR, "vgg_normalised.pth"))
    _download_hf(repo, decoder_remote, os.path.join(ADAIN_DIR, "decoder.pth"))


def download_face_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, "face_landmarker.task")
    if os.path.exists(dest):
        print(f"[+] {dest} déjà présent, on passe.")
        return
    print(f"[*] Téléchargement du modèle FaceLandmarker officiel ...\n    {FACE_MODEL_URL}")
    urllib.request.urlretrieve(FACE_MODEL_URL, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"    [+] Sauvegardé : {dest} ({size_mb:.1f} Mo)")


def main():
    parser = argparse.ArgumentParser(description="Téléchargement des poids du souvenir")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help="Dépôt HuggingFace des poids AdaIN")
    parser.add_argument("--vgg", default=DEFAULT_VGG_REMOTE,
                        help="Nom distant du fichier VGG normalisé")
    parser.add_argument("--decoder", default=DEFAULT_DECODER_REMOTE,
                        help="Nom distant du fichier décodeur")
    parser.add_argument("--with-face-model", action="store_true",
                        help="Télécharger aussi le modèle FaceLandmarker (requis)")
    parser.add_argument("--face-model-only", action="store_true",
                        help="Télécharger uniquement le modèle FaceLandmarker")
    args = parser.parse_args()

    print("=== Artimir — Téléchargement des poids du générateur de souvenir ===")
    try:
        if args.face_model_only:
            download_face_model()
        else:
            download_adain(args.repo, args.vgg, args.decoder)
            if args.with_face_model:
                download_face_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[-] Erreur : {exc}")
        print("    Le pipeline reste fonctionnel sans AdaIN (style ignoré) ;")
        print("    en revanche le modèle FaceLandmarker est indispensable.")
        raise SystemExit(1) from exc
    print("=== Terminé ===")


if __name__ == "__main__":
    main()
