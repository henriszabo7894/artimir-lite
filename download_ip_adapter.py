"""
Télécharge le checkpoint IP-Adapter (plus-face SDXL, ViT-H) + son encodeur
d'image CLIP, utilisés par souvenir_generator.py (pipeline principal).

Modèle : h94/IP-Adapter
- sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors
- models/image_encoder/*  (CLIP ViT-H, PAS FaceID, PAS insightface)

Destination : models/ip_adapter/  (structure du dépôt préservée pour que
pipe.load_ip_adapter(..., image_encoder_folder="../models/image_encoder") résolve
l'encodeur).

Usage :
    ./venv/bin/python download_ip_adapter.py
"""

import os

from huggingface_hub import snapshot_download

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IP_ADAPTER_DIR = os.path.join(_BASE_DIR, "models", "ip_adapter")
REPO_ID = "h94/IP-Adapter"

ALLOW_PATTERNS = [
    "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
    "models/image_encoder/config.json",
    "models/image_encoder/*.safetensors",
    "models/image_encoder/*.bin",
]


def _dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = os.path.join(root, name)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def main():
    os.makedirs(IP_ADAPTER_DIR, exist_ok=True)
    print(f"=== Téléchargement IP-Adapter -> {IP_ADAPTER_DIR} ===")
    print(f"[*] Dépôt : {REPO_ID}")
    try:
        snapshot_download(
            repo_id=REPO_ID, local_dir=IP_ADAPTER_DIR,
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[-] Échec du téléchargement : {exc}")
        raise SystemExit(1) from exc

    size_gb = _dir_size_bytes(IP_ADAPTER_DIR) / 1024 ** 3
    print(f"[+] Terminé. Taille totale locale : {size_gb:.2f} Go dans {IP_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
