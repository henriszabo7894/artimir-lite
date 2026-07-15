"""
Télécharge le modèle SDXL Inpainting utilisé par souvenir_generator.py.

Modèle : diffusers/stable-diffusion-xl-1.0-inpainting-0.1
Destination : models/sdxl_inpainting/

Usage :
    ./venv/bin/python download_sdxl.py

Affiche la taille totale téléchargée. Le pipeline le charge ensuite depuis ce
cache local (aucun nouvel appel réseau au runtime).
"""

import os

from huggingface_hub import snapshot_download

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SDXL_DIR = os.path.join(_BASE_DIR, "models", "sdxl_inpainting")
SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def _dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = os.path.join(root, name)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def main():
    os.makedirs(SDXL_DIR, exist_ok=True)
    print(f"=== Téléchargement SDXL Inpainting -> {SDXL_DIR} ===")
    print(f"[*] Modèle : {SDXL_MODEL_ID}")

    # variant fp16 en priorité (plus léger) ; on récupère aussi les configs/tokenizers.
    allow = [
        "*.json", "*.txt", "*.model",
        "*fp16*.safetensors",
    ]
    try:
        snapshot_download(
            repo_id=SDXL_MODEL_ID, local_dir=SDXL_DIR, allow_patterns=allow,
        )
        # Repli : si aucun poids fp16 n'existe, récupérer les safetensors standard.
        if not any(
            f.endswith(".safetensors") for f in _all_files(SDXL_DIR)
        ):
            print("[*] Aucun poids fp16 trouvé — récupération des safetensors standard.")
            snapshot_download(
                repo_id=SDXL_MODEL_ID, local_dir=SDXL_DIR,
                allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[-] Échec du téléchargement : {exc}")
        raise SystemExit(1) from exc

    size_gb = _dir_size_bytes(SDXL_DIR) / 1024 ** 3
    print(f"[+] Terminé. Taille totale locale : {size_gb:.2f} Go dans {SDXL_DIR}")


def _all_files(path):
    for root, _dirs, files in os.walk(path):
        for name in files:
            yield name


if __name__ == "__main__":
    main()
