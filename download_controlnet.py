"""
Télécharge le ControlNet SDXL depth utilisé par souvenir_generator.py
(generate_face_via_diffdiff) pour contraindre la structure du tableau
pendant la re-diffusion complète (Differential Diffusion).

Modèle : diffusers/controlnet-depth-sdxl-1.0-mid (variante distillée "mid")
- fp16 : 0.55 Go (vs 2.50 Go pour la variante complète, 0.32 Go pour -small)
- choisi comme compromis qualité/VRAM pour une RTX 5060 8 Go

Destination : models/controlnet_depth_sdxl/

Usage :
    ./venv/bin/python download_controlnet.py
"""

import os

from huggingface_hub import snapshot_download

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLNET_DIR = os.path.join(_BASE_DIR, "models", "controlnet_depth_sdxl")
REPO_ID = "diffusers/controlnet-depth-sdxl-1.0-mid"

# fp16 uniquement (0.55 Go) — le pipeline charge en torch.float16.
ALLOW_PATTERNS = [
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
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
    os.makedirs(CONTROLNET_DIR, exist_ok=True)
    print(f"=== Téléchargement ControlNet depth SDXL -> {CONTROLNET_DIR} ===")
    print(f"[*] Dépôt : {REPO_ID} (fp16, ~0.55 Go)")
    try:
        snapshot_download(
            repo_id=REPO_ID, local_dir=CONTROLNET_DIR,
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[-] Échec du téléchargement : {exc}")
        raise SystemExit(1) from exc

    size_gb = _dir_size_bytes(CONTROLNET_DIR) / 1024 ** 3
    print(f"[+] Terminé. Taille totale locale : {size_gb:.2f} Go dans {CONTROLNET_DIR}")


if __name__ == "__main__":
    main()
