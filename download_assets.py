"""
Télécharge les assets propres au projet (portraits + vidéo de narration)
depuis le dataset HuggingFace Henri-maker/artimir, pour ne pas avoir à les
committer/envoyer avec le code.
"""
import os
import shutil
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "Henri-maker/artimir"

ASSETS = [
    "joconde.jpg",
    "vangogh.jpg",
    "courbet.jpg",
    "driving_narration.mp4",
]


def download_assets(target_dir: str = "."):
    os.makedirs(target_dir, exist_ok=True)
    print(f"[*] Téléchargement de {len(ASSETS)} fichiers depuis {REPO_ID}...")

    for filename in tqdm(ASSETS, desc="Téléchargement assets"):
        target_dest = os.path.join(target_dir, filename)

        if os.path.exists(target_dest):
            tqdm.write(f"[+] {filename} déjà présent, on passe.")
            continue

        try:
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
            )
            shutil.copy(file_path, target_dest)
            tqdm.write(f"     [+] Sauvegardé : {target_dest}")
        except Exception as e:
            tqdm.write(f"     [-] Erreur rencontrée pour {filename} : {e}")


if __name__ == "__main__":
    print("=== Artimir 2.0 - Download assets projet ===")
    download_assets()
    print("=== Téléchargement terminé ===")
