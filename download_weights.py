import os
import shutil
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm

def download_liveportrait_models(weights_dir="./weights"):
    """
    Télécharge automatiquement les modèles spécifiques de LivePortrait 
    depuis Hugging Face vers le dossier cible.
    """
    print(f"[*] Vérification du dossier de destination: {weights_dir}")
    os.makedirs(weights_dir, exist_ok=True)
    
    repo_id = "KlingTeam/LivePortrait" # Repository officiel reconnu
    
    # Dictionnaire de correspondance nom_local -> chemin_distant
    file_mappings = {
        "appearance_feature_extractor.pth": "liveportrait/base_models/appearance_feature_extractor.pth",
        "motion_extractor.pth": "liveportrait/base_models/motion_extractor.pth",
        "warping_module.pth": "liveportrait/base_models/warping_module.pth",
        "spade_generator.pth": "liveportrait/base_models/spade_generator.pth",
        "stitching_retargeting_module.pth": "liveportrait/retargeting_models/stitching_retargeting_module.pth"
    }

    print(f"[*] Téléchargement de {len(file_mappings)} fichiers depuis {repo_id}...")
    
    # Barre de progression globale tqdm
    for local_name, remote_path in tqdm(file_mappings.items(), desc="Téléchargement global"):
        target_dest = os.path.join(weights_dir, local_name)
        
        # Si le fichier est déjà là, on peut le passer (gain de temps si connexion coupée)
        if os.path.exists(target_dest):
            tqdm.write(f"[+] {local_name} déjà présent, on passe.")
            continue

        try:
            # hf_hub_download a lui-même une belle barre de progression interne par fichier
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
            )
            
            # hf_hub_download stocke le fichier dans un cache.
            # On utilise shutil.copy pour le placer proprement dans notre ./weights/ nom_du_fichier.pth
            shutil.copy(file_path, target_dest)
            tqdm.write(f"     [+] Sauvegardé : {target_dest}")
            
        except Exception as e:
            tqdm.write(f"     [-] Erreur rencontrée pour {local_name} : {e}")

if __name__ == "__main__":
    print("=== Artimir 2.0 - Agent 1: Model Downloader ===")
    download_liveportrait_models()
    print("=== Téléchargement terminé ===")
