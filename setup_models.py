"""
Agent 1: DevOps & Environment - Modèle Downloader
Télécharge automatiquement les modèles de LivePortrait (ou fallback)
"""
import os
from huggingface_hub import snapshot_download

def download_models(weights_dir="./weights"):
    """
    Télécharge les poids pré-entraînés depuis HuggingFace.
    """
    print(f"[*] Vérification du dossier de destination: {weights_dir}")
    os.makedirs(weights_dir, exist_ok=True)
    
    repo_id = "KwaiVGI/LivePortrait" # Repository officiel (Exemple)
    
    print(f"[*] Téléchargement des poids du modèle depuis le repository: {repo_id}...")
    try:
        # Télécharge uniquement les fichiers .pth ou .onnx pertinents
        local_dir = snapshot_download(
            repo_id=repo_id, 
            local_dir=weights_dir,
            allow_patterns=["*.pth", "*.pt", "*.onnx", "*.json"],
            ignore_patterns=["*.md", "*.txt", ".git*"]
        )
        print(f"[+] Succès: Poids téléchargés et stockés dans {local_dir}")
        
    except Exception as e:
        print(f"[-] Erreur lors du téléchargement: {e}")
        print("[!] Assurez-vous d'avoir une connexion internet et que le repository existe.")

if __name__ == "__main__":
    print("=== Artimir 2.0 - Préparation des Modèles ===")
    download_models()
