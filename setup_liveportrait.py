"""
Récupère la copie vendored de LivePortrait (liveportrait_src/) depuis le
repository officiel GitHub, pour ne pas avoir à la committer/envoyer.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/KwaiVGI/LivePortrait.git"
TARGET_DIR = Path("./liveportrait_src")


def setup_liveportrait(target_dir: Path = TARGET_DIR):
    if target_dir.exists():
        print(f"[+] {target_dir} déjà présent, on passe.")
        return

    print(f"[*] Clonage de {REPO_URL} dans {target_dir}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(target_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[-] Échec du clonage: {e}")
        sys.exit(1)

    git_dir = target_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)

    print(f"[+] LivePortrait récupéré dans {target_dir}")


if __name__ == "__main__":
    print("=== Artimir 2.0 - Setup LivePortrait source ===")
    setup_liveportrait()
    print("=== Terminé ===")
