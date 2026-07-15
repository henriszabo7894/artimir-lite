"""
Test manuel du générateur de souvenir.

Usage :
    python test_souvenir.py --artwork joconde --frames_dir test_frames/

- Charge test_frames/*.jpg (et .png/.jpeg) comme frames visiteur simulés.
- Appelle SouvenirGenerator.generate_souvenir().
- Sauvegarde output/souvenir_test.jpg.
- Affiche le score du meilleur frame sélectionné.
"""

import argparse
import glob
import os
import sys
import time

import cv2

from souvenir_generator import SouvenirGenerator

ARTWORK_FILES = {
    "joconde": "joconde.jpg",
    "vangogh": "vangogh.jpg",
    "courbet": "courbet.jpg",
}

_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_frames(frames_dir):
    paths = []
    for pattern in _IMAGE_GLOBS:
        paths.extend(glob.glob(os.path.join(frames_dir, pattern)))
    paths = sorted(set(paths))

    frames = []
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)
        else:
            print(f"[WARN] Image illisible ignorée : {path}")
    return frames, paths


def main():
    parser = argparse.ArgumentParser(description="Test du générateur de souvenir Artimir")
    parser.add_argument("--artwork", default="joconde",
                        help="Nom de l'œuvre (joconde | vangogh | courbet)")
    parser.add_argument("--frames_dir", default="test_frames/",
                        help="Dossier contenant les frames visiteur simulés")
    parser.add_argument("--output", default="output/souvenir_test.jpg",
                        help="Chemin de sortie de l'image souvenir")
    parser.add_argument("--gender", default=None,
                        choices=("male", "female", "neutral"),
                        help="Force le genre injecté dans les prompts SDXL "
                             "(défaut : estimation automatique MediaPipe)")
    args = parser.parse_args()

    artwork_file = ARTWORK_FILES.get(args.artwork, f"{args.artwork}.jpg")
    if not os.path.exists(artwork_file):
        print(f"[ERREUR] Image du tableau introuvable : {artwork_file}")
        sys.exit(1)
    artwork_bgr = cv2.imread(artwork_file)
    if artwork_bgr is None:
        print(f"[ERREUR] Impossible de lire le tableau : {artwork_file}")
        sys.exit(1)

    frames, paths = load_frames(args.frames_dir)
    if not frames:
        print(f"[ERREUR] Aucun frame trouvé dans : {args.frames_dir}")
        sys.exit(1)
    print(f"[*] {len(frames)} frame(s) chargé(s) depuis {args.frames_dir}")

    generator = SouvenirGenerator()
    try:
        best_frame, best_score = generator.find_best_frame(frames)
        best_index = next(
            (i for i, f in enumerate(frames) if f is best_frame), -1
        )
        best_name = os.path.basename(paths[best_index]) if best_index >= 0 else "?"
        print(f"[*] Meilleur frame : {best_name} (score {best_score:.3f})")

        start = time.perf_counter()
        final = generator.generate_souvenir(
            frames, args.artwork, artwork_bgr, gender_override=args.gender,
        )
        elapsed = time.perf_counter() - start

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cv2.imwrite(args.output, final)
        method = generator.last_method or "inconnue"
        print(f"[*] Souvenir généré -> {args.output} ({final.shape[1]}x{final.shape[0]})")
        print(f"[*] Méthode ayant produit le fichier final : {method}")
        print(f"[*] Temps total generate_souvenir : {elapsed:.1f} s")

        # Résultat diffdiff sauvegardé sous un nom distinct pour ne pas le
        # confondre avec un souvenir_test.jpg d'un run précédent/fallback.
        if method == "diffdiff":
            diffdiff_path = os.path.join(
                os.path.dirname(args.output) or ".", "souvenir_diffdiff_result.jpg",
            )
            cv2.imwrite(diffdiff_path, final)
            print(f"[*] Résultat diffdiff sauvegardé -> {diffdiff_path}")

        try:
            import torch
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"[*] Pic VRAM process (depuis le dernier reset) : {peak:.2f} Go")
        except ImportError:
            pass
    finally:
        generator.close()


if __name__ == "__main__":
    main()
