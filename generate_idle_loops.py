"""
Génération offline de vidéos "idle loop" (mouvements de tête légers + clignements)
pour chaque tableau, via des keypoints synthétiques générés par fonctions périodiques
(pas de vidéo driver filmée).
"""

import copy
import math
import os

import cv2

from ai_engine import PortraitAnimator

# ==============================
# PARAMETRES
# ==============================
FPS = 25

# Position (fraction du cycle) où démarre le clignement ; la largeur de la fenêtre
# est calculée dynamiquement dans generate_variant() pour garantir un minimum de
# frames réelles quelle que soit la durée de la variante.
BLINK_START = 0.45
BLINK_MIN_FRAMES = 10
BLINK_MIN_FACTOR = 0.35  # facteur appliqué à exp pendant le clignement (battement de cils, pas un flash)

# Indices (sur les 21 keypoints implicites) correspondant aux paupières/yeux.
# stitching_retargeting_module["eye"] (cf. liveportrait_src/src/live_portrait_wrapper.py
# retarget_eye) serait l'outil correct, mais il attend des landmarks 2D dans le schéma
# à 203 points propre à LivePortrait (calc_eye_close_ratio) — schéma que ce repo ne
# produit pas (mapping_server.py utilise MediaPipe FaceMesh, indexation incompatible).
#
# Fallback : identifiés empiriquement en inspectant kp_info["kp"] (canonique, avant
# rotation/exp) sur les 3 portraits (joconde, vangogh, courbet) — 11/12/13 (oeil
# gauche) et 14/15/16 (oeil droit) forment deux clusters serrés, symétriques,
# nettement distincts du cluster bouche/lèvres (17-20). Au sein de CHAQUE oeil, le
# tri par y (négatif = vers le haut) est identique et stable sur les 3 tableaux :
#   11 / 14 = point le plus haut  (paupière haute)
#   12 / 15 = point intermédiaire (~iris, laissé inchangé)
#   13 / 16 = point le plus bas   (paupière basse)
EYE_KP_INDICES = [11, 12, 13, 14, 15, 16]
UPPER_EYE_KP_INDICES = [11, 14]
LOWER_EYE_KP_INDICES = [13, 16]

# Deux stratégies de clignement, calibrées par tableau via test_blink_calibration.py :
# - "symmetric" : paupière haute descend (+y), paupière basse monte (-y) — ferme
#   vraiment l'oeil au lieu de translater tout le cluster. Fonctionne bien sur
#   vangogh (amount ~0.035 — 0.06 calibré sur planche statique fusionnait les
#   paupières en usage réel, cf. BLINK_CONFIG). Sur courbet, la fermeture des yeux
#   sature très vite (déjà quasi complète à 0.01) alors que le mouvement de bouche
#   (corrélation apprise par le réseau, pas un chevauchement d'indices — kp confirme
#   yeux/bouche bien séparés géométriquement) croît à peu près linéairement avec
#   amount : au-delà de ~0.015-0.018 la bouche s'entrouvre visiblement. D'où un
#   amount bas (0.01) qui ferme déjà l'oeil sans perturber la bouche.
# - "uniform" : translation des 6 indices dans le même sens (ancienne approche).
#   Donne un effet "regard qui descend" plutôt qu'une vraie fermeture, MAIS c'est
#   la seule qui ne casse pas le rendu sur la Joconde : la stratégie "symmetric"
#   échoue structurellement sur ce portrait (testé jusqu'à amount=0.4 — l'oeil ne se
#   ferme jamais proprement, le visage se déforme progressivement à la place).
BLINK_CONFIG = {
    "joconde": {"strategy": "uniform", "amount": 0.02},
    "vangogh": {"strategy": "symmetric", "amount": 0.035},
    "courbet": {"strategy": "symmetric", "amount": 0.01},
}
DEFAULT_BLINK_CONFIG = {"strategy": "symmetric", "amount": 0.025}

# Par tableau : liste de variantes (duration_sec, amplitude_yaw, amplitude_pitch, amplitude_roll, phase_offset)
VARIANTS = {
    "joconde": [
        (3.5, 1.5, 0.8, 0.3, 0.0),
        (4.0, 2.0, 1.1, 0.5, 0.6),
        (4.5, 2.5, 1.5, 0.8, 1.2),
    ],
    "vangogh": [
        (3.8, 1.8, 1.0, 0.4, 0.2),
        (4.2, 2.2, 1.3, 0.6, 0.9),
        (4.5, 2.5, 1.5, 0.8, 1.5),
    ],
    "courbet": [
        (3.5, 1.6, 0.9, 0.35, 0.4),
        (4.0, 2.0, 1.2, 0.55, 1.0),
        (4.5, 2.4, 1.4, 0.75, 1.7),
    ],
}

ARTWORKS = {
    "joconde": "joconde.jpg",
    "vangogh": "vangogh.jpg",
    "courbet": "courbet.jpg",
}

OUTPUT_DIR = "idle_loops"
RESOLUTION = 512


def crop_square_face_zone(img):
    h, w = img.shape[:2]
    crop_size = min(h, w)

    start_x = (w - crop_size) // 2
    start_y = max(0, int((h - crop_size) * 0.2))

    cropped = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cv2.resize(cropped, (RESOLUTION, RESOLUTION))


def blink_envelope(phase, blink_start, blink_end):
    """Retourne un facteur d'atténuation d'expression dans [BLINK_MIN_FACTOR, 1.0]
    en fonction de la position dans le cycle (phase dans [0, 1)). En dehors de la
    fenêtre de clignement, retourne 1.0 (pas d'atténuation)."""
    if not (blink_start <= phase <= blink_end):
        return 1.0

    window_progress = (phase - blink_start) / (blink_end - blink_start)
    ease = 1.0 - math.sin(math.pi * window_progress)  # 1 -> 0 -> 1, doux aux bords
    return BLINK_MIN_FACTOR + (1.0 - BLINK_MIN_FACTOR) * ease


def clone_kp_info(kp_info):
    cloned = {}
    for k, v in kp_info.items():
        if hasattr(v, "clone"):
            cloned[k] = v.clone()
        else:
            cloned[k] = copy.deepcopy(v)
    return cloned


def generate_variant(animator, artwork_name, variant_idx, duration_sec, amp_yaw, amp_pitch, amp_roll, phase_offset):
    total_frames = int(duration_sec * FPS)
    period = total_frames / FPS

    blink_config = BLINK_CONFIG.get(artwork_name, DEFAULT_BLINK_CONFIG)

    out_path = os.path.join(OUTPUT_DIR, f"{artwork_name}_idle_{variant_idx}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (RESOLUTION, RESOLUTION))

    phase_shift_rad = 2 * math.pi * phase_offset

    blink_frac_width = max(0.05, BLINK_MIN_FRAMES / total_frames)
    blink_start = BLINK_START
    blink_end = blink_start + blink_frac_width

    for frame_idx in range(total_frames):
        t = frame_idx / FPS
        phase = (frame_idx / total_frames + phase_offset) % 1.0

        synthetic_kp_info = clone_kp_info(animator.source_kp_info)

        # phase_shift_rad est une constante additive : n'affecte pas la continuité
        # du bouclage (sin(2*pi + shift) == sin(shift) == valeur en t=0).
        synthetic_kp_info["yaw"] = synthetic_kp_info["yaw"] + amp_yaw * math.sin(2 * math.pi * t / period + phase_shift_rad)
        synthetic_kp_info["pitch"] = synthetic_kp_info["pitch"] + amp_pitch * math.sin(2 * math.pi * t / (period * 1.3) + phase_shift_rad)
        synthetic_kp_info["roll"] = synthetic_kp_info["roll"] + amp_roll * math.sin(2 * math.pi * t / (period * 0.7) + phase_shift_rad)

        blink_factor = blink_envelope(phase, blink_start, blink_end)
        if blink_factor < 1.0:
            # Additif, pas multiplicatif (cf. BLINK_CONFIG ci-dessus). Stratégie et
            # amplitude dépendent du tableau — cf. commentaire sur BLINK_CONFIG.
            closing = blink_config["amount"] * (1.0 - blink_factor)
            if blink_config["strategy"] == "symmetric":
                synthetic_kp_info["exp"][:, UPPER_EYE_KP_INDICES, 1] += closing
                synthetic_kp_info["exp"][:, LOWER_EYE_KP_INDICES, 1] -= closing
            else:
                synthetic_kp_info["exp"][:, EYE_KP_INDICES, 1] += closing

        frame = animator.generate_frame_from_kp_info(
            synthetic_kp_info, motion_scale=1.0, expression_scale=0.8
        )
        writer.write(frame)

        if frame_idx % 25 == 0 or frame_idx == total_frames - 1:
            print(f"[{artwork_name}][variant {variant_idx}] frame {frame_idx + 1}/{total_frames}")

    writer.release()
    print(f"[{artwork_name}][variant {variant_idx}] écrit -> {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    animator = PortraitAnimator()

    for artwork_name, image_filename in ARTWORKS.items():
        if not os.path.exists(image_filename):
            print(f"[!] {image_filename} introuvable, tableau {artwork_name} ignoré.")
            continue

        print(f"=== {artwork_name} ===")

        raw_source = cv2.imread(image_filename)
        source_img = crop_square_face_zone(raw_source)

        animator.reset_cache()
        animator.init_source(source_img)

        for variant_idx, (duration_sec, amp_yaw, amp_pitch, amp_roll, phase_offset) in enumerate(
            VARIANTS.get(artwork_name, [])
        ):
            generate_variant(
                animator, artwork_name, variant_idx,
                duration_sec, amp_yaw, amp_pitch, amp_roll, phase_offset
            )

    print("Terminé.")


if __name__ == "__main__":
    main()
