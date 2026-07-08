"""
Calibration de EYE_CLOSE_AMOUNT (clignement additif haut/bas, cf. generate_idle_loops.py).

Echelle des coordonnées kp/exp (voir liveportrait_src/src/) :
- make_coordinate_grid (modules/util.py) construit la grille de sampling utilisée par
  dense_motion.py en normalisant chaque axe sur [-1, 1] ("2 * (x/(w-1)) - 1"), quelle
  que soit la résolution réelle : kp_driving/kp_source vivent dans ce même repère
  [-1, 1], PAS en pixels ni en coordonnées "256x256 fixe".
- transform_keypoint applique : kp = (kp_canonique @ rot + exp) * scale + t. Comme
  notre delta est ajouté à exp (avant ce *scale), son effet en pixels final est déjà
  proportionnel à ce même scale — le ratio déplacement/hauteur-oeil se conserve tel
  quel, pas besoin de connaître le scale en valeur absolue (voir calibration ci-dessous).

Logique haut/bas (cf. generate_idle_loops.py pour le détail du diagnostic) :
- Au sein de chaque oeil (cluster EYE_KP_INDICES), le point le plus haut
  (UPPER_EYE_KP_INDICES = [11, 14]) et le plus bas (LOWER_EYE_KP_INDICES = [13, 16])
  sont déplacés en sens opposé : haute +y (descend), basse -y (monte) — pour que les
  paupières se rapprochent au lieu de translater tout le cluster (ce qui déplaçait
  l'iris et donnait un effet de regard qui descend plutôt qu'un clignement).
- Le point intermédiaire (12, 15, ~iris) n'est pas touché.

Ce script :
1. Mesure la hauteur d'oeil (écart haut/bas en unités kp) sur les 3 portraits.
2. Génère une planche de comparaison (5 frames fixes côte à côte) PAR TABLEAU pour
   EYE_CLOSE_AMOUNT = 0.01, 0.015, 0.02, 0.025, 0.03, avec la logique haut/bas.
"""

import copy

import cv2
import numpy as np

from ai_engine import PortraitAnimator
from generate_idle_loops import (
    crop_square_face_zone,
    UPPER_EYE_KP_INDICES,
    LOWER_EYE_KP_INDICES,
)

TEST_AMOUNTS = [0.01, 0.015, 0.02, 0.025, 0.03]

# Multiplicateur (1.0 - blink_factor) utilisé en production au pic du clignement
# (BLINK_MIN_FACTOR = 0.35 dans generate_idle_loops.py -> 1.0 - 0.35 = 0.65).
PRODUCTION_PEAK_MULTIPLIER = 0.65

ARTWORKS = {
    "joconde": "joconde.jpg",
    "vangogh": "vangogh.jpg",
    "courbet": "courbet.jpg",
}


def clone_kp_info(kp_info):
    cloned = {}
    for k, v in kp_info.items():
        if hasattr(v, "clone"):
            cloned[k] = v.clone()
        else:
            cloned[k] = copy.deepcopy(v)
    return cloned


def measure_eye_height_kp_units(source_kp_info):
    kp = source_kp_info["kp"][0].detach().cpu().numpy()  # (21, 3), canonique

    left_upper, left_lower = [i for i in UPPER_EYE_KP_INDICES if kp[i][0] < 0], [i for i in LOWER_EYE_KP_INDICES if kp[i][0] < 0]
    right_upper, right_lower = [i for i in UPPER_EYE_KP_INDICES if kp[i][0] >= 0], [i for i in LOWER_EYE_KP_INDICES if kp[i][0] >= 0]

    left_span = kp[left_lower[0], 1] - kp[left_upper[0], 1] if left_upper and left_lower else 0.0
    right_span = kp[right_lower[0], 1] - kp[right_upper[0], 1] if right_upper and right_lower else 0.0

    return float((left_span + right_span) / 2.0), float(left_span), float(right_span)


def render_with_amount(animator, eye_close_amount, peak_multiplier):
    synthetic_kp_info = clone_kp_info(animator.source_kp_info)
    closing = eye_close_amount * peak_multiplier
    synthetic_kp_info["exp"][:, UPPER_EYE_KP_INDICES, 1] += closing
    synthetic_kp_info["exp"][:, LOWER_EYE_KP_INDICES, 1] -= closing

    return animator.generate_frame_from_kp_info(
        synthetic_kp_info, motion_scale=1.0, expression_scale=0.8
    )


def generate_board(animator, artwork_name, image_filename):
    raw_source = cv2.imread(image_filename)
    source_img = crop_square_face_zone(raw_source)
    animator.reset_cache()
    animator.init_source(source_img)

    avg_span, left_span, right_span = measure_eye_height_kp_units(animator.source_kp_info)
    print(f"[calib][{artwork_name}] hauteur oeil (unités kp) — gauche: {left_span:.4f}  "
          f"droite: {right_span:.4f}  moyenne: {avg_span:.4f}")
    print(f"[calib][{artwork_name}] fermeture totale (2x amount) recommandée pour 15-20% : "
          f"{0.15 * avg_span:.4f} .. {0.20 * avg_span:.4f}  "
          f"(soit amount ~ {0.075 * avg_span:.4f} .. {0.10 * avg_span:.4f})")

    frames = []
    for amount in TEST_AMOUNTS:
        frame = render_with_amount(animator, amount, PRODUCTION_PEAK_MULTIPLIER)
        label = f"amount={amount}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        frames.append(frame)
        print(f"[calib][{artwork_name}] rendu -> {label}")

    combined = np.hstack(frames)
    out_path = f"blink_calibration_{artwork_name}.png"
    cv2.imwrite(out_path, combined)
    print(f"[calib][{artwork_name}] planche de comparaison écrite -> {out_path}")


def main():
    animator = PortraitAnimator()

    for artwork_name, image_filename in ARTWORKS.items():
        print(f"=== {artwork_name} ===")
        generate_board(animator, artwork_name, image_filename)


if __name__ == "__main__":
    main()
