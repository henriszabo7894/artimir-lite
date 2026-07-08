"""
ARTIMIR - Effet de parallaxe temps réel (couches de profondeur)

Utilise la carte de profondeur pré-calculée d'un tableau (voir generate_depth_map.py)
et la position de tête du visiteur pour décaler légèrement les pixels de l'image,
donnant une illusion de profondeur/fenêtre 3D.

Principe : les zones proches (valeur de profondeur haute) se déplacent plus,
les zones lointaines (valeur basse) se déplacent moins - comme en regardant
à travers une fenêtre réelle.
"""

import numpy as np
import cv2


class EffetParallaxe:
    def __init__(self, chemin_depth_npy: str, intensite_px: float = 15.0, lissage: float = 0.3, flou_depth_kernel: int = 51):
        """
        chemin_depth_npy : chemin vers le fichier .npy généré par generate_depth_map.py
        intensite_px     : déplacement max en pixels pour la zone la plus proche
                            (subtil ~5-8px, prononcé ~20-30px, à ajuster selon le ressenti)
        lissage           : 0 = pas de lissage temporel (réactif mais peut trembler),
                            1 = lissage max (doux mais lent à suivre). 0.3 est un bon départ.
        flou_depth_kernel : taille du flou gaussien appliqué à la carte de profondeur pour
                            adoucir les transitions brutales (réduit les déchirures visibles
                            aux bords des objets). Doit être impair. 0 = pas de flou.
        """
        depth_brute = np.load(chemin_depth_npy).astype(np.float32)
        self._depth_brute = depth_brute
        self.flou_depth_kernel = flou_depth_kernel
        self.depth_map = self._appliquer_flou(depth_brute, flou_depth_kernel)

        self.intensite_px = intensite_px
        self.lissage = lissage

        self.depth_map_resized_cache = {}

        self.offset_x_lisse = 0.0
        self.offset_y_lisse = 0.0

    @staticmethod
    def _appliquer_flou(depth_brute, kernel):
        if kernel <= 0:
            return depth_brute
        k = kernel if kernel % 2 == 1 else kernel + 1
        return cv2.GaussianBlur(depth_brute, (k, k), 0)

    def set_flou(self, kernel: int):
        self.flou_depth_kernel = kernel
        self.depth_map = self._appliquer_flou(self._depth_brute, kernel)
        self.depth_map_resized_cache = {}  # invalide le cache, la depth map a changé

    def set_intensite(self, intensite_px: float):
        self.intensite_px = intensite_px

    def _obtenir_depth_map_pour_taille(self, hauteur: int, largeur: int):
        cle = (hauteur, largeur)
        if cle not in self.depth_map_resized_cache:
            resized = cv2.resize(self.depth_map, (largeur, hauteur), interpolation=cv2.INTER_LINEAR)
            self.depth_map_resized_cache[cle] = resized
        return self.depth_map_resized_cache[cle]

    def appliquer(self, frame_bgr, yaw_degres: float, pitch_degres: float, yaw_max_degres: float = 40.0, pitch_max_degres: float = 30.0):
        """
        frame_bgr    : image de sortie du pipeline (après LivePortrait), à décaler
        yaw_degres   : rotation horizontale de tête détectée (depuis le motion_extractor)
        pitch_degres : rotation verticale de tête détectée
        yaw_max/pitch_max : valeurs utilisées pour normaliser -1..1 (adapter selon l'amplitude réelle observée)

        Retourne l'image avec l'effet de parallaxe appliqué.
        """
        hauteur, largeur = frame_bgr.shape[:2]
        depth_map = self._obtenir_depth_map_pour_taille(hauteur, largeur)

        # Normalisation de la position de tête en -1..1
        offset_x_cible = np.clip(yaw_degres / yaw_max_degres, -1.0, 1.0)
        offset_y_cible = np.clip(pitch_degres / pitch_max_degres, -1.0, 1.0)

        # Lissage temporel pour éviter les tremblements
        self.offset_x_lisse = self.lissage * self.offset_x_lisse + (1 - self.lissage) * offset_x_cible
        self.offset_y_lisse = self.lissage * self.offset_y_lisse + (1 - self.lissage) * offset_y_cible

        # Champ de déplacement : proportionnel à la profondeur (proche = bouge plus)
        dx = depth_map * self.offset_x_lisse * self.intensite_px
        dy = depth_map * self.offset_y_lisse * self.intensite_px

        map_x, map_y = np.meshgrid(np.arange(largeur, dtype=np.float32), np.arange(hauteur, dtype=np.float32))

        # Le signe (-dx) donne l'effet fenêtre classique : quand la tête va à droite,
        # les objets proches semblent glisser vers la gauche relativement au fond.
        # Si l'effet paraît inversé au test, remplacer (map_x - dx) par (map_x + dx).
        map_x_final = (map_x - dx).astype(np.float32)
        map_y_final = (map_y - dy).astype(np.float32)

        resultat = cv2.remap(frame_bgr, map_x_final, map_y_final, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Léger zoom + recadrage central : cache les artefacts de bord (pixels étirés/répétés)
        # créés par le décalage. La marge est proportionnelle à l'intensité utilisée.
        marge_ratio = min(0.15, (self.intensite_px / largeur) * 2.5)
        if marge_ratio > 0.001:
            marge_x = int(largeur * marge_ratio)
            marge_y = int(hauteur * marge_ratio)
            resultat_recadre = resultat[marge_y:hauteur - marge_y, marge_x:largeur - marge_x]
            resultat = cv2.resize(resultat_recadre, (largeur, hauteur), interpolation=cv2.INTER_LINEAR)

        return resultat


# --- Exemple d'intégration dans mapping_server.py ---
#
# from parallax_effect import EffetParallaxe
#
# parallaxe_joconde = EffetParallaxe("joconde_depth.npy", intensite_px=15.0)
#
# # Dans la boucle de génération de frame, après avoir obtenu yaw/pitch du motion_extractor :
# frame_finale = parallaxe_joconde.appliquer(frame_liveportrait, yaw, pitch)
#
# # Pour tester subtil vs prononcé sans relancer le serveur, exposer un endpoint :
# # @app.post("/api/parallaxe/intensite")
# # def set_intensite(px: float):
# #     parallaxe_joconde.set_intensite(px)
# #     return {"intensite_px": px}
