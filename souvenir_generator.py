"""
Générateur de photo souvenir Artimir.

Module autonome : ne dépend PAS de mapping_server.py et ne le modifie pas.
Prend une poignée de frames webcam d'un visiteur, sélectionne la meilleure,
projette le visage dans un tableau (warp + seamlessClone), applique un léger
transfert de style AdaIN, puis un cadre décoratif + filigrane Kinophos.

Pipeline complet : SouvenirGenerator.generate_souvenir(frames, artwork_name, artwork_bgr)

Le pipeline ne plante jamais : chaque étape est protégée individuellement et
dégrade proprement (log warning + résultat partiel) en cas d'échec.

Arborescence de données attendue (relative au dossier du projet) :
- souvenir_data/{artwork}_landmarks.json  -> landmarks de référence du tableau (auto-générés)
- souvenir_data/frame_counters.json       -> compteur de rotation des cadres par œuvre
- souvenir_frames/{artwork}_frame_{1,2,3}.png -> cadres PNG RGBA (optionnels)
- models/adain/vgg_normalised.pth, models/adain/decoder.pth -> poids AdaIN (optionnels)
"""

from __future__ import annotations

import gc
import json
import logging
import os

import cv2
import numpy as np

# Ce build de mediapipe (0.10.x) n'expose que l'API "Tasks" (pas mp.solutions.face_mesh).
# On utilise donc FaceLandmarker, dont la topologie de landmarks est identique à
# l'ancien FaceMesh (maillage 468 + iris), ce qui laisse tous les indices inchangés.
try:
    from mediapipe import Image as MpImage, ImageFormat
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MEDIAPIPE_OK = True
except ImportError:  # pragma: no cover - environnement sans mediapipe
    MpImage = None
    ImageFormat = None
    _mp_python = None
    _mp_vision = None
    _MEDIAPIPE_OK = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("souvenir")

# --- Chemins de données ---------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "souvenir_data")
FRAMES_DIR = os.path.join(_BASE_DIR, "souvenir_frames")
ADAIN_DIR = os.path.join(_BASE_DIR, "models", "adain")

# Modèle FaceLandmarker de MediaPipe Tasks (bundle .task). Cherché dans l'ordre.
_FACE_MODEL_CANDIDATES = (
    os.path.join(_BASE_DIR, "models", "face_landmarker.task"),
    os.path.join(_BASE_DIR, "face_landmarker.task"),
)

# --- SDXL Inpainting ------------------------------------------------------
SDXL_DIR = os.path.join(_BASE_DIR, "models", "sdxl_inpainting")
SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# --- IP-Adapter (conditionnement identité via CLIP vision, PAS FaceID) -----
IP_ADAPTER_DIR = os.path.join(_BASE_DIR, "models", "ip_adapter")
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
# Contient un "/" -> diffusers l'utilise tel quel comme subfolder sous IP_ADAPTER_DIR
# (models/ip_adapter/models/image_encoder).
IP_ADAPTER_IMAGE_ENCODER = "models/image_encoder"
IP_ADAPTER_SCALE = 0.65
# Zone neutre (genre ambigu, pas de signal texte fort) : identité visuelle du
# visiteur renforcée pour contrer le biais de genre du tableau. Ne pas monter
# au-delà : risque de sur-photoréalisme (rupture avec la matière picturale).
IP_ADAPTER_SCALE_NEUTRAL = 0.75

# --- Differential Diffusion (une passe, carte de force continue) -----------
# ControlNet depth SDXL (variante distillée "mid", 0.55 Go fp16) : contraint
# la structure du tableau pendant la re-diffusion du canevas complet.
CONTROLNET_DIR = os.path.join(_BASE_DIR, "models", "controlnet_depth_sdxl")
CONTROLNET_MODEL_ID = "diffusers/controlnet-depth-sdxl-1.0-mid"
CONTROLNET_SCALE = 0.55
# SOUVENIR_DIFFDIFF_NO_CONTROLNET=1 -> pipeline diffdiff SANS ControlNet
# (SDXL inpaint + IP-Adapter + carte de force seulement). Permet de mesurer
# si le ControlNet fait déborder le budget VRAM sur 8 Go, indépendamment des
# fuites entre étages de fallback.
DIFFDIFF_USE_CONTROLNET = (
    os.environ.get("SOUVENIR_DIFFDIFF_NO_CONTROLNET", "0") != "1"
)
# Depth maps pré-calculées du projet (effet parallax) : {artwork}_depth.npy à la
# racine, float32 [0,1], même résolution que le tableau. Réutilisées comme
# condition ControlNet ; fallback controlnet_aux MidasDetector si absentes.
DEPTH_NPY_TEMPLATE = os.path.join(_BASE_DIR, "{artwork}_depth.npy")

DIFFDIFF_STRENGTH = 0.80        # strength SDXL globale (re-diffusion du canevas)
DIFFDIFF_FACE_VALUE = 0.88      # valeur de la carte de force au centre du visage
DIFFDIFF_STEPS = 25             # * strength 0.80 -> 20 pas de débruitage effectifs
# Largeur de la bande de décroissance (smoothstep) autour du hull du visage,
# en fraction du côté max de la bbox visage (plancher 90 px). À l'échelle de
# génération 1024², cela correspond aux ~80-100 px visés : front, tempes,
# racine des cheveux et début du cou sont inclus dans la transition.
DIFFDIFF_BAND_FRAC = 0.35
DIFFDIFF_BAND_MIN_PX = 90
# Côté max de l'image retournée : la génération se fait sur un canevas 1024²,
# inutile de ré-agrandir au-delà (la Joconde native fait 9565 px de haut).
DIFFDIFF_OUTPUT_MAX_SIDE = 2048

# Suffixe négatif anti-« effet Jésus » : décourage barbe/moustache.
_FACE_NEGATIVE = ", beard, mustache"

# Estimation grossière du genre via proportions faciales (ratio inter-yeux /
# largeur mâchoire). Seuils calibrés sur joconde≈0.74 (F) vs vangogh/courbet≈0.86 (M),
# et validés sur visages de test : face_dark≈0.853 (homme jeune), face_test≈0.786
# (homme âgé avec lunettes — les lunettes ne biaisent PAS les landmarks, vérifié
# via cohérence coins-des-yeux/inter-iris ; sa morphologie est réellement à 0.786).
# Bande [FEMALE_MAX, MALE_MIN] = "neutral" (aucune injection de genre dans le
# prompt) : large exprès pour absorber les morphologies atypiques (âge, calvitie)
# plutôt que de trancher avec confiance sur un cas limite.
_GENDER_FEMALE_MAX = 0.76
_GENDER_MALE_MIN = 0.84

# Injections de prompt selon le genre estimé (positif ; l'opposé va au négatif).
# "neutral" n'est PAS un prompt vide : sans contre-poids, SDXL suit le biais du
# tableau lui-même (la Joconde tire le rendu vers le féminin : sourcils fins,
# lèvres rougies, peau lissée). En zone neutre on décrit donc la fidélité
# structurelle au visage de référence (IP-Adapter) au lieu d'un mot de genre,
# et on rejette explicitement l'embellissement/maquillage au négatif.
_GENDER_PROMPT = {
    "male": (", masculine face, male portrait", ", feminine face, female portrait"),
    "female": (", feminine face, female portrait", ", masculine face, male portrait"),
    # Version compacte (budget tokens CLIP, cf. _TEXTURE_POSITIVE) : la
    # formulation longue faisait déborder le prompt neutral au-delà de 77
    # tokens et tronquait tout le bloc texture picturale.
    "neutral": (
        ", preserving the exact facial structure, wrinkles and apparent age "
        "of the reference face",
        ", beautified face, makeup, lipstick, altered facial structure",
    ),
}

# Suffixes « rendu peint » : éclairage sfumato, matière picturale (touche,
# patine) + anti-texture photographique et anti-accessoires modernes (les
# bijoux de la photo de référence sont transférés par l'IP-Adapter -> ils
# doivent être rejetés explicitement au négatif).
# ATTENTION budget tokens : CLIP tronque à 77 tokens et coupe la FIN du
# prompt. Ces suffixes sont concaténés après SDXL_PROMPTS[artwork] et
# _GENDER_PROMPT[gender] ; le pire cas (joconde + neutral) tient à 74 tokens.
# Vérifier avec le tokenizer avant tout ajout, sinon la fin du prompt est
# silencieusement ignorée.
_TEXTURE_POSITIVE = (
    ", soft sfumato lighting, smoky transitions between light and shadow, "
    "visible brushstrokes, oil paint texture, aged canvas patina"
)
_TEXTURE_NEGATIVE = (
    ", modern jewelry, earrings, piercings, contemporary accessories, "
    "glossy skin, photographic sheen, skin pores, camera flash, "
    "studio lighting, photographic skin texture, digital photo, "
    "airbrushed, CGI look"
)

SDXL_PROMPTS = {
    "joconde": {
        "positive": "oil painting portrait, Renaissance master, Leonardo da Vinci style, "
                    "sfumato technique, warm amber palette, soft shadows, museum quality, "
                    "detailed face",
        "negative": "photorealistic, modern, cartoon, deformed, ugly, blurry, watermark"
                    + _FACE_NEGATIVE,
    },
    "vangogh": {
        "positive": "oil painting portrait, Post-Impressionist, Vincent van Gogh style, "
                    "thick impasto brushstrokes, swirling vivid colors, expressive texture",
        "negative": "photorealistic, modern, smooth, cartoon, deformed, blurry"
                    + _FACE_NEGATIVE,
    },
    "courbet": {
        "positive": "Realist oil painting portrait, Gustave Courbet style, dark earthy tones, "
                    "dramatic chiaroscuro, 19th century French painting",
        "negative": "photorealistic, modern, bright colors, cartoon, deformed, blurry"
                    + _FACE_NEGATIVE,
    },
}

# --- Indices MediaPipe FaceMesh -------------------------------------------
# Maillage de base (sans les points d'iris ajoutés par refine_landmarks).
_MESH_POINT_COUNT = 468

# Pose : nez, coins des yeux, coins de la bouche (cf. prompt).
_IDX_NOSE = 1
_IDX_EYE_L = 33
_IDX_EYE_R = 263
_IDX_MOUTH_L = 61
_IDX_MOUTH_R = 291

# EAR — œil gauche puis œil droit (ordre p1..p6 du Eye Aspect Ratio).
_EAR_LEFT = (33, 160, 158, 133, 153, 144)
_EAR_RIGHT = (362, 387, 385, 263, 380, 373)

# Seuils de scoring (cf. prompt).
_YAW_OK_DEG = 15.0
_PITCH_OK_DEG = 10.0
_YAW_ZERO_DEG = 45.0   # frontalité tombe à 0 à cet angle
_PITCH_ZERO_DEG = 40.0
_EAR_OPEN = 0.25       # yeux bien ouverts -> 1.0
_EAR_CLOSED = 0.15     # yeux fermés -> 0.0

_FRAME_VARIANTS = (1, 2, 3)


class SouvenirGenerator:
    """Génère une photo souvenir à partir de frames webcam et d'un tableau."""

    def __init__(self, device: str | None = None, face_model_path: str | None = None):
        self._landmarker = None
        self._face_model_path = face_model_path
        self._adain = None          # (torch, device, encoder, decoder) ou False si indisponible
        self._sdxl_pipe = None      # pipeline SDXL inpaint, ou False si indisponible
        self._ip_adapter_loaded = False  # IP-Adapter chargé sur le pipeline SDXL ?
        self._diffdiff_pipe = None  # pipeline SDXL+ControlNet (diffdiff), ou False
        self._diffdiff_ip_loaded = False   # IP-Adapter chargé sur le pipeline diffdiff ?
        self._diffdiff_sequential = False  # offload séquentiel (retry après OOM)
        self.last_method = None     # méthode ayant produit le dernier souvenir
        self._device_pref = device
        os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # MediaPipe (Tasks FaceLandmarker)                                  #
    # ------------------------------------------------------------------ #
    def _resolve_face_model(self):
        if self._face_model_path and os.path.exists(self._face_model_path):
            return self._face_model_path
        for candidate in _FACE_MODEL_CANDIDATES:
            if os.path.exists(candidate):
                return candidate
        raise RuntimeError(
            "Modèle FaceLandmarker introuvable. Placez 'face_landmarker.task' dans "
            "models/ (voir download_adain_weights.py --with-face-model)."
        )

    def _get_landmarker(self):
        if not _MEDIAPIPE_OK:
            raise RuntimeError("mediapipe indisponible")
        if self._landmarker is None:
            model_path = self._resolve_face_model()
            options = _mp_vision.FaceLandmarkerOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=_mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.4,
            )
            self._landmarker = _mp_vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    def _detect_landmarks(self, image_bgr):
        """Retourne la liste de landmarks normalisés (.x/.y/.z) du 1er visage, ou None."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        # _get_landmarker() vérifie _MEDIAPIPE_OK et lève une RuntimeError claire
        # ("mediapipe indisponible") si l'import a échoué — AVANT de toucher MpImage.
        landmarker = self._get_landmarker()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    @staticmethod
    def _landmarks_px(landmark, w, h):
        """Landmarks -> np.ndarray (468, 2) en pixels (maillage de base)."""
        pts = np.empty((_MESH_POINT_COUNT, 2), dtype=np.float32)
        for i in range(_MESH_POINT_COUNT):
            lm = landmark[i]
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
        return pts

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #
    def score_frame(self, frame_bgr) -> float:
        """Score de qualité de capture d'un frame webcam BGR, dans [0, 1].

        0.6 * frontalité + 0.4 * EAR. Retourne 0.0 si aucun visage.
        """
        try:
            landmark = self._detect_landmarks(frame_bgr)
        except RuntimeError:
            return 0.0
        if landmark is None:
            return 0.0

        h, w = frame_bgr.shape[:2]
        frontality = self._frontality_score(landmark, w, h)
        ear = self._ear_score(landmark, w, h)
        return float(0.6 * frontality + 0.4 * ear)

    def _frontality_score(self, landmark, w, h) -> float:
        """Estime yaw/pitch/roll depuis les landmarks 3D et note la frontalité."""
        def p3(idx):
            lm = landmark[idx]
            # z est ~à la même échelle que x (largeur) selon MediaPipe.
            return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float64)

        eye_l, eye_r = p3(_IDX_EYE_L), p3(_IDX_EYE_R)
        mouth_mid = (p3(_IDX_MOUTH_L) + p3(_IDX_MOUTH_R)) / 2.0
        eye_mid = (eye_l + eye_r) / 2.0

        eye_vec = eye_r - eye_l                # axe inter-oculaire
        vert_vec = mouth_mid - eye_mid         # axe vertical du visage

        # yaw : inclinaison en profondeur de la ligne des yeux (rotation gauche/droite).
        yaw = np.degrees(np.arctan2(eye_vec[2], abs(eye_vec[0]) + 1e-6))
        # pitch : inclinaison en profondeur de l'axe vertical (hochement).
        pitch = np.degrees(np.arctan2(vert_vec[2], abs(vert_vec[1]) + 1e-6))
        # roll : rotation dans le plan image (informatif, non pénalisé).
        _roll = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))

        yaw_score = self._linear_falloff(abs(yaw), _YAW_OK_DEG, _YAW_ZERO_DEG)
        pitch_score = self._linear_falloff(abs(pitch), _PITCH_OK_DEG, _PITCH_ZERO_DEG)
        return float(min(yaw_score, pitch_score))

    @staticmethod
    def _linear_falloff(value, ok_threshold, zero_threshold) -> float:
        """1.0 sous ok_threshold, 0.0 au-delà de zero_threshold, linéaire entre."""
        if value <= ok_threshold:
            return 1.0
        if value >= zero_threshold:
            return 0.0
        return 1.0 - (value - ok_threshold) / (zero_threshold - ok_threshold)

    def _ear_score(self, landmark, w, h) -> float:
        ear = 0.5 * (self._eye_ar(landmark, _EAR_LEFT, w, h)
                     + self._eye_ar(landmark, _EAR_RIGHT, w, h))
        if ear >= _EAR_OPEN:
            return 1.0
        if ear <= _EAR_CLOSED:
            return 0.0
        return float((ear - _EAR_CLOSED) / (_EAR_OPEN - _EAR_CLOSED))

    @staticmethod
    def _eye_ar(landmark, idx6, w, h) -> float:
        """Eye Aspect Ratio classique : (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
        def pt(i):
            lm = landmark[i]
            return np.array([lm.x * w, lm.y * h], dtype=np.float64)

        p1, p2, p3, p4, p5, p6 = (pt(i) for i in idx6)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return 0.0
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        return float(vertical / (2.0 * horizontal))

    def find_best_frame(self, frames):
        """Retourne (meilleur_frame, score).

        Fallback : si aucun frame ne dépasse 0.3, on retourne quand même celui
        au score maximal.
        """
        if not frames:
            raise ValueError("find_best_frame: liste de frames vide")

        best_frame = frames[0]
        best_score = -1.0
        for frame in frames:
            score = self.score_frame(frame)
            if score > best_score:
                best_score, best_frame = score, frame

        if best_score < 0.3:
            logger.warning(
                "Aucun frame au-dessus du seuil (meilleur score %.3f) — "
                "utilisation du moins mauvais.", max(best_score, 0.0),
            )
        return best_frame, float(max(best_score, 0.0))

    # ------------------------------------------------------------------ #
    # Warp visage -> tableau                                            #
    # ------------------------------------------------------------------ #
    def _compute_artwork_landmarks(self, artwork_bgr, artwork_name):
        """Détecte les landmarks du tableau et les met en cache sur disque."""
        landmark = self._detect_landmarks(artwork_bgr)
        if landmark is None:
            raise RuntimeError(
                f"Aucun visage détecté dans le tableau '{artwork_name}'"
            )
        h, w = artwork_bgr.shape[:2]
        pts = self._landmarks_px(landmark, w, h)

        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"{artwork_name}_landmarks.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"width": w, "height": h, "landmarks": pts.tolist()},
                handle,
            )
        logger.info("Landmarks du tableau '%s' calculés et sauvegardés -> %s",
                    artwork_name, path)
        return pts

    def _load_artwork_landmarks(self, artwork_bgr, artwork_name):
        path = os.path.join(DATA_DIR, f"{artwork_name}_landmarks.json")
        h, w = artwork_bgr.shape[:2]
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                pts = np.asarray(data["landmarks"], dtype=np.float32)
                # Re-échelonne si le tableau fourni n'a pas la résolution du cache.
                if data.get("width") and data.get("height"):
                    sx = w / float(data["width"])
                    sy = h / float(data["height"])
                    if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
                        pts = pts * np.array([sx, sy], dtype=np.float32)
                if pts.shape == (_MESH_POINT_COUNT, 2):
                    return pts
                logger.warning("Cache landmarks '%s' invalide, recalcul.", artwork_name)
            except (ValueError, KeyError, OSError) as exc:
                logger.warning("Lecture cache landmarks '%s' échouée (%s), recalcul.",
                               artwork_name, exc)
        return self._compute_artwork_landmarks(artwork_bgr, artwork_name)

    def warp_face_to_artwork(self, visitor_frame_bgr, artwork_bgr, artwork_name,
                             preserve_carnation=True):
        """Projette le visage du visiteur dans le tableau.

        Triangulation de Delaunay + warp affine par triangle, puis fusion.
        Retourne un BGR à la résolution native du tableau. Lève une RuntimeError
        si aucun visage visiteur n'est détecté.

        preserve_carnation :
          - True  (défaut) : composite alpha à bords adoucis -> conserve les
            VRAIES couleurs (carnation) du visiteur.
          - False : cv2.seamlessClone (Poisson) -> fondu très propre mais recolore
            le visage vers la palette du tableau (carnation perdue). Conservé pour
            comparaison.
        """
        landmark = self._detect_landmarks(visitor_frame_bgr)
        if landmark is None:
            raise RuntimeError("Aucun visage détecté dans le frame visiteur")

        vh, vw = visitor_frame_bgr.shape[:2]
        visitor_pts = self._landmarks_px(landmark, vw, vh)
        artwork_pts = self._load_artwork_landmarks(artwork_bgr, artwork_name)

        ah, aw = artwork_bgr.shape[:2]
        # Clamp dans les bornes des images respectives.
        visitor_pts[:, 0] = np.clip(visitor_pts[:, 0], 0, vw - 1)
        visitor_pts[:, 1] = np.clip(visitor_pts[:, 1], 0, vh - 1)
        artwork_pts[:, 0] = np.clip(artwork_pts[:, 0], 0, aw - 1)
        artwork_pts[:, 1] = np.clip(artwork_pts[:, 1], 0, ah - 1)

        triangles = self._delaunay_triangles((0, 0, aw, ah), artwork_pts)
        if not triangles:
            raise RuntimeError("Triangulation vide")

        warped_canvas = np.zeros_like(artwork_bgr)
        for i, j, k in triangles:
            self._warp_triangle(
                visitor_frame_bgr, warped_canvas,
                visitor_pts[[i, j, k]], artwork_pts[[i, j, k]],
            )

        hull = cv2.convexHull(artwork_pts.astype(np.int32))
        mask = np.zeros((ah, aw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        rect = cv2.boundingRect(hull)

        if preserve_carnation:
            return self._alpha_composite_face(warped_canvas, artwork_bgr, mask, rect)

        center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        try:
            return cv2.seamlessClone(
                warped_canvas, artwork_bgr, mask, center, cv2.NORMAL_CLONE,
            )
        except cv2.error as exc:
            # seamlessClone peut échouer sur des masques dégénérés : repli alpha.
            logger.warning("seamlessClone échoué (%s), fondu alpha simple.", exc)
            return self._alpha_composite_face(warped_canvas, artwork_bgr, mask, rect)

    @staticmethod
    def _alpha_composite_face(warped_canvas, artwork_bgr, mask, rect):
        """Composite alpha du visage warpé sur le tableau, avec bords adoucis au
        GaussianBlur. Préserve les couleurs du visiteur (aucune correction Poisson).
        Le feather est proportionnel à la taille du visage pour fondre la transition
        de teinte au bord du hull.
        """
        feather = max(6, int(min(rect[2], rect[3]) * 0.03))
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather)
        alpha3 = np.clip(alpha, 0.0, 1.0)[:, :, None]
        return (warped_canvas.astype(np.float32) * alpha3
                + artwork_bgr.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)

    @staticmethod
    def _delaunay_triangles(rect, points):
        """Triangulation de Delaunay -> liste de triplets d'indices de landmarks."""
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((float(p[0]), float(p[1])))

        triangle_list = subdiv.getTriangleList()
        triangles = []
        for t in triangle_list:
            corners = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            idx = []
            for cx, cy in corners:
                dist = np.sum((points - np.array([cx, cy])) ** 2, axis=1)
                j = int(np.argmin(dist))
                if dist[j] > 2.0:  # coin hors landmarks (sommet du super-triangle)
                    break
                idx.append(j)
            if len(idx) == 3 and len(set(idx)) == 3:
                triangles.append(tuple(idx))
        return triangles

    @staticmethod
    def _warp_triangle(src_img, dst_img, t_src, t_dst):
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return

        t1 = np.float32([[p[0] - r1[0], p[1] - r1[1]] for p in t_src])
        t2 = np.float32([[p[0] - r2[0], p[1] - r2[1]] for p in t_dst])

        src_crop = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        if src_crop.size == 0:
            return

        affine = cv2.getAffineTransform(t1, t2)
        warped = cv2.warpAffine(
            src_crop, affine, (r2[2], r2[3]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2), (1.0, 1.0, 1.0), cv2.LINE_AA, 0)

        roi = dst_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
        if roi.shape[:2] != warped.shape[:2]:
            return
        dst_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = (
            roi * (1.0 - mask) + warped * mask
        ).astype(dst_img.dtype)

    # ------------------------------------------------------------------ #
    # Transfert de style AdaIN                                          #
    # ------------------------------------------------------------------ #
    def _get_adain(self):
        """Charge (paresseusement) l'encodeur VGG et le décodeur AdaIN.

        Retourne un tuple (torch, device, encoder, decoder) ou None si les poids
        sont absents / le chargement échoue.
        """
        if self._adain is not None:
            return self._adain or None

        try:
            import torch
            from torch import nn
        except ImportError:
            logger.warning("PyTorch indisponible — transfert de style désactivé.")
            self._adain = False
            return None

        vgg_path = os.path.join(ADAIN_DIR, "vgg_normalised.pth")
        dec_path = os.path.join(ADAIN_DIR, "decoder.pth")
        if not (os.path.exists(vgg_path) and os.path.exists(dec_path)):
            logger.warning(
                "Poids AdaIN absents (%s / %s) — style ignoré. "
                "Exécuter download_adain_weights.py.", vgg_path, dec_path,
            )
            self._adain = False
            return None

        try:
            device = torch.device(
                self._device_pref
                or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            vgg = _build_adain_vgg(nn)
            vgg.load_state_dict(_load_adain_state_dict(torch, vgg_path))
            encoder = nn.Sequential(*list(vgg.children())[:31])  # jusqu'à relu4_1

            decoder = _build_adain_decoder(nn)
            decoder.load_state_dict(_load_adain_state_dict(torch, dec_path))

            encoder = encoder.to(device).float().eval()
            decoder = decoder.to(device).float().eval()
            for param in list(encoder.parameters()) + list(decoder.parameters()):
                param.requires_grad_(False)

            self._adain = (torch, device, encoder, decoder)
            logger.info("Modèle AdaIN chargé sur %s (FP32).", device)
            return self._adain
        except Exception as exc:  # noqa: BLE001 - on veut dégrader proprement
            logger.warning("Chargement AdaIN échoué (%s) — style ignoré.", exc)
            self._adain = False
            return None

    def apply_style_transfer(self, content_bgr, style_bgr, alpha=0.6):
        """Transfert de style AdaIN léger (content = warp, style = tableau).

        Traitement interne en 512x512, FP32. Retourne un BGR à la résolution de
        content_bgr. Lève une RuntimeError si le modèle AdaIN est indisponible.
        """
        adain = self._get_adain()
        if adain is None:
            raise RuntimeError("Modèle AdaIN indisponible")
        torch, device, encoder, decoder = adain

        oh, ow = content_bgr.shape[:2]

        def to_tensor(img_bgr):
            rgb = cv2.cvtColor(cv2.resize(img_bgr, (512, 512)), cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return tensor.to(device).float()

        with torch.no_grad():
            content_f = encoder(to_tensor(content_bgr))
            style_f = encoder(to_tensor(style_bgr))
            feat = _adain_normalize(content_f, style_f)
            feat = feat * alpha + content_f * (1.0 - alpha)
            output = decoder(feat).clamp(0.0, 1.0)

        out = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_bgr = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.resize(out_bgr, (ow, oh))

    def apply_style_transfer_face(self, warped_bgr, artwork_bgr, artwork_name,
                                  margin=0.30, alpha=0.35):
        """Applique AdaIN uniquement sur la région du visage puis recompose.

        Évite le flou dû à un aller-retour 512x512 sur toute la toile : on ne
        stylise que la boîte englobante du visage (hull des landmarks du tableau
        déjà calculés, + marge), et on la refond dans l'image via un masque à
        bords gaussiens. Retourne un BGR à la résolution de warped_bgr.
        Lève une RuntimeError si le modèle AdaIN est indisponible.
        """
        if self._get_adain() is None:
            raise RuntimeError("Modèle AdaIN indisponible")

        # Le visage du visiteur a été projeté à l'emplacement du visage du
        # tableau -> on réutilise les landmarks (cache) du tableau.
        artwork_pts = self._load_artwork_landmarks(artwork_bgr, artwork_name)
        h, w = warped_bgr.shape[:2]
        hull = cv2.convexHull(artwork_pts.astype(np.int32))
        bx, by, bw, bh = cv2.boundingRect(hull)

        # Marge de 20% autour de la boîte, bornée à l'image.
        mx, my = int(bw * margin), int(bh * margin)
        x0, y0 = max(0, bx - mx), max(0, by - my)
        x1, y1 = min(w, bx + bw + mx), min(h, by + bh + my)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise RuntimeError("Région de visage dégénérée pour le style")

        content_crop = warped_bgr[y0:y1, x0:x1]
        style_crop = artwork_bgr[y0:y1, x0:x1]
        styled_crop = self.apply_style_transfer(content_crop, style_crop, alpha=alpha)

        # Masque à bords gaussiens : 1 à l'intérieur, fondu vers 0 sur le pourtour
        # du crop pour masquer la jointure avec la zone non stylisée.
        ch, cw = content_crop.shape[:2]
        feather = max(4, int(min(ch, cw) * 0.12))
        mask = np.zeros((ch, cw), dtype=np.float32)
        cv2.rectangle(mask, (feather, feather), (cw - feather, ch - feather), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), feather / 2.0)
        mask3 = mask[:, :, None]

        result = warped_bgr.copy()
        region = result[y0:y1, x0:x1].astype(np.float32)
        result[y0:y1, x0:x1] = (
            styled_crop.astype(np.float32) * mask3 + region * (1.0 - mask3)
        ).astype(np.uint8)
        return result

    # ------------------------------------------------------------------ #
    # Gestion VRAM — nettoyage entre étages de fallback                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_cuda_mem(torch, label):
        logger.info(
            "VRAM [%s] : allocated=%.2f Go | reserved=%.2f Go",
            label,
            torch.cuda.memory_allocated() / 1024 ** 3,
            torch.cuda.memory_reserved() / 1024 ** 3,
        )

    def _free_cuda(self, label):
        """Libère la VRAM entre deux étages de fallback.

        gc.collect() (casse les cycles retenant des tenseurs), puis
        torch.cuda.synchronize() (attend les opérations async qui retiennent
        encore de la mémoire) AVANT torch.cuda.empty_cache(). Logue
        allocated/reserved avant et après pour vérifier la libération réelle.
        """
        try:
            import torch
        except ImportError:
            gc.collect()
            return
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            self._log_cuda_mem(torch, f"{label} — avant nettoyage")
        gc.collect()
        if has_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self._log_cuda_mem(torch, f"{label} — après nettoyage")

    def _teardown_pipe(self, pipe, label):
        """Détache les hooks d'offload accelerate, supprime toutes les
        références aux gros modules du pipeline (UNet, VAE, ControlNet,
        encodeurs texte/image), puis libère la VRAM (_free_cuda)."""
        try:
            pipe.remove_all_hooks()
        except Exception as exc:  # noqa: BLE001
            logger.warning("remove_all_hooks échoué (%s) — poursuite du "
                           "nettoyage.", exc)
        for name in ("unet", "vae", "controlnet", "text_encoder",
                     "text_encoder_2", "image_encoder", "feature_extractor"):
            if getattr(pipe, name, None) is not None:
                try:
                    setattr(pipe, name, None)
                except Exception:  # noqa: BLE001
                    pass
        del pipe
        self._free_cuda(label)

    def _unload_diffdiff_pipeline(self, label):
        """Décharge le pipeline diffdiff (s'il est chargé) et purge la VRAM."""
        pipe = self._diffdiff_pipe
        if pipe is False:  # indisponible (pas de poids/diffusers) : rien chargé
            self._free_cuda(label)
            return
        self._diffdiff_pipe = None
        self._diffdiff_ip_loaded = False
        if pipe is not None:
            self._teardown_pipe(pipe, label)
        else:
            self._free_cuda(label)

    def _unload_sdxl_pipeline(self, label):
        """Décharge le pipeline SDXL inpaint (s'il est chargé) et purge la VRAM."""
        pipe = self._sdxl_pipe
        if pipe is False:
            self._free_cuda(label)
            return
        self._sdxl_pipe = None
        self._ip_adapter_loaded = False
        if pipe is not None:
            self._teardown_pipe(pipe, label)
        else:
            self._free_cuda(label)

    # ------------------------------------------------------------------ #
    # SDXL Inpainting                                                   #
    # ------------------------------------------------------------------ #
    def _get_sdxl_pipeline(self):
        """Charge (paresseusement, singleton) le pipeline SDXL inpaint en FP16.

        Retourne le pipeline ou None si diffusers / le modèle / la VRAM ne
        permettent pas de le charger (le pipeline appelant bascule alors sur AdaIN).
        LivePortrait doit avoir été déchargé du GPU avant l'appel.
        """
        if self._sdxl_pipe is not None:
            return self._sdxl_pipe or None

        try:
            import torch
            from diffusers import (
                EulerDiscreteScheduler,
                StableDiffusionXLInpaintPipeline,
            )
        except ImportError as exc:
            logger.warning(
                "diffusers/torch indisponibles (%s) — SDXL désactivé "
                "(pip install diffusers accelerate transformers).", exc,
            )
            self._sdxl_pipe = False
            return None

        if not torch.cuda.is_available():
            logger.warning("CUDA indisponible — SDXL désactivé.")
            self._sdxl_pipe = False
            return None

        # Modèle : cache local models/sdxl_inpainting/ sinon téléchargement HF.
        model_source = SDXL_DIR
        if not os.path.exists(os.path.join(SDXL_DIR, "model_index.json")):
            try:
                from huggingface_hub import snapshot_download
                logger.info("Modèle SDXL absent en local — téléchargement dans %s ...", SDXL_DIR)
                model_source = snapshot_download(
                    repo_id=SDXL_MODEL_ID, local_dir=SDXL_DIR,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Téléchargement SDXL échoué (%s) — SDXL désactivé.", exc)
                self._sdxl_pipe = False
                return None

        try:
            # variant fp16 si présent, sinon poids standard chargés en fp16.
            try:
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    model_source, torch_dtype=torch.float16,
                    variant="fp16", use_safetensors=True,
                )
            except (OSError, ValueError):
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    model_source, torch_dtype=torch.float16, use_safetensors=True,
                )

            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

            # IP-Adapter chargé AVANT l'offload pour que ses modules + l'encodeur
            # d'image reçoivent bien les hooks de déchargement CPU.
            self._ip_adapter_loaded = self._try_load_ip_adapter(pipe)

            # 8 Go de VRAM : on décharge module par module sur CPU si trop juste.
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes < 7 * 1024 ** 3:
                logger.info("VRAM libre %.1f Go < 7 Go — enable_model_cpu_offload().",
                            free_bytes / 1024 ** 3)
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            # Réduit le pic mémoire du VAE au décodage 1024² (API pipe.enable_*
            # dépréciée en diffusers >= 0.40 -> passer par pipe.vae).
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            else:  # diffusers < 0.40
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
            pipe.set_progress_bar_config(disable=True)

            self._sdxl_pipe = pipe
            logger.info("Pipeline SDXL inpaint chargé (FP16).")
            return self._sdxl_pipe
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chargement SDXL échoué (%s) — SDXL désactivé.", exc)
            self._sdxl_pipe = False
            return None

    def _try_load_ip_adapter(self, pipe):
        """Charge IP-Adapter (plus-face SDXL, ViT-H) sur le pipeline. Retourne
        True si OK, False sinon (le pipeline bascule alors sur le warp legacy).
        """
        weight = os.path.join(IP_ADAPTER_DIR, IP_ADAPTER_SUBFOLDER, IP_ADAPTER_WEIGHT)
        if not os.path.exists(weight):
            logger.warning(
                "IP-Adapter absent (%s) — pipeline warp legacy. "
                "Exécuter download_ip_adapter.py.", weight,
            )
            return False
        try:
            pipe.load_ip_adapter(
                IP_ADAPTER_DIR,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=IP_ADAPTER_WEIGHT,
                image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER,
            )
            pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
            logger.info("IP-Adapter chargé (scale=%.2f).", IP_ADAPTER_SCALE)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chargement IP-Adapter échoué (%s) — warp legacy.", exc)
            return False

    def _extract_face_crop(self, frame_bgr, margin=0.4):
        """Recadre le visage du visiteur (bbox des landmarks + marge) pour le
        conditionnement IP-Adapter."""
        landmark = self._detect_landmarks(frame_bgr)
        if landmark is None:
            raise RuntimeError("Aucun visage visiteur détecté")
        h, w = frame_bgr.shape[:2]
        pts = self._landmarks_px(landmark, w, h)
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        mx, my = int(bw * margin), int(bh * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        return frame_bgr[y0:y1, x0:x1]

    def _create_face_mask(self, image_bgr, landmarks_cache=None):
        """Masque uint8 (255 = zone à inpainter) : hull du visage seul (+15% de
        marge), à bords adoucis. Pas d'extension cheveux — SDXL ne repeint que le
        visage, la chevelure du tableau est conservée.

        landmarks_cache : points (N,2) déjà calculés dans le repère de image_bgr.
        Sinon, re-détection MediaPipe sur image_bgr.
        """
        h, w = image_bgr.shape[:2]
        if landmarks_cache is not None:
            pts = np.asarray(landmarks_cache, dtype=np.float32)
        else:
            landmark = self._detect_landmarks(image_bgr)
            if landmark is None:
                raise RuntimeError("Aucun visage détecté pour le masque SDXL")
            pts = self._landmarks_px(landmark, w, h)

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2).astype(np.float32)
        centroid = hull.mean(axis=0)

        mask = np.zeros((h, w), dtype=np.uint8)
        face = ((hull - centroid) * 1.15 + centroid).astype(np.int32)
        cv2.fillConvexPoly(mask, face, 255)

        # Léger lissage des bords pour une recomposition sans jointure.
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        mask = cv2.GaussianBlur(mask, (0, 0), 8)
        return mask

    @staticmethod
    def _letterbox_1024(image_bgr):
        """Redimensionne en 1024x1024 en conservant le ratio (padding noir centré).

        Retourne (canvas1024, scale, pad_left, pad_top, new_w, new_h).
        """
        h, w = image_bgr.shape[:2]
        scale = 1024.0 / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h))
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        pad_left, pad_top = (1024 - new_w) // 2, (1024 - new_h) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas, scale, pad_left, pad_top, new_w, new_h

    @staticmethod
    def _unletterbox(canvas1024, target_hw, pad_left, pad_top, new_w, new_h):
        inner = canvas1024[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        return cv2.resize(inner, (target_hw[1], target_hw[0]))

    def estimate_gender(self, frame_bgr):
        """Estimation grossière du genre du visiteur via proportions faciales.

        Ratio = distance inter-yeux (coins externes) / largeur de mâchoire.
        Retourne 'male' | 'female' | 'neutral'. Heuristique géométrique
        approximative (pas un vrai classifieur) : la bande centrale renvoie
        'neutral' pour ne rien injecter en cas de doute.
        """
        try:
            landmark = self._detect_landmarks(frame_bgr)
        except RuntimeError:
            return "neutral"
        if landmark is None:
            return "neutral"

        h, w = frame_bgr.shape[:2]
        pts = self._landmarks_px(landmark, w, h)
        inter_eye = float(np.linalg.norm(pts[33] - pts[263]))
        jaw_width = float(np.linalg.norm(pts[172] - pts[397]))
        if jaw_width < 1e-6:
            return "neutral"

        ratio = inter_eye / jaw_width
        if ratio <= _GENDER_FEMALE_MAX:
            gender = "female"
        elif ratio >= _GENDER_MALE_MIN:
            gender = "male"
        else:
            gender = "neutral"

        logger.info(
            "Genre estimé automatiquement : %s (ratio inter-yeux/mâchoire = %.3f, seuils : "
            "female<=%.2f < neutral < %.2f<=male) — landmarks yeux 33=(%.0f,%.0f) "
            "263=(%.0f,%.0f), mâchoire 172=(%.0f,%.0f) 397=(%.0f,%.0f)",
            gender, ratio, _GENDER_FEMALE_MAX, _GENDER_MALE_MIN,
            pts[33][0], pts[33][1], pts[263][0], pts[263][1],
            pts[172][0], pts[172][1], pts[397][0], pts[397][1],
        )
        return gender

    def _resolve_gender(self, frame_bgr, gender_override=None):
        """Genre effectif pour les prompts : override manuel prioritaire,
        sinon heuristique estimate_gender(). Un override invalide est ignoré
        (warning) plutôt que de faire échouer la génération."""
        if gender_override is not None:
            if gender_override in _GENDER_PROMPT:
                logger.info("Genre forcé manuellement : %s (heuristique "
                            "MediaPipe ignorée).", gender_override)
                return gender_override
            logger.warning("gender_override invalide (%r) — retour à "
                           "l'heuristique.", gender_override)
        return self.estimate_gender(frame_bgr)

    @staticmethod
    def _ip_scale_for(gender):
        """Scale IP-Adapter selon le genre estimé : renforcé en zone neutre
        (pas de signal texte fort -> l'identité visuelle doit contrer le biais
        de genre du tableau), nominal sinon."""
        scale = (IP_ADAPTER_SCALE_NEUTRAL if gender == "neutral"
                 else IP_ADAPTER_SCALE)
        logger.info("Scale IP-Adapter : %.2f (genre estimé : %s).", scale, gender)
        return scale

    @staticmethod
    def _blend_identity(warped_crop, artwork_crop, visitor_weight=0.4):
        """Mélange pondéré entre le visage warpé (visiteur) et le visage original
        du tableau. visitor_weight = poids du visiteur ; le reste vient du tableau.
        Permet de ne garder qu'une influence partielle des traits du visiteur.
        """
        return cv2.addWeighted(
            warped_crop, visitor_weight, artwork_crop, 1.0 - visitor_weight, 0.0,
        )

    @staticmethod
    def _harmonize_chroma_lab(styled_bgr, artwork_bgr, mask):
        """Transfert de couleur type Reinhard limité à la chrominance (a,b en LAB).

        Recale moyenne/écart-type des canaux a,b du visage stylisé sur ceux du
        tableau (zone du masque), en CONSERVANT la luminance L (modelé/identité,
        carnation en valeur). Retourne (bgr_corrigé, stats) où stats = {"a":(sμ,sσ,
        tμ,tσ), "b":(...)} pour le log.
        """
        face = mask > 127
        if not face.any():
            return styled_bgr, None

        styled_lab = cv2.cvtColor(styled_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        art_lab = cv2.cvtColor(artwork_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        stats = {}
        for ch, name in ((1, "a"), (2, "b")):
            src = styled_lab[:, :, ch][face]
            tgt = art_lab[:, :, ch][face]
            s_mean, s_std = float(src.mean()), float(src.std() + 1e-6)
            t_mean, t_std = float(tgt.mean()), float(tgt.std() + 1e-6)
            styled_lab[:, :, ch] = (
                (styled_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean
            )
            stats[name] = (round(s_mean, 1), round(s_std, 1),
                           round(t_mean, 1), round(t_std, 1))

        out = cv2.cvtColor(np.clip(styled_lab, 0, 255).astype(np.uint8),
                           cv2.COLOR_LAB2BGR)
        return out, stats

    @staticmethod
    def _add_contact_shadow(composite_bgr, mask, opacity=0.18):
        """Ombre de contact multiplicative le long du contour BAS du masque
        (mâchoire, menton, raccord cou) pour ancrer visuellement le visage.
        """
        m = (mask > 127).astype(np.uint8) * 255
        ys = np.where(m > 0)[0]
        if ys.size == 0:
            return composite_bgr
        y_top, y_bot = int(ys.min()), int(ys.max())
        face_h = max(1, y_bot - y_top)

        # Bande située juste sous le masque = dilatation - masque.
        k = max(3, int(face_h * 0.06))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        band = cv2.subtract(cv2.dilate(m, kernel), m)
        # Ne garder que la moitié basse (mâchoire / menton / cou).
        band[: y_top + int(face_h * 0.55), :] = 0

        shadow = cv2.GaussianBlur(band.astype(np.float32) / 255.0, (0, 0), k)
        shadow = np.clip(shadow * opacity, 0.0, 1.0)[:, :, None]
        return (composite_bgr.astype(np.float32) * (1.0 - shadow)).astype(np.uint8)

    def apply_sdxl_inpainting(self, warped_bgr, artwork_bgr, artwork_name,
                              gender="neutral", visitor_weight=0.35):
        """Repeint la région du visage avec SDXL inpaint (style du tableau) puis
        recompose dans le tableau pleine résolution.

        Le visage n'est stylisé qu'au sein d'un crop autour du visage redimensionné
        en 1024² (le visage occupe donc tout le budget SDXL, pas ~150 px), puis
        refondu avec un masque à bords gaussiens. Lève une RuntimeError si SDXL est
        indisponible (le pipeline appelant bascule alors sur AdaIN).
        """
        from PIL import Image

        pipe = self._get_sdxl_pipeline()
        if pipe is None:
            raise RuntimeError("Pipeline SDXL indisponible")

        # Prompt de base + injection dynamique selon le genre estimé.
        base = SDXL_PROMPTS.get(artwork_name, SDXL_PROMPTS["joconde"])
        pos_suffix, neg_suffix = _GENDER_PROMPT.get(gender, ("", ""))
        positive_prompt = base["positive"] + pos_suffix + _TEXTURE_POSITIVE
        negative_prompt = base["negative"] + neg_suffix + _TEXTURE_NEGATIVE

        # Crop autour du visage (landmarks du tableau = position du visage dans warped).
        artwork_pts = self._load_artwork_landmarks(artwork_bgr, artwork_name)
        h, w = warped_bgr.shape[:2]
        hull = cv2.convexHull(artwork_pts.astype(np.int32))
        bx, by, bw, bh = cv2.boundingRect(hull)
        ctx = 0.35  # contexte symétrique autour du visage
        x0 = max(0, int(bx - bw * ctx))
        y0 = max(0, int(by - bh * ctx))
        x1 = min(w, int(bx + bw * (1 + ctx)))
        y1 = min(h, int(by + bh * (1 + ctx)))
        warped_crop = warped_bgr[y0:y1, x0:x1]
        if warped_crop.shape[0] < 8 or warped_crop.shape[1] < 8:
            raise RuntimeError("Région de visage dégénérée pour SDXL")

        # Blend identité : mélange du visage warpé (visiteur) et du visage
        # original du tableau AVANT l'inpainting. visitor_weight règle l'identité
        # (géométrie/couleur en entrée) ; strength règle indépendamment à quel
        # point SDXL repeint la matière picturale par-dessus.
        artwork_crop = artwork_bgr[y0:y1, x0:x1]
        crop = self._blend_identity(warped_crop, artwork_crop, visitor_weight)

        # Letterbox 1024² + landmarks transposés dans ce repère pour le masque.
        square, scale, pad_l, pad_t, new_w, new_h = self._letterbox_1024(crop)
        pts_square = artwork_pts.copy()
        pts_square[:, 0] = (pts_square[:, 0] - x0) * scale + pad_l
        pts_square[:, 1] = (pts_square[:, 1] - y0) * scale + pad_t
        mask_square = self._create_face_mask(square, landmarks_cache=pts_square)

        pil_image = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask_square)

        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=25,
            strength=0.58,
            guidance_scale=7.5,
            width=1024,
            height=1024,
        ).images[0]

        out_square = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

        # Retour au repère du crop, puis recomposition pleine résolution.
        styled_crop = self._unletterbox(
            out_square, crop.shape[:2], pad_l, pad_t, new_w, new_h,
        )
        mask_crop = self._unletterbox(
            cv2.cvtColor(mask_square, cv2.COLOR_GRAY2BGR),
            crop.shape[:2], pad_l, pad_t, new_w, new_h,
        )[:, :, 0]

        _face_px = mask_crop > 127

        def _skin_rgb(img_bgr):
            if not _face_px.any():
                return None
            mean_bgr = img_bgr[_face_px].reshape(-1, 3).mean(axis=0)
            return tuple(round(float(v), 1) for v in mean_bgr[::-1])  # BGR -> RGB

        # (4) Harmonisation chroma LAB : recale a,b du visage sur le tableau
        #     (Reinhard, chrominance seule), luminance L conservée.
        harmonized, lab_stats = self._harmonize_chroma_lab(
            styled_crop, artwork_crop, mask_crop,
        )

        logger.info(
            "Carnation (RGB, zone visage) — warpé=%s | tableau=%s | blend(vw=%.2f)=%s "
            "| SDXL=%s | SDXL+LAB=%s",
            _skin_rgb(warped_crop), _skin_rgb(artwork_crop), visitor_weight,
            _skin_rgb(crop), _skin_rgb(styled_crop), _skin_rgb(harmonized),
        )
        if lab_stats:
            logger.info(
                "Correction LAB chroma (moy/écart src->cible) — "
                "a: %s->%s | b: %s->%s",
                lab_stats["a"][:2], lab_stats["a"][2:],
                lab_stats["b"][:2], lab_stats["b"][2:],
            )
        styled_crop = harmonized

        # (5) Ombre de contact sous la mâchoire pour ancrer le visage.
        styled_crop = self._add_contact_shadow(styled_crop, mask_crop)

        # (1) Masque doux LARGE : dégradé ~16% de la largeur du visage, qui
        #     s'étend dans la chevelure/le voile au lieu de s'arrêter à la mâchoire.
        feather = max(6, int(bw * 0.16))
        soft = cv2.GaussianBlur(mask_crop.astype(np.float32) / 255.0, (0, 0), feather)
        soft3 = np.clip(soft, 0.0, 1.0)[:, :, None]

        result_full = warped_bgr.copy()
        region = result_full[y0:y1, x0:x1].astype(np.float32)
        result_full[y0:y1, x0:x1] = (
            styled_crop.astype(np.float32) * soft3 + region * (1.0 - soft3)
        ).astype(np.uint8)
        return result_full

    def generate_face_via_ip_adapter(self, visitor_face_bgr, artwork_bgr,
                                     artwork_name, mask, gender_override=None):
        """Pipeline principal : IP-Adapter (identité CLIP vision) + SDXL inpaint.

        La zone du visage (mask) est ENTIÈREMENT régénérée par SDXL — aucun pixel
        du visiteur n'est copié/collé (fini l'effet « masque collé »). L'identité
        vient du conditionnement IP-Adapter (crop visage original), et une image de
        guidage FLOUE donne juste position/couleur approximatives. Retourne le
        tableau pleine résolution avec le visage généré recomposé.
        Lève une RuntimeError si IP-Adapter/SDXL est indisponible.
        """
        from PIL import Image

        pipe = self._get_sdxl_pipeline()
        if pipe is None or not self._ip_adapter_loaded:
            raise RuntimeError("Pipeline IP-Adapter indisponible")

        # Prompt style peinture + genre (override manuel ou heuristique).
        gender = self._resolve_gender(visitor_face_bgr, gender_override)
        base = SDXL_PROMPTS.get(artwork_name, SDXL_PROMPTS["joconde"])
        pos_suffix, neg_suffix = _GENDER_PROMPT.get(gender, ("", ""))
        positive_prompt = base["positive"] + pos_suffix + _TEXTURE_POSITIVE
        negative_prompt = base["negative"] + neg_suffix + _TEXTURE_NEGATIVE

        # Région = bbox du masque + contexte.
        h, w = artwork_bgr.shape[:2]
        ys, xs = np.where(mask > 127)
        if ys.size == 0:
            raise RuntimeError("Masque de visage vide")
        mx0, my0, mx1, my1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = mx1 - mx0, my1 - my0
        ctx = 0.35
        x0 = max(0, int(mx0 - bw * ctx))
        y0 = max(0, int(my0 - bh * ctx))
        x1 = min(w, int(mx1 + bw * ctx))
        y1 = min(h, int(my1 + bh * ctx))
        artwork_crop = artwork_bgr[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        # Guidage FAIBLE : visage visiteur fortement flouté, posé dans la zone
        # masquée (position + couleur approx.) ; hors masque = tableau original.
        vis_resized = cv2.resize(visitor_face_bgr, (x1 - x0, y1 - y0))
        ksize = max(9, ((min(x1 - x0, y1 - y0) // 8) | 1))  # noyau impair large
        vis_blur = cv2.GaussianBlur(vis_resized, (ksize, ksize), 0)
        soft_m = cv2.GaussianBlur(mask_crop.astype(np.float32) / 255.0, (0, 0), 8)[:, :, None]
        guidance = (vis_blur.astype(np.float32) * soft_m
                    + artwork_crop.astype(np.float32) * (1.0 - soft_m)).astype(np.uint8)

        # Letterbox 1024² (guidage + masque, même transform).
        square, scale, pad_l, pad_t, new_w, new_h = self._letterbox_1024(guidance)
        mask_sq = self._letterbox_1024(
            cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR),
        )[0][:, :, 0]

        pil_image = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask_sq)
        ip_image = Image.fromarray(cv2.cvtColor(visitor_face_bgr, cv2.COLOR_BGR2RGB))

        pipe.set_ip_adapter_scale(self._ip_scale_for(gender))
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            ip_adapter_image=ip_image,
            num_inference_steps=25,
            strength=0.80,
            guidance_scale=7.5,
            width=1024,
            height=1024,
        ).images[0]

        out_square = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        styled_crop = self._unletterbox(
            out_square, artwork_crop.shape[:2], pad_l, pad_t, new_w, new_h,
        )

        # Post-traitements conservés : harmonisation LAB, ombre de contact.
        face_px = mask_crop > 127

        def _skin_rgb(img_bgr):
            if not face_px.any():
                return None
            return tuple(round(float(v), 1)
                         for v in img_bgr[face_px].reshape(-1, 3).mean(axis=0)[::-1])

        harmonized, lab_stats = self._harmonize_chroma_lab(
            styled_crop, artwork_crop, mask_crop,
        )
        logger.info(
            "IP-Adapter (RGB, zone visage) — visiteur=%s | tableau=%s | "
            "SDXL=%s | SDXL+LAB=%s",
            _skin_rgb(cv2.resize(visitor_face_bgr, mask_crop.shape[::-1])),
            _skin_rgb(artwork_crop), _skin_rgb(styled_crop), _skin_rgb(harmonized),
        )
        styled_crop = self._add_contact_shadow(harmonized, mask_crop)

        # Recomposition pleine résolution (masque doux large ~16% largeur visage).
        feather = max(6, int(bw * 0.16))
        soft = cv2.GaussianBlur(mask_crop.astype(np.float32) / 255.0, (0, 0), feather)
        soft3 = np.clip(soft, 0.0, 1.0)[:, :, None]
        result_full = artwork_bgr.copy()
        region = result_full[y0:y1, x0:x1].astype(np.float32)
        result_full[y0:y1, x0:x1] = (
            styled_crop.astype(np.float32) * soft3 + region * (1.0 - soft3)
        ).astype(np.uint8)
        return result_full

    # ------------------------------------------------------------------ #
    # Differential Diffusion (une passe, carte de force continue)       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_debug_heatmap(map01, filename):
        """Sauvegarde une visualisation heatmap d'une carte float [0,1] dans
        output/ (côté max 1024 pour rester léger). Ne lève jamais."""
        try:
            out_dir = os.path.join(_BASE_DIR, "output")
            os.makedirs(out_dir, exist_ok=True)
            vis = map01
            h, w = vis.shape[:2]
            scale = 1024.0 / max(h, w)
            if scale < 1.0:
                vis = cv2.resize(vis, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            heat = cv2.applyColorMap(
                np.clip(vis * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO,
            )
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, heat)
            logger.info("Heatmap debug -> %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sauvegarde heatmap debug échouée (%s).", exc)

    def _generate_strength_map(self, landmarks, image_shape,
                               face_value=DIFFDIFF_FACE_VALUE):
        """Carte de force continue float32 [0,1], même résolution que le tableau.

        - Cœur du visage (hull des landmarks, étendu vers le haut pour couvrir
          front et racine des cheveux) : face_value (~0.88).
        - Décroissance radiale en smoothstep jusqu'à 0.0 sur une large bande
          autour du visage (tempes, début du cou inclus).
        - 0.0 partout ailleurs : le tableau original est verrouillé sur ses
          latents d'origine par le re-bruitage diffdiff.

        Sauvegarde une heatmap de contrôle dans output/debug_strength_map.png.
        """
        h, w = image_shape[:2]
        pts = np.asarray(landmarks, dtype=np.float32).copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        hull = cv2.convexHull(pts.astype(np.int32))
        _bx, _by, bw, bh = cv2.boundingRect(hull)

        # Cœur : hull rempli + union avec une copie décalée vers le haut (le
        # maillage MediaPipe s'arrête aux sourcils -> front / racine des cheveux).
        core = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(core, hull, 255)
        up = int(round(bh * 0.18))
        if up > 0:
            shifted = np.zeros_like(core)
            shifted[:h - up, :] = core[up:, :]
            core = cv2.bitwise_or(core, shifted)

        # Bande proportionnelle à la taille du visage : à l'échelle de
        # génération 1024², cela correspond aux ~80-100 px visés.
        band = max(float(DIFFDIFF_BAND_MIN_PX), DIFFDIFF_BAND_FRAC * max(bw, bh))

        dist = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 5)
        t = np.clip(1.0 - dist / band, 0.0, 1.0)
        strength = (face_value * t * t * (3.0 - 2.0 * t)).astype(np.float32)

        self._save_debug_heatmap(strength, "debug_strength_map.png")
        return strength

    def _generate_controlnet_condition(self, artwork_bgr, artwork_name=None):
        """Condition depth (BGR uint8, résolution du tableau) pour le ControlNet.

        Réutilise la depth map pré-calculée du projet ({artwork}_depth.npy,
        déjà générée pour l'effet parallax) si présente ; sinon extraction
        MiDaS via controlnet_aux sur le tableau ORIGINAL (pas le visiteur).
        Convention ControlNet depth : proche = clair (MiDaS) ; auto-inversion
        si la carte chargée est manifestement inversée.
        """
        h, w = artwork_bgr.shape[:2]
        depth = None
        if artwork_name:
            npy_path = DEPTH_NPY_TEMPLATE.format(artwork=artwork_name)
            if os.path.exists(npy_path):
                try:
                    depth = np.load(npy_path).astype(np.float32)
                    if depth.shape[:2] != (h, w):
                        depth = cv2.resize(depth, (w, h),
                                           interpolation=cv2.INTER_LINEAR)
                    logger.info("Condition ControlNet : depth %s réutilisée.", npy_path)
                except (OSError, ValueError) as exc:
                    logger.warning("Lecture %s échouée (%s) — extraction MiDaS.",
                                   npy_path, exc)
                    depth = None

        if depth is None:
            from PIL import Image
            from controlnet_aux import MidasDetector
            logger.info("Extraction depth MiDaS (controlnet_aux) du tableau ...")
            midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            pil = Image.fromarray(cv2.cvtColor(artwork_bgr, cv2.COLOR_BGR2RGB))
            depth_pil = midas(pil)
            depth = np.asarray(depth_pil.convert("L"), dtype=np.float32) / 255.0
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        lo, hi = float(depth.min()), float(depth.max())
        if hi - lo > 1e-6:
            depth = (depth - lo) / (hi - lo)

        # Convention proche=clair : sur un portrait, le sujet (centre) doit être
        # plus clair que le fond (coins). Sinon la carte est inversée.
        ch, cw = h // 2, w // 2
        center = float(depth[ch - h // 8: ch + h // 8,
                             cw - w // 8: cw + w // 8].mean())
        k = max(2, min(h, w) // 20)
        corners = float(np.mean([depth[:k, :k].mean(), depth[:k, -k:].mean(),
                                 depth[-k:, :k].mean(), depth[-k:, -k:].mean()]))
        if center < corners:
            logger.info("Depth inversée (centre %.2f < coins %.2f) — inversion.",
                        center, corners)
            depth = 1.0 - depth

        self._save_debug_heatmap(depth, "debug_controlnet_depth.png")
        return cv2.cvtColor((depth * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _get_diffdiff_pipeline(self):
        """Charge (paresseusement, singleton) SDXL inpaint + ControlNet depth +
        IP-Adapter pour la génération Differential Diffusion.

        Pipeline indépendant de self._sdxl_pipe : celui-ci n'est chargé que si
        le fallback IP-Adapter est réellement atteint (évite de doubler la RAM
        en fonctionnement nominal). Retourne le pipeline ou None.
        """
        if self._diffdiff_pipe is not None:
            return self._diffdiff_pipe or None

        try:
            import torch
            from diffusers import (
                ControlNetModel,
                EulerDiscreteScheduler,
                StableDiffusionXLControlNetInpaintPipeline,
                StableDiffusionXLInpaintPipeline,
            )
        except ImportError as exc:
            logger.warning("diffusers/torch indisponibles (%s) — diffdiff désactivé.", exc)
            self._diffdiff_pipe = False
            return None

        if not torch.cuda.is_available():
            logger.warning("CUDA indisponible — diffdiff désactivé.")
            self._diffdiff_pipe = False
            return None

        if not os.path.exists(os.path.join(SDXL_DIR, "model_index.json")):
            logger.warning("Modèle SDXL absent (%s) — diffdiff désactivé "
                           "(exécuter download_sdxl.py).", SDXL_DIR)
            self._diffdiff_pipe = False
            return None

        controlnet_source = CONTROLNET_DIR
        if DIFFDIFF_USE_CONTROLNET and \
                not os.path.exists(os.path.join(CONTROLNET_DIR, "config.json")):
            try:
                from huggingface_hub import snapshot_download
                logger.info("ControlNet absent en local — téléchargement (~0.55 Go) "
                            "dans %s ...", CONTROLNET_DIR)
                controlnet_source = snapshot_download(
                    repo_id=CONTROLNET_MODEL_ID, local_dir=CONTROLNET_DIR,
                    allow_patterns=["config.json",
                                    "diffusion_pytorch_model.fp16.safetensors"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Téléchargement ControlNet échoué (%s) — diffdiff "
                               "désactivé.", exc)
                self._diffdiff_pipe = False
                return None

        try:
            if DIFFDIFF_USE_CONTROLNET:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_source, torch_dtype=torch.float16,
                    variant="fp16", use_safetensors=True,
                )
                pipe_cls = StableDiffusionXLControlNetInpaintPipeline
                extra_kwargs = {"controlnet": controlnet}
            else:
                logger.info("Diffdiff SANS ControlNet "
                            "(SOUVENIR_DIFFDIFF_NO_CONTROLNET=1).")
                pipe_cls = StableDiffusionXLInpaintPipeline
                extra_kwargs = {}
            try:
                pipe = pipe_cls.from_pretrained(
                    SDXL_DIR, torch_dtype=torch.float16,
                    variant="fp16", use_safetensors=True, **extra_kwargs,
                )
            except (OSError, ValueError):
                pipe = pipe_cls.from_pretrained(
                    SDXL_DIR, torch_dtype=torch.float16,
                    use_safetensors=True, **extra_kwargs,
                )

            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

            # Le mask_processor du pipeline inpaint BINARISE le masque par
            # défaut (do_binarize=True) — incompatible avec une carte de force
            # continue : on le remplace par un équivalent non-binarisant.
            from diffusers.image_processor import VaeImageProcessor
            pipe.mask_processor = VaeImageProcessor(
                vae_scale_factor=pipe.vae_scale_factor, do_normalize=False,
                do_binarize=False, do_convert_grayscale=True,
            )

            # IP-Adapter AVANT l'offload (hooks corrects sur ses modules).
            # Sans IP-Adapter, plus d'identité visiteur -> diffdiff inutilisable.
            self._diffdiff_ip_loaded = self._try_load_ip_adapter(pipe)
            if not self._diffdiff_ip_loaded:
                self._diffdiff_pipe = False
                return None

            # 8 Go de VRAM et UNet + ControlNet actifs à chaque pas : offload
            # obligatoire. Le mode séquentiel (plus lent) n'est activé qu'en
            # retry après OOM (voir generate_face_via_diffdiff).
            if self._diffdiff_sequential:
                logger.info("Offload séquentiel activé (retry après OOM).")
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            else:  # diffusers < 0.40
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
            pipe.set_progress_bar_config(disable=True)

            self._diffdiff_pipe = pipe
            logger.info("Pipeline diffdiff chargé (SDXL inpaint%s + IP-Adapter, "
                        "FP16%s).",
                        " + ControlNet depth" if DIFFDIFF_USE_CONTROLNET else "",
                        ", offload séquentiel" if self._diffdiff_sequential else "")
            return pipe
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chargement pipeline diffdiff échoué (%s) — diffdiff "
                           "désactivé.", exc)
            self._diffdiff_pipe = False
            return None

    @staticmethod
    def _encode_artwork_latents(pipe, pil_image, torch):
        """Encode l'image letterboxée en latents VAE (1, 4, H/8, W/8), à
        l'échelle scaling_factor, sur le device d'exécution du pipeline.

        Le VAE SDXL produit des NaN en fp16 (force_upcast=True dans sa config) :
        on l'upcast temporairement en float32 comme le fait diffusers en interne.
        """
        image = pipe.image_processor.preprocess(pil_image)  # (1,3,H,W), [-1,1]
        device = pipe._execution_device
        vae_dtype = pipe.vae.dtype
        upcast = (vae_dtype == torch.float16
                  and bool(getattr(pipe.vae.config, "force_upcast", False)))
        try:
            if upcast:
                pipe.vae.to(dtype=torch.float32)
            with torch.no_grad():
                posterior = pipe.vae.encode(
                    image.to(device=device, dtype=pipe.vae.dtype),
                )
                latents = posterior.latent_dist.sample()
        finally:
            if upcast:
                pipe.vae.to(dtype=vae_dtype)
        latents = (latents * pipe.vae.config.scaling_factor).to(
            device=device, dtype=pipe.unet.dtype,
        )
        if torch.isnan(latents).any():
            raise RuntimeError("NaN dans les latents VAE du tableau")
        return latents

    def generate_face_via_diffdiff(self, visitor_face_bgr, artwork_bgr,
                                   artwork_name, mask_full=None,
                                   gender_override=None):
        """Génération en UNE seule passe sur tout le canevas (Differential
        Diffusion) : plus de frontière « masque collé », car il n'y a jamais de
        composite — un seul processus de diffusion continu.

        Une carte de force continue module PAR PIXEL le pas de temps auquel
        chaque région rejoint le débruitage : centre du visage fortement
        re-généré, décroissance smoothstep autour, et le reste du tableau est
        re-verrouillé à chaque pas sur ses latents d'origine re-bruités. Un
        ControlNet depth contraint pose/composition du tableau ; l'IP-Adapter
        porte l'identité du visiteur.

        Implémentation : le checkpoint SDXL local est la variante INPAINTING
        (UNet 9 canaux), incompatible avec le pipeline communautaire diffdiff
        (img2img, 4 canaux). Le re-bruitage par pixel est donc appliqué via
        callback_on_step_end sur StableDiffusionXLControlNetInpaintPipeline,
        la carte de force servant aussi de masque continu au canal du UNet.

        mask_full : accepté pour symétrie d'API avec generate_face_via_ip_adapter,
        non utilisé (la carte de force est dérivée des landmarks du tableau).

        Retourne le canevas complet re-diffusé, au ratio natif du tableau
        (côté max plafonné à DIFFDIFF_OUTPUT_MAX_SIDE). Lève une exception en
        cas d'indisponibilité ou d'échec (generate_souvenir bascule alors sur
        generate_face_via_ip_adapter).
        """
        import time

        import torch
        from PIL import Image

        pipe = self._get_diffdiff_pipeline()
        if pipe is None:
            raise RuntimeError("Pipeline Differential Diffusion indisponible")

        # Prompt style peinture + genre (override manuel ou heuristique).
        gender = self._resolve_gender(visitor_face_bgr, gender_override)
        base = SDXL_PROMPTS.get(artwork_name, SDXL_PROMPTS["joconde"])
        pos_suffix, neg_suffix = _GENDER_PROMPT.get(gender, ("", ""))
        positive_prompt = base["positive"] + pos_suffix + _TEXTURE_POSITIVE
        negative_prompt = base["negative"] + neg_suffix + _TEXTURE_NEGATIVE

        # Carte de force + condition depth (si ControlNet actif), à la
        # résolution native du tableau.
        artwork_pts = self._load_artwork_landmarks(artwork_bgr, artwork_name)
        strength_map = self._generate_strength_map(artwork_pts, artwork_bgr.shape)

        # Letterbox 1024² commun : canevas COMPLET (pas de crop), carte, depth.
        square, _scale, pad_l, pad_t, new_w, new_h = self._letterbox_1024(artwork_bgr)
        map_sq = self._letterbox_1024(
            cv2.cvtColor(np.clip(strength_map * 255.0, 0, 255).astype(np.uint8),
                         cv2.COLOR_GRAY2BGR),
        )[0][:, :, 0].astype(np.float32) / 255.0

        pil_image = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        pil_map = Image.fromarray(
            np.clip(map_sq * 255.0, 0, 255).astype(np.uint8),
        )
        control_kwargs = {}
        if DIFFDIFF_USE_CONTROLNET:
            depth_cond = self._generate_controlnet_condition(artwork_bgr, artwork_name)
            depth_sq = self._letterbox_1024(depth_cond)[0]
            control_kwargs = {
                "control_image": Image.fromarray(
                    cv2.cvtColor(depth_sq, cv2.COLOR_BGR2RGB),
                ),
                "controlnet_conditioning_scale": CONTROLNET_SCALE,
            }
        ip_image = Image.fromarray(cv2.cvtColor(visitor_face_bgr, cv2.COLOR_BGR2RGB))

        num_train_t = int(pipe.scheduler.config.num_train_timesteps)

        def _run(pipe):
            pipe.set_ip_adapter_scale(self._ip_scale_for(gender))

            # Latents d'origine + bruit FIXE pour le re-bruitage par pixel.
            orig_latents = self._encode_artwork_latents(pipe, pil_image, torch)
            fixed_noise = torch.randn_like(orig_latents)
            lat_h, lat_w = orig_latents.shape[-2:]
            map_latent = torch.from_numpy(
                cv2.resize(map_sq, (lat_w, lat_h), interpolation=cv2.INTER_AREA)
            ).to(device=orig_latents.device, dtype=orig_latents.dtype)[None, None]

            def _diffdiff_renoise(p, i, _t, kwargs):
                # Après le pas i, les latents sont au timestep suivant t_next.
                # Tout pixel dont la force est inférieure à la fraction de bruit
                # restante est REMIS sur l'original re-bruité à t_next : il ne
                # rejoint le débruitage libre que lorsque frac <= sa force
                # (force 0 = jamais -> tableau original préservé, en latent,
                # sans composite).
                latents = kwargs["latents"]
                timesteps = p.scheduler.timesteps
                if i + 1 < len(timesteps):
                    t_next = timesteps[i + 1]
                    frac = float(t_next) / num_train_t
                    renoised = p.scheduler.add_noise(
                        orig_latents, fixed_noise, t_next.reshape(1),
                    )
                else:
                    frac = 0.0  # dernier pas : fond = latents propres d'origine
                    renoised = orig_latents
                freeze = (map_latent < max(frac, 1e-4)).to(latents.dtype)
                kwargs["latents"] = (
                    renoised.to(device=latents.device, dtype=latents.dtype) * freeze
                    + latents * (1.0 - freeze)
                )
                return kwargs

            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=pil_image,
                mask_image=pil_map,      # carte CONTINUE, pas de masque binaire
                ip_adapter_image=ip_image,
                num_inference_steps=DIFFDIFF_STEPS,
                strength=DIFFDIFF_STRENGTH,
                guidance_scale=7.5,
                width=1024,
                height=1024,
                callback_on_step_end=_diffdiff_renoise,
                callback_on_step_end_tensor_inputs=["latents"],
                **control_kwargs,
            ).images[0]
            elapsed = time.perf_counter() - start
            peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            logger.info("Diffdiff (%s ControlNet) : génération en %.1f s, "
                        "pic VRAM %.2f Go.",
                        "avec" if DIFFDIFF_USE_CONTROLNET else "sans",
                        elapsed, peak_gb)
            return result

        result = None
        try:
            result = _run(pipe)
        except torch.cuda.OutOfMemoryError:
            # NE PAS recharger ici : tant que ce bloc except est actif, le
            # traceback de l'OOM retient toutes les frames internes du pipeline
            # (donc leurs tenseurs GPU). Le retry se fait après le bloc.
            logger.warning("OOM avec model_cpu_offload — libération complète du "
                           "pipeline puis rechargement en sequential_cpu_offload.")

        if result is None:
            # Retry séquentiel : libère ENTIÈREMENT la première tentative
            # (hooks d'offload, UNet/VAE/ControlNet/encodeurs) AVANT de
            # recharger quoi que ce soit.
            del pipe
            self._unload_diffdiff_pipeline(
                "diffdiff : model_cpu_offload -> sequential_cpu_offload",
            )
            self._diffdiff_sequential = True
            pipe = self._get_diffdiff_pipeline()
            if pipe is None:
                raise RuntimeError("Rechargement diffdiff (séquentiel) échoué")
            result = _run(pipe)

        out_square = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

        # Dé-letterbox vers le ratio natif, côté max plafonné : le canevas
        # généré fait 1024², ré-agrandir à la résolution native (Joconde :
        # 9565 px) n'apporterait aucun détail.
        ah, aw = artwork_bgr.shape[:2]
        out_scale = min(1.0, DIFFDIFF_OUTPUT_MAX_SIDE / max(ah, aw))
        target_hw = (max(1, int(round(ah * out_scale))),
                     max(1, int(round(aw * out_scale))))
        return self._unletterbox(out_square, target_hw, pad_l, pad_t, new_w, new_h)

    # ------------------------------------------------------------------ #
    # Cadre + filigrane                                                 #
    # ------------------------------------------------------------------ #
    def _next_frame_variant(self, artwork_name):
        """Variante de cadre courante (1..3) par rotation persistante par œuvre."""
        path = os.path.join(DATA_DIR, "frame_counters.json")
        counters = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    counters = json.load(handle)
            except (ValueError, OSError):
                counters = {}
        count = int(counters.get(artwork_name, 0))
        variant = _FRAME_VARIANTS[count % len(_FRAME_VARIANTS)]
        return variant, counters, count, path

    @staticmethod
    def _commit_frame_variant(counters, artwork_name, count, path):
        counters[artwork_name] = count + 1
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(counters, handle)
        except OSError as exc:
            logger.warning("Écriture compteur de cadre échouée (%s).", exc)

    def apply_frame_overlay(self, image_bgr, artwork_name):
        """Applique un cadre PNG RGBA (rotation 1..3) + filigrane Kinophos.

        Si le PNG de cadre est absent, retourne l'image inchangée sans erreur
        (et sans faire tourner le compteur).
        """
        variant, counters, count, counters_path = self._next_frame_variant(artwork_name)
        frame_path = os.path.join(
            FRAMES_DIR, f"{artwork_name}_frame_{variant}.png"
        )
        if not os.path.exists(frame_path):
            logger.info("Cadre absent (%s) — image renvoyée inchangée.", frame_path)
            return image_bgr

        frame_rgba = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if frame_rgba is None or frame_rgba.ndim != 3 or frame_rgba.shape[2] < 4:
            logger.warning("Cadre PNG invalide (%s) — ignoré.", frame_path)
            return image_bgr

        h, w = image_bgr.shape[:2]
        frame_rgba = cv2.resize(frame_rgba, (w, h), interpolation=cv2.INTER_AREA)
        overlay_bgr = frame_rgba[:, :, :3].astype(np.float32)
        alpha = (frame_rgba[:, :, 3:4].astype(np.float32) / 255.0)
        blended = (image_bgr.astype(np.float32) * (1.0 - alpha)
                   + overlay_bgr * alpha).astype(np.uint8)

        result = self._draw_watermark(blended)
        self._commit_frame_variant(counters, artwork_name, count, counters_path)
        return result

    @staticmethod
    def _draw_watermark(image_bgr):
        text = "Kinophos - Artimir"
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        scale, thickness = 0.5, 1
        h, w = image_bgr.shape[:2]
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        margin = max(8, int(h * 0.02))
        org = (w - tw - margin, h - margin - baseline)

        overlay = image_bgr.copy()
        cv2.putText(overlay, text, org, font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        # alpha 0.4 : seuls les pixels du texte diffèrent -> filigrane translucide.
        return cv2.addWeighted(overlay, 0.4, image_bgr, 0.6, 0.0)

    # ------------------------------------------------------------------ #
    # Pipeline complet                                                  #
    # ------------------------------------------------------------------ #
    def generate_souvenir(self, visitor_frames, artwork_name, artwork_bgr,
                          gender_override=None):
        """Pipeline souvenir principal. Ne lève jamais.

        gender_override : "male" | "female" | "neutral" pour forcer le genre
        injecté dans les prompts SDXL (l'heuristique MediaPipe est alors
        ignorée) ; None = estimation automatique.

        Chaîne d'essai :
        1. generate_face_via_diffdiff (Differential Diffusion, une passe, sans
           frontière) ;
        2. generate_face_via_ip_adapter (inpaint masqué, existant) ;
        3. generate_souvenir_legacy (warp + composite).
        Chaque niveau logue un warning avec la méthode en échec et la cause.
        Retourne le BGR final (cadre/filigrane appliqués).
        """
        try:
            best_frame, best_score = self.find_best_frame(visitor_frames)
            logger.info("Meilleur frame sélectionné (score %.3f).", best_score)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sélection de frame échouée (%s).", exc)
            best_frame = None

        # NB : chaque fallback est appelé APRÈS la fin du bloc except de l'étage
        # précédent — un fallback lancé DANS le except garde vivant le traceback
        # de l'exception (souvent un OOM), qui retient les frames internes du
        # pipeline et donc ses tenseurs GPU pendant tout l'étage suivant.
        styled = None
        method = None
        if best_frame is not None:
            try:
                visitor_face = self._extract_face_crop(best_frame)
                artwork_pts = self._load_artwork_landmarks(artwork_bgr, artwork_name)
                mask_full = self._create_face_mask(
                    artwork_bgr, landmarks_cache=artwork_pts,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Préparation visage/masque échouée (%s) — "
                    "fallback generate_souvenir_legacy.", exc,
                )
                visitor_face = None

            if visitor_face is not None:
                try:
                    styled = self.generate_face_via_diffdiff(
                        visitor_face, artwork_bgr, artwork_name, mask_full,
                        gender_override=gender_override,
                    )
                    method = "diffdiff"
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "generate_face_via_diffdiff échoué (%s) — "
                        "fallback generate_face_via_ip_adapter.", exc,
                    )

                if styled is None:
                    # Purge le pipeline diffdiff AVANT de charger l'étage suivant.
                    self._unload_diffdiff_pipeline(
                        "transition diffdiff -> ip_adapter",
                    )
                    try:
                        styled = self.generate_face_via_ip_adapter(
                            visitor_face, artwork_bgr, artwork_name, mask_full,
                            gender_override=gender_override,
                        )
                        method = "ip_adapter"
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "generate_face_via_ip_adapter échoué (%s) — "
                            "fallback generate_souvenir_legacy.", exc,
                        )

                if styled is None:
                    # Purge le pipeline SDXL/IP-Adapter AVANT l'étage legacy.
                    self._unload_sdxl_pipeline("transition ip_adapter -> legacy")

        if styled is not None:
            self.last_method = method
            logger.info("=== Méthode ayant produit l'image finale : %s ===", method)
            try:
                return self.apply_frame_overlay(styled, artwork_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cadre échoué (%s) — sans cadre.", exc)
                return styled

        return self.generate_souvenir_legacy(
            visitor_frames, artwork_name, artwork_bgr,
            gender_override=gender_override,
        )

    def generate_souvenir_legacy(self, visitor_frames, artwork_name, artwork_bgr,
                                 gender_override=None):
        """Ancien pipeline (warp Delaunay + composite alpha + SDXL inpaint), gardé
        comme fallback silencieux. Ne lève jamais d'exception.
        """
        # 1. Sélection du meilleur frame -----------------------------------
        try:
            best_frame, best_score = self.find_best_frame(visitor_frames)
            logger.info("Meilleur frame sélectionné (score %.3f).", best_score)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sélection de frame échouée (%s) — tableau brut.", exc)
            best_frame = None

        # 2. Warp visage -> tableau ----------------------------------------
        warp_ok = False
        if best_frame is not None:
            try:
                warped = self.warp_face_to_artwork(best_frame, artwork_bgr, artwork_name)
                warp_ok = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warp visage échoué (%s) — tableau original conservé.", exc)
                warped = artwork_bgr.copy()
        else:
            warped = artwork_bgr.copy()

        # 3. Stylisation du visage : SDXL inpaint, repli AdaIN, puis sans style ---
        # Comme dans generate_souvenir : les fallbacks tournent APRÈS le bloc
        # except (traceback OOM relâché), et la VRAM est purgée entre étages.
        styled = None
        method = "legacy_sans_warp" if not warp_ok else "legacy_sans_style"
        if warp_ok:
            gender = self._resolve_gender(best_frame, gender_override)
            try:
                styled = self.apply_sdxl_inpainting(
                    warped, artwork_bgr, artwork_name, gender=gender,
                )
                method = "legacy_sdxl_inpaint"
            except Exception as exc:  # noqa: BLE001
                logger.warning("SDXL échoué (%s) — fallback AdaIN.", exc)

            if styled is None:
                # Purge le pipeline SDXL AVANT de charger AdaIN.
                self._unload_sdxl_pipeline("transition legacy SDXL -> AdaIN")
                try:
                    styled = self.apply_style_transfer_face(
                        warped, artwork_bgr, artwork_name,
                    )
                    method = "legacy_adain"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Fallback AdaIN échoué (%s) — sans style.", exc)

        if styled is None:
            styled = warped

        self.last_method = method
        logger.info("=== Méthode ayant produit l'image finale : %s ===", method)

        # 4. Cadre + filigrane ---------------------------------------------
        try:
            final = self.apply_frame_overlay(styled, artwork_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Application du cadre échouée (%s) — sans cadre.", exc)
            final = styled

        return final

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


# ---------------------------------------------------------------------- #
# AdaIN : architectures et opérateur (d'après naoto0804/pytorch-AdaIN)   #
# ---------------------------------------------------------------------- #
def _load_adain_state_dict(torch, path):
    """Charge un state_dict AdaIN en retirant un éventuel préfixe 'net.'.

    Certains checkpoints encapsulent le nn.Sequential dans un attribut self.net et
    préfixent donc toutes les clés par 'net.' (cas du décodeur ici), ce que nos
    architectures « nues » n'attendent pas. Le strip n'a lieu que si TOUTES les clés
    sont préfixées, donc c'est un no-op sûr pour le VGG (non préfixé).
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict and all(k.startswith("net.") for k in state_dict):
        state_dict = {k[len("net."):]: value for k, value in state_dict.items()}
    return state_dict


def _adain_normalize(content_feat, style_feat, eps=1e-5):
    """AdaIN : aligne la moyenne/écart-type du contenu sur ceux du style."""
    size = content_feat.size()
    n, c = size[:2]

    def mean_std(feat):
        var = feat.view(n, c, -1).var(dim=2) + eps
        std = var.sqrt().view(n, c, 1, 1)
        mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
        return mean, std

    style_mean, style_std = mean_std(style_feat)
    content_mean, content_std = mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _build_adain_decoder(nn):
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


def _build_adain_vgg(nn):
    # VGG19 "normalised" complet ; on n'utilisera que les 31 premières couches.
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),          # relu1_1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),         # relu1_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),        # relu2_1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),       # relu2_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),       # relu3_1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),       # relu3_2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),       # relu3_3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),       # relu3_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),       # relu4_1  <- couche 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),       # relu4_2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),       # relu4_3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),       # relu4_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),       # relu5_1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),       # relu5_2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),       # relu5_3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),       # relu5_4
    )
