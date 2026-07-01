import cv2
import threading
import os
import time
import uvicorn
import subprocess
import signal
import math
import wave
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.responses import RedirectResponse

narration_cap = cv2.VideoCapture("driving_narration.mp4")

try:
    from ai_engine import PortraitAnimator
except ImportError:
    print("[-] ERREUR : 'ai_engine.py' est introuvable.")
    exit()

try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh
    _MEDIAPIPE_OK = True
except (ImportError, AttributeError):
    _MEDIAPIPE_OK = False
    print("[WARN] mediapipe non installé — coordonnées visage fixes (pip install mediapipe)")


class EngineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.frozen = False
        self.capture_requested = False
        self.artwork = "joconde"
        self.output_path = "output/portrait.jpg"
        self.last_frame = None
        self.stream_frame = None
        self.valid_artworks = ["joconde", "vangogh", "courbet"]

        self.narration_process = None
        self.narration_paused = False
        self.narration_active = False
        self.narration_start_time = 0
        self.narration_wav_path = None
        self.narration_audio_levels = []
        self.narration_base_frame = None  # image propre figée au démarrage narration
        self.face_zones = None            # {"mouth": (x1,y1,x2,y2), "eyes": (x1,y1,x2,y2)}
        self.age_group = "adulte"         # enfant | ado | adulte


state = EngineState()
app = FastAPI(title="Artimir API")


AUDIO_PATHS = {
    "joconde": {
        "enfant": "audio/audio_joconde_enfant.mp3.mpeg",
        "ado":    "audio/audio_joconde_ado.mp3.mpeg",
        "adulte": "audio/audio_joconde_adulte.mp3.mpeg",
    },
    "vangogh": {
        "enfant": "audio/audio_vangogh_enfant.mp3.mpeg",
        "ado":    "audio/audio_vangogh_ado.mp3.mpeg",
        "adulte": "audio/audio_vangogh_adulte.mp3.mpeg",
    },
    "courbet": {
        "enfant": "audio/audio_courbet_enfant.mp3.mpeg",
        "ado":    "audio/audio_courbet_ado.mp3.mpeg",
        "adulte": "audio/audio_courbet_adulte.mp3.mpeg",
    },
}


NARRATIONS = {
    "joconde": {
        "enfant": (
            "Regarde ce tableau. C’est la Joconde, peinte par Léonard de Vinci. "
            "Elle a un sourire mystérieux. On ne sait pas si elle est contente ou triste. "
            "C’est la peinture la plus connue du monde !"
        ),
        "ado": (
            "La Joconde, ou Mona Lisa, c’est l’œuvre de Léonard de Vinci. "
            "Ce qui la rend fascinante, c’est son sourire ambigu et son regard qui semble te suivre. "
            "Léonard a utilisé une technique appelée sfumato pour fondre les contours et créer cette douceur."
        ),
        "adulte": (
            "Léonard de Vinci est l’un des plus grands génies de la Renaissance. "
            "Il était peintre, inventeur, scientifique et ingénieur. "
            "La Joconde, aussi appelée Mona Lisa, est son œuvre la plus célèbre. "
            "Elle fascine par son sourire mystérieux, son regard vivant et la douceur de ses expressions. "
            "Cette peinture montre le talent exceptionnel de Léonard pour représenter l’humain, la lumière et les émotions."
        ),
    },

    "vangogh": {
        "enfant": (
            "Ce tableau, c’est Vincent Van Gogh qui l’a peint. C’est lui-même sur le tableau ! "
            "Il aimait les couleurs vives et les gros traits de pinceau. "
            "Il a peint des centaines de tableaux dans sa vie."
        ),
        "ado": (
            "Van Gogh a peint cet autoportrait avec des couleurs intenses et des coups de pinceau expressifs. "
            "Il a eu une vie difficile, mais il a produit plus de neuf cents tableaux. "
            "Son style a totalement changé la façon dont on pense la peinture moderne."
        ),
        "adulte": (
            "Vincent Van Gogh est un peintre néerlandais du dix-neuvième siècle. "
            "Il est connu pour ses couleurs puissantes, ses coups de pinceau expressifs et sa grande sensibilité. "
            "Son autoportrait permet de comprendre son regard intérieur et son émotion. "
            "Van Gogh a transformé la peinture en une manière très personnelle d’exprimer les sentiments. "
            "Aujourd’hui, il est considéré comme l’un des artistes les plus importants de l’histoire de l’art."
        ),
    },

    "courbet": {
        "enfant": (
            "Ce monsieur sur le tableau s’appelle Gustave Courbet. Il s’est peint lui-même. "
            "Il a l’air très surpris, tu ne trouves pas ? "
            "Il voulait montrer de vraies émotions dans ses peintures."
        ),
        "ado": (
            "Gustave Courbet, c’est le peintre réaliste par excellence. "
            "Dans ce tableau appelé Le Désespéré, il se représente lui-même avec une expression de détresse totale. "
            "Il voulait peindre la vérité des émotions humaines, sans embellir."
        ),
        "adulte": (
            "Gustave Courbet est un peintre français du dix-neuvième siècle. "
            "Il est l’un des grands représentants du réalisme. "
            "Son tableau Le Désespéré montre un homme bouleversé, avec un regard intense et une expression très forte. "
            "Cette œuvre est célèbre car elle transmet directement l’émotion au spectateur. "
            "Courbet voulait peindre la réalité, les sentiments humains et la vérité des visages."
        ),
    },
}


def crop_square_face_zone(img):
    h, w = img.shape[:2]
    crop_size = min(h, w)

    start_x = (w - crop_size) // 2
    start_y = max(0, int((h - crop_size) * 0.2))

    cropped = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cv2.resize(cropped, (512, 512))


def compute_audio_levels(wav_path, fps=30):
    levels = []

    try:
        with wave.open(wav_path, "rb") as wav:
            framerate = wav.getframerate()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()

            samples_per_frame = int(framerate / fps)

            while True:
                frames = wav.readframes(samples_per_frame)

                if not frames:
                    break

                if sample_width == 2:
                    data = np.frombuffer(frames, dtype=np.int16)
                else:
                    data = np.frombuffer(frames, dtype=np.uint8)

                if n_channels > 1:
                    data = data[::n_channels]

                if len(data) == 0:
                    levels.append(0)
                    continue

                volume = np.sqrt(np.mean(data.astype(np.float32) ** 2))
                levels.append(volume)

        if not levels:
            return []

        max_level = max(levels)

        if max_level == 0:
            return [0 for _ in levels]

        levels = [min(1.0, level / max_level) for level in levels]

        return levels

    except Exception as e:
        print("[AUDIO] Erreur analyse audio :", e)
        return []


def detect_face_landmarks(img):
    """
    Détecte les zones bouche et yeux avec MediaPipe Face Mesh.
    Retourne {"mouth": (x1,y1,x2,y2), "eyes": (x1,y1,x2,y2)} ou None.
    Appelée une seule fois au démarrage de la narration.
    """
    if not _MEDIAPIPE_OK or img is None:
        return None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4
    ) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        print("[MEDIAPIPE] Aucun visage détecté — coordonnées fixes utilisées")
        return None

    lm = res.multi_face_landmarks[0].landmark

    def px(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    # Lèvres : contour extérieur complet
    mouth_ids = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    mouth_pts = [px(i) for i in mouth_ids]
    mx = [p[0] for p in mouth_pts]
    my = [p[1] for p in mouth_pts]
    pad_m = max(8, int(h * 0.018))
    mouth_zone = (max(0, min(mx) - pad_m), max(0, min(my) - pad_m),
                  min(w, max(mx) + pad_m), min(h, max(my) + pad_m))

    # Yeux : œil gauche + œil droit réunis en une seule bande
    eye_ids = [33, 133, 160, 158, 153, 144,   # œil gauche
               362, 263, 387, 385, 380, 373]   # œil droit
    eye_pts = [px(i) for i in eye_ids]
    ex = [p[0] for p in eye_pts]
    ey = [p[1] for p in eye_pts]
    pad_e = max(10, int(h * 0.022))
    eyes_zone = (max(0, min(ex) - pad_e), max(0, min(ey) - pad_e),
                 min(w, max(ex) + pad_e), min(h, max(ey) + pad_e))

    print(f"[MEDIAPIPE] Bouche détectée : {mouth_zone} | Yeux : {eyes_zone}")
    return {"mouth": mouth_zone, "eyes": eyes_zone}


def animate_face_simple(frame, t, mouth_level=0.0, face_zones=None):
    if frame is None:
        return frame

    h, w = frame.shape[:2]

    # =========================
    # 1. MOUVEMENT DE TÊTE (sans zoom)
    # =========================
    angle = 1.2 * math.sin(t * 0.8)
    scale = 1.0                          # pas de zoom
    dx = 3 * math.sin(t * 0.9)
    dy = 2 * math.sin(t * 1.2)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    result = cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # =========================
    # 2. BOUCHE
    # =========================
    if face_zones and "mouth" in face_zones:
        mx1, my1, mx2, my2 = face_zones["mouth"]
    else:
        mx1, my1 = int(w * 0.40), int(h * 0.54)
        mx2, my2 = int(w * 0.60), int(h * 0.61)

    mouth = result[my1:my2, mx1:mx2].copy()
    mh, mw_m = mouth.shape[:2]

    if mh > 5 and mw_m > 5:
        amp = int(4 + mouth_level * 18)

        if amp > 0:
            stretched = cv2.resize(mouth, (mw_m, mh + amp))
            stretched = stretched[amp // 2:amp // 2 + mh, :]

            mask = np.zeros((mh, mw_m), dtype=np.float32)
            cv2.ellipse(
                mask,
                (mw_m // 2, mh // 2),
                (mw_m // 3, mh // 3),
                0, 0, 360, 1, -1
            )
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask3 = mask[:, :, None]

            blended = mouth.astype(np.float32) * (1 - mask3) + stretched.astype(np.float32) * mask3
            result[my1:my2, mx1:mx2] = blended.astype(np.uint8)

    # =========================
    # 3. YEUX
    # =========================
    if face_zones and "eyes" in face_zones:
        ex1, ey1, ex2, ey2 = face_zones["eyes"]
    else:
        ex1, ey1 = int(w * 0.32), int(h * 0.36)
        ex2, ey2 = int(w * 0.68), int(h * 0.44)

    eyes = result[ey1:ey2, ex1:ex2].copy()
    shift = int(3 * math.sin(t * 0.7))

    if eyes.size > 0 and shift != 0:
        moved = np.roll(eyes, shift, axis=1)

        if shift > 0:
            moved[:, :shift] = eyes[:, :shift]
        else:
            moved[:, shift:] = eyes[:, shift:]

        result[ey1:ey2, ex1:ex2] = cv2.addWeighted(eyes, 0.65, moved, 0.35, 0)

    return result


def stop_narration_process():
    with state.lock:
        proc = state.narration_process
        state.narration_process = None
        state.narration_paused = False
        state.narration_active = False
        state.narration_base_frame = None
        state.face_zones = None

    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass


def narration_finished_watcher(proc):
    proc.wait()

    with state.lock:
        if state.narration_process == proc:
            state.narration_process = None
            state.narration_paused = False
            state.narration_active = False
            state.frozen = False

    print("[NARRATION] Terminée -> retour au live")


def start_narration_for_current_artwork():
    stop_narration_process()

    with state.lock:
        artwork = state.artwork
        state.narration_active = True
        state.narration_paused = False
        state.narration_start_time = time.time()
        state.frozen = False
        state.narration_base_frame = state.last_frame.copy() if state.last_frame is not None else None
        _base_for_detection = state.narration_base_frame.copy() if state.narration_base_frame is not None else None

    # Détection MediaPipe hors du lock
    if _base_for_detection is not None:
        zones = detect_face_landmarks(_base_for_detection)
        with state.lock:
            state.face_zones = zones

    with state.lock:
        age_group = state.age_group

    if age_group not in ("enfant", "ado", "adulte"):
        age_group = "adulte"

    print(f"[NARRATION] {artwork} - {age_group}")

    audio_file = AUDIO_PATHS.get(artwork, {}).get(age_group)

    if not audio_file or not os.path.exists(audio_file):
        print("[ERREUR] Audio introuvable :", audio_file)
        with state.lock:
            state.narration_active = False
            state.narration_paused = False
            state.frozen = False
        return False

    RASPBERRY_IP = "192.168.0.20"

    proc = subprocess.Popen([
        "ffmpeg",
        "-re",
        "-i", audio_file,
        "-vn",
        "-filter:a", "volume=9.0",
        "-c:a", "aac",
        "-f", "mpegts",
        f"udp://{RASPBERRY_IP}:1235"
    ])

    with state.lock:
        state.narration_process = proc
        state.narration_paused = False
        state.narration_active = True
        state.narration_start_time = time.time()
        state.narration_audio_levels = []

    threading.Thread(
        target=narration_finished_watcher,
        args=(proc,),
        daemon=True
    ).start()

    print(f"[NARRATION] START -> {artwork} - {age_group}")
    return True

@app.get("/status")
def get_status():
    with state.lock:
        return {
            "running": state.running,
            "frozen": state.frozen,
            "artwork": state.artwork,
            "narration_active": state.narration_active,
            "narration_paused": state.narration_paused,
            "last_capture": state.output_path if os.path.exists(state.output_path) else None
        }
def generate_video_stream():
    while True:
        with state.lock:
            frame = state.stream_frame.copy() if state.stream_frame is not None else None

        if frame is not None:
            # On réduit le flux envoyé au Raspberry
            frame = cv2.resize(frame, (640, 640))

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 45]
            )

            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" +
                    buffer.tobytes() +
                    b"\r\n"
                )

        # 10 images/seconde environ pour éviter la latence
        time.sleep(0.10)



@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )
@app.get("/", response_class=HTMLResponse)
def fullscreen_display():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Artimir Display</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background: black;
            overflow: hidden;
        }

        img {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            background: black;
        }
    </style>
</head>
<body>
    <img src="/video_feed">
</body>
</html>
"""
@app.post("/command/age/{age_group}")
def set_age(age_group: str):
    if age_group not in ("enfant", "ado", "adulte"):
        raise HTTPException(status_code=400, detail="age_group invalide (enfant | ado | adulte)")

    with state.lock:
        state.age_group = age_group

    print(f"[AGE] -> {age_group}")
    return {"status": "ok", "age_group": age_group}


@app.post("/command/start/{artwork_name}")
def start_artwork(artwork_name: str):
    if artwork_name not in state.valid_artworks:
        raise HTTPException(status_code=400, detail=f"Artwork '{artwork_name}' inconnu")

    stop_narration_process()

    with state.lock:
        state.artwork = artwork_name
        state.frozen = False

    print(f"[API] START -> {artwork_name}")
    return {"status": "ok", "action": "start", "artwork": artwork_name}


@app.post("/command/freeze")
def freeze_engine():
    with state.lock:
        state.frozen = True

    print("[API] FREEZE")
    return {"status": "ok", "action": "freeze"}


@app.post("/command/resume")
def resume_engine():
    with state.lock:
        state.frozen = False

    print("[API] RESUME")
    return {"status": "ok", "action": "resume"}


@app.post("/command/capture")
def capture_frame():
    with state.lock:
        state.capture_requested = True

    print("[API] CAPTURE REQUESTED")
    return {"status": "ok", "action": "capture", "file": state.output_path}


@app.get("/capture/file")
def get_capture_file():
    if os.path.exists(state.output_path):
        return FileResponse(
            state.output_path,
            media_type="image/jpeg",
            filename="portrait.jpg"
        )

    raise HTTPException(status_code=404, detail="Portrait non trouvé sur le PC")


@app.post("/command/stop")
def stop_engine():
    stop_narration_process()

    with state.lock:
        state.running = False

    print("[API] STOP SERVER")
    return {"status": "ok", "action": "stop"}


@app.post("/command/narration/start/{age_group}")
def narration_start(age_group: str):
    if age_group not in ("enfant", "ado", "adulte"):
        age_group = "adulte"

    with state.lock:
        state.age_group = age_group

    ok = start_narration_for_current_artwork()

    if not ok:
        raise HTTPException(status_code=500, detail="Impossible de lancer la narration")

    return {"status": "ok", "action": "narration_start", "age_group": age_group}


@app.post("/command/narration/pause")
def narration_pause():
    with state.lock:
        proc = state.narration_process

        if proc and proc.poll() is None:
            os.kill(proc.pid, signal.SIGSTOP)
            state.narration_paused = True
            state.narration_active = True
            print("[NARRATION] PAUSE")
            return {"status": "ok", "action": "narration_pause"}

    return {"status": "ok", "action": "nothing_to_pause"}


@app.post("/command/narration/resume")
def narration_resume():
    with state.lock:
        proc = state.narration_process

        if proc and proc.poll() is None:
            os.kill(proc.pid, signal.SIGCONT)
            state.narration_paused = False
            state.narration_active = True
            print("[NARRATION] RESUME")
            return {"status": "ok", "action": "narration_resume"}

    return {"status": "ok", "action": "nothing_to_resume"}


@app.post("/command/narration/stop")
def narration_stop():
    stop_narration_process()

    with state.lock:
        state.frozen = False

    print("[NARRATION] STOP")
    return {"status": "ok", "action": "narration_stop"}


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

ffmpeg_process = None

def start_ffmpeg_stream():
    global ffmpeg_process

    command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "512x512",
        "-r", "20",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "mpegts",
        "udp://0.0.0.0:1234"
    ]

    ffmpeg_process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE
    )

    return ffmpeg_process


def main_loop():
    try:
        animator = PortraitAnimator()
    except Exception as e:
        print(f"[-] Erreur Init IA : {e}")
        return

    artworks_files = {
        "joconde": "joconde.jpg",
        "vangogh": "vangogh.jpg",
        "courbet": "courbet.jpg"
    }

    artworks_cache = {}

    for name, path in artworks_files.items():
        img = cv2.imread(path)

        if img is not None:
            artworks_cache[name] = img
            print(f"[INIT] {path} chargé.")
        else:
            print(f"[ALERTE] Fichier manquant : {path}")

    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    time.sleep(1)

    if not cap.isOpened():
        print("[-] ERREUR : Webcam introuvable.")
        return

    threading.Thread(target=run_server, daemon=True).start()
    ffmpeg_stream = start_ffmpeg_stream()

    current_artwork_name = None
    source_img = None

    print("\n[MOTEUR] Serveur Mapping prêt. En attente du Raspberry Pi...")

    while True:
        with state.lock:
            if not state.running:
                break

            target_artwork = state.artwork
            is_frozen = state.frozen
            capture_req = state.capture_requested
            narration_active = state.narration_active
            narration_paused = state.narration_paused
            narration_start_time = state.narration_start_time
            audio_levels = state.narration_audio_levels

        if target_artwork != current_artwork_name:
            raw_source = artworks_cache.get(target_artwork)

            if raw_source is not None:
                source_img = crop_square_face_zone(raw_source)
                animator.reset_cache()

                with state.lock:
                    state.last_frame = None

                current_artwork_name = target_artwork
                print(f"[MOTEUR] Bascule sur : {target_artwork}")

        ret, frame = cap.read()

        if not ret:
            print("[-] ERREUR : impossible de lire la webcam.")
            break

        frame = cv2.flip(frame, 1)

        if source_img is not None:

            if narration_active:
                if narration_paused:
                    with state.lock:
                        final_frame = state.last_frame.copy() if state.last_frame is not None else source_img.copy()
                else:
                    ret_anim, driving_frame = narration_cap.read()

                    if not ret_anim:
                        narration_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_anim, driving_frame = narration_cap.read()

                    if ret_anim:
                        driving_resized = cv2.resize(driving_frame, (512, 512))
                        final_frame = animator.generate_frame(source_img, driving_resized, narration_mode=True)
                    else:
                        final_frame = source_img.copy()

                    with state.lock:
                        state.last_frame = final_frame.copy()

            elif is_frozen:
                with state.lock:
                    final_frame = state.last_frame.copy() if state.last_frame is not None else source_img.copy()

            else:
                driving_resized = cv2.resize(frame, (512, 512))
                final_frame = animator.generate_frame(source_img, driving_resized)

                with state.lock:
                    state.last_frame = final_frame.copy()

        else:
            final_frame = frame
        
        with state.lock:
            state.stream_frame = final_frame.copy()

        if capture_req:
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(state.output_path, final_frame)

            print(f"[PHOTO] Portrait enregistré dans {state.output_path}")

            with state.lock:
                state.capture_requested = False

        try:
            ffmpeg_stream.stdin.write(final_frame.tobytes())
        except Exception:
            pass

        cv2.imshow("ARTIMIR - SERVEUR GPU", final_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_narration_process()
    cap.release()
    cv2.destroyAllWindows()
    print("[MOTEUR] Arrêt proprement effectué.")

if __name__ == "__main__":
    main_loop()