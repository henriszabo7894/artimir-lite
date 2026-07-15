"""
Test headless du pipeline SESSION -> SOUVENIR de mapping_server.py.

But : valider le câblage (buffer de frames pendant la session, sélection du
meilleur candidat, appel de generate_souvenir avec des frames RÉELLES, écriture
du fichier, exposition via l'endpoint) SANS webcam physique ni boucle de rendu.

Le flux C920 est simulé en poussant de vrais visages (test_frames/*) dans
state.webcam_frame, exactement comme le fait main_loop() à chaque itération.
La sélection/scoring/génération, elle, n'est PAS simulée : c'est le vrai code.

Sur le poste GPU réel, le test « vrai C920 » consiste simplement à lancer
`python mapping_server.py` puis à piloter les endpoints
/command/session/{start,capture,status} et /souvenir/{id}.

Usage :
    python test_session_souvenir.py --frames_dir test_frames/ --artwork joconde
"""

import argparse
import glob
import os
import threading
import time

import cv2

import mapping_server as ms

_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_frames(frames_dir):
    paths = []
    for pattern in _IMAGE_GLOBS:
        paths.extend(glob.glob(os.path.join(frames_dir, pattern)))
    paths = sorted(set(paths))
    frames = [img for img in (cv2.imread(p) for p in paths) if img is not None]
    return frames


def main():
    parser = argparse.ArgumentParser(description="Test session -> souvenir (headless)")
    parser.add_argument("--frames_dir", default="test_frames/")
    parser.add_argument("--artwork", default="joconde")
    parser.add_argument("--feed_seconds", type=float, default=6.0,
                        help="Durée d'alimentation du buffer avant capture")
    args = parser.parse_args()

    frames = load_frames(args.frames_dir)
    if not frames:
        print(f"[TEST] Aucun frame dans {args.frames_dir}")
        raise SystemExit(1)
    print(f"[TEST] {len(frames)} frame(s) réel(s) chargé(s) — simulation flux C920.")

    # Échantillonnage rapide pour ne pas attendre : on ne teste pas le timing.
    ms.SESSION_SAMPLE_INTERVAL = 0.3

    # Le worker lit state.running (True par défaut) et boucle.
    worker = threading.Thread(target=ms.session_worker, daemon=True)
    worker.start()

    # 1) Démarrage de session (comme POST /command/session/start).
    session_id = ms.start_session(args.artwork)

    # 2) On « joue » le flux webcam : main_loop publierait state.webcam_frame à
    #    chaque itération. Ici on cycle sur les vrais visages de test_frames.
    print(f"[TEST] Alimentation du buffer pendant {args.feed_seconds:.0f}s...")
    t_end = time.time() + args.feed_seconds
    i = 0
    while time.time() < t_end:
        with ms.state.lock:
            ms.state.webcam_frame = frames[i % len(frames)].copy()
        i += 1
        time.sleep(0.05)

    with ms.state.lock:
        buffered = len(ms.state.souvenir_buffer)
        scores = [round(s, 3) for (s, _f) in ms.state.souvenir_buffer]
    print(f"[TEST] Buffer rempli : {buffered} candidat(s), scores={scores}")
    assert buffered > 0, "Le buffer ne s'est pas rempli — scoring live KO"

    # 3) Capture explicite (comme POST /command/session/capture).
    with ms.state.lock:
        ms.state.session_capture_requested = True
    print("[TEST] Capture demandée -> génération en cours...")

    # 4) Attente du résultat (le worker passe generating -> resultReady/error).
    deadline = time.time() + 600
    while time.time() < deadline:
        with ms.state.lock:
            status = ms.state.session_status
            path = ms.state.souvenir_path
        if status in (ms.SESSION_RESULT_READY, ms.SESSION_ERROR):
            break
        time.sleep(1.0)

    print(f"[TEST] Statut final : {status}")
    if status != ms.SESSION_RESULT_READY:
        print("[TEST] Génération en échec (voir logs ci-dessus).")
        raise SystemExit(2)

    assert path and os.path.exists(path), f"Fichier souvenir absent : {path}"
    img = cv2.imread(path)
    print(f"[TEST] Souvenir écrit : {path} ({img.shape[1]}x{img.shape[0]}) — OK")

    # 5) Vérifie l'exposition via l'endpoint FastAPI GET /souvenir/{id}.
    from fastapi.testclient import TestClient
    client = TestClient(ms.app)
    r = client.get(f"/souvenir/{session_id}")
    print(f"[TEST] GET /souvenir/{session_id} -> HTTP {r.status_code}, "
          f"{len(r.content)} octets, {r.headers.get('content-type')}")
    assert r.status_code == 200 and len(r.content) > 0, "Endpoint souvenir KO"

    s = client.get("/command/session/status").json()
    print(f"[TEST] /command/session/status -> {s}")

    # 6) Vérifie la purge du buffer entre sessions (Tâche 3).
    with ms.state.lock:
        remaining = len(ms.state.souvenir_buffer)
        active = ms.state.session_active
    print(f"[TEST] Après session : buffer={remaining}, active={active}")
    assert remaining == 0 and not active, "Buffer non purgé en fin de session !"

    print("\n[TEST] === SUCCÈS : buffer -> meilleur frame -> generate_souvenir "
          "(frames réelles) -> fichier -> endpoint -> purge ===")


if __name__ == "__main__":
    main()
