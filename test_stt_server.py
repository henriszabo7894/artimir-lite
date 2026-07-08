"""Serveur de test autonome pour valider le pipeline micro (telephone) -> transcription
avant integration dans mapping_server.py. Completement independant du reste du projet.

Usage : python test_stt_server.py
Puis ouvrir l'URL affichee au demarrage depuis le telephone (meme reseau local).
"""

import socket
import tempfile
import time

import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel

PORT = 8001
WHISPER_MODEL_SIZE = "small"

app = FastAPI(title="Artimir STT Test")

print(f"Chargement du modele Whisper ({WHISPER_MODEL_SIZE}) ...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
print("Modele Whisper charge.")


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


PAGE_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Test STT Artimir</title>
<style>
    body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 24px;
        font-family: sans-serif;
        background: #111;
        color: #eee;
        padding: 24px;
        box-sizing: border-box;
        text-align: center;
    }
    #talkBtn {
        width: 70vw;
        max-width: 320px;
        height: 70vw;
        max-height: 320px;
        border-radius: 50%;
        border: none;
        font-size: 1.3rem;
        font-weight: bold;
        background: #d33;
        color: white;
        cursor: pointer;
    }
    #talkBtn.recording {
        background: #2a2;
    }
    #status {
        font-size: 1rem;
        color: #aaa;
        min-height: 1.5em;
    }
    #result {
        font-size: 1.2rem;
        max-width: 90vw;
        white-space: pre-wrap;
    }
</style>
</head>
<body>
    <button id="talkBtn">Appuyer pour parler</button>
    <div id="status"></div>
    <div id="result"></div>

<script>
    const talkBtn = document.getElementById("talkBtn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");

    let mediaRecorder = null;
    let chunks = [];
    let recording = false;

    talkBtn.addEventListener("click", async () => {
        if (!recording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                chunks = [];
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach((track) => track.stop());

                    const blob = new Blob(chunks, { type: "audio/webm" });
                    statusEl.textContent = "Transcription en cours...";

                    const formData = new FormData();
                    formData.append("audio", blob, "recording.webm");

                    try {
                        const resp = await fetch("/transcribe", { method: "POST", body: formData });
                        const data = await resp.json();
                        resultEl.textContent = data.text;
                        statusEl.textContent = `Transcrit en ${data.duration_sec.toFixed(2)}s`;
                    } catch (err) {
                        statusEl.textContent = "Erreur : " + err;
                    }
                };

                mediaRecorder.start();
                recording = true;
                talkBtn.textContent = "Appuyer pour arreter";
                talkBtn.classList.add("recording");
                statusEl.textContent = "Enregistrement en cours...";
                resultEl.textContent = "";
            } catch (err) {
                statusEl.textContent = "Erreur micro : " + err;
            }
        } else {
            mediaRecorder.stop();
            recording = false;
            talkBtn.textContent = "Appuyer pour parler";
            talkBtn.classList.remove("recording");
        }
    });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE_HTML


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        start = time.perf_counter()
        segments, _info = whisper_model.transcribe(tmp.name, language="fr")
        text = "".join(segment.text for segment in segments).strip()
        duration_sec = time.perf_counter() - start

    return {"text": text, "duration_sec": duration_sec}


if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"\nOuvre cette URL sur ton telephone (meme reseau local) :")
    print(f"  http://{local_ip}:{PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
