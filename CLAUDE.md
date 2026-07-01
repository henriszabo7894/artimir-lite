# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Artimir 2.0** — a real-time face mapping installation that animates painted portraits (La Joconde, Van Gogh self-portrait, Courbet's Le Désespéré) using a webcam feed and the LivePortrait deep learning model. Designed to run on a GPU workstation (RTX 5080) with commands arriving from a Raspberry Pi over HTTP.

## Network topology

| Host | IP | Role |
|------|----|------|
| PC Linux (GPU) | 192.168.0.10 | Runs `mapping_server.py`, serves HTTP :8000, streams video and audio |
| Raspberry Pi | 192.168.0.20 | Displays video, plays audio, sends HTTP commands |

**Video**: PC exposes an MJPEG stream at `GET /video_feed`; Raspberry Pi reads it with `mpv` (`scripts/raspberry/start_display.sh`). Also broadcasts raw frames as UDP MPEGTS on port 1234 (via ffmpeg subprocess in `start_ffmpeg_stream()`).

**Audio**: PC streams narration audio with ffmpeg as AAC/MPEGTS over UDP to `192.168.0.20:1235`; Raspberry Pi receives with `mpv --no-video` (`scripts/raspberry/start_audio.sh`).

**Raspberry Pi startup**: `scripts/raspberry/start_artimir.sh` launches both display and audio in parallel tmux sessions.

## Setup

```bash
# Activate virtualenv
source .venv/bin/activate

# Install torch/torchvision separately with CUDA support matching your GPU driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Verify CUDA is working
python test_cuda.py

# Download LivePortrait weights (~500MB) from HuggingFace into ./weights/
python download_weights.py

# Fetch the vendored LivePortrait source into ./liveportrait_src/ (from KwaiVGI/LivePortrait on GitHub)
python setup_liveportrait.py

# Fetch project-specific assets (portrait images + narration video) from the
# Henri-maker/artimir HuggingFace dataset
python download_assets.py
```

The repo is kept intentionally light: `weights/`, `liveportrait_src/`, the portrait images (`joconde.jpg`, `vangogh.jpg`, `courbet.jpg`), and `driving_narration.mp4` are all git-ignored and fetched on demand by the scripts above rather than committed.

## Running

```bash
# Full server mode (production): starts FastAPI + webcam pipeline, listens on :8000
python mapping_server.py

# Simple real-time demo (no HTTP server, adds FPS/latency overlay, joconde only)
python main.py
```

## Architecture

The system has three logical layers:

### 1. AI Engine (`ai_engine.py`) — `PortraitAnimator`
Loads 5 LivePortrait PyTorch models from `./weights/` into VRAM at startup:
- `appearance_feature_extractor` — extracts visual features from the source portrait (cached after first frame)
- `motion_extractor` — extracts 3D keypoints (pitch/yaw/roll + 21 keypoints) from both source and driving frames
- `warping_module` — warps source features using relative keypoint delta
- `spade_generator` (SPADEDecoder) — decodes warped features into an RGB image
- `stitching_retargeting_module` — contains 3 sub-networks (stitching, lip, eye retargeting)

**Key inference logic** (`generate_frame`): on first call, caches source features and initial driving keypoints. Each subsequent call computes a delta between the current and initial driving keypoints, then applies tunable weights to drive the portrait. Two operating modes:

| Mode | `MOTION` | `EXPRESSION` | `STABILIZER` |
|------|----------|--------------|--------------|
| Live (webcam) | 0.30 | 0.50 | 0.85 |
| Narration (`narration_mode=True`) | 0.00 | 0.65 | 0.55 |

The model config lives at `liveportrait_src/src/config/models.yaml`.

### 2. Mapping Server (`mapping_server.py`) — main production entry point
Combines the AI engine with a FastAPI HTTP server on port 8000. Manages three artworks (`joconde`, `vangogh`, `courbet`) and a shared `EngineState` protected by a threading lock.

**Age group system**: each artwork has three narration variants (`enfant` | `ado` | `adulte`) stored as pre-recorded audio files in `audio/` (`audio_{artwork}_{age}.mp3.mpeg`). The active age group is stored in `state.age_group`.

**Narration mode**: when narration is active, the main loop reads frames from `driving_narration.mp4` (looped) instead of the webcam and feeds them through `generate_frame(narration_mode=True)`. Audio is simultaneously streamed via ffmpeg UDP to the Raspberry Pi. When narration ends naturally (watcher thread), `state.frozen` is set back to `False`.

**Engine states**:
- **live**: webcam drives the AI in real time
- **frozen**: last rendered frame is held
- **narration**: `driving_narration.mp4` drives the AI; audio streams to Pi simultaneously

### 3. REST API endpoints

| Method | Path | Effect |
|--------|------|--------|
| GET | `/` | Fullscreen HTML page showing `/video_feed` |
| GET | `/video_feed` | MJPEG stream (640×640, quality 45, ~10fps) |
| GET | `/status` | Returns running/frozen/artwork/narration state |
| POST | `/command/start/{artwork_name}` | Switch active artwork (`joconde`\|`vangogh`\|`courbet`) |
| POST | `/command/age/{age_group}` | Set age group (`enfant`\|`ado`\|`adulte`) for narration |
| POST | `/command/freeze` | Hold on current frame |
| POST | `/command/resume` | Return to live animation |
| POST | `/command/capture` | Save current frame to `output/portrait.jpg` |
| GET | `/capture/file` | Download saved portrait JPEG |
| POST | `/command/stop` | Shut down server |
| POST | `/command/narration/start/{age_group}` | Set age group + start narration for current artwork |
| POST | `/command/narration/pause` | SIGSTOP audio process |
| POST | `/command/narration/resume` | SIGCONT audio process |
| POST | `/command/narration/stop` | Kill narration, return to live |

### 4. LivePortrait source (`liveportrait_src/src/`)
Vendored copy of the LivePortrait model code. `ai_engine.py` injects this path into `sys.path` at import time. Do not modify files under `liveportrait_src/` — treat as a third-party dependency.

## Key constraints

- All source portrait images are cropped to 512×512 with a face-biased vertical crop (`start_y = (h - crop_size) * 0.2`) before being fed to the model.
- The AI engine resizes all driving frames to 256×256 internally (`preprocess`) regardless of input size.
- `PortraitAnimator.reset_cache()` must be called when switching artworks to clear cached source features and initial keypoints.
- Thread safety: all reads/writes to `EngineState` fields must be done inside `with state.lock`. MediaPipe face detection (`detect_face_landmarks`) is intentionally run outside the lock since it's slow.
- The `animate_face_simple` function and `compute_audio_levels` are defined but currently unused — they are legacy code from an earlier approach to narration animation.
- Output images go to `output/` (git-ignored). Audio files go to `audio/` (git-ignored).
