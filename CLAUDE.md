# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Neiro is a music source separation web app. It wraps an ensemble deep-learning separation
pipeline (adapted from MVSep-MDX23) behind a FastAPI backend and a single-page vanilla-JS
frontend, with two deployment targets: a local CPU dev server and a Modal-hosted GPU service.
The separation pipeline itself (`inference.py`, `modules/`) is a vendored fork — treat it as
mostly stable library code; day-to-day work happens in `app.py`, `modal_app.py`, and
`static/index.html`.

## Commands

```bash
make setup    # uv sync --extra web — installs deps into .venv
make web      # uv run uvicorn app:app --reload --port 8000 — local CPU dev server
make deploy   # deploys to Modal (GPU) under both the "neiro" and legacy "song-lab" app
              # names, so previously shared song-lab URLs keep working
make run INPUT=song.mp3 [OUTPUT=./output] [CPU=1]   # CLI separation via inference.py
make clean    # remove .venv, __pycache__, output/
```

There is no test suite or linter configured in this repo.

Direct inference invocation (bypassing the web layer):

```bash
uv run python inference.py \
  --input_audio song.mp3 --output_folder ./output \
  --use_BSRoformer --use_Kim_MelRoformer --use_InstVoc \
  --weight_BSRoformer 9.18 --weight_Kim_MelRoformer 10 --weight_InstVoc 3.39 \
  --BigShifts 3 --vocals_only
```

Key `inference.py` flags: `--large_gpu` (keep all models resident in VRAM), `--cpu` (force CPU,
slow), `--BigShifts N` (shift-average passes, 1=off/3–11=typical), `--output_format`
(`PCM_16`/`FLOAT`/`FLAC`), `--filter_vocals`, `--input_gain`/`--restore_gain`.

## Architecture

### Two parallel FastAPI apps, one shared contract

`app.py` (local) and the `fastapi_app` closure inside `modal_app.py` (Modal) are **independent,
duplicated implementations** of the same API surface (`/`, `/api/separate`, `/api/youtube`,
`/api/share/save`, `/api/result/{job_id}`, `/api/audio-proxy`, R2 helpers, `DEFAULTS` dict). They are
not imported from a shared module — when changing request/response shapes, defaults, or
endpoint behavior, **update both files**. `app.py`'s Modal GPU path calls into `modal_app.py`'s
`separate` function via `ma.separate.remote_gen.aio(...)`; its CPU path calls `inference.py`
directly in a background thread.

### Request flow

1. Client uploads audio (or a YouTube URL, downloaded server-side via `yt-dlp` into an mp3) to
   `POST /api/separate` with a JSON `options` blob merged onto `DEFAULTS`.
2. **CPU path** (`app.py`, `cpu: true`): runs `inference.predict_with_model` in a background
   thread; `print` and `tqdm` are monkey-patched to push `{type: log|progress}` messages onto an
   `asyncio.Queue`, drained as SSE.
3. **GPU path**: `app.py` proxies to the Modal `separate` function (an `@app.function(gpu="H100")`
   generator); `modal_app.py`'s own `fastapi_app` does the equivalent locally when deployed. The
   generator yields `log`/`progress`/`result`/`error` dict messages; results are FLAC-encoded and
   base64'd, then re-encoded to MP3 for browser playback.
4. Model checkpoints live on a Modal `Volume` (`mvsep-models`) so they persist across container
   cold starts; `model_volume.commit()` is called after each run.
5. After separation, the web tier converts stems to MP3 and (when R2 is configured) stages them
   in Cloudflare R2 under `tmp/{job_id}/`; the SSE `done` message then carries `job_id` plus
   presigned GET URLs instead of base64 audio (if R2 is unset or staging fails it falls back to
   inline base64 and Save is unavailable). "Save to library" is `POST /api/share/save`, a
   server-side copy of `tmp/{job_id}/` → `results/{job_id}/` plus `manifest.json` — the browser
   never uploads audio. A bucket lifecycle rule (set at startup) expires `tmp/` after a day.
   Why not upload from the browser: presigned browser→R2 PUTs of stem-sized files failed
   unpredictably in Safari/WebKit ("Load failed" even at ~5-10 MB, fine in Chromium), and the
   older single-JSON-POST variant hit Modal's 150s web-endpoint timeout. Note `tmp/` objects are
   deliberately left in place after a save so the still-open player/download links keep working.
   `GET /api/result/{job_id}` returns presigned GET URLs for saved results (24h expiry). R2 is
   optional — configured via `.env` locally (`cp .env.example .env`) or a `cloudflare-r2` Modal
   secret in production.
6. YouTube ingestion (`/api/youtube`) calls a self-hosted Cobalt instance (`cobalt_app.py`, a
   separate Modal app — see there for why: yt-dlp run directly from Modal's IPs kept tripping
   bot checks) over HTTP, and strips "(Official Video)"-style junk from titles. Both `app.py`
   and `modal_app.py` need `COBALT_URL` / `COBALT_API_KEY` set (`.env` locally, the `cobalt`
   Modal secret in production — `make deploy-cobalt` deploys Cobalt and creates that secret).
   YouTube sometimes requires a proof-of-origin token Cobalt can't obtain from Modal's IP
   range, in which case it silently returns an empty file instead of erroring — `_download_yt`
   in both files treats an empty download as a hard error rather than passing it to the
   separation pipeline (see the note in `cobalt_app.py` on why there's no token-provider
   sidecar: it hits the same IP-blocking problem it's meant to solve).
7. Practice mode's speed control (`/api/speed`) re-renders every stem at a new tempo via
   Rubber Band (`pyrubberband`, shelling out to the `rubberband` CLI — `_stretch_stem` in
   both files) without shifting pitch. It only runs while paused: the frontend disables the
   slider during playback and while a request is in flight, always resends the pristine
   (rate=1) stem bytes it kept from the initial load rather than the currently-loaded
   buffers (so repeated speed changes don't compound re-stretches), and reprocesses each
   stem concurrently to stay well under Modal's 150s web endpoint timeout even on long
   songs. Rubber Band was chosen after an earlier real-time AudioWorklet approach
   (SoundTouchJS) produced audible quality degradation on polyphonic stems — see git
   history — and offline batch processing has no such real-time-DSP quality ceiling.
   Requires the `rubberband` CLI (`brew install rubberband` / `apt install rubberband-cli`
   locally; `rubberband-cli` in `modal_app.py`'s `web_image`).

### Separation pipeline (`inference.py`)

`EnsembleDemucsMDXMusicSeparationModel` orchestrates:

1. **Vocals ensemble** — weighted average of up to 6 models: `BSRoformer` / `Kim_MelRoformer`
   (PyTorch transformers, `.ckpt`+`.yaml`), `InstVoc` (MDXv3/TFC_TDF_net), `VitLarge`
   (Segm_Models_Net), `VOCFT`/`InstHQ4` (ONNX via `onnxruntime`).
2. **Instrumental** — computed as `mix - vocals`.
3. **4-stem decomposition** (bass/drums/other) — ensemble of 4 Demucs models (`htdemucs_ft`,
   `htdemucs`, `htdemucs_6s`, `hdemucs_mmi`) applied to the instrumental.

Models download automatically on first use and cache in `models/`; loading is lazy
(`initialize_model_if_needed`) and models offload from GPU after use unless `--large_gpu`.
`demix_new`/`demix_new_wrapper` handle chunked inference + BigShifts shift-averaging for the
transformer models; `demix`/`demix_wrapper` handle ONNX-based chunked inference for MDX models;
`lr_filter` is the Linkwitz-Riley crossover used when blending VOCFT's high/low bands.

Custom model modules live in `modules/`: `bs_roformer/bs_roformer.py`,
`bs_roformer/mel_band_roformer.py`, `tfc_tdf_v3.py` (MDXv3/InstVoc), `tfc_tdf_v2.py`
(MDXv2/VOCFT/InstHQ4), `segm_models.py` (VitLarge).

### Modal deployment split

`modal_app.py` builds two separate images: `gpu_image` (heavy ML deps, `modules/` +
`inference.py` baked in, runs `separate` on an H100) and `web_image` (lightweight — FastAPI,
boto3, yt-dlp + Node 20 — serves `static/`). Keep heavy dependencies out of `web_image` and
web-only dependencies out of `gpu_image`.

### Frontend

`static/index.html` is a single-file vanilla-JS SPA (Gruvbox dark theme) — no build step. It
POSTs to `/api/separate`, consumes the SSE stream for live progress/log updates, plays stems
inline, and calls `/api/share/save` to keep a result in the library.

## Known open issues (`fixes.md`)

Karaoke/lyric-sync UX has known rough edges: auto-scroll on lyric click, choppy "include
vocals" playback, vocals not time-synced, and no ability to queue a second song while karaoke
is active on the current one.
