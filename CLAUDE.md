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
make r2-setup [BUCKET=neiro]   # one-time R2 bucket config (CORS + tmp/ expiry) via wrangler
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

### One web app, two deployments

Every route (`/`, `/api/separate`, `/api/youtube`, `/api/share/save`, `/api/library`,
`/api/result/{job_id}`, `/api/audio-proxy`, `/api/speed`) is defined once, in
`neiro_web.build_app()`. `app.py` (local dev server) and `modal_app.py`'s `fastapi_app`
(Modal) each just call it, passing in the few things that genuinely differ:

- `static_dir` — `static` locally, `/app/static` in the Modal web image.
- `gpu_separate` — the Modal `separate` generator. Modal passes `separate.remote_gen.aio`;
  `app.py` passes a wrapper that imports `modal_app` lazily, so the local server starts (and
  CPU mode works) without Modal set up.
- `cpu_separate` — local only (`app.py`'s `_cpu_separate`, which runs `inference.py`
  in-process). Without it, as on Modal, `cpu: true` requests run on the GPU.
- `remote_stretch` — Modal only: hands R2-backed speed changes to `stretch_stems`. Without
  it, as locally, speed changes run in-process.

The non-route logic lives in `neiro_common.py`: `DEFAULTS`, separation output handling
(`finalize_outputs`), SSE streaming (`queue_to_sse`), YouTube/Cobalt ingestion, the R2 helpers
(client, staging, save/library/manifest) and the speed-change helpers.

Import constraints: `neiro_common.py` must keep its top-level imports stdlib-only, because
`modal_app.py` imports it at the top, so it loads in every Modal container including the GPU
one. `neiro_web.py` imports FastAPI, so `modal_app.py` imports it only inside `fastapi_app`,
and only the web image includes it. Both reach the Modal images only via
`add_local_python_source(...)` — a new shared module needs adding there too.

### Request flow

1. Client uploads audio (or a YouTube URL, downloaded server-side via `yt-dlp` into an mp3) to
   `POST /api/separate` with a JSON `options` blob merged onto `DEFAULTS`.
2. **CPU path** (local server only, `cpu: true`): `app.py`'s `_cpu_separate` runs
   `inference.predict_with_model` in a background thread; `print` and `tqdm` are monkey-patched
   to push `{type: log|progress}` messages onto an `asyncio.Queue`, drained as SSE.
3. **GPU path** (everything else): the web app streams from the Modal `separate` function (an
   `@app.function(gpu="H100")` generator) — whether it's running locally or on Modal. The
   generator yields `log`/`progress`/`result`/`error` dict messages; results are FLAC-encoded and
   base64'd, then re-encoded to MP3 for browser playback.
4. Model checkpoints live on a Modal `Volume` (`mvsep-models`) so they persist across container
   cold starts; `model_volume.commit()` is called after each run.
5. After separation, the web tier converts stems to MP3 and (when R2 is configured) stages them
   in Cloudflare R2 under `tmp/{job_id}/`; the SSE `done` message then carries `job_id` plus
   presigned GET URLs instead of base64 audio (if R2 is unset or staging fails it falls back to
   inline base64 and Save is unavailable). "Save to library" is `POST /api/share/save`, a
   server-side copy of `tmp/{job_id}/` → `results/{job_id}/` plus `manifest.json` — the browser
   never uploads audio. A bucket lifecycle rule expires `tmp/` after a day; it and the
   bucket's CORS policy (`r2-cors.json`) are set once with `make r2-setup`, which runs
   `wrangler` under the user's own Cloudflare login. The app doesn't set them itself: that
   needs an Admin Read & Write token (account-wide, can delete buckets), whereas the app's
   long-lived token is Object Read & Write scoped to the one bucket.
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
   range, in which case it silently returns an empty file instead of erroring — `download_yt`
   (in `neiro_common.py`) treats an empty download as a hard error rather than passing it to the
   separation pipeline (see the note in `cobalt_app.py` on why there's no token-provider
   sidecar: it hits the same IP-blocking problem it's meant to solve).
7. Practice mode's speed control (`/api/speed`) re-renders every stem at a new tempo via
   Rubber Band (the `rubberband` CLI run directly on WAV files, with ffmpeg decoding before
   and encoding FLAC after — `stream_stretch` in `neiro_common.py`; decoding the MP3s in Python via
   soundfile was ~6x slower on long tracks) without shifting pitch. It only runs while
   paused (the frontend disables the slider during playback), and every request starts
   from the pristine (rate=1) stems rather than the
   currently-loaded buffers (so repeated speed changes don't compound re-stretches). When
   the stems are in R2 the client sends just their location (`source: {prefix: "tmp/<id>" |
   "results/<id>", files}`) and the server reads them from the bucket; otherwise (or on a
   404 when a staged copy expired) the client uploads the pristine bytes as base64.
   Profiling a 21-minute, 6-stem track showed base64-over-JSON transfer (~250 MB up,
   ~530 MB down) and Python-side MP3 decoding as large costs next to the stretch itself;
   MP3 output was benchmarked and rejected (encoding costs more than the smaller download
   saves). Each stem is cut into 30s chunks stretched in parallel and rejoined with a
   correlation-normalised crossfade.
   **The result is streamed** (`stream_stretch` in `neiro_common.py`): the response is SSE —
   a `meta` event with the new timeline, then each (stem, ~30s segment) as a small FLAC
   (presigned R2 URL under `tmp/`, or inline base64 without R2) the moment its two chunks are
   done, playhead-first (the request's `start_fraction`), then `done`. The player
   (`track` in `buildPracticeCard`) is a timeline of per-stem segment buffers: it schedules
   ready segments back-to-back on the AudioContext clock, lets you play as soon as the
   playhead's segment has every stem, and waits (⏳) at any segment that hasn't arrived,
   resuming when it does; a failure partway through restores the last fully-loaded audio.
   The speed controls stay usable while segments load: applying another speed aborts the
   in-flight request, and the dropped connection stops the stretch behind it (in-process
   via `_in_thread`; on Modal via a flag in the `neiro-stretch-cancels` `modal.Dict` that
   `stretch_stems` polls — closing a `remote_gen` stream doesn't cancel the remote call, and
   `FunctionCall.cancel` can't look up `remote_gen` calls). Segment downloads (`fetchStemBytes`)
   abort and retry when a 5s window brings under 10% of the median recent download rate
   (floor 16 KB/s) — connections occasionally stall or trickle for 25s+. 1.0x is decoded
   locally from the pristine bytes. On Modal, R2-backed requests run on a separate `stretch_stems` generator
   function (12 CPUs reserved — what the playhead's first segment needs, 2 chunks × 6
   stems — bursting to 32; a fixed 32 was no faster to first playback and cost ~2x, since
   reserved cores are billed while idle — and 8 GB), working
   in `/dev/shm` because the container filesystem was the bottleneck for the many chunk
   files; the web container only gets ~5-6 cores in practice.
   Rubber Band was chosen after an earlier real-time AudioWorklet approach (SoundTouchJS)
   produced audible quality degradation on polyphonic stems — see git history — and
   offline batch processing has no such real-time-DSP quality ceiling.
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
