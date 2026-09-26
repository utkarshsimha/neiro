"""
Neiro — full stack on Modal.

  Deploy:  uv run modal deploy modal_app.py
  URL:     printed after deploy (e.g. https://simha-utkarsh--neiro-fastapi-app.modal.run)

Architecture
  fastapi_app  — lightweight web container; serves HTML + API; calls separate()
  separate     — A10G GPU container; runs inference; yields progress + FLAC bytes
"""

import base64
import builtins
import os
import queue as _queue
import tempfile
import threading

import modal

from neiro_common import r2_segment_deliverer, run_speed_stream

app = modal.App("neiro")

model_volume = modal.Volume.from_name("mvsep-models", create_if_missing=True)

# ── GPU image — heavy ML deps ──────────────────────────────────────────────
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .run_commands(
        "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124",
        "pip install "
        "numpy soundfile scipy tqdm librosa demucs pyyaml ml-collections "
        "samplerate 'segmentation-models-pytorch==0.3.3' six 'beartype==0.14.1' "
        "'rotary-embedding-torch==0.3.5' onnxruntime-gpu",
    )
    .add_local_dir("modules", remote_path="/app/modules")
    .add_local_file("inference.py", remote_path="/app/inference.py")
    # Imported at the top of this file, so every container needs it (stdlib-only at import).
    .add_local_python_source("neiro_common")
)

# ── Web image — FastAPI only ───────────────────────────────────────────────
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "rubberband-cli", "libsndfile1")
    .pip_install("fastapi[standard]", "python-multipart", "boto3", "numpy", "soundfile", "pyrubberband")
    .add_local_dir("static", remote_path="/app/static")
    .add_local_python_source("neiro_common")
    .add_local_python_source("neiro_web")  # the shared routes; imported only in fastapi_app
)


# ── GPU separation function ────────────────────────────────────────────────

@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=3600,
    volumes={"/app/models": model_volume},
)
def separate(audio_bytes: bytes, options_dict: dict):
    """
    Generator: runs music separation on GPU.
    Yields dicts:
      {"type": "log",      "message": str}
      {"type": "progress", "n": int, "total": int}
      {"type": "result",   "files": {name: base64_str}}
      {"type": "error",    "message": str}
    """
    import sys
    sys.path.insert(0, "/app")
    import inference as inf

    msg_queue: _queue.Queue = _queue.Queue()

    class PatchedTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self._items = list(iterable) if iterable is not None else []
            self._total = kwargs.get("total", len(self._items))
            self._n = 0

        def __iter__(self):
            for item in self._items:
                yield item
                self._n += 1
                msg_queue.put({"type": "progress", "n": self._n, "total": self._total})

    _orig_print = builtins.print

    def _patched_print(*args, **kwargs):
        msg_queue.put({"type": "log", "message": " ".join(str(a) for a in args)})

    builtins.print = _patched_print
    inf.tqdm = PatchedTqdm

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp3")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        with open(input_path, "wb") as fh:
            fh.write(audio_bytes)

        opts = {
            **options_dict,
            "input_audio": [input_path],
            "output_folder": output_dir,
            "output_format": "FLAC",
        }
        inf.options = opts  # separate_music_file reads module-level options

        error = None

        def _run():
            nonlocal error
            try:
                inf.predict_with_model(opts)
            except Exception:
                import traceback
                error = traceback.format_exc()
            finally:
                msg_queue.put(None)

        threading.Thread(target=_run, daemon=True).start()

        while True:
            msg = msg_queue.get()
            if msg is None:
                break
            yield msg

        builtins.print = _orig_print

        if error:
            yield {"type": "error", "message": error}
            return

        files: dict[str, str] = {}
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "rb") as fh:
                files[fname] = base64.b64encode(fh.read()).decode()

        model_volume.commit()
        yield {"type": "result", "files": files}


# ── Web endpoint ───────────────────────────────────────────────────────────

def _optional_secret(name: str) -> modal.Secret:
    """The named secret if it exists, else an empty one, so deploys work without it.

    Always returns exactly one secret: Modal checks that a container declares the same
    number of dependencies as the deployed function, so this can't vary between deploy
    time and container start. Inside the container the returned object is just a
    placeholder that gets bound to whichever secret was attached at deploy.
    """
    secret = modal.Secret.from_name(name)
    if not modal.is_local():
        return secret
    try:
        secret.hydrate()
    except modal.exception.NotFoundError:
        print(f"Modal secret '{name}' not found — deploying without it.")
        return modal.Secret.from_dict({})
    return secret


# ── Speed-change stretching on a big CPU box ────────────────────────────────

# Reserve 12 cores but let the container burst to 32 (Modal bills max(reserved, used)).
# 12 is what the playhead's segment needs: two chunks of each of six stems, stretched at
# once. Benchmarked on six 21-minute stems, that got the first segment ready as fast as a
# fixed 32-core reservation (~1.6s after decoding) at about half the cost (~$0.006 vs
# ~$0.012+ per speed change): 32 reserved cores are billed from start-up through the idle
# window, but the actual work is only ~300 core-seconds, and bursting (~15 cores on
# average) finishes the rest far faster than real time. 8 reserved was ~0.5s slower to the
# first segment. Running the non-urgent chunks under `nice` made no measurable difference.
STRETCH_CPUS = (12, 32)
STRETCH_POOL = STRETCH_CPUS[1]  # threads for chunk work: enough to use the whole burst

# Cancellation flags for in-flight stretch_stems calls, keyed by out_prefix (unique per
# request). Closing a `remote_gen` stream doesn't stop the remote call, and FunctionCall
# .cancel() can't look up `remote_gen` calls (only spawned ones, and generators can't be
# spawned) — so when the browser drops a speed-change stream (e.g. it changed speed again),
# the web tier sets a flag here that stretch_stems polls.
stretch_cancels = modal.Dict.from_name("neiro-stretch-cancels", create_if_missing=True)


@app.function(
    image=web_image,
    cpu=STRETCH_CPUS,
    # The stretch's working files live in /dev/shm (see below), which counts as memory:
    # it peaked at ~3.3 GB for six 21-minute stems (sources, decoded WAVs, chunk files in
    # flight). A request, not a hard cap — usage above it is billed, not killed.
    memory=8192,
    timeout=300,
    # Idle containers are billed at their reservation, so don't linger long; the price is
    # a cold start (a few seconds) on a speed change that comes after this.
    scaledown_window=10,
    secrets=[_optional_secret("cloudflare-r2")],
)
def stretch_stems(keys: dict, rate: float, chunk_s: float, out_prefix: str, start_fraction: float = 0.0):
    """Generator: stretch every stem of a track (read from R2 at `keys` {stem: key}) to
    `rate`, yielding neiro_common.stream_stretch's events as segments land in R2 under
    `out_prefix/` — the web tier relays them to the browser as they come (see /api/speed).

    Runs apart from the web container because the stretch is CPU-bound and the web
    container only gets ~5-6 cores in practice: profiling a 21-minute, 6-stem track showed
    ~300 core-seconds of Rubber Band work taking ~57s there even when split into chunks.
    Here every stem's chunks share one pool sized to this function's burst limit.

    All the working files live in /dev/shm (RAM): chunking does a lot of small file
    writes/reads (~6 GB of chunk WAVs for six 21-minute stems), and Modal's sandboxed
    container filesystem made that the bottleneck — benchmarked at 32 cores, six stems
    took 41s with the work in /tmp vs 15s in /dev/shm.

    Stops early — cancelling the chunk work that hasn't started — once the caller sets
    this request's flag in `stretch_cancels`; a watcher thread checks it every 0.5s."""
    import tempfile
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    cancelled, finished = threading.Event(), threading.Event()

    def watch_for_cancel():
        while not finished.is_set():
            try:
                if stretch_cancels.get(out_prefix):
                    cancelled.set()
                    return
            except Exception:
                pass  # a flaky lookup just means checking again shortly
            time.sleep(0.5)

    threading.Thread(target=watch_for_cancel, daemon=True).start()
    inputs = {stem: ("r2", key) for stem, key in keys.items()}
    pool = ThreadPoolExecutor(max_workers=STRETCH_POOL)
    try:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as workdir:
            # A missing stem raises FileNotFoundError here, which Modal re-raises in the
            # caller before any event — /api/speed turns that into a 404.
            events = run_speed_stream(inputs, rate, workdir, pool, r2_segment_deliverer(out_prefix),
                                      chunk_s, start_fraction)
            try:
                for ev in events:
                    if cancelled.is_set():
                        break
                    yield ev
            finally:
                events.close()  # cancels stream_stretch's pending chunk work
    finally:
        finished.set()
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            stretch_cancels.pop(out_prefix, None)
        except Exception:
            pass


@app.function(
    image=web_image,
    timeout=3600,
    scaledown_window=300,
    # R2 is optional: without it results fall back to inline audio and Save is unavailable.
    secrets=[_optional_secret("cloudflare-r2"), modal.Secret.from_name("cobalt")],
)
@modal.asgi_app()
def fastapi_app():
    # Imported here, not at the top: neiro_web needs FastAPI, which only the web image has
    # (the GPU container imports this module too).
    from neiro_web import build_app

    async def remote_stretch(keys: dict, rate: float, out_prefix: str, chunk_s: float, start_fraction: float):
        # R2-backed speed changes run on the big-CPU stretch_stems function rather than in
        # this container (see its docstring for why); its events stream back as they happen.
        # If we're closed before the end (the browser dropped the stream), tell it to stop.
        import asyncio

        try:
            async for ev in stretch_stems.remote_gen.aio(keys, rate, chunk_s, out_prefix, start_fraction):
                yield ev
        except (GeneratorExit, asyncio.CancelledError):
            try:
                await stretch_cancels.put.aio(out_prefix, True)
            except Exception:
                pass  # worst case the stretch runs to completion unobserved
            raise

    return build_app(static_dir="/app/static", gpu_separate=separate.remote_gen.aio,
                     remote_stretch=remote_stretch)

