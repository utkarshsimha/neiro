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

from neiro_common import (
    DEFAULTS, INDEX_HEADERS, JOB_ID_RE, SPEED_FILE_RE, SPEED_SOURCE_RE, STAGED_PREFIX,
    clean_yt_title, delete_job, download_yt, finalize_outputs, get_r2, list_library,
    queue_to_sse, r2_configured, save_staged, stretch_stem, update_manifest, yt_url_ok,
)

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

STRETCH_CPUS = 32


@app.function(
    image=web_image,
    cpu=STRETCH_CPUS,
    # Also sizes /dev/shm, where the stretch works (see below): ~1 GB per 21-minute stem.
    memory=16384,
    timeout=300,
    # Idle containers are billed at their full reservation, and 32 idle cores cost more
    # per minute than a whole 21-minute speed change's actual work (~300 core-seconds),
    # so shut down quickly; the price is a cold start (a few seconds) on the next change.
    scaledown_window=10,
    secrets=[_optional_secret("cloudflare-r2")],
)
def stretch_stems(keys: dict, rate: float, chunk_s: float, out_prefix: str) -> dict:
    """Stretch every stem of a track (read from R2 at `keys` {stem: key}) to `rate`, write
    the FLACs under `out_prefix/` and return {stem: presigned url}.

    Runs apart from the web container because the stretch is CPU-bound and the web
    container only gets ~5-6 cores in practice: profiling a 21-minute, 6-stem track showed
    ~300 core-seconds of Rubber Band work taking ~57s there even when split into chunks.
    Here every stem's chunks share one pool sized to this function's reservation.

    All the working files live in /dev/shm (RAM): chunking does a lot of small file
    writes/reads (~6 GB of chunk WAVs for six 21-minute stems), and Modal's sandboxed
    container filesystem made that the bottleneck — benchmarked at 32 cores, six stems
    took 41s with the work in /tmp vs 15s in /dev/shm."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from botocore.exceptions import ClientError

    r2, bucket = get_r2(), os.environ["R2_BUCKET_NAME"]
    chunk_pool = ThreadPoolExecutor(max_workers=STRETCH_CPUS)

    def process_one(item):
        stem, key = item
        with tempfile.TemporaryDirectory(dir="/dev/shm") as workdir:
            src = os.path.join(workdir, "src")
            try:
                r2.download_file(bucket, key, src)
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                    raise FileNotFoundError(key)  # re-raised in the caller → HTTP 404
                raise
            out_path = stretch_stem(src, rate, workdir, chunk_pool, chunk_s)
            out_key = f"{out_prefix}/{stem}.flac"
            r2.upload_file(out_path, bucket, out_key, ExtraArgs={"ContentType": "audio/flac"})
            return stem, r2.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": out_key}, ExpiresIn=86400)

    try:
        with ThreadPoolExecutor(max_workers=len(keys)) as stems_pool:
            return dict(stems_pool.map(process_one, keys.items()))
    finally:
        chunk_pool.shutdown(cancel_futures=True)


@app.function(
    image=web_image,
    timeout=3600,
    scaledown_window=300,
    # R2 is optional: without it results fall back to inline audio and Save is unavailable.
    secrets=[_optional_secret("cloudflare-r2"), modal.Secret.from_name("cobalt")],
)
@modal.asgi_app()
def fastapi_app():
    import asyncio
    import base64
    import json
    import re
    import uuid

    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI(title="Neiro")
    web.mount("/static", StaticFiles(directory="/app/static"), name="static")

    @web.post("/api/youtube")
    async def api_youtube(request: Request):
        data = await request.json()
        url = (data.get("url") or "").strip()
        if not yt_url_ok(url):
            raise HTTPException(status_code=400, detail="Only YouTube URLs are accepted")
        try:
            audio_bytes, raw_title = await asyncio.get_event_loop().run_in_executor(
                None, download_yt, url
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Download failed: {e}")
        title = clean_yt_title(raw_title)
        safe = re.sub(r'[<>:"/\\|?*]', '', title).strip() or 'audio'
        filename = safe + '.mp3'
        return {"audio": base64.b64encode(audio_bytes).decode(), "filename": filename, "title": title}

    @web.get("/")
    async def index():
        return FileResponse("/app/static/index.html", headers=INDEX_HEADERS)

    @web.get("/result/{job_id}")
    async def share_page(job_id: str):
        return FileResponse("/app/static/index.html", headers=INDEX_HEADERS)

    @web.post("/api/share/save")
    async def api_share_save(request: Request):
        """Promote a staged separation result into the library — see app.py's api_share_save."""
        data = await request.json()
        job_id = data.get("job_id") or ""
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(status_code=400, detail="Invalid job_id")
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, save_staged, job_id, data.get("mode"), data.get("title"), data.get("artist"),
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="This run expired before it was saved — separate the track again.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"uuid": job_id}

    @web.get("/api/library")
    async def api_library():
        try:
            items = await asyncio.get_event_loop().run_in_executor(None, list_library)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"items": items}

    @web.patch("/api/library/{job_id}")
    async def api_library_rename(job_id: str, request: Request):
        data = await request.json()
        try:
            manifest = await asyncio.get_event_loop().run_in_executor(None, update_manifest, job_id, data)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Not found: {e}")
        return manifest

    @web.delete("/api/library/{job_id}")
    async def api_library_delete(job_id: str):
        try:
            await asyncio.get_event_loop().run_in_executor(None, delete_job, job_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Not found: {e}")
        return {"deleted": job_id}

    @web.get("/api/audio-proxy")
    async def audio_proxy(url: str):
        import asyncio
        import urllib.request
        try:
            def _fetch():
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.read(), r.headers.get("Content-Type", "audio/flac")
            data, ct = await asyncio.to_thread(_fetch)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        return Response(content=data, media_type=ct,
                        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=3600"})


    @web.post("/api/speed")
    async def api_speed(request: Request):
        """Re-render every stem at a new speed, pitch unchanged — see app.py's api_speed for
        the two request shapes (stems read from R2, or uploaded as base64) and delivery."""
        from botocore.exceptions import ClientError

        data = await request.json()
        try:
            rate = float(data.get("rate"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="rate must be a number")
        if not (0.1 <= rate <= 2.0):
            raise HTTPException(status_code=400, detail="rate must be 0.1-2.0")

        source = data.get("source")
        if source:
            prefix, names = source.get("prefix") or "", source.get("files") or {}
            if not r2_configured():
                raise HTTPException(status_code=400, detail="R2 is not configured")
            if not (SPEED_SOURCE_RE.match(prefix) and names
                    and all(isinstance(n, str) and SPEED_FILE_RE.match(n) for n in names.values())):
                raise HTTPException(status_code=400, detail="Invalid source")
            inputs = {stem: ("r2", f"{prefix}/{name}") for stem, name in names.items()}
        else:
            files = data.get("files") or {}
            if not files:
                raise HTTPException(status_code=400, detail="No stems given")
            inputs = {stem: ("b64", b64) for stem, b64 in files.items()}

        deliver_via_r2 = r2_configured()
        out_id = str(uuid.uuid4())

        if source:
            # Stems in R2 → stretch them in 30s chunks on the big-CPU stretch_stems
            # function rather than in this container (see its docstring for why).
            keys = {stem: ref for stem, (_, ref) in inputs.items()}
            try:
                urls = await stretch_stems.remote.aio(keys, rate, 30.0, f"{STAGED_PREFIX}/{out_id}")
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="A stem is no longer in storage")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Speed change failed: {e}")
            return {"delivery": "url", "files": urls}

        r2 = get_r2() if deliver_via_r2 else None  # boto3 clients are thread-safe
        bucket = os.environ.get("R2_BUCKET_NAME")

        def process_one(stem: str, kind: str, ref: str) -> str:
            with tempfile.TemporaryDirectory() as workdir:
                src = os.path.join(workdir, "src")
                if kind == "r2":
                    try:
                        r2.download_file(bucket, ref, src)
                    except ClientError as e:
                        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                            raise FileNotFoundError(ref)
                        raise
                else:
                    with open(src, "wb") as fh:
                        fh.write(base64.b64decode(ref))
                out_path = stretch_stem(src, rate, workdir)
                if deliver_via_r2:
                    key = f"{STAGED_PREFIX}/{out_id}/{stem}.flac"
                    r2.upload_file(out_path, bucket, key, ExtraArgs={"ContentType": "audio/flac"})
                    result = r2.generate_presigned_url(
                        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400)
                else:
                    with open(out_path, "rb") as fh:
                        result = base64.b64encode(fh.read()).decode()
                return result

        loop = asyncio.get_running_loop()

        async def run_one(stem, kind, ref):
            try:
                return stem, await loop.run_in_executor(None, process_one, stem, kind, ref)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Stem {stem} is no longer in storage")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Speed change failed on {stem}: {e}")

        results = await asyncio.gather(*(run_one(s, k, r) for s, (k, r) in inputs.items()))
        return {"delivery": "url" if deliver_via_r2 else "base64", "files": dict(results)}

    @web.get("/api/result/{job_id}")
    async def get_result(job_id: str):
        r2 = get_r2()
        bucket = os.environ["R2_BUCKET_NAME"]
        try:
            manifest = json.loads(
                r2.get_object(Bucket=bucket, Key=f"results/{job_id}/manifest.json")["Body"].read()
            )
        except Exception:
            raise HTTPException(status_code=404, detail="Result not found")
        urls = {
            fname: r2.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": f"results/{job_id}/{fname}"},
                ExpiresIn=86400,
            )
            for fname in manifest["files"]
        }
        return {
            "files": urls,
            "mode": manifest.get("mode") or ("karaoke" if len(manifest["files"]) <= 2 else "practice"),
            "title": manifest.get("title"),
            "artist": manifest.get("artist"),
        }

    @web.post("/api/separate")
    async def api_separate(
        file: UploadFile = File(...),
        options: str = Form("{}"),
    ):
        audio_bytes = await file.read()
        opts = {**DEFAULTS, **json.loads(options)}
        opts.pop("input_audio", None)
        opts.pop("output_folder", None)
        opts["cpu"] = False  # always GPU on Modal

        q: asyncio.Queue = asyncio.Queue()

        loop = asyncio.get_event_loop()

        async def _task():
            try:
                async for msg in separate.remote_gen.aio(audio_bytes, opts):
                    if msg.get("type") == "result":
                        with tempfile.TemporaryDirectory() as td:
                            for fname, b64 in msg["files"].items():
                                with open(os.path.join(td, fname), "wb") as fh:
                                    fh.write(base64.b64decode(b64))
                            del msg
                            await q.put(await loop.run_in_executor(None, finalize_outputs, td))
                    else:
                        await q.put(msg)
            except Exception as exc:
                import traceback
                await q.put({"type": "error", "message": traceback.format_exc()})
            finally:
                await q.put(None)

        asyncio.create_task(_task())
        return StreamingResponse(
            queue_to_sse(q),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return web
