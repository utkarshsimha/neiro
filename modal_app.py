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
import subprocess
import tempfile
import threading

import modal

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
)

# ── Web image — FastAPI only ───────────────────────────────────────────────
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "python-multipart", "boto3")
    .add_local_dir("static", remote_path="/app/static")
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

@app.function(
    image=web_image,
    timeout=3600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("cloudflare-r2"), modal.Secret.from_name("cobalt")],
)
@modal.asgi_app()
def fastapi_app():
    import asyncio
    import base64
    import datetime as _dt
    import json
    import re
    import uuid

    import boto3
    from botocore.config import Config
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI(title="Neiro")
    web.mount("/static", StaticFiles(directory="/app/static"), name="static")

    def _get_r2():
        return boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(s3={"addressing_style": "path"}),
        )

    def _upload_to_r2(files: dict, mode: str = None, title: str = None, artist: str = None) -> str:
        r2 = _get_r2()
        bucket = os.environ["R2_BUCKET_NAME"]
        job_id = str(uuid.uuid4())
        for fname, b64 in files.items():
            r2.put_object(
                Bucket=bucket,
                Key=f"results/{job_id}/{fname}",
                Body=base64.b64decode(b64),
                ContentType="audio/mpeg" if fname.endswith(".mp3") else "application/octet-stream",
            )
        manifest = json.dumps({
            "files": list(files.keys()),
            "mode": mode if mode in ("karaoke", "practice") else "practice",
            "title": title or None,
            "artist": artist or None,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        })
        r2.put_object(
            Bucket=bucket,
            Key=f"results/{job_id}/manifest.json",
            Body=manifest.encode(),
            ContentType="application/json",
        )
        return job_id

    def _list_library() -> list:
        r2 = _get_r2()
        bucket = os.environ["R2_BUCKET_NAME"]
        items = []
        paginator = r2.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix="results/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("/manifest.json"):
                    continue
                job_id = obj["Key"].split("/")[1]
                try:
                    manifest = json.loads(r2.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
                except Exception:
                    continue
                files = manifest.get("files", [])
                items.append({
                    "job_id": job_id,
                    "title": manifest.get("title"),
                    "artist": manifest.get("artist"),
                    "mode": manifest.get("mode") or ("karaoke" if len(files) <= 2 else "practice"),
                    "created_at": manifest.get("created_at") or obj["LastModified"].isoformat(),
                    "stems": sorted({f.rsplit(".", 1)[0].split("_")[-1] for f in files}),
                })
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    def _update_manifest(job_id: str, data: dict) -> dict:
        r2 = _get_r2()
        bucket = os.environ["R2_BUCKET_NAME"]
        key = f"results/{job_id}/manifest.json"
        manifest = json.loads(r2.get_object(Bucket=bucket, Key=key)["Body"].read())
        if "title" in data:
            manifest["title"] = data["title"] or None
        if "artist" in data:
            manifest["artist"] = data["artist"] or None
        if data.get("mode") in ("karaoke", "practice"):
            manifest["mode"] = data["mode"]
        manifest.setdefault("created_at", _dt.datetime.now(_dt.timezone.utc).isoformat())
        r2.put_object(Bucket=bucket, Key=key, Body=json.dumps(manifest).encode(), ContentType="application/json")
        return manifest

    def _delete_job(job_id: str) -> None:
        r2 = _get_r2()
        bucket = os.environ["R2_BUCKET_NAME"]
        prefix = f"results/{job_id}/"
        r2.head_object(Bucket=bucket, Key=f"{prefix}manifest.json")  # 404s if the job doesn't exist
        paginator = r2.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                r2.delete_objects(Bucket=bucket, Delete={"Objects": keys})

    # Configure R2 CORS once at startup so browsers can fetch audio directly
    try:
        _get_r2().put_bucket_cors(
            Bucket=os.environ["R2_BUCKET_NAME"],
            CORSConfiguration={"CORSRules": [{"AllowedHeaders": ["*"], "AllowedMethods": ["GET", "HEAD"],
                                               "AllowedOrigins": ["*"], "MaxAgeSeconds": 86400}]},
        )
    except Exception:
        pass

    DEFAULTS: dict = {
        "large_gpu": False,
        "single_onnx": False,
        "cpu": False,
        "overlap_demucs": 0.1,
        "overlap_VOCFT": 0.1,
        "overlap_InstHQ4": 0.1,
        "overlap_VitLarge": 1,
        "overlap_InstVoc": 2,
        "overlap_BSRoformer": 2,
        "weight_InstVoc": 3.39,
        "weight_VOCFT": 1.0,
        "weight_InstHQ4": 1.0,
        "weight_VitLarge": 1.0,
        "weight_BSRoformer": 9.18,
        "weight_Kim_MelRoformer": 10.0,
        "BigShifts": 3,
        "use_BSRoformer6Stem": True,
        "vocals_only": True,
        "use_BSRoformer": True,
        "use_Kim_MelRoformer": True,
        "use_InstVoc": True,
        "use_VitLarge": False,
        "use_InstHQ4": False,
        "use_VOCFT": False,
        "BSRoformer_model": "ep_317_1297",
        "output_format": "FLAC",
        "input_gain": 0,
        "restore_gain": False,
        "filter_vocals": False,
    }

    def _flac_to_mp3(flac_path: str) -> str:
        mp3_path = flac_path[:-5] + ".mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", flac_path, "-b:a", "192k", "-ar", "44100", mp3_path],
            capture_output=True, check=True,
        )
        return mp3_path

    async def _queue_to_sse(q: asyncio.Queue):
        """Drain queue as SSE, sending keepalives every 15 s of silence."""
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=8.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"

    _YT_JUNK = re.compile(
        r'\s*[\(\[]\s*(?:official\s+)?(?:music\s+)?'
        r'(?:video|audio|lyrics?|hq|4k|mv|visualizer|live|clip|explicit)\s*[\)\]]\s*',
        re.IGNORECASE,
    )
    _YT_RE = re.compile(r'^https?://(www\.)?(youtube\.com/watch|youtu\.be/|youtube\.com/shorts/)')

    def _clean_yt_title(title: str) -> str:
        return re.sub(r'\s+', ' ', _YT_JUNK.sub('', title)).strip()

    def _yt_url_ok(url: str) -> bool:
        return bool(_YT_RE.match(url.strip()))

    def _download_yt(url: str) -> tuple[bytes, str]:
        """Extract audio via a self-hosted Cobalt instance (see cobalt_app.py)."""
        import json
        import urllib.error
        import urllib.request

        cobalt_url = os.environ.get("COBALT_URL", "").rstrip("/")
        if not cobalt_url:
            raise RuntimeError("COBALT_URL is not configured (see the 'cobalt' Modal secret).")

        req = urllib.request.Request(
            cobalt_url + "/",
            data=json.dumps({
                "url": url,
                "downloadMode": "audio",
                "audioFormat": "mp3",
                "audioBitrate": "320",
                "filenameStyle": "pretty",
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Api-Key {os.environ.get('COBALT_API_KEY', '')}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                meta = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Cobalt request failed: {e.read().decode(errors='replace')}")

        if meta.get("status") not in ("tunnel", "redirect"):
            raise RuntimeError(f"Cobalt error: {meta.get('error', meta)}")

        dl_req = urllib.request.Request(meta["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(dl_req, timeout=300) as resp:
            audio_bytes = resp.read()

        title = os.path.splitext(meta.get("filename", "audio.mp3"))[0]
        return audio_bytes, title

    @web.post("/api/youtube")
    async def api_youtube(request: Request):
        data = await request.json()
        url = (data.get("url") or "").strip()
        if not _yt_url_ok(url):
            raise HTTPException(status_code=400, detail="Only YouTube URLs are accepted")
        try:
            audio_bytes, raw_title = await asyncio.get_event_loop().run_in_executor(
                None, _download_yt, url
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Download failed: {e}")
        title = _clean_yt_title(raw_title)
        safe = re.sub(r'[<>:"/\\|?*]', '', title).strip() or 'audio'
        filename = safe + '.mp3'
        return {"audio": base64.b64encode(audio_bytes).decode(), "filename": filename, "title": title}

    @web.get("/")
    async def index():
        return FileResponse("/app/static/index.html")

    @web.get("/result/{job_id}")
    async def share_page(job_id: str):
        return FileResponse("/app/static/index.html")

    @web.post("/api/share")
    async def api_share(request: Request):
        data = await request.json()
        try:
            job_id = await asyncio.get_event_loop().run_in_executor(
                None, _upload_to_r2, data["files"], data.get("mode"), data.get("title"), data.get("artist"),
            )
            return {"uuid": job_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web.get("/api/library")
    async def api_library():
        try:
            items = await asyncio.get_event_loop().run_in_executor(None, _list_library)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"items": items}

    @web.patch("/api/library/{job_id}")
    async def api_library_rename(job_id: str, request: Request):
        data = await request.json()
        try:
            manifest = await asyncio.get_event_loop().run_in_executor(None, _update_manifest, job_id, data)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Not found: {e}")
        return manifest

    @web.delete("/api/library/{job_id}")
    async def api_library_delete(job_id: str):
        try:
            await asyncio.get_event_loop().run_in_executor(None, _delete_job, job_id)
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

    @web.get("/api/result/{job_id}")
    async def get_result(job_id: str):
        r2 = _get_r2()
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

        async def _convert(fname, b64, fpath):
            if fname.endswith(".flac"):
                try:
                    mp3 = await loop.run_in_executor(None, _flac_to_mp3, fpath)
                    with open(mp3, "rb") as fh:
                        return (os.path.basename(mp3), base64.b64encode(fh.read()).decode())
                except Exception:
                    return (fname, b64)
            return (fname, b64)

        async def _task():
            try:
                async for msg in separate.remote_gen.aio(audio_bytes, opts):
                    if msg.get("type") == "result":
                        files: dict[str, str] = {}
                        with tempfile.TemporaryDirectory() as td:
                            entries = []
                            for fname, b64 in msg["files"].items():
                                fpath = os.path.join(td, fname)
                                with open(fpath, "wb") as fh:
                                    fh.write(base64.b64decode(b64))
                                entries.append((fname, b64, fpath))
                            results = await asyncio.gather(*[_convert(f, b, p) for f, b, p in entries])
                            files = dict(results)
                        await q.put({"type": "done", "files": files})
                    else:
                        await q.put(msg)
            except Exception as exc:
                import traceback
                await q.put({"type": "error", "message": traceback.format_exc()})
            finally:
                await q.put(None)

        asyncio.create_task(_task())
        return StreamingResponse(
            _queue_to_sse(q),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return web
