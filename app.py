"""
Local FastAPI server for Neiro music source separation.

  CPU mode  — streams inference progress + base64 files in a single POST response
  GPU mode  — proxies Modal generator the same way

Start with:  uv run uvicorn app:app --reload --port 8000
"""

import asyncio
import base64
import builtins
import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="Neiro")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def _configure_r2_cors():
    try:
        _get_r2().put_bucket_cors(
            Bucket=os.environ["R2_BUCKET_NAME"],
            CORSConfiguration={"CORSRules": [{"AllowedHeaders": ["*"], "AllowedMethods": ["GET", "HEAD"],
                                               "AllowedOrigins": ["*"], "MaxAgeSeconds": 86400}]},
        )
    except Exception:
        pass


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/result/{job_id}")
async def share_page(job_id: str):
    return FileResponse("static/index.html")


@app.post("/api/share")
async def api_share(request: Request):
    data = await request.json()
    try:
        loop = asyncio.get_event_loop()
        job_id = await loop.run_in_executor(
            None, _upload_to_r2, data["files"], data.get("mode"), data.get("title"), data.get("artist"),
        )
        return {"uuid": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/library")
async def api_library():
    loop = asyncio.get_event_loop()
    try:
        items = await loop.run_in_executor(None, _list_library)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"items": items}


@app.patch("/api/library/{job_id}")
async def api_library_rename(job_id: str, request: Request):
    data = await request.json()
    loop = asyncio.get_event_loop()
    try:
        manifest = await loop.run_in_executor(None, _update_manifest, job_id, data)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Not found: {e}")
    return manifest


@app.delete("/api/library/{job_id}")
async def api_library_delete(job_id: str):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _delete_job, job_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Not found: {e}")
    return {"deleted": job_id}


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
        raise RuntimeError(
            "COBALT_URL is not configured — deploy cobalt_app.py and set "
            "COBALT_URL / COBALT_API_KEY (see .env.example)."
        )

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


@app.post("/api/youtube")
async def api_youtube(request: Request):
    data = await request.json()
    url = (data.get("url") or "").strip()
    if not _yt_url_ok(url):
        raise HTTPException(400, "Only YouTube URLs are accepted")
    try:
        loop = asyncio.get_event_loop()
        audio_bytes, raw_title = await loop.run_in_executor(None, _download_yt, url)
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")
    title = _clean_yt_title(raw_title)
    safe = re.sub(r'[<>:"/\\|?*]', '', title).strip() or 'audio'
    filename = safe + '.mp3'
    return {"audio": base64.b64encode(audio_bytes).decode(), "filename": filename, "title": title}


@app.get("/api/audio-proxy")
async def audio_proxy(url: str):
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


@app.get("/api/result/{job_id}")
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


# ---------------------------------------------------------------------------
# R2 helpers
# ---------------------------------------------------------------------------

def _get_r2():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(s3={"addressing_style": "path"}),
    )


def _upload_to_r2(files: dict, mode: str | None = None, title: str | None = None, artist: str | None = None) -> str:
    """Upload base64 files to R2, return UUID."""
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
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    r2.put_object(
        Bucket=bucket,
        Key=f"results/{job_id}/manifest.json",
        Body=manifest.encode(),
        ContentType="application/json",
    )
    return job_id


def _list_library() -> list[dict]:
    """List every saved job's manifest, newest first."""
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
    """Patch title/artist/mode onto an existing job's manifest."""
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
    manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    r2.put_object(Bucket=bucket, Key=key, Body=json.dumps(manifest).encode(), ContentType="application/json")
    return manifest


def _delete_job(job_id: str) -> None:
    """Delete every object under a job's results/ prefix, manifest included."""
    r2 = _get_r2()
    bucket = os.environ["R2_BUCKET_NAME"]
    prefix = f"results/{job_id}/"
    r2.head_object(Bucket=bucket, Key=f"{prefix}manifest.json")  # 404s if the job doesn't exist
    paginator = r2.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            r2.delete_objects(Bucket=bucket, Delete={"Objects": keys})


# ---------------------------------------------------------------------------
# Defaults (mirrors argparse defaults in inference.py)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_inf = None
_inf_lock = threading.Lock()


def _get_inf():
    global _inf
    if _inf is None:
        with _inf_lock:
            if _inf is None:
                import inference as m
                _inf = m
    return _inf


def _flac_to_mp3(flac_path: str) -> str:
    mp3_path = flac_path[:-5] + ".mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", flac_path, "-b:a", "192k", "-ar", "44100", mp3_path],
        capture_output=True, check=True,
    )
    return mp3_path


def _encode_outputs(output_dir: str) -> dict[str, str]:
    """Read each FLAC in output_dir, convert to MP3, return {filename: base64}."""
    files: dict[str, str] = {}
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if fname.endswith(".flac"):
            try:
                mp3 = _flac_to_mp3(fpath)
                key = os.path.basename(mp3)
                with open(mp3, "rb") as fh:
                    files[key] = base64.b64encode(fh.read()).decode()
            except Exception:
                with open(fpath, "rb") as fh:
                    files[fname] = base64.b64encode(fh.read()).decode()
        else:
            with open(fpath, "rb") as fh:
                files[fname] = base64.b64encode(fh.read()).decode()
    return files


# ---------------------------------------------------------------------------
# Streaming helpers — convert sync/async sources to a keepalive-aware SSE gen
# ---------------------------------------------------------------------------

async def _queue_to_sse(q: asyncio.Queue):
    """Drain an asyncio.Queue as SSE lines, sending keepalives every 15 s."""
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=8.0)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue
        if msg is None:
            break
        yield f"data: {json.dumps(msg)}\n\n"


# ---------------------------------------------------------------------------
# POST /api/separate — returns a single streaming SSE response
# ---------------------------------------------------------------------------

@app.post("/api/separate")
async def api_separate(
    file: UploadFile = File(...),
    options: str = Form("{}"),
):
    audio_bytes = await file.read()
    opts = {**DEFAULTS, **json.loads(options)}
    use_cpu = bool(opts.get("cpu", False))

    # ── CPU path ──────────────────────────────────────────────────────────
    if use_cpu:
        ext = Path(file.filename).suffix or ".mp3"
        tmpdir = tempfile.mkdtemp()
        input_path = os.path.join(tmpdir, f"input{ext}")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)
        with open(input_path, "wb") as fh:
            fh.write(audio_bytes)
        opts["input_audio"] = [input_path]
        opts["output_folder"] = output_dir

        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _thread():
            inf = _get_inf()

            def _put(msg):
                loop.call_soon_threadsafe(q.put_nowait, msg)

            class PatchedTqdm:
                def __init__(self, iterable=None, *args, **kwargs):
                    self._items = list(iterable) if iterable is not None else []
                    self._total = kwargs.get("total", len(self._items))
                    self._n = 0

                def __iter__(self):
                    for item in self._items:
                        yield item
                        self._n += 1
                        _put({"type": "progress", "n": self._n, "total": self._total})

            _orig = builtins.print
            builtins.print = lambda *a, **k: _put({"type": "log", "message": " ".join(str(x) for x in a)})
            inf.tqdm = PatchedTqdm
            inf.options = opts

            try:
                inf.predict_with_model(opts)
            except Exception as exc:
                import traceback
                _put({"type": "error", "message": traceback.format_exc()})
            else:
                files = _encode_outputs(output_dir)
                _put({"type": "done", "files": files})
            finally:
                builtins.print = _orig
                _put(None)

        threading.Thread(target=_thread, daemon=True).start()
        return StreamingResponse(
            _queue_to_sse(q),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Modal GPU path ─────────────────────────────────────────────────────
    opts.pop("input_audio", None)
    opts.pop("output_folder", None)
    opts["cpu"] = False

    q: asyncio.Queue = asyncio.Queue()

    loop = asyncio.get_event_loop()

    async def _convert_file(fname, b64, fpath):
        if fname.endswith(".flac"):
            try:
                mp3 = await loop.run_in_executor(None, _flac_to_mp3, fpath)
                with open(mp3, "rb") as fh:
                    return (os.path.basename(mp3), base64.b64encode(fh.read()).decode())
            except Exception:
                return (fname, b64)
        return (fname, b64)

    async def _modal_task():
        import modal_app as ma
        try:
            async for msg in ma.separate.remote_gen.aio(audio_bytes, opts):
                if msg.get("type") == "result":
                    files: dict[str, str] = {}
                    with tempfile.TemporaryDirectory() as td:
                        entries = []
                        for fname, b64 in msg["files"].items():
                            fpath = os.path.join(td, fname)
                            with open(fpath, "wb") as fh:
                                fh.write(base64.b64decode(b64))
                            entries.append((fname, b64, fpath))
                        results = await asyncio.gather(*[_convert_file(f, b, p) for f, b, p in entries])
                        files = dict(results)
                    await q.put({"type": "done", "files": files})
                else:
                    await q.put(msg)
        except Exception as exc:
            import traceback
            await q.put({"type": "error", "message": traceback.format_exc()})
        finally:
            await q.put(None)

    asyncio.create_task(_modal_task())
    return StreamingResponse(
        _queue_to_sse(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
