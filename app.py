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
async def _configure_r2_bucket():
    if not _r2_configured():
        return
    r2, bucket = _get_r2(), os.environ["R2_BUCKET_NAME"]
    try:
        r2.put_bucket_cors(
            Bucket=bucket,
            CORSConfiguration={"CORSRules": [{"AllowedHeaders": ["*"], "AllowedMethods": ["GET", "HEAD"],
                                               "AllowedOrigins": ["*"], "MaxAgeSeconds": 86400}]},
        )
    except Exception:
        pass
    try:
        # Unsaved separation results are staged under tmp/ — expire them after a day.
        try:
            rules = r2.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
        except Exception:
            rules = []
        rules = [r for r in rules if r.get("ID") != "expire-unsaved-runs"]
        rules.append({"ID": "expire-unsaved-runs", "Status": "Enabled",
                      "Filter": {"Prefix": f"{_STAGED_PREFIX}/"}, "Expiration": {"Days": 1}})
        r2.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": rules})
    except Exception:
        pass


_INDEX_HEADERS = {"Cache-Control": "no-cache"}  # always revalidate — this is a single-file SPA,
# so a stale cached copy after a deploy silently keeps calling removed/changed API routes


@app.get("/")
async def index():
    return FileResponse("static/index.html", headers=_INDEX_HEADERS)


@app.get("/result/{job_id}")
async def share_page(job_id: str):
    return FileResponse("static/index.html", headers=_INDEX_HEADERS)


@app.post("/api/share/save")
async def api_share_save(request: Request):
    """Promote a staged separation result into the library.

    Stems are already in R2 (staged under tmp/ by /api/separate), so saving is just a
    server-side copy + manifest — the browser never uploads audio. Browser-to-R2
    uploads of stem-sized files failed unpredictably in Safari/WebKit ("Load failed").
    """
    data = await request.json()
    job_id = data.get("job_id") or ""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job_id")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _save_staged, job_id, data.get("mode"), data.get("title"), data.get("artist"),
        )
    except FileNotFoundError:
        raise HTTPException(404, "This run expired before it was saved — separate the track again.")
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"uuid": job_id}


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

    if not audio_bytes:
        # Cobalt sometimes returns an HTTP 200 with an empty body when YouTube
        # silently rejects the upstream fetch (e.g. a video that now requires
        # a proof-of-origin token) — surface this instead of letting an empty
        # file reach the separation pipeline.
        raise RuntimeError(
            "Cobalt returned an empty file for this video — YouTube likely "
            "rejected the download (this can happen for videos that need a "
            "proof-of-origin token). Try a different video or link."
        )

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


def _run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[-500:]}")


def _stretch_stem(src_path: str, rate: float, workdir: str) -> str:
    """Pitch-preserving time-stretch via Rubber Band — chosen over WSOLA/phase-vocoder
    options (e.g. the AudioWorklet approach tried earlier) because it's specifically tuned
    for polyphonic full mixes, not just monophonic/speech material.

    Runs the rubberband CLI directly on files — the same `rubberband -q --tempo <rate>`
    call on 16-bit WAV that pyrubberband made — with ffmpeg decoding before and encoding
    FLAC after. Decoding the MP3 in Python via soundfile took ~11s per stem on a 21-minute
    track on Modal; ffmpeg does it in under a second. Returns the output FLAC's path."""
    wav_in = os.path.join(workdir, "in.wav")
    wav_out = os.path.join(workdir, "stretched.wav")
    flac_out = os.path.join(workdir, "out.flac")
    _run_cmd(["ffmpeg", "-loglevel", "error", "-y", "-i", src_path, "-c:a", "pcm_s16le", wav_in])
    if rate == 1.0:
        wav_out = wav_in  # like pyrubberband, don't run a no-op stretch
    else:
        _run_cmd(["rubberband", "-q", "--tempo", str(rate), wav_in, wav_out])
    _run_cmd(["ffmpeg", "-loglevel", "error", "-y", "-i", wav_out, "-c:a", "flac", flac_out])
    return flac_out


# Where /api/speed may read stems from in R2: a staged (tmp/) or saved (results/) job.
_SPEED_SOURCE_RE = re.compile(r"^(tmp|results)/[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
_SPEED_FILE_RE = re.compile(r"^(?!\.)[^/\\]+\.(mp3|flac)$")  # a bare filename within that job


@app.post("/api/speed")
async def api_speed(request: Request):
    """Re-render every stem at a new playback speed with pitch unchanged.

    Runs while paused (see static/index.html's applySpeed), always from the pristine
    (rate=1) stems at the target absolute rate, so repeated speed changes never compound
    re-stretches of an already-stretched buffer. Two request shapes:

      {rate, source: {prefix: "tmp/<job_id>" | "results/<job_id>", files: {stem: filename}}}
          The stems are already in R2 (staged or saved), so the server reads them from
          there and the browser uploads nothing. 404 if they're gone (e.g. tmp/ expired);
          the client then falls back to uploading.
      {rate, files: {stem: base64}}
          Fallback when the result isn't in R2.

    With R2 configured the stretched FLACs are written under tmp/<new id>/ (expired with
    the rest of tmp/) and the response carries presigned URLs ({"delivery": "url"});
    otherwise they come back inline as base64 ({"delivery": "base64"}). Profiling showed
    base64 over JSON dominating on long tracks (~250 MB up, ~530 MB down for 21 minutes).
    Each stem is processed concurrently (the heavy lifting is in ffmpeg/rubberband
    subprocesses, so this parallelizes despite the GIL) to keep total latency close to a
    single stem's processing time rather than their sum.
    """
    from botocore.exceptions import ClientError

    data = await request.json()
    try:
        rate = float(data.get("rate"))
    except (TypeError, ValueError):
        raise HTTPException(400, "rate must be a number")
    if not (0.1 <= rate <= 2.0):
        raise HTTPException(400, "rate must be 0.1-2.0")

    source = data.get("source")
    if source:
        prefix, names = source.get("prefix") or "", source.get("files") or {}
        if not _r2_configured():
            raise HTTPException(400, "R2 is not configured")
        if not (_SPEED_SOURCE_RE.match(prefix) and names
                and all(isinstance(n, str) and _SPEED_FILE_RE.match(n) for n in names.values())):
            raise HTTPException(400, "Invalid source")
        inputs = {stem: ("r2", f"{prefix}/{name}") for stem, name in names.items()}
    else:
        files = data.get("files") or {}
        if not files:
            raise HTTPException(400, "No stems given")
        inputs = {stem: ("b64", b64) for stem, b64 in files.items()}

    deliver_via_r2 = _r2_configured()
    r2 = _get_r2() if deliver_via_r2 else None  # boto3 clients are thread-safe
    bucket = os.environ.get("R2_BUCKET_NAME")
    out_id = str(uuid.uuid4())

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
            out_path = _stretch_stem(src, rate, workdir)
            if deliver_via_r2:
                key = f"{_STAGED_PREFIX}/{out_id}/{stem}.flac"
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
            raise HTTPException(404, f"Stem {stem} is no longer in storage")
        except Exception as e:
            raise HTTPException(500, f"Speed change failed on {stem}: {e}")

    results = await asyncio.gather(*(run_one(s, k, r) for s, (k, r) in inputs.items()))
    return {"delivery": "url" if deliver_via_r2 else "base64", "files": dict(results)}


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


def _r2_configured() -> bool:
    return all(os.environ.get(k) for k in
               ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"))


_STAGED_PREFIX = "tmp"  # unsaved separation results; expired by the lifecycle rule in _configure_r2_bucket
_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")


def _content_type_for(fname: str) -> str:
    if fname.endswith(".mp3"):
        return "audio/mpeg"
    return "audio/flac" if fname.endswith(".flac") else "application/octet-stream"


def _stage_to_r2(paths: dict[str, str]) -> dict:
    """Upload finished stems to R2 under tmp/{job_id}/ and return the SSE 'done' message
    carrying presigned GET URLs instead of the audio itself."""
    from concurrent.futures import ThreadPoolExecutor

    r2, bucket = _get_r2(), os.environ["R2_BUCKET_NAME"]
    job_id = str(uuid.uuid4())

    def put(item):
        name, path = item
        with open(path, "rb") as fh:
            r2.put_object(Bucket=bucket, Key=f"{_STAGED_PREFIX}/{job_id}/{name}",
                          Body=fh.read(), ContentType=_content_type_for(name))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(put, paths.items()))
    urls = {
        name: r2.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": f"{_STAGED_PREFIX}/{job_id}/{name}",
                    # so the ⬇ buttons download (cross-origin <a download> is ignored) — <audio> is unaffected
                    "ResponseContentDisposition": f'attachment; filename="{name}"'},
            ExpiresIn=86400)
        for name in paths
    }
    return {"type": "done", "job_id": job_id, "files": urls}


def _deliver_outputs(paths: dict[str, str]) -> dict:
    """Build the SSE 'done' message: staged-in-R2 URLs when R2 is configured (Save is then a
    server-side copy), otherwise — or if staging fails — the audio inline as base64."""
    if _r2_configured():
        try:
            return _stage_to_r2(paths)
        except Exception:
            import traceback
            traceback.print_exc()
    files = {}
    for name, path in paths.items():
        with open(path, "rb") as fh:
            files[name] = base64.b64encode(fh.read()).decode()
    return {"type": "done", "files": files}


def _save_staged(job_id: str, mode: str | None, title: str | None, artist: str | None) -> None:
    r2, bucket = _get_r2(), os.environ["R2_BUCKET_NAME"]
    src = f"{_STAGED_PREFIX}/{job_id}/"
    names = []
    for page in r2.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=src):
        names += [o["Key"][len(src):] for o in page.get("Contents", [])]
    if not names:
        try:
            r2.head_object(Bucket=bucket, Key=f"results/{job_id}/manifest.json")  # already saved
        except Exception:
            raise FileNotFoundError(job_id)
        return
    for name in names:
        r2.copy_object(Bucket=bucket, Key=f"results/{job_id}/{name}",
                       CopySource={"Bucket": bucket, "Key": src + name})
    _write_manifest(job_id, names, mode, title, artist)


def _write_manifest(
    job_id: str, files: list[str],
    mode: str | None = None, title: str | None = None, artist: str | None = None,
) -> str:
    """Write manifest.json for a job's stems already in R2."""
    r2 = _get_r2()
    bucket = os.environ["R2_BUCKET_NAME"]
    manifest = json.dumps({
        "files": files,
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


def _finalize_outputs(output_dir: str) -> dict:
    """Convert each FLAC in output_dir to MP3 (in parallel) and build the SSE 'done' message."""
    from concurrent.futures import ThreadPoolExecutor

    names = os.listdir(output_dir)

    def resolve(fname):
        fpath = os.path.join(output_dir, fname)
        if fname.endswith(".flac"):
            try:
                mp3 = _flac_to_mp3(fpath)
                return os.path.basename(mp3), mp3
            except Exception:
                pass  # keep the FLAC
        return fname, fpath

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = dict(pool.map(resolve, names))
    return _deliver_outputs(paths)


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
                _put(_finalize_outputs(output_dir))
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

    async def _modal_task():
        import modal_app as ma
        try:
            async for msg in ma.separate.remote_gen.aio(audio_bytes, opts):
                if msg.get("type") == "result":
                    with tempfile.TemporaryDirectory() as td:
                        for fname, b64 in msg["files"].items():
                            with open(os.path.join(td, fname), "wb") as fh:
                                fh.write(base64.b64decode(b64))
                        del msg
                        await q.put(await loop.run_in_executor(None, _finalize_outputs, td))
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
