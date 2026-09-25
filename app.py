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
import tempfile
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from neiro_common import (
    DEFAULTS, INDEX_HEADERS, JOB_ID_RE, SPEED_FILE_RE, SPEED_SOURCE_RE, STAGED_PREFIX,
    clean_yt_title, delete_job, download_yt, finalize_outputs, get_r2, list_library,
    queue_to_sse, r2_configured, save_staged, stretch_stem, update_manifest, yt_url_ok,
)

load_dotenv()

app = FastAPI(title="Neiro")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html", headers=INDEX_HEADERS)


@app.get("/result/{job_id}")
async def share_page(job_id: str):
    return FileResponse("static/index.html", headers=INDEX_HEADERS)


@app.post("/api/share/save")
async def api_share_save(request: Request):
    """Promote a staged separation result into the library.

    Stems are already in R2 (staged under tmp/ by /api/separate), so saving is just a
    server-side copy + manifest — the browser never uploads audio. Browser-to-R2
    uploads of stem-sized files failed unpredictably in Safari/WebKit ("Load failed").
    """
    data = await request.json()
    job_id = data.get("job_id") or ""
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job_id")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, save_staged, job_id, data.get("mode"), data.get("title"), data.get("artist"),
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
        items = await loop.run_in_executor(None, list_library)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"items": items}


@app.patch("/api/library/{job_id}")
async def api_library_rename(job_id: str, request: Request):
    data = await request.json()
    loop = asyncio.get_event_loop()
    try:
        manifest = await loop.run_in_executor(None, update_manifest, job_id, data)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Not found: {e}")
    return manifest


@app.delete("/api/library/{job_id}")
async def api_library_delete(job_id: str):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, delete_job, job_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Not found: {e}")
    return {"deleted": job_id}


@app.post("/api/youtube")
async def api_youtube(request: Request):
    data = await request.json()
    url = (data.get("url") or "").strip()
    if not yt_url_ok(url):
        raise HTTPException(400, "Only YouTube URLs are accepted")
    try:
        loop = asyncio.get_event_loop()
        audio_bytes, raw_title = await loop.run_in_executor(None, download_yt, url)
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")
    title = clean_yt_title(raw_title)
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
        if not r2_configured():
            raise HTTPException(400, "R2 is not configured")
        if not (SPEED_SOURCE_RE.match(prefix) and names
                and all(isinstance(n, str) and SPEED_FILE_RE.match(n) for n in names.values())):
            raise HTTPException(400, "Invalid source")
        inputs = {stem: ("r2", f"{prefix}/{name}") for stem, name in names.items()}
    else:
        files = data.get("files") or {}
        if not files:
            raise HTTPException(400, "No stems given")
        inputs = {stem: ("b64", b64) for stem, b64 in files.items()}

    deliver_via_r2 = r2_configured()
    r2 = get_r2() if deliver_via_r2 else None  # boto3 clients are thread-safe
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
            raise HTTPException(404, f"Stem {stem} is no longer in storage")
        except Exception as e:
            raise HTTPException(500, f"Speed change failed on {stem}: {e}")

    results = await asyncio.gather(*(run_one(s, k, r) for s, (k, r) in inputs.items()))
    return {"delivery": "url" if deliver_via_r2 else "base64", "files": dict(results)}


@app.get("/api/result/{job_id}")
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
                _put(finalize_outputs(output_dir))
            finally:
                builtins.print = _orig
                _put(None)

        threading.Thread(target=_thread, daemon=True).start()
        return StreamingResponse(
            queue_to_sse(q),
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
                        await q.put(await loop.run_in_executor(None, finalize_outputs, td))
                else:
                    await q.put(msg)
        except Exception as exc:
            import traceback
            await q.put({"type": "error", "message": traceback.format_exc()})
        finally:
            await q.put(None)

    asyncio.create_task(_modal_task())
    return StreamingResponse(
        queue_to_sse(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
