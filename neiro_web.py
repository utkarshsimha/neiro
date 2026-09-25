"""The Neiro web API — one set of route handlers for both deployments.

app.py (local dev server) and modal_app.py's fastapi_app (Modal) both build their FastAPI app
with build_app(); only the few things that genuinely differ between them are passed in:

  static_dir      where static/ lives (the repo locally, /app/static in the Modal image)
  gpu_separate    the Modal `separate` generator (`.remote_gen.aio`) — async-iterates
                  log/progress/result/error messages for one separation
  cpu_separate    local only: run inference.py in-process (see app.py); None on Modal, where
                  every separation runs on the GPU
  remote_stretch  Modal only: hand R2-backed speed changes to the big-CPU stretch_stems
                  function; None locally, where the stretch runs in-process

Unlike neiro_common, this imports FastAPI at module level, so it must only be imported where
FastAPI is installed: app.py, and inside modal_app.fastapi_app (never at modal_app's top
level, which the GPU container also loads).
"""
import asyncio
import base64
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from neiro_common import (
    DEFAULTS, INDEX_HEADERS, JOB_ID_RE, SPEED_FILE_RE, SPEED_SOURCE_RE, STAGED_PREFIX,
    clean_yt_title, delete_job, download_yt, finalize_outputs, get_r2, list_library,
    queue_to_sse, r2_configured, save_staged, stretch_stem, update_manifest, yt_url_ok,
)

# (audio_bytes, options) -> async iterator of {type: log|progress|result|error} messages
GpuSeparate = Callable[[bytes, dict], AsyncIterator[dict]]
# (audio_bytes, filename, options, put) -> output dir of FLAC stems; runs in a worker
# thread and reports progress through the thread-safe `put(message)`.
CpuSeparate = Callable[[bytes, str, dict, Callable[[dict], None]], str]
# (keys {stem: R2 key}, rate, out_prefix) -> {stem: presigned url}
RemoteStretch = Callable[[dict, float, str], Awaitable[dict]]


def _sse(q: asyncio.Queue) -> StreamingResponse:
    return StreamingResponse(
        queue_to_sse(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def build_app(static_dir: str, gpu_separate: GpuSeparate, cpu_separate: CpuSeparate | None = None,
              remote_stretch: RemoteStretch | None = None) -> FastAPI:
    web = FastAPI(title="Neiro")
    web.mount("/static", StaticFiles(directory=static_dir), name="static")
    index_html = os.path.join(static_dir, "index.html")

    @web.get("/")
    async def index():
        return FileResponse(index_html, headers=INDEX_HEADERS)

    @web.get("/result/{job_id}")
    async def share_page(job_id: str):
        return FileResponse(index_html, headers=INDEX_HEADERS)

    # ── Library ──────────────────────────────────────────────────────────────

    @web.post("/api/share/save")
    async def api_share_save(request: Request):
        """Promote a staged separation result into the library.

        Stems are already in R2 (staged under tmp/ by /api/separate), so saving is just a
        server-side copy + manifest — the browser never uploads audio. Browser-to-R2
        uploads of stem-sized files failed unpredictably in Safari/WebKit ("Load failed").
        """
        data = await request.json()
        job_id = data.get("job_id") or ""
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(status_code=400, detail="Invalid job_id")
        try:
            await asyncio.get_running_loop().run_in_executor(
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
            items = await asyncio.get_running_loop().run_in_executor(None, list_library)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"items": items}

    @web.patch("/api/library/{job_id}")
    async def api_library_rename(job_id: str, request: Request):
        data = await request.json()
        try:
            manifest = await asyncio.get_running_loop().run_in_executor(None, update_manifest, job_id, data)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Not found: {e}")
        return manifest

    @web.delete("/api/library/{job_id}")
    async def api_library_delete(job_id: str):
        try:
            await asyncio.get_running_loop().run_in_executor(None, delete_job, job_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Not found: {e}")
        return {"deleted": job_id}

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

    # ── Ingestion and playback helpers ───────────────────────────────────────

    @web.post("/api/youtube")
    async def api_youtube(request: Request):
        data = await request.json()
        url = (data.get("url") or "").strip()
        if not yt_url_ok(url):
            raise HTTPException(status_code=400, detail="Only YouTube URLs are accepted")
        try:
            audio_bytes, raw_title = await asyncio.get_running_loop().run_in_executor(None, download_yt, url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Download failed: {e}")
        title = clean_yt_title(raw_title)
        safe = re.sub(r'[<>:"/\\|?*]', '', title).strip() or 'audio'
        filename = safe + '.mp3'
        return {"audio": base64.b64encode(audio_bytes).decode(), "filename": filename, "title": title}

    @web.get("/api/audio-proxy")
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

    # ── Separation ───────────────────────────────────────────────────────────

    @web.post("/api/separate")
    async def api_separate(
        file: UploadFile = File(...),
        options: str = Form("{}"),
    ):
        """Run a separation, streaming log/progress messages as SSE and ending with a 'done'
        message (staged-in-R2 URLs, or inline base64 — see neiro_common.deliver_outputs).

        `cpu: true` runs it in-process via `cpu_separate` where there is one (the local
        server); otherwise — and always on Modal — it runs on the GPU `separate` function."""
        audio_bytes = await file.read()
        opts = {**DEFAULTS, **json.loads(options)}
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        if opts.get("cpu") and cpu_separate:
            def put(msg):
                loop.call_soon_threadsafe(q.put_nowait, msg)

            def run():
                try:
                    output_dir = cpu_separate(audio_bytes, file.filename or "input.mp3", opts, put)
                except Exception:
                    import traceback
                    put({"type": "error", "message": traceback.format_exc()})
                else:
                    put(finalize_outputs(output_dir))
                finally:
                    put(None)

            threading.Thread(target=run, daemon=True).start()
            return _sse(q)

        opts.pop("input_audio", None)
        opts.pop("output_folder", None)
        opts["cpu"] = False

        async def run_gpu():
            try:
                async for msg in gpu_separate(audio_bytes, opts):
                    if msg.get("type") == "result":
                        with tempfile.TemporaryDirectory() as td:
                            for fname, b64 in msg["files"].items():
                                with open(os.path.join(td, fname), "wb") as fh:
                                    fh.write(base64.b64decode(b64))
                            del msg
                            await q.put(await loop.run_in_executor(None, finalize_outputs, td))
                    else:
                        await q.put(msg)
            except Exception:
                import traceback
                await q.put({"type": "error", "message": traceback.format_exc()})
            finally:
                await q.put(None)

        asyncio.create_task(run_gpu())
        return _sse(q)

    # ── Practice-mode speed changes ──────────────────────────────────────────

    @web.post("/api/speed")
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

        R2-backed requests go to `remote_stretch` when there is one (Modal's stretch_stems —
        see its docstring). Otherwise each stem is processed concurrently in-process (the
        heavy lifting is in ffmpeg/rubberband subprocesses, so this parallelizes despite the
        GIL) to keep total latency close to a single stem's processing time.
        """
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

        if source and remote_stretch:
            keys = {stem: ref for stem, (_, ref) in inputs.items()}
            try:
                urls = await remote_stretch(keys, rate, f"{STAGED_PREFIX}/{out_id}")
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
                    return r2.generate_presigned_url(
                        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400)
                with open(out_path, "rb") as fh:
                    return base64.b64encode(fh.read()).decode()

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

    return web
