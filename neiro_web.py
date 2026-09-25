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
from collections.abc import AsyncIterator, Callable, Iterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from neiro_common import (
    DEFAULTS, INDEX_HEADERS, JOB_ID_RE, SPEED_FILE_RE, SPEED_SOURCE_RE, STAGED_PREFIX, STEM_NAME_RE,
    clean_yt_title, delete_job, download_yt, finalize_outputs, get_r2, inline_segment_deliverer,
    list_library, queue_to_sse, r2_configured, r2_segment_deliverer, run_speed_stream, save_staged,
    update_manifest, yt_url_ok,
)

# (audio_bytes, options) -> async iterator of {type: log|progress|result|error} messages
GpuSeparate = Callable[[bytes, dict], AsyncIterator[dict]]
# (audio_bytes, filename, options, put) -> output dir of FLAC stems; runs in a worker
# thread and reports progress through the thread-safe `put(message)`.
CpuSeparate = Callable[[bytes, str, dict, Callable[[dict], None]], str]
# (keys {stem: R2 key}, rate, out_prefix, chunk_s, start_fraction) -> async iterator of
# neiro_common.stream_stretch events (segments delivered as presigned R2 URLs)
RemoteStretch = Callable[[dict, float, str, float, float], AsyncIterator[dict]]


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
        """Re-render every stem at a new playback speed with pitch unchanged, streamed.

        Runs while paused (see static/index.html's applySpeed), always from the pristine
        (rate=1) stems at the target absolute rate, so repeated speed changes never compound
        re-stretches of an already-stretched buffer. Two request shapes:

          {rate, start_fraction?, source: {prefix: "tmp/<job_id>" | "results/<job_id>",
                                           files: {stem: filename}}}
              The stems are already in R2 (staged or saved), so the server reads them from
              there and the browser uploads nothing. 404 if they're gone (e.g. tmp/ expired);
              the client then falls back to uploading.
          {rate, start_fraction?, files: {stem: base64}}
              Fallback when the result isn't in R2.

        The response is an SSE stream of neiro_common.stream_stretch's events: a `meta`
        with the new timeline, then one `segment` per stem × ~30s piece as each finishes —
        playhead-first, starting from `start_fraction` of the way through — each with a
        presigned R2 URL (under tmp/<new id>/, expired with the rest of tmp/) or, without
        R2, inline base64; then `done`, or `error` if something fails partway. The browser
        starts playing as soon as every stem's segment under the playhead has arrived
        instead of waiting for whole files (~30s of stretch + ~15s of download and decode
        for a 21-minute track). The first event is awaited before responding, so failures
        up front (like a missing R2 stem) still get a proper HTTP status.

        R2-backed requests go to `remote_stretch` when there is one (Modal's stretch_stems —
        see its docstring); everything else is stretched in this process.
        """
        data = await request.json()
        try:
            rate = float(data.get("rate"))
            start_fraction = float(data.get("start_fraction") or 0.0)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="rate and start_fraction must be numbers")
        if not (0.1 <= rate <= 2.0):
            raise HTTPException(status_code=400, detail="rate must be 0.1-2.0")
        if not (0.0 <= start_fraction <= 1.0):
            raise HTTPException(status_code=400, detail="start_fraction must be 0-1")
        # Chunk length for the parallel stretch, which is also the length of each streamed
        # segment — see neiro_common.stream_stretch.
        chunk_s = 30.0

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
        if not all(STEM_NAME_RE.match(stem) for stem in inputs):
            raise HTTPException(status_code=400, detail="Invalid stem name")

        out_prefix = f"{STAGED_PREFIX}/{uuid.uuid4()}"
        if source and remote_stretch:
            keys = {stem: ref for stem, (_, ref) in inputs.items()}
            events = remote_stretch(keys, rate, out_prefix, chunk_s, start_fraction)
        else:
            deliver = r2_segment_deliverer(out_prefix) if r2_configured() else inline_segment_deliverer
            events = _in_thread(lambda workdir, pool: run_speed_stream(
                inputs, rate, workdir, pool, deliver, chunk_s, start_fraction))

        try:
            first = await anext(events)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="A stem is no longer in storage")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Speed change failed: {e}")

        async def stream():
            # If the client goes away mid-stream (e.g. it changed speed again, which aborts
            # this request), Starlette cancels this generator; closing `events` then stops
            # the stretch behind it (in-process: _in_thread; Modal: remote_stretch's flag).
            try:
                yield f"data: {json.dumps(first)}\n\n"
                async for ev in events:
                    yield f"data: {json.dumps(ev)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                await events.aclose()

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return web


async def _in_thread(make_events: Callable[[str, object], Iterator[dict]]) -> AsyncIterator[dict]:
    """Run a blocking event generator in a worker thread (with its own temp dir and stretch
    pool) and iterate its events here. Stopping early — the client went away — closes the
    generator, which cancels the stretch work that hasn't started."""
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    end = object()

    def run():
        pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        try:
            with tempfile.TemporaryDirectory() as workdir:
                gen = make_events(workdir, pool)
                try:
                    for ev in gen:
                        loop.call_soon_threadsafe(q.put_nowait, ev)
                        if stop.is_set():
                            break
                finally:
                    gen.close()
        except BaseException as e:  # noqa: BLE001 — re-raised on the event loop side
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            loop.call_soon_threadsafe(q.put_nowait, end)

    threading.Thread(target=run, daemon=True).start()
    try:
        while (item := await q.get()) is not end:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
