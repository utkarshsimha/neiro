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
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="Neiro")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        job_id = await loop.run_in_executor(None, _upload_to_r2, data["files"])
        return {"uuid": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    return {"files": urls}


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


def _upload_to_r2(files: dict) -> str:
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
    manifest = json.dumps({"files": list(files.keys())})
    r2.put_object(
        Bucket=bucket,
        Key=f"results/{job_id}/manifest.json",
        Body=manifest.encode(),
        ContentType="application/json",
    )
    return job_id


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
        ["ffmpeg", "-y", "-i", flac_path, "-q:a", "2", mp3_path],
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

    async def _modal_task():
        import modal_app as ma
        try:
            async for msg in ma.separate.remote_gen.aio(audio_bytes, opts):
                if msg.get("type") == "result":
                    files: dict[str, str] = {}
                    with tempfile.TemporaryDirectory() as td:
                        for fname, b64 in msg["files"].items():
                            fpath = os.path.join(td, fname)
                            with open(fpath, "wb") as fh:
                                fh.write(base64.b64decode(b64))
                            if fname.endswith(".flac"):
                                try:
                                    mp3 = _flac_to_mp3(fpath)
                                    with open(mp3, "rb") as fh:
                                        files[os.path.basename(mp3)] = base64.b64encode(fh.read()).decode()
                                except Exception:
                                    files[fname] = b64
                            else:
                                files[fname] = b64
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
