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
    .apt_install("ffmpeg")  # for FLAC → MP3 conversion
    .pip_install("fastapi[standard]", "python-multipart", "boto3")
    .add_local_dir("static", remote_path="/app/static")
)


# ── GPU separation function ────────────────────────────────────────────────

@app.function(
    image=gpu_image,
    gpu="A10G",
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
    secrets=[modal.Secret.from_name("cloudflare-r2")],
)
@modal.asgi_app()
def fastapi_app():
    import asyncio
    import base64
    import json
    import uuid

    import boto3
    from botocore.config import Config
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, StreamingResponse
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

    def _upload_to_r2(files: dict) -> str:
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
            ["ffmpeg", "-y", "-i", flac_path, "-q:a", "2", mp3_path],
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
                None, _upload_to_r2, data["files"]
            )
            return {"uuid": job_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
        return {"files": urls}

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

        async def _task():
            try:
                async for msg in separate.remote_gen.aio(audio_bytes, opts):
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

        asyncio.create_task(_task())
        return StreamingResponse(
            _queue_to_sse(q),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return web
