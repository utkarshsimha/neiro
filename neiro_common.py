"""Helpers shared by the local server (app.py) and the Modal deployment (modal_app.py).

app.py and modal_app.py are two separate implementations of the same API (see CLAUDE.md);
code that's genuinely identical between them lives here instead of being duplicated.
modal_app.py adds this module to its images with `add_local_python_source`.

Keep imports at module level to the standard library: modal_app.py imports this at the top,
so it's loaded in every Modal container, including the GPU one. Heavier dependencies
(boto3, numpy, soundfile) are imported inside the functions that need them.
"""
import asyncio
import base64
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone

# Always revalidate the SPA shell — it's a single file, so a stale cached copy after a deploy
# silently keeps calling removed/changed API routes.
INDEX_HEADERS = {"Cache-Control": "no-cache"}

# ── Separation: request defaults and output handling ────────────────────────

# Merged under each /api/separate request's `options` (mirrors inference.py's argparse defaults).
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


def flac_to_mp3(flac_path: str) -> str:
    mp3_path = flac_path[:-5] + ".mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", flac_path, "-b:a", "192k", "-ar", "44100", mp3_path],
        capture_output=True, check=True,
    )
    return mp3_path


def finalize_outputs(output_dir: str) -> dict:
    """Convert each FLAC in output_dir to MP3 (in parallel) and build the SSE 'done' message."""
    from concurrent.futures import ThreadPoolExecutor

    names = os.listdir(output_dir)

    def resolve(fname):
        fpath = os.path.join(output_dir, fname)
        if fname.endswith(".flac"):
            try:
                mp3 = flac_to_mp3(fpath)
                return os.path.basename(mp3), mp3
            except Exception:
                pass  # keep the FLAC
        return fname, fpath

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = dict(pool.map(resolve, names))
    return deliver_outputs(paths)


async def queue_to_sse(q: asyncio.Queue):
    """Drain an asyncio.Queue as SSE lines until a None sentinel, sending a keepalive comment
    after every 8 s of silence so proxies don't drop the connection."""
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=8.0)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue
        if msg is None:
            break
        yield f"data: {json.dumps(msg)}\n\n"


# ── YouTube ingestion (via the self-hosted Cobalt in cobalt_app.py) ─────────

_YT_JUNK = re.compile(
    r'\s*[\(\[]\s*(?:official\s+)?(?:music\s+)?'
    r'(?:video|audio|lyrics?|hq|4k|mv|visualizer|live|clip|explicit)\s*[\)\]]\s*',
    re.IGNORECASE,
)
_YT_RE = re.compile(r'^https?://(www\.)?(youtube\.com/watch|youtu\.be/|youtube\.com/shorts/)')


def clean_yt_title(title: str) -> str:
    """Strip "(Official Video)"-style junk from a YouTube title."""
    return re.sub(r'\s+', ' ', _YT_JUNK.sub('', title)).strip()


def yt_url_ok(url: str) -> bool:
    return bool(_YT_RE.match(url.strip()))


def download_yt(url: str) -> tuple[bytes, str]:
    """Extract audio via a self-hosted Cobalt instance (see cobalt_app.py)."""
    import urllib.error
    import urllib.request

    cobalt_url = os.environ.get("COBALT_URL", "").rstrip("/")
    if not cobalt_url:
        raise RuntimeError(
            "COBALT_URL is not configured — deploy cobalt_app.py and set COBALT_URL / "
            "COBALT_API_KEY (.env locally, see .env.example; the 'cobalt' Modal secret in production)."
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


# ── R2 storage: staged results, the library, manifests ──────────────────────
#
# Configured by R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET_NAME —
# from .env locally, the `cloudflare-r2` Modal secret in production. R2 is optional: without
# it results come back inline and Save is unavailable (see deliver_outputs).

STAGED_PREFIX = "tmp"  # unsaved separation results; expired by the lifecycle rule `make r2-setup` sets
JOB_ID_RE = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")


def get_r2():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(s3={"addressing_style": "path"}),
    )


def r2_configured() -> bool:
    return all(os.environ.get(k) for k in
               ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"))


def content_type_for(fname: str) -> str:
    if fname.endswith(".mp3"):
        return "audio/mpeg"
    return "audio/flac" if fname.endswith(".flac") else "application/octet-stream"


def stage_to_r2(paths: dict[str, str]) -> dict:
    """Upload finished stems to R2 under tmp/{job_id}/ and return the SSE 'done' message
    carrying presigned GET URLs instead of the audio itself."""
    from concurrent.futures import ThreadPoolExecutor

    r2, bucket = get_r2(), os.environ["R2_BUCKET_NAME"]
    job_id = str(uuid.uuid4())

    def put(item):
        name, path = item
        with open(path, "rb") as fh:
            r2.put_object(Bucket=bucket, Key=f"{STAGED_PREFIX}/{job_id}/{name}",
                          Body=fh.read(), ContentType=content_type_for(name))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(put, paths.items()))
    urls = {
        name: r2.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": f"{STAGED_PREFIX}/{job_id}/{name}",
                    # so the ⬇ buttons download (cross-origin <a download> is ignored) — <audio> is unaffected
                    "ResponseContentDisposition": f'attachment; filename="{name}"'},
            ExpiresIn=86400)
        for name in paths
    }
    return {"type": "done", "job_id": job_id, "files": urls}


def deliver_outputs(paths: dict[str, str]) -> dict:
    """Build the SSE 'done' message: staged-in-R2 URLs when R2 is configured (Save is then a
    server-side copy), otherwise — or if staging fails — the audio inline as base64."""
    if r2_configured():
        try:
            return stage_to_r2(paths)
        except Exception:
            import traceback
            traceback.print_exc()
    files = {}
    for name, path in paths.items():
        with open(path, "rb") as fh:
            files[name] = base64.b64encode(fh.read()).decode()
    return {"type": "done", "files": files}


def save_staged(job_id: str, mode: str | None, title: str | None, artist: str | None) -> None:
    """Promote a staged run (tmp/{job_id}/) into the library (results/{job_id}/ + manifest).

    A server-side copy, so the browser never uploads audio. Idempotent: if tmp/ is empty but
    the job is already saved, returns quietly; raises FileNotFoundError if neither exists
    (the staged copy expired before it was saved). tmp/ is deliberately left in place so the
    still-open player/download links keep working."""
    r2, bucket = get_r2(), os.environ["R2_BUCKET_NAME"]
    src = f"{STAGED_PREFIX}/{job_id}/"
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
    write_manifest(job_id, names, mode, title, artist)


def write_manifest(
    job_id: str, files: list[str],
    mode: str | None = None, title: str | None = None, artist: str | None = None,
) -> str:
    """Write manifest.json for a job's stems already in R2."""
    r2 = get_r2()
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


def list_library() -> list[dict]:
    """List every saved job's manifest, newest first."""
    r2 = get_r2()
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


def update_manifest(job_id: str, data: dict) -> dict:
    """Patch title/artist/mode onto an existing job's manifest."""
    r2 = get_r2()
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


def delete_job(job_id: str) -> None:
    """Delete every object under a job's results/ prefix, manifest included."""
    r2 = get_r2()
    bucket = os.environ["R2_BUCKET_NAME"]
    prefix = f"results/{job_id}/"
    r2.head_object(Bucket=bucket, Key=f"{prefix}manifest.json")  # 404s if the job doesn't exist
    paginator = r2.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            r2.delete_objects(Bucket=bucket, Delete={"Objects": keys})


# ── Practice-mode speed changes (/api/speed) ────────────────────────────────

# Where /api/speed may read stems from in R2: a staged (tmp/) or saved (results/) job.
SPEED_SOURCE_RE = re.compile(r"^(tmp|results)/[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
SPEED_FILE_RE = re.compile(r"^(?!\.)[^/\\]+\.(mp3|flac)$")  # a bare filename within that job


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[-500:]}")


def stretch_wav_chunked(wav_in: str, wav_out: str, rate: float, workdir: str, pool,
                        chunk_s: float, pad_s: float = 2.0, xfade_s: float = 0.05) -> None:
    """Rubber Band over time chunks in parallel, rejoined with short crossfades.

    One rubberband process per stem is single-stream (~41s for a 21-minute stem on Modal),
    so this cuts the stem into `chunk_s` pieces and stretches each in its own process (on
    `pool`, shared across stems) with `pad_s` of extra context on both sides, so Rubber
    Band's start-up/tail behaviour falls outside the part that's kept. Each piece's kept
    region is then crossfaded into its neighbours over `xfade_s`, centred on the boundary.
    Boundaries depend only on the input length, so every stem of a track is cut at the
    same places and stays in sync. Streams to `wav_out` rather than holding the whole stem
    in memory (a 21-minute stereo stem is ~450 MB as float32).

    Output isn't bit-identical to a single pass — Rubber Band's phase state depends on
    everything it has processed, so a chunk starting fresh renders the same audio with
    different phases — but on a click-train test every transient landed at the same time
    as in a single pass, and blind listening at the seams couldn't tell them apart."""
    import numpy as np
    import soundfile as sf

    info = sf.info(wav_in)
    n, sr = info.frames, info.samplerate
    chunk, pad = int(chunk_s * sr), int(pad_s * sr)
    bounds = list(range(0, n, chunk)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < chunk // 2:
        del bounds[-2]  # fold a short tail into the previous chunk
    n_chunks = len(bounds) - 1
    if n_chunks == 1:
        run_cmd(["rubberband", "-q", "--tempo", str(rate), wav_in, wav_out])
        return

    def stretch_chunk(k: int) -> tuple[int, str]:
        e0, e1 = max(0, bounds[k] - pad), min(n, bounds[k + 1] + pad)
        src, dst = (os.path.join(workdir, f"chunk{k}_{s}.wav") for s in ("in", "out"))
        sf.write(src, sf.read(wav_in, start=e0, stop=e1, dtype="int16")[0], sr, subtype="PCM_16")
        run_cmd(["rubberband", "-q", "--tempo", str(rate), src, dst])
        os.remove(src)
        return e0, dst

    n_out = round(n / rate)
    xf = max(2, int(xfade_s * sr))
    half = xf // 2
    fade_in = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf, dtype=np.float32)))[:, None]
    fade_out = fade_in[::-1]  # fade_in + fade_out == 1 across the crossfade

    def crossfade(tail, head):
        """Crossfade the two chunks' renderings of the same overlap, level-matched.

        Adjacent chunks render the overlap with different phases, so a plain crossfade
        partly cancels and dips — ~3 dB mid-fade when uncorrelated — while an equal-power
        one overshoots when they're partly correlated (measured ~50% on sustained tones).
        Normalising by the measured correlation `rho` keeps the summed level flat either
        way: this reduces to a plain crossfade at rho=1 and to equal-power at rho=0.
        Measured against a single pass, seams came out within the same level variation as
        mid-chunk audio."""
        rho = float(np.sum(tail * head) / (np.sqrt(np.sum(tail**2) * np.sum(head**2)) + 1e-12))
        rho = min(max(rho, 0.0), 1.0)  # anti-correlated overlaps would need a big, risky boost
        norm = np.sqrt(fade_in**2 + fade_out**2 + 2 * fade_in * fade_out * rho)
        return (tail * fade_out + head * fade_in) / norm

    carry = None  # previous chunk's overlap (unfaded), waiting to be crossfaded with this one
    with sf.SoundFile(wav_out, "w", sr, info.channels, subtype="PCM_16") as out:
        # pool.map yields in order, so assembly proceeds while later chunks still stretch.
        for k, (e0, dst) in enumerate(pool.map(stretch_chunk, range(n_chunks))):
            piece = sf.read(dst, dtype="float32", always_2d=True)[0]
            os.remove(dst)
            # This chunk's span in the output: its own region, plus half a crossfade either side.
            s = 0 if k == 0 else round(bounds[k] / rate) - half
            e = n_out if k == n_chunks - 1 else round(bounds[k + 1] / rate) - half + xf
            local = s - round(e0 / rate)  # where output sample `s` falls within this piece
            seg = piece[local:local + (e - s)]
            if len(seg) < e - s:  # rubberband's output length can be off by a few samples
                seg = np.pad(seg, ((0, e - s - len(seg)), (0, 0)))
            if k > 0:
                seg[:xf] = crossfade(carry, seg[:xf])
            if k < n_chunks - 1:
                carry = seg[-xf:].copy()
                seg = seg[:-xf]
            out.write(np.clip(seg, -1.0, 1.0))


def stretch_stem(src_path: str, rate: float, workdir: str, pool=None,
                 chunk_s: float | None = None) -> str:
    """Pitch-preserving time-stretch via Rubber Band — chosen over WSOLA/phase-vocoder
    options (e.g. the AudioWorklet approach tried earlier) because it's specifically tuned
    for polyphonic full mixes, not just monophonic/speech material.

    Runs the rubberband CLI directly on files — the same `rubberband -q --tempo <rate>`
    call on 16-bit WAV that pyrubberband made — with ffmpeg decoding before and encoding
    FLAC after. Decoding the MP3 in Python via soundfile took ~11s per stem on a 21-minute
    track on Modal; ffmpeg does it in under a second. With `chunk_s` (and a thread `pool`)
    the stretch is split into parallel chunks — see stretch_wav_chunked. Returns the output
    FLAC's path."""
    wav_in = os.path.join(workdir, "in.wav")
    wav_out = os.path.join(workdir, "stretched.wav")
    flac_out = os.path.join(workdir, "out.flac")
    run_cmd(["ffmpeg", "-loglevel", "error", "-y", "-i", src_path, "-c:a", "pcm_s16le", wav_in])
    if rate == 1.0:
        wav_out = wav_in  # like pyrubberband, don't run a no-op stretch
    elif chunk_s:
        stretch_wav_chunked(wav_in, wav_out, rate, workdir, pool, chunk_s)
    else:
        run_cmd(["rubberband", "-q", "--tempo", str(rate), wav_in, wav_out])
    run_cmd(["ffmpeg", "-loglevel", "error", "-y", "-i", wav_out, "-c:a", "flac", flac_out])
    return flac_out
