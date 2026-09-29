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
from typing import BinaryIO

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
    # The practice player's frame index (see build_frame_index), while the stems are at hand.
    frame_index = None
    stems = practice_mp3s(paths)
    if len(stems) >= 3:
        try:
            index = build_frame_index({n: open(paths[n], "rb").read() for n in stems})
            r2.put_object(Bucket=bucket, Key=f"{STAGED_PREFIX}/{job_id}/{FRAME_INDEX}",
                          Body=json.dumps(index).encode(), ContentType="application/json")
            frame_index = frames_for_client(r2, bucket, f"{STAGED_PREFIX}/{job_id}", index)
        except Exception:
            import traceback
            traceback.print_exc()  # the player downloads the stems whole instead
    urls = {
        name: r2.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": f"{STAGED_PREFIX}/{job_id}/{name}",
                    # so the ⬇ buttons download (cross-origin <a download> is ignored) — <audio> is unaffected
                    "ResponseContentDisposition": f'attachment; filename="{name}"'},
            ExpiresIn=86400)
        for name in paths
    }
    return {"type": "done", "job_id": job_id, "files": urls, "frame_index": frame_index}


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
    stems = [n for n in names if n != FRAME_INDEX]  # the frame index isn't a stem
    write_manifest(job_id, stems, mode, title, artist)


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


# ── Frame index: 1.0x playback as byte ranges of the stems ──────────────────
#
# MP3 is a sequence of frames (1152 samples each) that the browser can decode from any
# frame boundary, as long as it's given a couple of frames before it: a frame can borrow
# bits from the frames just before it, and its output overlaps the previous one's. So the
# player fetches each ~30s segment of the original stems as a byte range straight from R2
# (with WARMUP_FRAMES in front, decoded and discarded) instead of the server decoding and
# re-encoding them. Tested in Chrome and Firefox on six 21-minute stems: a slice decodes to
# exactly the predicted length and position, only its first ~1.9 frames differ from
# decoding the whole file, and the rest is bit-identical (Firefox: within 1 LSB).

def _id3v2_end(data: bytes) -> int:
    """Where an MP3's audio starts: after its ID3v2 tag, if it has one (ffmpeg writes one)."""
    if data[:3] != b"ID3" or len(data) < 10:
        return 0
    size = ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14
            | (data[8] & 0x7F) << 7 | data[9] & 0x7F)  # 7 bits per byte ("syncsafe")
    return 10 + size + (10 if data[5] & 0x10 else 0)  # header, tag, footer


def mp3_gapless(head: bytes) -> tuple[int, int] | None:
    """(decoded length, delay) in samples of an MP3, from its first bytes — or None if it has
    no Xing/Info frame with a LAME tag (which LAME and ffmpeg write). The length is what
    ffmpeg decodes: frames × samples-per-frame − encoder delay − padding, the gapless trim
    its mp3 demuxer applies. The delay is how far into the raw decoder output the audio
    starts: the encoder's delay from the LAME tag plus the MP3 decoder's own 529 samples."""
    i = _id3v2_end(head)
    if len(head) < i + 4 or head[i] != 0xFF or head[i + 1] & 0xE0 != 0xE0:
        return None
    mpeg1 = (head[i + 1] >> 3) & 3 == 3
    if (head[i + 1] >> 1) & 3 != 1:  # not Layer III
        return None
    mono = head[i + 3] >> 6 == 3
    x = i + 4 + ((17 if mono else 32) if mpeg1 else (9 if mono else 17))  # after the side info
    if head[x:x + 4] not in (b"Xing", b"Info"):
        return None
    flags = int.from_bytes(head[x + 4:x + 8], "big")
    if not flags & 1:  # no frame count
        return None
    frames = int.from_bytes(head[x + 8:x + 12], "big")
    lame = x + 12 + (4 if flags & 2 else 0) + (100 if flags & 4 else 0) + (4 if flags & 8 else 0)
    if len(head) < lame + 24:
        return None
    delay_padding = int.from_bytes(head[lame + 21:lame + 24], "big")
    delay, padding = delay_padding >> 12, delay_padding & 0xFFF
    return frames * (1152 if mpeg1 else 576) - delay - padding, delay + 529


FRAME_INDEX = "frames.json"  # next to a job's stems — not a stem, so kept out of the manifest
PRACTICE_STEMS = ("vocals", "bass", "drums", "guitar", "piano", "other")  # as in static/index.html
WARMUP_FRAMES = 3            # slices need 2; one to spare
_MP3_BITRATES = {True: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
                 False: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
_MP3_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_frames(data: bytes) -> tuple[int, int, list[int]]:
    """(sample rate, samples per frame, byte offsets of every audio frame plus the end of the
    last) for a Layer III MP3 — skipping any ID3v2 tag and the Xing/Info frame, which holds
    no audio. Raises ValueError if the frame headers don't chain."""
    i = _id3v2_end(data)
    offsets, sr, mpeg1 = [], None, None
    while i + 4 <= len(data):
        b1, b2 = data[i + 1], data[i + 2]
        if data[i] != 0xFF or b1 & 0xE0 != 0xE0 or (b1 >> 1) & 3 != 1 or b2 >> 4 in (0, 15):
            if data[i:i + 3] == b"TAG" and len(data) - i == 128:
                break  # an ID3v1 tag at the end
            raise ValueError(f"no MP3 frame at byte {i}")
        ver = (b1 >> 3) & 3
        mpeg1, sr = ver == 3, _MP3_RATES[ver][(b2 >> 2) & 3]
        offsets.append(i)
        i += (144 if mpeg1 else 72) * _MP3_BITRATES[mpeg1][b2 >> 4] * 1000 // sr + ((b2 >> 1) & 1)
    if not offsets:
        raise ValueError("no MP3 frames")
    first = data[offsets[0]:offsets[1] if len(offsets) > 1 else i]
    if b"Xing" in first[:64] or b"Info" in first[:64]:
        offsets = offsets[1:]
    return sr, 1152 if mpeg1 else 576, offsets + [i]


def build_frame_index(files: dict[str, bytes], segment_s: float = 30.0) -> dict | None:
    """The player's layout for fetching `files` ({name: MP3 bytes}, one job's stems) as
    byte ranges: ~`segment_s` segments on frame boundaries, shared by every stem, and each
    stem's byte range per segment (from WARMUP_FRAMES before the segment to its end). None
    if they aren't all gapless-tagged MP3s with the same layout, in which case the player
    downloads the stems whole instead. Segment j covers audio frames start..end, i.e. track
    samples start×spf − delay up to end×spf − delay (from 0 for the first; to `length` for
    the last)."""
    stems = {}
    for name, data in files.items():
        gapless = mp3_gapless(data[:1 << 16])
        if not gapless:
            return None
        try:
            sr, spf, offsets = mp3_frames(data)
        except ValueError:
            return None
        stems[name] = (sr, spf, gapless, offsets)
    layouts = {(sr, spf, gapless, len(offsets)) for sr, spf, gapless, offsets in stems.values()}
    if len(layouts) != 1:
        return None
    sr, spf, (length, delay), n_offsets = layouts.pop()
    n_frames = n_offsets - 1  # the offsets end with the end of the last frame
    per = round(segment_s * sr / spf)
    starts = list(range(0, n_frames, per))
    if len(starts) > 1 and n_frames - starts[-1] < per // 2:
        starts.pop()  # fold a short tail into the previous segment
    segments = [{"first": max(0, a - WARMUP_FRAMES), "start": a, "end": b}
                for a, b in zip(starts, starts[1:] + [n_frames])]
    return {"sample_rate": sr, "samples_per_frame": spf, "delay": delay, "length": length,
            "segments": segments,
            "bytes": {name: [[offsets[s["first"]], offsets[s["end"]]] for s in segments]
                      for name, (_, _, _, offsets) in stems.items()}}


def practice_mp3s(names) -> list[str]:
    """The files the practice player plays: MP3s named ..._<practice stem>.mp3."""
    return [n for n in names if n.endswith(".mp3") and n[:-4].split("_")[-1] in PRACTICE_STEMS]


def frames_for_client(r2, bucket: str, prefix: str, index: dict | None) -> dict | None:
    """A frame index plus presigned URLs for fetching the ranges. They're distinct from the
    stems' other URLs (a Cache-Control override) so the browser can't answer these CORS range
    requests from a cached non-CORS response of the same URL — the stem cards' <audio>
    elements fetch those, and that made R2 fetches fail CORS."""
    if not index:
        return None
    urls = {name: r2.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": f"{prefix}/{name}",
                                      "ResponseCacheControl": "no-store"}, ExpiresIn=86400)
            for name in index["bytes"]}
    return {**index, "urls": urls}


def ensure_frame_index(prefix: str, names: list[str]) -> dict | None:
    """The job's frame index (see build_frame_index), for the browser: read from R2, or —
    for a result saved before there were frame indexes — built from its stems (one download
    of each) and stored. None if it's not a practice-mode result or the stems don't qualify
    (stored too, as `null`, so they're only checked once)."""
    stems = practice_mp3s(names)
    if len(stems) < 3:
        return None
    r2, bucket = get_r2(), os.environ["R2_BUCKET_NAME"]
    try:
        stored = r2.get_object(Bucket=bucket, Key=f"{prefix}/{FRAME_INDEX}")["Body"].read()
        index = json.loads(stored)
    except r2.exceptions.NoSuchKey:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(stems)) as pool:
            data = dict(zip(stems, pool.map(
                lambda n: r2.get_object(Bucket=bucket, Key=f"{prefix}/{n}")["Body"].read(), stems)))
        index = build_frame_index(data)
        r2.put_object(Bucket=bucket, Key=f"{prefix}/{FRAME_INDEX}",
                      Body=json.dumps(index).encode(), ContentType="application/json")
    return frames_for_client(r2, bucket, prefix, index)


# ── Practice-mode speed changes (/api/speed) ────────────────────────────────

# Where /api/speed may read stems from in R2: a staged (tmp/) or saved (results/) job.
SPEED_SOURCE_RE = re.compile(r"^(tmp|results)/[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
SPEED_FILE_RE = re.compile(r"^(?!\.)[^/\\]+\.(mp3|flac)$")  # a bare filename within that job


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[-500:]}")


def _crossfade_ramps(xf: int):
    """Raised-cosine fade-in/fade-out ramps of `xf` samples (they sum to 1), shaped (xf, 1)."""
    import numpy as np

    fade_in = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf, dtype=np.float32)))[:, None]
    return fade_in, fade_in[::-1]


def _to_int16(x):
    """Float audio → int16, done here rather than by libsndfile, whose float→int16 conversion
    differs by up to 1 LSB between output formats (WAV vs FLAC) — so the samples don't
    depend on which container they're written to."""
    import numpy as np

    return np.rint(np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def _seam_crossfade(tail, head, fade_in, fade_out):
    """Crossfade two adjacent chunks' renderings of the same overlap, level-matched.

    Adjacent chunks render the overlap with different phases, so a plain crossfade
    partly cancels and dips — ~3 dB mid-fade when uncorrelated — while an equal-power
    one overshoots when they're partly correlated (measured ~50% on sustained tones).
    Normalising by the measured correlation `rho` keeps the summed level flat either
    way: this reduces to a plain crossfade at rho=1 and to equal-power at rho=0.
    Measured against a single pass, seams came out within the same level variation as
    mid-chunk audio."""
    import numpy as np

    rho = float(np.sum(tail * head) / (np.sqrt(np.sum(tail**2) * np.sum(head**2)) + 1e-12))
    rho = min(max(rho, 0.0), 1.0)  # anti-correlated overlaps would need a big, risky boost
    norm = np.sqrt(fade_in**2 + fade_out**2 + 2 * fade_in * fade_out * rho)
    return (tail * fade_out + head * fade_in) / norm


_READ_BYTES = 1 << 16  # how much of an input StreamingDecode reads at a time


def mp3_length(head: bytes) -> int | None:
    """The exact decoded length, in samples, of an MP3 from its first bytes — or None if it
    has no Xing/Info frame with a LAME tag (which LAME and ffmpeg write). It's what ffmpeg
    decodes: frames × samples-per-frame − encoder delay − padding, the same gapless trim
    ffmpeg's mp3 demuxer applies. Lets stream_stretch lay out the segments before the stem
    has finished downloading."""
    i = 0
    if head[:3] == b"ID3" and len(head) >= 10:  # skip an ID3v2 tag (ffmpeg writes one)
        size = ((head[6] & 0x7F) << 21 | (head[7] & 0x7F) << 14
                | (head[8] & 0x7F) << 7 | head[9] & 0x7F)  # 7 bits per byte ("syncsafe")
        i = 10 + size + (10 if head[5] & 0x10 else 0)  # header, tag, footer
    if len(head) < i + 4 or head[i] != 0xFF or head[i + 1] & 0xE0 != 0xE0:
        return None
    mpeg1 = (head[i + 1] >> 3) & 3 == 3
    if (head[i + 1] >> 1) & 3 != 1:  # not Layer III
        return None
    mono = head[i + 3] >> 6 == 3
    x = i + 4 + ((17 if mono else 32) if mpeg1 else (9 if mono else 17))  # after the side info
    if head[x:x + 4] not in (b"Xing", b"Info"):
        return None
    flags = int.from_bytes(head[x + 4:x + 8], "big")
    if not flags & 1:  # no frame count
        return None
    frames = int.from_bytes(head[x + 8:x + 12], "big")
    lame = x + 12 + (4 if flags & 2 else 0) + (100 if flags & 4 else 0) + (4 if flags & 8 else 0)
    if len(head) < lame + 24:
        return None
    delay_padding = int.from_bytes(head[lame + 21:lame + 24], "big")
    return frames * (1152 if mpeg1 else 576) - (delay_padding >> 12) - (delay_padding & 0xFFF)


class StreamingDecode:
    """One input decoded by ffmpeg to 16-bit PCM as its bytes arrive (an R2 download, or a
    file), into a raw file at `path` whose start can be read while the rest is still
    downloading and decoding. On Modal's web container, downloading six 21-minute stems in
    full and then decoding them took 5-8s, where the first chunks need only a few seconds
    of each.

    `src` is a readable binary file object, read to the end and closed by a feeder thread.
    Use format(), length() and read(); close() stops the decode."""

    def __init__(self, src: BinaryIO, path: str):
        import tempfile
        import threading

        self._path = path
        self._cond = threading.Condition()
        self._frames = 0            # decoded so far
        self._done = False
        self._error = None
        self._sr = self._channels = None
        self._predicted = None      # exact length from the MP3's header, if it has one
        self._head_read = False
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-map_metadata", "-1",
             "-fflags", "+bitexact", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr)
        threading.Thread(target=self._feed, args=(src,), daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _set(self, **state):
        with self._cond:
            for name, value in state.items():
                setattr(self, name, value)
            self._cond.notify_all()

    def _feed(self, src: BinaryIO) -> None:
        """Pipe `src` into ffmpeg, noting the MP3 header's length on the way."""
        try:
            head = src.read(_READ_BYTES)
            self._set(_predicted=mp3_length(head), _head_read=True)
            while head:
                self._proc.stdin.write(head)
                head = src.read(_READ_BYTES)
        except (BrokenPipeError, ValueError, OSError):
            pass  # ffmpeg exited (an error, reported by _drain) or we were closed
        except Exception as e:  # noqa: BLE001 — e.g. the download failed; surfaced to readers
            self._set(_error=e)
        finally:
            self._set(_head_read=True)
            for f in (self._proc.stdin, src):
                try:
                    f.close()
                except Exception:
                    pass

    def _drain(self) -> None:
        """Copy ffmpeg's output to the raw file, noting how much is decoded."""
        out = self._proc.stdout
        try:
            # ffmpeg's WAV header on a pipe has placeholder sizes; take the format from 'fmt '
            # and treat everything after 'data' as samples.
            if out.read(12)[:4] != b"RIFF":
                raise RuntimeError("ffmpeg produced no audio")
            while True:
                chunk_id, size = out.read(4), int.from_bytes(out.read(4), "little")
                if chunk_id == b"data":
                    break
                body = out.read(size + (size & 1))
                if chunk_id == b"fmt ":
                    self._set(_channels=int.from_bytes(body[2:4], "little"),
                              _sr=int.from_bytes(body[4:8], "little"))
            frame_bytes, total = 2 * self._channels, 0
            with open(self._path, "wb") as fh:
                while chunk := out.read1(_READ_BYTES):
                    fh.write(chunk)
                    fh.flush()
                    total += len(chunk)
                    self._set(_frames=total // frame_bytes)
            if self._proc.wait() != 0:
                self._stderr.seek(0)
                stderr = self._stderr.read().decode(errors="replace").strip()
                raise RuntimeError(f"ffmpeg failed: {stderr[-500:]}")
        except Exception as e:  # noqa: BLE001 — surfaced to readers
            if self._error is None:
                self._set(_error=e)
        finally:
            self._set(_done=True)

    def _wait(self, ready) -> None:
        with self._cond:
            self._cond.wait_for(lambda: ready() or self._done or self._error)
            if self._error:
                raise self._error

    def format(self) -> tuple[int, int]:
        """(sample rate, channels), once ffmpeg has started producing audio."""
        self._wait(lambda: self._sr is not None)
        if self._sr is None:
            raise RuntimeError("ffmpeg produced no audio")
        return self._sr, self._channels

    def length(self) -> int:
        """Total length in samples: from the MP3's header if it has one, else once fully decoded."""
        self._wait(lambda: self._head_read)
        if self._predicted is not None:
            return self._predicted
        self._wait(lambda: False)
        return self._frames

    def read(self, start: int, stop: int) -> "numpy.ndarray":
        """Samples start..stop (frames × channels, int16), waiting until they're decoded;
        zero-padded past the end."""
        import numpy as np

        # Decoding from a pipe, ffmpeg skips the MP3's encoder delay but not its end padding
        # (~a few hundred near-silent samples): the header's length is where the audio ends.
        end = self._predicted if self._predicted is not None else float("inf")
        self._wait(lambda: self._frames >= min(stop, end))
        avail = max(0, min(stop, self._frames, end) - start)
        audio = np.fromfile(self._path, dtype="<i2", count=avail * self._channels,
                            offset=start * self._channels * 2).reshape(-1, self._channels)
        if len(audio) < stop - start:
            audio = np.pad(audio, ((0, stop - start - len(audio)), (0, 0)))
        return audio

    def close(self) -> None:
        try:
            self._proc.kill()
        except Exception:
            pass


STEM_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")  # stem names end up in R2 keys


def stream_stretch(srcs: dict[str, str | BinaryIO], rate: float, workdir: str, pool, deliver,
                   chunk_s: float = 30.0, start_fraction: float = 0.0,
                   pad_s: float = 2.0, xfade_s: float = 0.05):
    """Pitch-preserving time-stretch of every stem in `srcs` ({stem: audio file path, or a
    readable binary stream such as an R2 download}) to `rate`, yielding the output piece by
    piece as it's ready, so the browser can start playing before the rest is done.

    The stretch is Rubber Band — chosen over WSOLA/phase-vocoder options (e.g. the
    AudioWorklet approach tried earlier) because it's tuned for polyphonic full mixes —
    run as the rubberband CLI on 16-bit WAV (the same `rubberband -q --tempo <rate>` call
    pyrubberband made). ffmpeg decodes the MP3s (soundfile took ~11s per 21-minute stem on
    Modal, ffmpeg under a second), as they download (StreamingDecode): with the length from
    the MP3 header (mp3_length) the layout is known up front, and each chunk waits only for
    the audio it needs.

    One rubberband process is single-stream (~41s for a 21-minute stem on Modal), so each
    stem is cut into `chunk_s` pieces stretched in parallel on `pool` (shared across
    stems), each with `pad_s` of extra context on both sides so Rubber Band's start-up and
    tail behaviour fall outside the part that's kept, and neighbouring chunks are joined
    with an `xfade_s` level-matched crossfade (_seam_crossfade) centred on the boundary.
    Boundaries depend only on the input length, so every stem is cut at the same places
    and stays in sync. The result isn't bit-identical to a single pass — Rubber Band's
    phase state depends on everything it has processed, so a chunk starting fresh renders
    the same audio with different phases — but on a click-train test every transient
    landed where a single pass put it, and blind listening at the seams couldn't tell
    them apart.

    The output timeline is split into segments — segment j runs from the crossfade into
    chunk j up to the crossfade into chunk j+1, so it needs only chunks j-1 and j — and
    each (stem, segment) is delivered as a small FLAC the moment its two chunks are done.
    Chunks are submitted to `pool` playhead-first and interleaved across stems — the
    segment at `start_fraction` of the track, then onwards to the end, then back to the
    start — so every stem's segment under the playhead finishes first.

    `deliver(stem, index, flac_bytes)` returns what the client needs to fetch a segment
    ({"url": ...} or {"b64": ...}); it's called from worker threads. Yields:

      {"type": "meta", "sample_rate", "duration", "segments": [start seconds...], "first"}
      {"type": "segment", "stem", "index", **deliver(...)}   — one per stem × segment
      {"type": "done"}

    Raises the first error from any worker."""
    # Every stem starts decoding at once, as it downloads; the decoders are stopped however
    # the stretch ends (finished, failed, or the client went away).
    decoders = {stem: StreamingDecode(open(src, "rb") if isinstance(src, str) else src,
                                      os.path.join(workdir, f"{stem}.pcm"))
                for stem, src in srcs.items()}
    try:
        yield from _stretch_decoded(decoders, rate, workdir, pool, deliver, chunk_s=chunk_s,
                                    start_fraction=start_fraction, pad_s=pad_s, xfade_s=xfade_s)
    finally:
        for decoder in decoders.values():
            decoder.close()


def _stretch_decoded(decoders: dict[str, StreamingDecode], rate: float, workdir: str, pool,
                     deliver, *, chunk_s: float, start_fraction: float, pad_s: float,
                     xfade_s: float):
    """stream_stretch, once its inputs are decoding."""
    import io
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    import numpy as np
    import soundfile as sf

    stems = list(decoders)
    sr = decoders[stems[0]].format()[0]
    # One layout for all stems (they should be the same length; if not, shorter ones are
    # zero-padded), so segment boundaries — and sync — are shared. For MP3s the length is
    # in their header, so this doesn't wait for the decode.
    n = max(decoder.length() for decoder in decoders.values())

    chunk, pad = int(chunk_s * sr), int(pad_s * sr)
    bounds = list(range(0, n, chunk)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < chunk // 2:
        del bounds[-2]  # fold a short tail into the previous chunk
    n_chunks = len(bounds) - 1
    n_out = round(n / rate)
    xf = max(2, int(xfade_s * sr))
    half = xf // 2
    fade_in, fade_out = _crossfade_ramps(xf)
    starts = [0] + [round(bounds[k] / rate) - half for k in range(1, n_chunks)]
    ends = starts[1:] + [n_out]

    playhead = min(max(start_fraction, 0.0), 1.0) * n_out
    first = max(j for j in range(n_chunks) if starts[j] <= playhead)
    yield {"type": "meta", "sample_rate": sr, "duration": n_out / sr,
           "segments": [s / sr for s in starts], "first": first}

    # Segment order: playhead onwards, then backwards; each segment j needs chunks j-1 and j.
    order = ([first - 1] if first > 0 else []) + list(range(first, n_chunks)) + list(range(first - 2, -1, -1))

    events = queue.Queue()  # segment events, or an exception from a worker
    lock = threading.Lock()
    done = {s: {} for s in stems}          # stem -> {chunk k: (e0, stretched path)}
    started = {s: set() for s in stems}    # segments whose assembly has been kicked off
    # Chunk k's file is read by segments k and k+1; delete it once both have used it.
    reads_left = {s: {k: (2 if k + 1 < n_chunks else 1) for k in range(n_chunks)} for s in stems}
    io_pool = ThreadPoolExecutor(max_workers=8)  # assembly/encode/delivery, off the stretch pool

    def reporting_errors(fn):
        def wrapped(*args):
            try:
                fn(*args)
            except BaseException as e:  # noqa: BLE001 — handed to the consumer to raise
                events.put(e)
        return wrapped

    def stretch_chunk(stem: str, k: int) -> tuple[int, str]:
        e0, e1 = max(0, bounds[k] - pad), min(n, bounds[k + 1] + pad)
        src, dst = (os.path.join(workdir, f"{stem}_{k}_{s}.wav") for s in ("in", "out"))
        sf.write(src, decoders[stem].read(e0, e1), sr, subtype="PCM_16")
        if rate == 1.0:
            os.replace(src, dst)
        else:
            run_cmd(["rubberband", "-q", "--tempo", str(rate), src, dst])
            os.remove(src)
        return e0, dst

    def read_out(stem: str, k: int, start: int, frames: int):
        """`frames` samples of chunk k's stretched output, from output-timeline sample `start`."""
        e0, path = done[stem][k]
        local = start - round(e0 / rate)
        seg = sf.read(path, start=local, stop=local + frames, dtype="float32", always_2d=True)[0]
        if len(seg) < frames:  # rubberband's output length can be off by a few samples
            seg = np.pad(seg, ((0, frames - len(seg)), (0, 0)))
        return seg

    def release(stem: str, k: int) -> None:
        with lock:
            reads_left[stem][k] -= 1
            last_read = reads_left[stem][k] == 0
        if last_read:
            os.remove(done[stem][k][1])

    @reporting_errors
    def assemble(stem: str, j: int) -> None:
        s, e = starts[j], ends[j]
        seg = read_out(stem, j, s, e - s)
        if j > 0:
            tail = read_out(stem, j - 1, s, xf)  # chunk j-1's rendering of the same overlap
            seg[:xf] = _seam_crossfade(tail, seg[:xf], fade_in, fade_out)
        buf = io.BytesIO()
        sf.write(buf, _to_int16(seg), sr, format="FLAC", subtype="PCM_16")
        payload = deliver(stem, j, buf.getvalue())
        events.put({"type": "segment", "stem": stem, "index": j, **payload})
        release(stem, j)
        if j > 0:
            release(stem, j - 1)

    @reporting_errors
    def on_chunk_done(stem: str, k: int, fut) -> None:
        e0, path = fut.result()
        with lock:
            done[stem][k] = (e0, path)
            ready = [j for j in (k, k + 1)
                     if j < n_chunks and j not in started[stem]
                     and j in done[stem] and (j == 0 or j - 1 in done[stem])]
            started[stem].update(ready)
        for j in ready:
            io_pool.submit(assemble, stem, j)

    futures = []
    try:
        for k in order:
            for stem in stems:  # interleaved, so all stems' chunks for a segment finish together
                fut = pool.submit(stretch_chunk, stem, k)
                fut.add_done_callback(partial(on_chunk_done, stem, k))
                futures.append(fut)
        for _ in range(n_chunks * len(stems)):
            ev = events.get()
            if isinstance(ev, BaseException):
                raise ev
            yield ev
        yield {"type": "done"}
    finally:
        for fut in futures:  # on error or client disconnect, drop what hasn't started
            fut.cancel()
        io_pool.shutdown(wait=False, cancel_futures=True)


def open_speed_inputs(inputs: dict) -> dict[str, BinaryIO]:
    """Open /api/speed inputs — {stem: ("r2", key) | ("b64", data)} — as readable byte
    streams, concurrently: an R2 object's download is only started, so stream_stretch can
    decode it as it arrives. Raises FileNotFoundError if an R2 stem is gone (e.g. an expired
    tmp/ copy), which the API turns into a 404 so the client can re-upload."""
    import io
    from concurrent.futures import ThreadPoolExecutor

    from botocore.exceptions import ClientError

    needs_r2 = any(kind == "r2" for kind, _ in inputs.values())
    r2, bucket = (get_r2(), os.environ["R2_BUCKET_NAME"]) if needs_r2 else (None, None)

    def one(item):
        stem, (kind, ref) = item
        if kind != "r2":
            return stem, io.BytesIO(base64.b64decode(ref))
        try:
            return stem, r2.get_object(Bucket=bucket, Key=ref)["Body"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                raise FileNotFoundError(ref)
            raise

    with ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        return dict(pool.map(one, inputs.items()))


def run_speed_stream(inputs: dict, rate: float, workdir: str, pool, deliver,
                     chunk_s: float, start_fraction: float):
    """The whole of a streamed speed change: open the inputs (see open_speed_inputs), then
    yield stream_stretch's events. Shared by the in-process path and Modal's stretch_stems."""
    srcs = open_speed_inputs(inputs)
    yield from stream_stretch(srcs, rate, workdir, pool, deliver, chunk_s, start_fraction)


def r2_segment_deliverer(out_prefix: str):
    """A stream_stretch `deliver` that uploads each segment to R2 and returns a presigned URL."""
    r2, bucket = get_r2(), os.environ["R2_BUCKET_NAME"]

    def deliver(stem: str, index: int, data: bytes) -> dict:
        key = f"{out_prefix}/{stem}/{index:04d}.flac"
        r2.put_object(Bucket=bucket, Key=key, Body=data, ContentType="audio/flac")
        return {"url": r2.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400)}

    return deliver


def inline_segment_deliverer(stem: str, index: int, data: bytes) -> dict:
    """A stream_stretch `deliver` for when R2 isn't configured: the segment rides in the event."""
    return {"b64": base64.b64encode(data).decode()}
