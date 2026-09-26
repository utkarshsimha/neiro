"""
Local FastAPI server for Neiro music source separation.

  CPU mode  — runs inference.py in-process, streaming its progress over SSE
  GPU mode  — proxies the Modal `separate` generator the same way

The routes themselves are shared with the Modal deployment (neiro_web.build_app); this file
only supplies what's local: .env config, the in-process CPU separation, and the route to Modal.

Start with:  uv run uvicorn app:app --reload --port 8000
"""

import builtins
import os
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv

from neiro_web import build_app

load_dotenv()

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


def _cpu_separate(audio_bytes: bytes, filename: str, opts: dict, put) -> str:
    """Run inference.py in this process (called in a worker thread by /api/separate).

    inference.py reports progress via print and tqdm, so both are monkey-patched to forward
    {type: log|progress} messages through `put` while it runs. Returns the output directory
    of FLAC stems."""
    inf = _get_inf()
    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, f"input{Path(filename).suffix or '.mp3'}")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir)
    with open(input_path, "wb") as fh:
        fh.write(audio_bytes)
    opts["input_audio"] = [input_path]
    opts["output_folder"] = output_dir

    class PatchedTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self._items = list(iterable) if iterable is not None else []
            self._total = kwargs.get("total", len(self._items))
            self._n = 0

        def __iter__(self):
            for item in self._items:
                yield item
                self._n += 1
                put({"type": "progress", "n": self._n, "total": self._total})

    _orig = builtins.print
    builtins.print = lambda *a, **k: put({"type": "log", "message": " ".join(str(x) for x in a)})
    inf.tqdm = PatchedTqdm
    inf.options = opts
    try:
        inf.predict_with_model(opts)
    finally:
        builtins.print = _orig
    return output_dir


async def _gpu_separate(audio_bytes: bytes, opts: dict):
    # Imported on first use so the local server starts (and CPU mode works) without Modal set up.
    import modal_app as ma
    async for msg in ma.separate.remote_gen.aio(audio_bytes, opts):
        yield msg


app = build_app(static_dir="static", gpu_separate=_gpu_separate, cpu_separate=_cpu_separate)
