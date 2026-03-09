# Neiro

> Separate any song into stems using a weighted ensemble of deep-learning models.
> Runs on Modal (A10G GPU) or locally on CPU.

## What it does

- **2-stem** — vocals + instrumental (fast)
- **4-stem** — vocals, bass, drums, other
- **6-stem** — vocals, bass, drums, guitar, piano, other
- Web UI with inline audio players and seek controls
- Shareable links via Cloudflare R2 (on-demand, user-triggered)

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `brew install uv` or `pip install uv`
- ffmpeg — `brew install ffmpeg` or `apt install ffmpeg`

## Local setup (CPU)

```bash
git clone https://github.com/simha-utkarsh/neiro
cd neiro

cp .env.example .env      # fill in R2 creds if you want share links (optional)
make setup                # installs all Python deps into .venv
make web                  # starts http://localhost:8000
```

## Deploy to Modal (A10G GPU)

```bash
# One-time: create the R2 secret in Modal
modal secret create cloudflare-r2 \
  R2_ACCOUNT_ID=<id> \
  R2_ACCESS_KEY_ID=<key> \
  R2_SECRET_ACCESS_KEY=<secret> \
  R2_BUCKET_NAME=neiro

make deploy               # deploys to Modal; prints the public URL
```

## CLI usage

```bash
# Vocals + instrumental from a single file
make run INPUT=song.mp3 OUTPUT=./output

# Force CPU
make run INPUT=song.mp3 CPU=1

# Direct inference.py usage
uv run python inference.py \
  --input_audio song.mp3 \
  --output_folder ./output \
  --use_BSRoformer --use_Kim_MelRoformer --use_InstVoc \
  --weight_BSRoformer 9.18 --weight_Kim_MelRoformer 10 --weight_InstVoc 3.39 \
  --BigShifts 3 --vocals_only
```

## Makefile targets

```
make help     Show all targets
make setup    Install all deps (uv sync --extra web)
make web      Run local dev server on :8000
make deploy   Deploy to Modal
make run      Separate a file: make run INPUT=song.mp3 [CPU=1]
make clean    Remove .venv, __pycache__, output/
```

## Architecture

```
inference.py          Ensemble separation pipeline (BSRoformer, MelBandRoformer,
                      InstVoc, Demucs) — runs locally or inside Modal GPU container
app.py                Local FastAPI server (CPU + Modal GPU proxy, SSE streaming)
modal_app.py          Modal deployment: GPU container + FastAPI web container
static/index.html     Web UI (Gruvbox dark theme, inline audio players, R2 share)
modules/              Vendored model implementations (see Attribution below)
models/               Model checkpoints — downloaded automatically on first run
```

Model weights are downloaded automatically to `models/` on first use via `torch.hub` and direct URLs. Use `--large_gpu` to keep all models loaded in VRAM for batch processing (requires ~11 GB).

## Attribution

Neiro is built on top of several excellent open-source projects:

- Separation pipeline adapted from [jarredou/MVSEP-MDX23-Colab_v2](https://github.com/jarredou/MVSEP-MDX23-Colab_v2), which builds on [ZFTurbo/MVSEP-MDX23-music-separation-model](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model) — MIT License
- BSRoformer architecture from [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) — MIT License
- MelBandRoformer from [KimberleyJSN/melbandroformer](https://github.com/KimberleyJSN/melbandroformer)
- Demucs models from [facebookresearch/demucs](https://github.com/facebookresearch/demucs) — MIT License
- Pre-trained models by [Anjok07](https://github.com/Anjok07), [aufr33](https://github.com/aufr33), viperx, and [Kimberley Jensen](https://github.com/KimberleyJensen)
