# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MVSep-MDX23 Colab Fork v2.5 — a music source separation tool that separates audio into vocals, instrumental, bass, drums, and other stems using a weighted ensemble of deep learning models.

## Running Inference

```bash
# Install dependencies
pip install -r requirements.txt

# Vocals + instrumental only (faster)
python inference.py \
  --input_audio song.wav \
  --output_folder ./output \
  --vocals_only \
  --use_BSRoformer --use_Kim_MelRoformer --use_InstVoc \
  --weight_BSRoformer 9.18 --weight_Kim_MelRoformer 10 --weight_InstVoc 3.39 \
  --BigShifts 3

# 4-stem separation (vocals/bass/drums/other)
python inference.py \
  --input_audio song.wav \
  --output_folder ./output \
  --use_BSRoformer --use_Kim_MelRoformer --use_InstVoc

# Batch processing a folder (use --large_gpu to keep models in VRAM)
python inference.py \
  --input_audio /path/to/folder/*.wav \
  --output_folder ./output \
  --vocals_only --large_gpu \
  --use_BSRoformer --use_Kim_MelRoformer --use_InstVoc
```

Key CLI flags:
- `--large_gpu` — keep all models loaded in GPU memory (requires ~11GB VRAM), faster for batches
- `--cpu` — force CPU inference (very slow)
- `--BigShifts N` — number of shift-average passes (1=off, 3–11=typical; higher = slower but potentially better)
- `--output_format` — `PCM_16`, `FLOAT`, or `FLAC`
- `--filter_vocals` — highpass filter below 50Hz on vocals stem
- `--input_gain` / `--restore_gain` — adjust input volume and restore after separation

## Architecture

### Separation pipeline (`inference.py`)

`EnsembleDemucsMDXMusicSeparationModel` orchestrates the full pipeline:

1. **Vocals ensemble** — weighted average of up to 6 model outputs:
   - `BSRoformer` / `Kim_MelRoformer` — PyTorch transformer models (`.ckpt` + `.yaml`)
   - `InstVoc` (MDXv3 / TFC_TDF_net) — PyTorch model
   - `VitLarge` (Segm_Models_Net) — PyTorch model
   - `VOCFT` / `InstHQ4` — ONNX models via `onnxruntime`

2. **Instrumental** — computed as `mix - vocals`

3. **4-stem decomposition** (bass/drums/other) — ensemble of 4 Demucs models (`htdemucs_ft`, `htdemucs`, `htdemucs_6s`, `hdemucs_mmi`) applied to the instrumental

### Model loading

Models are downloaded automatically on first use (via `torch.hub` / direct URLs) and cached in `models/`. Loading is lazy (`initialize_model_if_needed`) and models are offloaded from GPU after use unless `--large_gpu` is set.

### Key demixing functions

- `demix_new` / `demix_new_wrapper` — chunked inference with fade windows for BSRoformer/MelRoformer/InstVoc/VitLarge; wrapper adds BigShifts shift-average
- `demix` / `demix_wrapper` — ONNX-based chunked inference for MDX models (VOCFT, InstHQ4)
- `lr_filter` — Linkwitz-Riley crossover filter used for high/low band blending when VOCFT is enabled

### Custom model modules (`modules/`)

- `bs_roformer/bs_roformer.py` — BSRoformer transformer
- `bs_roformer/mel_band_roformer.py` — MelBandRoformer (Kim's model)
- `tfc_tdf_v3.py` — TFC-TDF network (MDXv3 / InstVoc)
- `tfc_tdf_v2.py` — TFC-TDF network (MDXv2 / VOCFT, InstHQ4)
- `segm_models.py` — VitLarge segmentation model wrapper

### Model configs (`models/`)

YAML configs define audio processing parameters (sample rate, FFT settings, chunk sizes) and model architecture. Downloaded automatically alongside checkpoints.

## Colab Usage

Open `MVSep-MDX23-Colab.ipynb` in Google Colab. The notebook installs dependencies, mounts Google Drive, and exposes all separation options as form widgets that map directly to `inference.py` CLI arguments.
