# Contributing to Neiro

Thanks for looking into contributing! This covers local development, how the codebase
is organized, and how to stand up your own Modal deployment (including the optional
share/library and YouTube-import features) so you can test end-to-end before opening a PR.

There's no CI, linter, or test suite configured in this repo — verification is manual
(see [Testing your changes](#testing-your-changes)).

## Getting set up

Follow the "Setup" section in the [README](README.md) to install `uv`, `ffmpeg`, and
the project dependencies, and get `make web` running locally on CPU. That's enough to
work on the separation pipeline, the web UI, or the local FastAPI server without
touching Modal at all.

## How the codebase is organized

See [CLAUDE.md](CLAUDE.md) for the full architecture writeup. The single most
important thing to know before making a change:

> **The API is defined once, in `neiro_web.py`**, and both the local server (`app.py`)
> and the Modal deployment (`modal_app.py`) build their app from it. Change routes there,
> and put shared non-route logic in `neiro_common.py`. `app.py` and `modal_app.py` only
> hold what genuinely differs between the two (local CPU inference, Modal functions and
> images) — if you find yourself adding the same code to both, it belongs in one of the
> shared modules instead.

Other things worth knowing before you dig in:

- `inference.py` and `modules/` are a vendored fork of the separation pipeline — treat
  them as stable library code unless you're specifically working on the models
  themselves.
- `static/index.html` is a single-file vanilla-JS SPA with no build step or bundler.
  There's no `npm install` — if you need a JS library, it has to be loaded via a
  `<script>` tag or dynamic `import()` from a CDN (see how `cobalt_app.py`'s sibling
  YouTube-import code and the (now-removed) SoundTouch experiment did this) since
  there's no bundler to vendor a package through.
- `cobalt_app.py` is a separate, independent Modal app (self-hosted
  [Cobalt](https://github.com/imputnet/cobalt)) that `app.py`/`modal_app.py` call into
  for YouTube URL imports. You only need to deploy it if you're working on that
  feature — everything else works without it.
- Code style: no comments unless something is genuinely non-obvious (a workaround, a
  hidden constraint), no speculative abstractions, no backwards-compatibility shims for
  code you're removing — just delete it. Keep changes scoped to what you're actually
  asked to do.

## Testing your changes

There's no automated test suite, so before opening a PR:

- **Backend/API changes**: run `make web` and exercise the affected endpoint(s)
  directly (curl or the UI) against the local CPU server.
- **Frontend changes**: open `http://localhost:8000` and click through the actual
  flow — upload a short file, run a separation, and use the feature you changed. For
  anything audio-related (the practice mixer, karaoke player), actually listen to it;
  passing a syntax check doesn't mean the audio graph is wired correctly.
- **Modal-specific changes** (GPU path, `modal_app.py`, `cobalt_app.py`): you'll need
  your own Modal deployment to test against — see below. `make deploy` only takes a
  few seconds once your Modal account is set up, so iterating against a real
  deployment is fast.
- If you change **both** `app.py` and `modal_app.py` for the same feature, test both
  paths (local CPU and Modal GPU) — they're separate code, and it's easy to fix one
  and forget the other.

## Deploying your own instance to Modal

You don't need this for most contributions (UI tweaks, pipeline changes, docs) — only
if you're testing something that requires the GPU path, the share/library feature, or
YouTube import.

The default path below is your own free Modal account — Modal's free tier includes
$30/month in compute credits, enough for a lot of testing. If you'd rather not set
that up, or want to test against the exact same deployment/secrets the maintainer
runs in production, **message [@utkarshsimha](https://github.com/utkarshsimha) to be
added as a member of his Modal workspace** instead of following the steps below.
Note that as a workspace member you'd be deploying into the same live `neiro` app and
sharing its billing and secrets, so this is meant for people actively collaborating,
not a default ask for every contribution.

### 1. Modal account

1. [Create a Modal account](https://modal.com/signup).
2. `uv run modal token new` to authenticate the CLI.
3. `uv run modal deploy modal_app.py` — deploys the main app under your own workspace,
   printed as `https://<your-workspace>--neiro-fastapi-app.modal.run`.

Skip `make deploy-legacy` / the `song-lab` alias — that only exists to keep the
original maintainer's previously-shared links working, and doesn't apply to your own
deployment.

### 2. R2 storage (optional — share links & library)

Without this, separation still works but "Save to library" and shareable links are
silently unavailable. See the "R2 storage setup" section in the [README](README.md)
for creating a Cloudflare R2 bucket and API token, then either:

- **Local**: `cp .env.example .env` and fill in `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` /
  `R2_SECRET_ACCESS_KEY` / `R2_BUCKET_NAME`.
- **Modal**: `modal secret create cloudflare-r2 R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET_NAME=...`

### 3. Cobalt (optional — YouTube import)

The YouTube-URL-to-audio feature runs through a self-hosted [Cobalt](https://github.com/imputnet/cobalt)
instance, deployed as its own Modal app (`cobalt_app.py`). One thing you **must**
change before deploying it yourself:

`cobalt_app.py` hardcodes `COBALT_URL` to the maintainer's own Modal workspace
subdomain:

```python
COBALT_URL = "https://simha-utkarsh--neiro-cobalt-cobalt.modal.run"
```

Modal's web endpoint URLs are deterministic (`https://<workspace>--<app-name>-<function-name>.modal.run`),
so update this to match **your** workspace name before deploying, or Cobalt will sign
its tunnel links with the wrong URL and downloads will fail. Then:

```bash
make deploy-cobalt   # deploys cobalt_app.py and creates/rotates the "cobalt" Modal secret
```

The command prints the `COBALT_URL` / `COBALT_API_KEY` to also add to your local
`.env` if you want YouTube import working with `make web` too.

Note: we tried adding a proof-of-origin token sidecar (`imputnet/yt-session-generator`)
to make Cobalt more reliable, but it hits the same Modal-IP-based blocking it's meant
to work around and just times out — see the comment in `cobalt_app.py`. Some videos
will still fail to import; that's a known, currently-unsolved limitation, not a bug in
your setup.

## Submitting changes

- Keep commits focused — one logical change per commit, with a message explaining
  *why*, not just what (see recent commit history for the style this project uses).
- If your change touches `app.py` or `modal_app.py`, mention in the PR description
  whether you tested the local (CPU) path, the Modal (GPU) path, or both.
- Check [fixes.md](fixes.md) for known open issues before starting — you might be
  able to pick one up, or your change might interact with one of them.
- No formal issue/PR template right now — a clear description of what changed and why
  is enough.
