"""
Cobalt — self-hosted YouTube/media extraction service, run on Modal.

  Deploy:  uv run modal deploy cobalt_app.py
  URL:     COBALT_URL below (must stay in sync with the deployed URL — see note)

Why this exists: shelling out to yt-dlp from Modal's shared IP range kept
tripping YouTube's bot checks, and browser cookies expire/rotate constantly.
Cobalt (https://github.com/imputnet/cobalt) runs its own extraction pipeline
behind a small REST API; app.py / modal_app.py's `/api/youtube` handlers call
into this instance instead of invoking yt-dlp directly. Set COBALT_URL and
COBALT_API_KEY (see .env.example) so they can reach it.

Built from source rather than the published ghcr.io/imputnet/cobalt image
because that image is Alpine/musl-based and Modal's runtime needs a
glibc-linked container to attach a Python interpreter to. Pinned to the
commit matching the ghcr.io/imputnet/cobalt:11 (v11.7.1) release.

The shared auth key (`COBALT_API_KEY`, sent as `Authorization: Api-Key <key>`)
lives only in the `cobalt` Modal secret, never in source — run
`make deploy-cobalt` (creates/rotates that secret, then deploys this file).
"""

import json
import os
import subprocess

import modal

app = modal.App("neiro-cobalt")

# Cobalt signs its tunnel links with its own external URL, so this has to be
# known at image-build time. Modal's web_server URLs are deterministic:
# https://<workspace>--<app-name>-<function-name>.modal.run — if you rename
# the app or the function below, update this to match before deploying.
COBALT_URL = "https://simha-utkarsh--neiro-cobalt-cobalt.modal.run"

_COBALT_COMMIT = "a636575b09de1fc55d9b8cd98cac88f5f2f16b42"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "corepack enable",
        f"git clone https://github.com/imputnet/cobalt.git /app "
        f"&& cd /app && git checkout {_COBALT_COMMIT}",
        "corepack prepare pnpm@9.6.0 --activate",
        "cd /app && pnpm install --frozen-lockfile",
    )
    .env({
        "API_URL": COBALT_URL,
        "API_PORT": "9000",
        "CORS_WILDCARD": "0",
        "CORS_URL": COBALT_URL,
        "API_AUTH_REQUIRED": "1",
        "API_KEY_URL": "file:///keys.json",
    })
)


@app.function(
    image=image,
    min_containers=0,
    scaledown_window=180,
    secrets=[modal.Secret.from_name("cobalt")],
)
@modal.web_server(9000, startup_timeout=60)
def cobalt():
    api_key = os.environ["COBALT_API_KEY"]
    with open("/keys.json", "w") as f:
        json.dump({api_key: {"name": "neiro-backend", "limit": "unlimited"}}, f)
    subprocess.Popen(["node", "src/cobalt.js"], cwd="/app/api")
