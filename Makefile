.PHONY: setup sync run web deploy clean help yt-cookies

# Default target
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Install all deps (creates .venv, installs web extras)
	uv sync --extra web

sync: ## Re-sync without updating locked versions (CI-safe)
	uv sync --frozen --extra web

# Usage: make run INPUT=song.wav OUTPUT=./output [CPU=1]
INPUT  ?= input.wav
OUTPUT ?= ./output
CPU    ?= 0

_CPU_FLAG = $(if $(filter 1,$(CPU)),--cpu,)

run: ## Separate a file: make run INPUT=song.mp3 [OUTPUT=./output] [CPU=1]
	uv run python inference.py \
		--input_audio "$(INPUT)" \
		--output_folder "$(OUTPUT)" \
		--use_BSRoformer6Stem \
		--BigShifts 3 \
		--output_format FLAC \
		$(_CPU_FLAG)

web: ## Run local dev server on :8000
	uv sync --extra web
	uv run uvicorn app:app --reload --port 8000

deploy: ## Deploy to Modal (GPU)
	uv run modal deploy modal_app.py

# Usage: make yt-cookies FILE=~/.neiro-yt-cookies.txt
FILE ?= $(HOME)/.neiro-yt-cookies.txt

yt-cookies: ## Push a freshly-exported cookies.txt to the Modal yt-cookies secret
	@test -s "$(FILE)" || { echo "No cookies file at $(FILE) — export one first (see app.py's _YT_COOKIES_HELP)"; exit 1; }
	uv run modal secret create yt-cookies YT_COOKIES_CONTENT="$$(cat $(FILE))" --force

clean: ## Remove .venv, __pycache__, output/
	rm -rf .venv __pycache__ output/ out/
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
