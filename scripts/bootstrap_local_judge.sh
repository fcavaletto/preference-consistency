#!/usr/bin/env bash
# Install Ollama (if needed), start it, pull the default 3B judge.
set -euo pipefail
MODEL="${MODEL:-llama3.2:3b}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if ! command -v ollama >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Install Homebrew or Ollama from https://ollama.com then re-run." >&2
    exit 1
  fi
  echo "Installing Ollama via Homebrew…"
  HOMEBREW_NO_AUTO_UPDATE=1 brew install ollama
fi

if ! curl -sf "${HOST}/api/tags" >/dev/null; then
  echo "Starting ollama serve…"
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do
    if curl -sf "${HOST}/api/tags" >/dev/null; then
      break
    fi
    sleep 1
  done
fi

if ! curl -sf "${HOST}/api/tags" >/dev/null; then
  echo "Ollama did not become reachable at ${HOST}. Open the Ollama app and retry." >&2
  echo "Last log: /tmp/ollama-serve.log" >&2
  exit 1
fi

echo "Pulling ${MODEL} (about 2 GB, one-time)…"
ollama pull "${MODEL}"
echo "Local judge ready: ${MODEL}"
echo "Keep the daemon up: open the Ollama app, or leave \`ollama serve\` running in a terminal."
