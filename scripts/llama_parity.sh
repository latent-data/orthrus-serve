#!/usr/bin/env bash
# Reference parity endpoint: vanilla Qwen3-8B (Q8_0 GGUF) served via llama.cpp
# on the same machine, same port 8080, same Qwen3 chat template + tool-call wire
# format as orthrus-serve. Stop orthrus-serve first (port clash), then point a
# benchmark at it with --base-url to compare against the non-diffusion baseline.
set -euo pipefail

llama-server \
  -hf Qwen/Qwen3-8B-GGUF:Q8_0 \
  --jinja \
  --chat-template-file <(curl -s https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json | jq -r .chat_template) \
  -ngl 99 \
  --host 0.0.0.0 --port 8080
