#!/usr/bin/env bash
# Downloads the model files Ember uses from Hugging Face (Apache-2.0 licensed).
set -e
cd "$(dirname "$0")"
get() { # repo file
  mkdir -p "$1"
  [ -s "$1/$2" ] && return
  curl -fL --retry 5 -C - -o "$1/$2.part" "https://huggingface.co/Qwen/$1/resolve/main/$2"
  mv "$1/$2.part" "$1/$2"
}
for f in config.json generation_config.json tokenizer.json tokenizer_config.json LICENSE model.safetensors; do get Qwen3-0.6B $f; done
for f in config.json generation_config.json tokenizer.json tokenizer_config.json LICENSE model.safetensors.index.json \
         model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do get Qwen3-1.7B $f; done
echo "models ready"
