#!/usr/bin/env bash
# download_and_build.sh — Fast Mode B Delta Pack Assembler (0 tensor synthesis required!)
#
# What this script does:
# 1. Downloads the pre-built Delta Pack (~12.74 GiB: model-00001 & model-00002 with Abliterated wo_b + TensorFold fixes,
#    config.json, model.safetensors.index.json, and tokenizer files) from Hugging Face:
#      momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2
# 2. Downloads (or reuses existing cache of) pure expert shards 00003..00015 (~155.25 GiB) from:
#      vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw
# 3. Hardlinks (`ln -f`) all 15 shards into $HF_CACHE/hub/models--local--dsv41-2.50bpw-ablit-tp2
#    using ZERO extra disk space, ready for TensorFold's `./scripts/prepare.sh`.

set -euo pipefail

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
OUT_DIR="${OUT_DIR:-$HOME/models/dsv41-2.50bpw-ablit-tp2}"
REVISION="e1ablit250bpw000000000000000000000000001"

DELTA_REPO="momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2"
VCRUZ_REPO="vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw"
VCRUZ_REV="c0331d62d770d40a06e05a89ee3307e8778a7ff7"

command -v hf >/dev/null || { echo "ERROR: 'hf' CLI not found. Install via: pip install -U huggingface_hub"; exit 1; }

echo "[Step 1/3] Downloading pre-built Delta Pack (shards 1 & 2 + index, ~12.74 GiB) from Hugging Face ($DELTA_REPO)..."
hf download "$DELTA_REPO" --cache-dir "$HF_CACHE/hub"

echo "[Step 2/3] Downloading pure routed-expert shards 00003..00015 (~155.25 GiB, resumable) from $VCRUZ_REPO..."
INC_ARGS=()
for idx in $(seq -f "%05g" 3 15); do
  INC_ARGS+=(--include "model-${idx}-of-00017.safetensors")
done
hf download "$VCRUZ_REPO" --revision "$VCRUZ_REV" --cache-dir "$HF_CACHE/hub" "${INC_ARGS[@]}"

DELTA_SNAP="$(ls -td "$HF_CACHE/hub/models--${DELTA_REPO//\//--}/snapshots/"* | head -1)"
VCRUZ_SNAP="$HF_CACHE/hub/models--${VCRUZ_REPO//\//--}/snapshots/$VCRUZ_REV"

echo "[Step 3/3] Assembling 15-shard checkpoint via zero-space hardlinks..."
mkdir -p "$OUT_DIR"

# Hardlink Delta Pack files (shards 1 & 2, index, config, tokenizers)
for f in \
  model-00001-of-00015.safetensors \
  model-00002-of-00015.safetensors \
  model.safetensors.index.json \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  generation_config.json \
  chat_template.jinja \
  engram_token_map.u32 \
  LICENSE; do
  if [[ -f "$DELTA_SNAP/$f" ]]; then
    ln -f "$(readlink -f "$DELTA_SNAP/$f")" "$OUT_DIR/$f"
  fi
done

# Hardlink shards 3..15 from vcruz305 (00003..00015-of-00017 -> 00003..00015-of-00015)
for idx in $(seq -f "%05g" 3 15); do
  src="$VCRUZ_SNAP/model-${idx}-of-00017.safetensors"
  dst="$OUT_DIR/model-${idx}-of-00015.safetensors"
  [[ -f "$src" ]] || { echo "ERROR: Missing $src"; exit 1; }
  ln -f "$(readlink -f "$src")" "$dst"
done

# Register into Hugging Face cache layout for TensorFold recipe (`MODEL_ID=local/dsv41-2.50bpw-ablit-tp2`)
LOCAL_HUB="$HF_CACHE/hub/models--local--dsv41-2.50bpw-ablit-tp2"
mkdir -p "$LOCAL_HUB/snapshots/$REVISION" "$LOCAL_HUB/refs"
for f in "$OUT_DIR"/*; do
  [[ -f "$f" ]] && ln -f "$f" "$LOCAL_HUB/snapshots/$REVISION/$(basename "$f")"
done
printf '%s' "$REVISION" > "$LOCAL_HUB/refs/main"

echo "================================================================================"
echo "SUCCESS! 15-shard checkpoint linked at: $OUT_DIR"
echo "HF Cache registered at: $LOCAL_HUB/snapshots/$REVISION"
echo "Next step: Copy configs/local.sh.example to scripts/local.sh and run ./scripts/prepare.sh"
echo "================================================================================"
