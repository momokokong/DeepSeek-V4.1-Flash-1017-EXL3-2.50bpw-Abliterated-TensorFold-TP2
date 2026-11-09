---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4.1-Flash-1017
  - vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw
  - bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw
  - drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated
tags:
  - exl3
  - tensorfold
  - dgx-spark
  - gb10
  - abliterated
  - uncensored
  - deepseek-v4.1
  - multimodal
pipeline_tag: image-text-to-text
---

# DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2 (Delta Pack for 2× DGX Spark)

[![GitHub Recipe & Scripts](https://img.shields.io/badge/GitHub-Deployment%20Recipe%20%26%20Scripts-181717?logo=github)](https://github.com/momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2)

This repository hosts the **12.74 GiB Pre-built Delta Pack (`model-00001-of-00015.safetensors` + `model-00002-of-00015.safetensors` + `model.safetensors.index.json`)** for running **DeepSeek-V4.1-Flash-1017 (Abliterated / Uncensored)** at **2.50 bpw EXL3 (`TP=2`)** across **2× NVIDIA DGX Spark (GB10 128 GB)** nodes using the native **TensorFold** engine.

For the full deployment recipe, automated zero-copy linker script (`download_and_build.sh`), systemd service, and watchdog scripts, visit the companion GitHub repository:
👉 **[https://github.com/momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2](https://github.com/momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2)**

---

## What Is Included in This Delta Pack?

| File | Size | Description |
|---|---|---|
| **`model-00001-of-00015.safetensors`** | `6.74 GiB` (`7,237,136,672` B) | **3,571 non-expert & abliterated tensors**:<br/>• Base 2.5bpw EXL3 non-expert tensors from `bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw`<br/>• Grafted **104 abliterated `layers.10..35.attn.wo_b` EXL3 tensors** (`651 MiB`) from `drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`<br/>• TensorFold compatibility fixes: `mtp.2.markov_head.{embed,head}.weight` (native `BF16`/`F16`), `layers.{1,14}.engram.{q,k}_weight` widened to `F32`, and all 43 `ffn.gate.bias_vl` vision routing biases |
| **`model-00002-of-00015.safetensors`** | `6.00 GiB` (`6,442,515,560` B) | **8,519 pure routed-expert EXL3 tensors** extracted from `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw` shard 2 (with the 218 incompatible FP8 non-expert tensors stripped) |
| **`model.safetensors.index.json`** | `14.7 MB` | Complete 15-shard index (`192,499` keys, `167.99 GiB` total) mapping shards `00001..00002` (from this repo) and shards `00003..00015` (`155.25 GiB`, hardlinked from `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw`) |
| **`config.json` & Tokenizers** | `6.9 MB` | TensorFold-compatible EXL3 configuration, `engram_token_map.u32`, `chat_template.jinja`, and tokenizer files |

---

## Why Use This 2.50bpw Hybrid Pack Over 2.9bpw?

1. **Saves ~14 GiB of Unified Memory Per Node**:
   - Reduces the per-node TensorFold `rank-cache` from `98.85 GiB` (2.9bpw) down to **`84.72 GiB`** (`89.92 GiB` total resident footprint).
2. **`~16.5 – 17.5 GiB` Free `MemAvailable` on Both Sparks**:
   - Even at `CONTEXT=250000`, `POOL=524288`, and `PARALLEL=4`, both nodes maintain **`16.5+ GiB` of free RAM** during 78K long-context stress tests—allowing you to co-locate Agent Gateways, local Embedding/Reranker servers, and ComfyUI without triggering `memguard` or OOM kills.
3. **Full Speed, Vision & Speculative Decoding**:
   - **Cold Boot Time**: **`33.7 s`** (`81.2 GiB` expert cache read in `16.2 s`)
   - **Single-Stream Decode**: **`64.25 tok/s`** median (`tools/bench.py --suite decode`), up to **`90.51 tok/s`** on code generation
   - **4-Way Concurrent Decode**: **`124.62 tok/s`** (`tools/bench.py --suite multi`), up to **`153.58 tok/s`** on 4-way structured generation
   - **Cold Prefill (`cached=0`)**: **`2,134.8 tok/s`** @ 8K | **`2,184.4 tok/s`** @ 32K | **`2,152.0 tok/s`** @ 78K
   - **Needle in a Haystack (`32K` & `78K`, depths `0.25, 0.50, 0.75`)**: **`6 / 6 (100% CORRECT)`**
