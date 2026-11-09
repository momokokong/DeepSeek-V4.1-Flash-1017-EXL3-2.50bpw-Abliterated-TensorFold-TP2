# DeepSeek-V4.1-Flash-1017 EXL3 2.50bpw Abliterated (`TP=2`) on 2× NVIDIA DGX Spark (TensorFold Native Engine)

[![Hugging Face Delta Pack](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Delta%20Pack%20(Shards%201%20%26%202)-yellow)](https://huggingface.co/momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2)
[![Engine](https://img.shields.io/badge/Engine-TensorFold%20Zig%20TP%3D2-blue)](#)
[![Quantization](https://img.shields.io/badge/EXL3-2.50%20bpw%20(mul1)-green)](#)
[![Uncensored](https://img.shields.io/badge/Abliterated-layers.10..35.attn.wo__b-orange)](#)
[![Hardware](https://img.shields.io/badge/Hardware-2%C3%97%20DGX%20Spark%20(GB10%20128GB)-76b900)](#)

A zero-compromise deployment recipe and hybrid checkpoint toolkit to run **DeepSeek-V4.1-Flash-1017 (Abliterated / Uncensored)** at **2.50 bpw EXL3 (`TP=2`)** across two **NVIDIA DGX Spark (GB10 128 GB)** nodes using the native **TensorFold** Zig/CUDA engine.

> [!IMPORTANT]
> **Where to Download the Model Weights (Hugging Face Delta Pack)**:
> This GitHub repository contains **only the deployment recipe, scripts, and configurations** (no large `.safetensors` files are stored in Git).
> - **Pre-built Delta Pack (`12.74 GiB` — `model-00001` & `model-00002` + `model.safetensors.index.json`)**: Download from [**🤗 Hugging Face: `momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2`**](https://huggingface.co/momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2)
> - **Pure Routed-Expert Shards (`155.25 GiB` — `model-00003` through `model-00015`)**: Automatically linked from [**`vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw`**](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw) via [`scripts/download_and_build.sh`](./scripts/download_and_build.sh) using zero extra disk space (`ln -f`).

---

## Why This Recipe Exists (Solving the 2× DGX Spark Memory Wall)

Running the standard **2.9 bpw** EXL3 checkpoint on two 128 GB DGX Sparks consumes **`~103.7 GiB` per node** (`98.85 GiB` rank-cache), leaving only **`~2.4 – 3.1 GiB` of `MemAvailable`** after long-context prompts. Any co-located service (Agent Gateways, local Embedding servers, ComfyUI, or 4-way concurrent prefill spikes) immediately trips the `memguard` floor or triggers the Linux OOM killer.

Meanwhile, `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw` shrinks the routed experts to **2.50 bpw**, but stores its non-expert layers in `FP8 (F8_E4M3)` inside shards 1 & 2—which the native **TensorFold** engine rejects (`plan.zig` requires EXL3 or BF16 non-experts and F32 Engram `q/k` weights).

### How Our 15-Shard (`167.99 GiB`) Hybrid Checkpoint Works
By inspecting the tensor layouts across upstream checkpoints, we discovered:
1. **95% of the model's footprint (`161.25 GiB`, 384 Routed Experts)** in `vcruz305`'s 2.50bpw pack uses standard `EXL3 mul1` variable-K trellises (`k2 = trellis.shape[2] / 8`), which TensorFold's `exl3_experts.zig` natively supports at full speed.
2. **100% of the abliteration / uncensoring modifications** in `drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated` reside in just **26 middle-layer Attention output projection matrices (`layers.10..35.attn.wo_b`, 104 EXL3 tensors, `651 MiB`)**.
3. The **MTP Drafter (3-layer speculative decoding heads)**, **Shared Experts**, **Vision Tower**, and **Engram tables (`190 GiB`)** remain **100% identical to official stock `deepseek-ai/DeepSeek-V4.1-Flash-1017`**—preserving full speculative acceptance rates and vision capabilities.

| Shard(s) | Size | Tensor Count | Source & Contents |
|---|---|---|---|
| **`model-00001-of-00015.safetensors`**<br/>*(Hosted on Hugging Face Delta Pack)* | `6.74 GiB` | `3,571` | • Base non-expert EXL3 layers from `bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw`<br/>• Grafted `104` abliterated `layers.10..35.attn.wo_b` tensors from `drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`<br/>• TensorFold loader compatibility fixes: `mtp.2.markov_head.{embed,head}.weight`, `layers.{1,14}.engram.{q,k}_weight` widened `BF16 → F32`, and all 43 `ffn.gate.bias_vl` tensors for 1017 vision routing |
| **`model-00002-of-00015.safetensors`**<br/>*(Hosted on Hugging Face Delta Pack)* | `6.00 GiB` | `8,519` | • Pure `.ffn.experts.` EXL3 tensors extracted from `vcruz305` shard 2 (stripping the 218 incompatible FP8 non-expert tensors) |
| **`model-00003..00015-of-00015.safetensors`**<br/>*(Hardlinked from `vcruz305`)* | `155.25 GiB` | `180,409` | • Pure `.ffn.experts.` EXL3 tensors hardlinked directly from `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw` (`model-00003..00015-of-00017.safetensors`, **0 extra disk usage**) |

---

## Verified Benchmarks & Memory Footprint (2× DGX Spark GB10)

Tested with `CONTEXT=250000`, `POOL=524288`, `PARALLEL=4`, RoCE dual-rail (`MTU 1500`, `gid_index=3`) via the official `tools/bench.py` suite:

### 1. Memory Comparison: 2.50bpw Hybrid vs. 2.9bpw

| Metric (Per DGX Spark Node, 121.6 GiB Usable) | **2.50bpw Hybrid (This Recipe)** | 2.9bpw EAQ v1 | Improvement |
|---|---|---|---|
| `rank-cache` Weight File Size | **`84.72 GiB`** (`90,971,455,304` B) | `98.85 GiB` | **−14.13 GiB / node** |
| Resident Model Footprint (`tensorfold` report) | **`89.92 GiB`** | `103.40 GiB` | **−13.48 GiB / node** |
| Cold Boot Time (from prebuilt `rank-cache`) | **`33.7 s`** (weights read in `16.2 s`) | `~45 s` | **Faster startup** |
| Context Window / Shared KV Pool / Concurrency | **`250,000` / `524,288` / `4 streams`** | `150,000` / `250,000` / `3 streams` | **+66% Context, +33% Streams** |
| **MemAvailable After 78K Needle & 4-Way Stress** | **Head: `16.63 GiB` / Worker: `16.53 GiB`** | Head: `3.71 GiB` / Worker: `3.16 GiB` | **+13.3 GiB free RAM for co-located apps!** |

> With **`~16.5 – 17.5 GiB` of free unified memory on both nodes**, you can comfortably co-locate **Agent Gateways**, multiple `llama-server` **Embedding/Reranker** instances, and even **ComfyUI** (`--cache-none` / staged offload) without ever hitting `MEMGUARD_GIB=2.5`.

### 2. Speed & Accuracy Benchmarks (`tools/bench.py`, Cold Cache)

| Suite | Configuration | Result | Notes |
|---|---|---|---|
| **Single-Stream Decode** | Short prompt → 256 tokens (`--suite decode`) | **`64.25 tok/s`** (TTFT `0.15s`) | Up to **`90.51 tok/s`** on code generation (A* pathfinding, 512 tokens) |
| **Multi-Stream Decode** | `--suite multi --streams 1,2,4` | **×1**: `63.17 tok/s`<br/>**×2**: `90.22 tok/s` (`46.65`/stream)<br/>**×4**: **`124.62 tok/s`** (`33.07`/stream) | Up to **`153.58 tok/s`** aggregate on 4-way structured explanations |
| **Cold Prefill** | `--suite prefill --prefill-sizes 8000,32000,78000` (`cached=0`) | **8K**: `2,134.8 tok/s` (TTFT `3.60s`)<br/>**32K**: `2,184.4 tok/s` (TTFT `14.04s`)<br/>**78K**: **`2,152.0 tok/s`** (TTFT `34.73s`) | Zero throughput degradation across repeated 78K cold prefills |
| **Needle in a Haystack** | `32K` & `78K` @ depths `0.25, 0.50, 0.75` | **`6 / 6 (100% CORRECT)`** | Decode speed post-32K prefill: `~89.9 tok/s`; post-78K prefill: `~75.0 tok/s` |
| **Vision Multimodal** | 3-shape color/geometry recognition | **`1.09 s`** end-to-end | 100% accurate with 1017 `gate_bias_vl.safetensors` |
| **DSML Tool Calling** | Multi-tool JSON function calling | **100% valid `tool_calls`** | Clean JSON arguments, zero raw DSML tag leakage |

---

## Quick Start Guide

### Prerequisites
1. Two NVIDIA DGX Spark (GB10) nodes connected via QSFP/RoCE.
2. The base 2× DGX Spark TensorFold repository cloned at `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`.
3. `huggingface_hub` CLI installed (`pip install -U huggingface_hub`).

### Step 1: Download the Hugging Face Delta Pack & Link Shards 3–15
Run [`scripts/download_and_build.sh`](./scripts/download_and_build.sh) on your **Head node (Rank 0)**. It downloads the pre-built Delta Pack (`model-00001` & `model-00002` + index, `12.74 GiB`) from [🤗 Hugging Face (`momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2`)](https://huggingface.co/momokokong/DeepSeek-V4.1-Flash-1017-EXL3-2.50bpw-Abliterated-TensorFold-TP2) and hardlinks shards `00003..00015` from `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw` into `~/.cache/huggingface/hub`:

```bash
chmod +x scripts/download_and_build.sh
./scripts/download_and_build.sh
```

*(Optional: If you prefer to synthesize `model-00001` and `model-00002` from scratch using the upstream repositories instead of downloading the pre-built Delta Pack, use [`scripts/build_e1_hybrid_pack.py`](./scripts/build_e1_hybrid_pack.py).)*

### Step 2: Configure `scripts/local.sh` & Build Rank Caches (CPU-Only)
Copy [`configs/local.sh.example`](./configs/local.sh.example) to `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/scripts/local.sh` (update `WORKER=user@192.168.200.13` to match your Rank 1 node's username and RoCE IP):

```bash
cp configs/local.sh.example ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/scripts/local.sh
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
./scripts/prepare.sh
```
`prepare.sh` runs purely on the CPU (can be executed with zero downtime while another LLM server is still running on the GPUs) and builds the `84.72 GiB` `rank0of2` and `rank1of2` weight caches on each Spark.

### Step 3: Launch via Systemd & Enable Watchdog
Install the user systemd service on the Head node:
```bash
mkdir -p ~/.config/systemd/user
cp configs/dsv41-tensorfold.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now dsv41-tensorfold.service
```

---

## Credits & Acknowledgments
- **Base Model**: [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- **2.50bpw Routed Experts**: [`vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw) (derived from `Blackfrost-AI/DeepSeek-V4.1-Flash-1017-DERISKED-FP8`)
- **2.5bpw Non-Expert EXL3 Base & TensorFold Engine**: [`bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw`](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw) & `bertholomus/TensorFold`
- **Abliterated `wo_b` Overlay**: [`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)
- **2× DGX Spark Orchestration Recipe**: Mia-AiLab
