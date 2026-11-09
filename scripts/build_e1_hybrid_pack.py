#!/usr/bin/env python3
"""
build_e1_hybrid_pack.py — Assemble the 2.50bpw Abliterated EXL3 TP=2 Hybrid Checkpoint
for DeepSeek-V4.1-Flash-1017 on 2x NVIDIA DGX Spark (TensorFold Native Engine).

What this script does (Pure Python stdlib, 0 GPU required):
1. Synthesizes `model-00001-of-00015.safetensors` (6.74 GiB, 3,571 tensors):
   - Base non-expert EXL3 tensors from `bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw` (`model-00001-of-00037.safetensors`)
   - Grafts 104 abliterated attention output projection tensors (`layers.10..35.attn.wo_b.{trellis,suh,svh,mul1}`)
     from `drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated` (`exl3_wo_b_l10_35_mia29.safetensors`)
   - Applies TensorFold loader compatibility fixes in-place:
     * Renames `mtp.2.markov_head.markov_w1.weight` -> `mtp.2.markov_head.embed.weight` (BF16)
     * Renames `mtp.2.markov_head.markov_w2.weight` -> `mtp.2.markov_head.head.weight` (F16)
     * Widens `layers.{1,14}.engram.{q_weight,k_weight}` from BF16 -> F32
     * Injects `mtp.{0,1,2}.ffn.gate.bias_vl` (BF16) from `vcruz305`'s shard 17 so all 43 vision routing biases exist
2. Extracts pure routed-expert EXL3 tensors (`8,519` `.ffn.experts.` tensors, 6.00 GiB) from
   `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw` (`model-00002-of-00017.safetensors`) into
   `model-00002-of-00015.safetensors`, stripping the 218 FP8 non-expert tensors that TensorFold rejects.
3. Hardlinks (`os.link`, 0 extra disk space) `model-00003..00015-of-00017.safetensors` from `vcruz305`
   as `model-00003..00015-of-00015.safetensors` (155.25 GiB, 180,409 pure expert tensors).
4. Generates `model.safetensors.index.json` (192,499 keys) and copies `config.json` & tokenizer files.
5. Optionally links the assembled checkpoint into `$HF_CACHE/hub/models--local--dsv41-2.50bpw-ablit-tp2`
   for seamless use with the 2x DGX Spark TensorFold recipe (`scripts/prepare.sh`).
"""

import argparse
import array
import json
import os
import re
import shutil
import struct
import sys

MARKOV_RE = re.compile(r"^(mtp\.\d+)\.markov_head\.markov_w([12])\.weight$")
ENGRAM_QK_RE = re.compile(r"^layers\.\d+\.engram\.[qk]_weight$")


def read_safetensors_header(path: str):
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise RuntimeError(f"{path}: truncated safetensors header")
        n = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", {"format": "pt"})
    return 8 + n, header, metadata


def widen_bf16_to_f32(data: bytes) -> bytes:
    h = array.array("H")
    h.frombytes(data)
    if sys.byteorder != "little":
        h.byteswap()
    w = array.array("I", (x << 16 for x in h))
    if sys.byteorder != "little":
        w.byteswap()
    return w.tobytes()


def write_safetensors(out_path: str, records: list):
    """
    records: list of (name, dtype, shape, byte_length, writer_callable(out_file))
    Streams tensor payloads directly to disk without holding the full 6+ GiB shard in RAM.
    """
    header = {}
    offset = 0
    for name, dtype, shape, length, _ in records:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + length],
        }
        offset += length
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hb)) % 8
    if pad:
        hb += b" " * pad

    tmp_path = out_path + ".partial"
    with open(tmp_path, "wb") as out_f:
        out_f.write(struct.pack("<Q", len(hb)))
        out_f.write(hb)
        for _, _, _, length, writer_fn in records:
            written = writer_fn(out_f)
            if written != length:
                raise RuntimeError(f"Short write on {out_path}: expected {length}, wrote {written}")
    os.replace(tmp_path, out_path)
    return os.path.getsize(out_path)


def copy_slice_writer(src_path: str, file_offset: int, length: int):
    def _write(out_f):
        remaining = length
        written = 0
        with open(src_path, "rb") as in_f:
            in_f.seek(file_offset)
            while remaining > 0:
                chunk = in_f.read(min(remaining, 16 * 1024 * 1024))
                if not chunk:
                    break
                out_f.write(chunk)
                written += len(chunk)
                remaining -= len(chunk)
        return written
    return _write


def bytes_writer(payload: bytes):
    def _write(out_f):
        out_f.write(payload)
        return len(payload)
    return _write


def build_shard1(bert_shard1: str, vcruz_shard17: str, ablit_sidecar: str, out_shard1: str):
    print(f"[1/4] Synthesizing {os.path.basename(out_shard1)} ...")
    b_base, b_hdr, _ = read_safetensors_header(bert_shard1)
    a_base, a_hdr, _ = read_safetensors_header(ablit_sidecar)
    v17_base, v17_hdr, _ = read_safetensors_header(vcruz_shard17)

    records = []
    indexed_keys = []
    replaced_ablit = 0

    for key in sorted(b_hdr.keys()):
        if ".ffn.experts." in key:
            continue
        info = b_hdr[key]
        dtype = info["dtype"]
        shape = info["shape"]
        a, b = info["data_offsets"]
        length = b - a

        # 1. Abliterated wo_b graft (layers 10..35)
        if key in a_hdr:
            ainfo = a_hdr[key]
            aa, ab = ainfo["data_offsets"]
            alength = ab - aa
            records.append((
                key,
                ainfo["dtype"],
                ainfo["shape"],
                alength,
                copy_slice_writer(ablit_sidecar, a_base + aa, alength),
            ))
            indexed_keys.append(key)
            replaced_ablit += 1
            continue

        # 2. Markov head rename for TensorFold loader
        m_markov = MARKOV_RE.match(key)
        if m_markov:
            new_sub = "embed" if m_markov.group(2) == "1" else "head"
            new_key = f"{m_markov.group(1)}.markov_head.{new_sub}.weight"
            records.append((
                new_key,
                dtype,
                shape,
                length,
                copy_slice_writer(bert_shard1, b_base + a, length),
            ))
            indexed_keys.append(new_key)
            continue

        # 3. Engram q_weight / k_weight BF16 -> F32 widening
        if ENGRAM_QK_RE.match(key) and dtype == "BF16":
            with open(bert_shard1, "rb") as f:
                f.seek(b_base + a)
                raw = f.read(length)
            f32_bytes = widen_bf16_to_f32(raw)
            records.append((
                key,
                "F32",
                shape,
                len(f32_bytes),
                bytes_writer(f32_bytes),
            ))
            indexed_keys.append(key)
            continue

        # Standard non-expert tensor
        records.append((
            key,
            dtype,
            shape,
            length,
            copy_slice_writer(bert_shard1, b_base + a, length),
        ))
        indexed_keys.append(key)

    # 4. Append mtp.{0,1,2}.ffn.gate.bias_vl from vcruz305 shard 17 if not already present
    existing = set(indexed_keys)
    for step in (0, 1, 2):
        bkey = f"mtp.{step}.ffn.gate.bias_vl"
        if bkey not in existing and bkey in v17_hdr:
            vinfo = v17_hdr[bkey]
            va, vb = vinfo["data_offsets"]
            vlen = vb - va
            records.append((
                bkey,
                vinfo["dtype"],
                vinfo["shape"],
                vlen,
                copy_slice_writer(vcruz_shard17, v17_base + va, vlen),
            ))
            indexed_keys.append(bkey)

    if replaced_ablit != 104:
        raise RuntimeError(f"Expected 104 abliterated wo_b tensors, replaced {replaced_ablit}")

    sz = write_safetensors(out_shard1, records)
    print(f"      -> Wrote {len(indexed_keys)} tensors ({sz / (1024**3):.2f} GiB), 104 abliterated wo_b tensors grafted.")
    return indexed_keys, sz


def build_shard2(vcruz_shard2: str, out_shard2: str):
    print(f"[2/4] Extracting pure routed experts into {os.path.basename(out_shard2)} ...")
    v2_base, v2_hdr, _ = read_safetensors_header(vcruz_shard2)
    records = []
    indexed_keys = []
    skipped_non_expert = 0

    for key in sorted(v2_hdr.keys()):
        if ".ffn.experts." not in key:
            skipped_non_expert += 1
            continue
        info = v2_hdr[key]
        a, b = info["data_offsets"]
        length = b - a
        records.append((
            key,
            info["dtype"],
            info["shape"],
            length,
            copy_slice_writer(vcruz_shard2, v2_base + a, length),
        ))
        indexed_keys.append(key)

    sz = write_safetensors(out_shard2, records)
    print(f"      -> Wrote {len(indexed_keys)} expert tensors ({sz / (1024**3):.2f} GiB), stripped {skipped_non_expert} FP8 non-expert tensors.")
    return indexed_keys, sz


def link_or_copy(src: str, dst: str):
    if os.path.exists(dst):
        os.remove(dst)
    src_real = os.path.realpath(src)
    try:
        os.link(src_real, dst)
    except OSError:
        shutil.copy2(src_real, dst)


def main():
    ap = argparse.ArgumentParser(description="Build 2.50bpw Abliterated EXL3 TP=2 Hybrid Pack for TensorFold")
    ap.add_argument("--vcruz-dir", required=True, help="Path to vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw snapshot")
    ap.add_argument("--bert-dir", required=True, help="Path to bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw snapshot (needs model-00001-of-00037.safetensors & config.json)")
    ap.add_argument("--ablit-file", required=True, help="Path to drowzeys exl3_wo_b_l10_35_mia29.safetensors")
    ap.add_argument("--out-dir", required=True, help="Output directory for the assembled 15-shard checkpoint")
    ap.add_argument("--register-hf-cache", default="", help="Optional path to ~/.cache/huggingface/hub to hardlink as local/dsv41-2.50bpw-ablit-tp2")
    ap.add_argument("--revision", default="e1ablit250bpw000000000000000000000000001", help="Snapshot revision hash when registering into HF cache")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bert_s1 = os.path.join(args.bert_dir, "model-00001-of-00037.safetensors")
    vcruz_s2 = os.path.join(args.vcruz_dir, "model-00002-of-00017.safetensors")
    vcruz_s17 = os.path.join(args.vcruz_dir, "model-00017-of-00017.safetensors")

    for req in (bert_s1, vcruz_s2, vcruz_s17, args.ablit_file):
        if not os.path.isfile(req):
            sys.exit(f"ERROR: Required input file not found: {req}")

    weight_map = {}
    total_size = 0

    # Shard 1
    s1_name = "model-00001-of-00015.safetensors"
    s1_keys, s1_sz = build_shard1(bert_s1, vcruz_s17, args.ablit_file, os.path.join(args.out_dir, s1_name))
    for k in s1_keys:
        weight_map[k] = s1_name
    total_size += s1_sz

    # Shard 2
    s2_name = "model-00002-of-00015.safetensors"
    s2_keys, s2_sz = build_shard2(vcruz_s2, os.path.join(args.out_dir, s2_name))
    for k in s2_keys:
        weight_map[k] = s2_name
    total_size += s2_sz

    # Shards 3..15 (Hardlinked from vcruz305 shards 3..15)
    print("[3/4] Hardlinking pure expert shards 00003..00015 (0 extra disk space) ...")
    for idx in range(3, 16):
        src_name = f"model-{idx:05d}-of-00017.safetensors"
        dst_name = f"model-{idx:05d}-of-00015.safetensors"
        src_path = os.path.join(args.vcruz_dir, src_name)
        dst_path = os.path.join(args.out_dir, dst_name)
        if not os.path.isfile(src_path):
            sys.exit(f"ERROR: Missing expert shard {src_path}")
        link_or_copy(src_path, dst_path)
        _, hdr, _ = read_safetensors_header(dst_path)
        for k in hdr.keys():
            weight_map[k] = dst_name
        total_size += os.path.getsize(dst_path)

    # Metadata & Index
    print("[4/4] Writing model.safetensors.index.json and config/tokenizer files ...")
    index_obj = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index_obj, f, indent=2)

    shutil.copy2(os.path.join(args.bert_dir, "config.json"), os.path.join(args.out_dir, "config.json"))
    for fname in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "engram_token_map.u32", "LICENSE"):
        for src_root in (args.vcruz_dir, args.bert_dir):
            cand = os.path.join(src_root, fname)
            if os.path.isfile(cand):
                link_or_copy(cand, os.path.join(args.out_dir, fname))
                break

    print(f"SUCCESS: Assembled 15 shards ({len(weight_map):,} keys, {total_size / (1024**3):.2f} GiB) in {args.out_dir}")

    if args.register_hf_cache:
        hub_dir = os.path.expanduser(args.register_hf_cache)
        snap_dir = os.path.join(hub_dir, "models--local--dsv41-2.50bpw-ablit-tp2", "snapshots", args.revision)
        refs_dir = os.path.join(hub_dir, "models--local--dsv41-2.50bpw-ablit-tp2", "refs")
        os.makedirs(snap_dir, exist_ok=True)
        os.makedirs(refs_dir, exist_ok=True)
        for fn in os.listdir(args.out_dir):
            s = os.path.join(args.out_dir, fn)
            d = os.path.join(snap_dir, fn)
            if os.path.isfile(s):
                link_or_copy(s, d)
        with open(os.path.join(refs_dir, "main"), "w", encoding="utf-8") as f:
            f.write(args.revision)
        print(f"REGISTERED in HF Cache: {snap_dir}")


if __name__ == "__main__":
    main()
