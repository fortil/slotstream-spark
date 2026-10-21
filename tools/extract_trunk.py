#!/usr/bin/env python3
"""Extract the dense trunk into one safetensors file (embed_tokens first and 4096-aligned).

Main mode keeps every language_model.* tensor except the routed experts (.mlp.switch_mlp.) and the
n-gram shards (.ngram_embedding.shard_); vision_tower.* is dropped. --mtp reads mtp.safetensors and keeps
every mtp.* tensor except mtp.layers.0.mlp.switch_mlp.*. Tensor bytes are copied verbatim (no norm is
changed; the -1 on centered norms belongs to the oracle only).

Layout: 8-byte header length, JSON header padded with spaces so that (8 + header_len) % 4096 == 0, then
embed_tokens.{weight,scales,biases} (when present), then the rest sorted by name, contiguous.

  extract_trunk.py --model DIR --out FILE [--mtp] [--allow-buffered] [--allow-unaligned]
"""
import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slotstream_spark.model.mlx_ckpt import ALIGN, AlignedWriter, Checkpoint  # noqa: E402

EMBED = ("embed_tokens.weight", "embed_tokens.scales", "embed_tokens.biases")


def keep_main(name):
    return (name.startswith("language_model.") and ".mlp.switch_mlp." not in name
            and ".ngram_embedding.shard_" not in name)


def keep_mtp(name):
    return name.startswith("mtp.") and not name.startswith("mtp.layers.0.mlp.switch_mlp.")


def order(names, mtp):
    """Embed tensors first (main mode), then the rest sorted by name."""
    embed = [n for e in EMBED for n in names if n.endswith("." + e) and n.startswith("language_model.model.")] \
        if not mtp else []
    rest = sorted(n for n in names if n not in set(embed))
    return embed + rest


def plan(ck, names):
    """-> (header dict without metadata, total data bytes)."""
    header, off = {}, 0
    for n in names:
        t = ck.info(n)
        header[n] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
        off += t.nbytes
    return header, off


def encode_header(header, meta):
    h = dict(header)
    h["__metadata__"] = meta
    raw = json.dumps(h, separators=(",", ":")).encode()
    hlen = -(-(8 + len(raw)) // ALIGN) * ALIGN - 8
    return struct.pack("<Q", hlen) + raw + b" " * (hlen - len(raw))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--allow-buffered", action="store_true")
    ap.add_argument("--allow-unaligned", action="store_true", help="skip the embed 4096 alignment check (synthetic tests)")
    a = ap.parse_args(argv)

    model = Path(a.model)
    ck = Checkpoint(model / "mtp.safetensors" if a.mtp else model, allow_buffered=a.allow_buffered)
    keep = keep_mtp if a.mtp else keep_main
    names = order([n for n in ck.tensors() if keep(n)], a.mtp)
    if not names:
        raise SystemExit("no trunk tensors selected")
    header, nbytes = plan(ck, names)
    meta = {"source_index_sha256": ck.index_sha256, "repacker": "tools/extract_trunk.py"}
    head = encode_header(header, meta)
    assert len(head) % ALIGN == 0
    embed_off = {n: len(head) + header[n]["data_offsets"][0] for n in names if not a.mtp and n.endswith(EMBED)}
    bad = [n for n, o in embed_off.items() if o % ALIGN]
    if bad and not a.allow_unaligned:
        raise SystemExit(f"embed tensors not 4096-aligned: {bad}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    t0 = time.time()
    with AlignedWriter(tmp) as wr:
        wr.write(head)
        for n in names:
            wr.write(ck.read_bytes(n))
    if os.stat(tmp).st_size != len(head) + nbytes:
        raise SystemExit("trunk size mismatch")
    os.replace(tmp, out)
    summary = {"path": str(out), "tensors": len(names), "tensor_bytes": nbytes, "file_bytes": len(head) + nbytes,
               "header_bytes": len(head), "embed_offsets": embed_off, "seconds": round(time.time() - t0, 1)}
    s = json.dumps(summary)
    cap = os.environ.get("CAP")
    if cap:
        with open(os.path.join(cap, f"extract_trunk{'_mtp' if a.mtp else ''}.json"), "w") as f:
            f.write(s + "\n")
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
