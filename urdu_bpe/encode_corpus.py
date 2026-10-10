#!/usr/bin/env python3
"""
encode_corpus.py
================
Encode train.txt / val.txt (and optionally test.txt) into flat uint16 arrays.

Example:
    python encode_corpus.py --tokenizer tokenizer.json \
        --input data/train.txt --output data/train.bin
    python encode_corpus.py --tokenizer tokenizer.json \
        --input data/val.txt   --output data/val.bin

The .bin files are raw little-endian-on-x86 uint16 values with no header; read
them with  np.fromfile(path, dtype=np.uint16)  (or np.memmap).
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import os
import sys

import numpy as np

from bpe import BPETokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# =============================================================================
# 2. ENCODING (streaming, chunked by whole lines)
# =============================================================================
def encode_file(tok, in_path, out_path, chunk_bytes=8 * 2**20, verify_first=True):
    """Encode `in_path` into `out_path`; returns (num_tokens, num_bytes).

    Chunks are made of whole lines.  Because the corpus is normalised (no runs
    of whitespace, no trailing spaces), encoding line-chunks gives exactly the
    same ids as encoding the whole file at once.
    """
    total_tokens = 0
    total_bytes = os.path.getsize(in_path)
    bar = tqdm(total=total_bytes, unit="B", unit_scale=True,
               desc=os.path.basename(in_path)) if tqdm else None
    first = True
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        buf, size = [], 0

        def flush():
            nonlocal buf, size, total_tokens, first
            if not buf:
                return
            text = b"".join(buf).decode("utf-8")
            ids = tok.encode(text)
            if ids and max(ids) >= 2**16:
                raise ValueError("token id does not fit in uint16")
            if first and verify_first:
                assert tok.decode(ids) == text, "round-trip failed on first chunk"
            first = False
            np.asarray(ids, dtype=np.uint16).tofile(fout)
            total_tokens += len(ids)
            if bar:
                bar.update(size)
            buf, size = [], 0

        for line in fin:  # binary mode: lines split only at b"\n"
            buf.append(line)
            size += len(line)
            if size >= chunk_bytes:
                flush()
        flush()
    if bar:
        bar.close()
    return total_tokens, total_bytes


# =============================================================================
# 3. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-limit", type=int, default=3_000_000,
                    help="max number of distinct words kept in the encoding cache")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer, cache_limit=args.cache_limit)
    n_tok, n_bytes = encode_file(tok, args.input, args.output)
    print(f"{args.input}: {n_bytes:,} bytes -> {n_tok:,} tokens "
          f"({n_bytes / max(n_tok, 1):.2f} bytes/token) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
