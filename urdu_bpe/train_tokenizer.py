#!/usr/bin/env python3
"""
train_tokenizer.py
==================
Train the byte-level BPE tokenizer on train.txt (Assignment 1, Part B).

Example:
    python train_tokenizer.py --train data/train.txt --out tokenizer.json

Writes
    tokenizer.json        all merges, in learned order (49,743 for 50k vocab)
    merges_readable.txt   id, parent ids, token bytes (hex) per merge
    training_log.json     wall-clock time + machine description (for the report)
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import json
import os
import platform
import sys
import time

from bpe import BPETokenizer, count_words


# =============================================================================
# 2. MACHINE DESCRIPTION (the assignment asks for it in the report)
# =============================================================================
def total_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2**30
    except Exception:
        return None


def machine_info():
    ram = total_ram_gb()
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(ram, 1) if ram else "unknown",
        "python": sys.version.split()[0],
    }


# =============================================================================
# 3. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="path to train.txt")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--readable", default="merges_readable.txt")
    ap.add_argument("--log", default="training_log.json")
    ap.add_argument("--vocab-size", type=int, default=50_000)
    ap.add_argument("--method", choices=["fast", "reference"], default="fast")
    ap.add_argument("--min-word-freq", type=int, default=1,
                    help="ignore words rarer than this (saves RAM; default 1 = use all)")
    ap.add_argument("--sample-mb", type=float, default=None,
                    help="train on the first N MB only (for quick experiments)")
    ap.add_argument("--allow-short", action="store_true",
                    help="do not fail if fewer merges than requested were possible")
    args = ap.parse_args()

    max_bytes = int(args.sample_mb * 2**20) if args.sample_mb else None

    t0 = time.time()
    counts = count_words(args.train, max_bytes=max_bytes)
    t1 = time.time()
    print(f"counted {sum(counts.values()):,} word occurrences, "
          f"{len(counts):,} distinct, in {t1 - t0:.0f}s")

    tok = BPETokenizer()
    tok.train(counts, vocab_size=args.vocab_size, method=args.method,
              min_word_freq=args.min_word_freq)
    t2 = time.time()

    expected = args.vocab_size - 257
    if len(tok.merges) != expected and not args.allow_short:
        sys.exit(f"ERROR: learned {len(tok.merges)} merges, expected {expected}. "
                 "The corpus is too small for this vocabulary size.")

    tok.save(args.out, args.readable)
    log = {
        "vocab_size": tok.vocab_size,
        "num_merges": len(tok.merges),
        "method": args.method,
        "min_word_freq": args.min_word_freq,
        "train_file": args.train,
        "sample_mb": args.sample_mb,
        "distinct_words": len(counts),
        "word_occurrences": sum(counts.values()),
        "seconds_counting": round(t1 - t0, 1),
        "seconds_training": round(t2 - t1, 1),
        "seconds_total": round(t2 - t0, 1),
        "machine": machine_info(),
    }
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
