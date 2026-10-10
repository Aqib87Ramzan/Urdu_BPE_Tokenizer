#!/usr/bin/env python3
"""
verify_equivalence.py
=====================
Check that the FAST trainer learns exactly the same merges as the REFERENCE
trainer on a ~1 MB sample of train.txt (assignment requirement for Part B).

Example:
    python verify_equivalence.py --train data/train.txt --sample-mb 1 --merges 1000
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import sys
import time

from bpe import _to_bytes, count_words, train_fast, train_reference


# =============================================================================
# 2. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--sample-mb", type=float, default=1.0)
    ap.add_argument("--merges", type=int, default=1000)
    args = ap.parse_args()

    counts = count_words(args.train, max_bytes=int(args.sample_mb * 2**20), progress=False)
    items = sorted((_to_bytes(w), c) for w, c in counts.items())
    print(f"sample: {len(items):,} distinct words; comparing {args.merges} merges")

    t = time.time()
    ref = train_reference(items, args.merges)
    t_ref = time.time() - t
    t = time.time()
    fast = train_fast(items, args.merges, verbose=False)
    t_fast = time.time() - t
    print(f"reference {t_ref:.1f}s   fast {t_fast:.1f}s")

    if ref == fast:
        print(f"OK: identical {len(ref)} merges")
        return 0
    first = next(i for i, (a, b) in enumerate(zip(ref, fast)) if a != b) \
        if len(ref) == len(fast) else min(len(ref), len(fast))
    print(f"MISMATCH at merge index {first}: reference={ref[first:first+1]} fast={fast[first:first+1]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
