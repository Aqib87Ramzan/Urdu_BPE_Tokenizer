#!/usr/bin/env python3
"""
prepare_corpus.py
=================
Cleaning pipeline for Part A:  raw sources  ->  train.txt / val.txt / test.txt

Steps (each one reports how many bytes it removed):
  1. extract plain text   (strip HTML / wiki markup / URLs / e-mails / menus)
  2. normalise Unicode    (normalize.normalize)
  3. language filter      (drop lines that are mostly not Urdu)
  4a. exact-duplicate documents removed
  4b. boilerplate lines that repeat across many documents removed
  5.  documents that became too short removed
  6. split BY DOCUMENT  98 / 1 / 1 with a fixed seed; "<|endoftext|>" after each
     document; train documents are shuffled.

Usage:
    python prepare_corpus.py --sources sources.json --out-dir data

sources.json is a list of objects, e.g.
  {"name": "urdu_wikipedia", "path": "raw/urdu_wikipedia.jsonl", "format": "jsonl",
   "field": "text", "genre": "encyclopedia", "url": "https://ur.wikipedia.org",
   "licence": "CC BY-SA 4.0", "collection_date": "2026-10-01",
   "own_collection": true, "evidence": "raw/urdu_wikipedia.evidence.csv"}
Supported formats: jsonl | csv | txt-blank (documents separated by blank lines)
                   | txt-lines (one sentence per line; grouped into pseudo-documents)
Files may be .gz / .xz / .bz2 compressed.
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import bz2
import csv
import gzip
import hashlib
import html
import json
import lzma
import os
import random
import re
import sys
import time
from collections import Counter

from normalize import normalize

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

EOT = "<|endoftext|>"

# =============================================================================
# 2. STEP 1 - MARKUP / BOILERPLATE STRIPPING
# =============================================================================
_TAG_RE = re.compile(r"<[^>\n]{1,500}>")
_SCRIPT_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")
_URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_WIKI_HEADING_RE = re.compile(r"(?m)^\s*=+[^=\n]*=+\s*$")
_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}|\{\|.*?\|\}", re.S)
_WIKI_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_WIKI_EXT_RE = re.compile(r"\[https?://\S+(?:\s+([^\]]*))?\]")
_WIKI_QUOTE_RE = re.compile(r"'{2,}")
_WIKI_CAT_RE = re.compile(r"(?mi)^\s*\[\[(?:category|زمرہ|file|فائل|image|تصویر):[^\n]*$")
# Urdu share/menu strings: a line that contains one AND is short is removed
_MENU_PHRASES = ("شیئر کریں", "مزید پڑھیں", "فالو کریں", "ہمارے بارے میں",
                 "رابطہ کریں", "تمام حقوق محفوظ", "پرائیویسی پالیسی", "فیس بک پر",
                 "ٹویٹر پر", "واٹس ایپ پر", "اشتہار")


def strip_markup(text):
    """Return plain text: no HTML, wiki markup, URLs, e-mails, menus."""
    if not text:
        return ""
    text = text.replace(EOT, " ")  # the special token must never occur in raw text
    text = _SCRIPT_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text)))
    text = _WIKI_CAT_RE.sub("", text)
    text = _WIKI_TEMPLATE_RE.sub(" ", text)
    text = _WIKI_LINK_RE.sub(r"\1", text)
    text = _WIKI_EXT_RE.sub(lambda m: m.group(1) or "", text)
    text = _WIKI_HEADING_RE.sub("", text)
    text = _WIKI_QUOTE_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    kept = []
    for ln in text.split("\n"):
        if len(ln) < 80 and any(p in ln for p in _MENU_PHRASES):
            continue
        kept.append(ln)
    return "\n".join(kept)


# =============================================================================
# 3. STEP 3 - LANGUAGE FILTER (line level)
# =============================================================================
_ALL_LETTERS_RE = re.compile(r"[^\W\d_]")
# Arabic-script LETTERS (no digits, no punctuation, no combining marks)
_AR_LETTER_RE = re.compile("[\u0621-\u063f\u0641-\u064a\u066e\u066f\u0671-\u06d3\u06d5"
                           "\u06fa-\u06ff\u0750-\u077f]")
# very common Urdu function words; Arabic / Persian / Pashto / Sindhi lines
# (same script!) rarely contain any of them
_URDU_STOPWORDS = {"کے", "کی", "کا", "کو", "ہے", "ہیں", "اور", "میں", "سے", "نے", "پر",
                   "کہ", "یہ", "وہ", "ہو", "تھا", "تھی", "تھے", "بھی", "ایک", "ہوں", "گا",
                   "گی", "گے", "کر", "کیا", "جو", "ہم", "آپ", "اس", "ان", "تو"}


def urdu_line_ok(line, min_ratio=0.5):
    """True if the line is mostly Urdu-script text (code-mixing is allowed)."""
    total = len(_ALL_LETTERS_RE.findall(line))
    if total == 0:
        return False
    ar = len(_AR_LETTER_RE.findall(line))
    if ar == 0 or ar / total < min_ratio:
        return False
    words = line.split()
    if len(words) >= 6 and not any(w in _URDU_STOPWORDS for w in words):
        return False
    return True


# =============================================================================
# 4. READING SOURCES
# =============================================================================
def open_any(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    if path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def iter_source_docs(src):
    """Yield raw document strings of one source."""
    fmt, path = src["format"], src["path"]
    field = src.get("field", "text")
    with open_any(path) as f:
        if fmt == "jsonl":
            fields = field.split(",")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parts = [str(rec.get(k, "")) for k in fields]
                yield "\n".join(p for p in parts if p)
        elif fmt == "csv":
            csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
            fields = field.split(",")
            for rec in csv.DictReader(f):
                parts = [str(rec.get(k, "") or "") for k in fields]
                yield "\n".join(p for p in parts if p)
        elif fmt == "txt-blank":
            buf = []
            for line in f:
                if line.strip():
                    buf.append(line.rstrip("\n"))
                elif buf:
                    yield "\n".join(buf)
                    buf = []
            if buf:
                yield "\n".join(buf)
        elif fmt == "txt-lines":
            n = int(src.get("group_size", 50))
            buf = []
            for line in f:
                if line.strip():
                    buf.append(line.rstrip("\n"))
                if len(buf) >= n:
                    yield "\n".join(buf)
                    buf = []
            if buf:
                yield "\n".join(buf)
        else:
            raise ValueError(f"unknown format {fmt!r} for source {src['name']}")


def blen(s):
    return len(s.encode("utf-8"))


STEPS = ["1_extract_text", "2_normalize", "3_language_filter",
         "4a_duplicate_documents", "4b_boilerplate_lines", "5_too_short_documents"]


# =============================================================================
# 5. STAGE 1: extract -> normalise -> language filter -> exact-duplicate docs
# =============================================================================
def stage1(sources, work, args, stats):
    seen = set()
    path = os.path.join(work, "stage1.jsonl")
    with open(path, "w", encoding="utf-8") as out:
        for si, src in enumerate(sources):
            st = stats["sources"][si]
            it = iter_source_docs(src)
            if tqdm:
                it = tqdm(it, desc=f"stage1 {src['name']}", unit="doc")
            for raw in it:
                b0 = blen(raw)
                st["raw_bytes"] += b0
                st["raw_docs"] += 1
                t1 = strip_markup(raw)
                b1 = blen(t1)
                stats["steps"]["1_extract_text"]["bytes"] += b0 - b1
                t2 = normalize(t1)
                b2 = blen(t2)
                stats["steps"]["2_normalize"]["bytes"] += b1 - b2
                lines = [ln for ln in t2.split("\n") if urdu_line_ok(ln, args.min_urdu_ratio)]
                t3 = "\n".join(lines)
                b3 = blen(t3)
                stats["steps"]["3_language_filter"]["bytes"] += b2 - b3
                if not t3:
                    stats["steps"]["3_language_filter"]["docs"] += 1
                    continue
                digest = hashlib.md5(t3.encode("utf-8")).digest()
                if digest in seen:
                    stats["steps"]["4a_duplicate_documents"]["bytes"] += b3
                    stats["steps"]["4a_duplicate_documents"]["docs"] += 1
                    continue
                seen.add(digest)
                out.write(json.dumps({"s": si, "t": t3}, ensure_ascii=False) + "\n")
    return path


# =============================================================================
# 6. STAGE 2: find boilerplate lines (same line in many different documents)
# =============================================================================
def find_boilerplate(stage1_path, min_docs):
    cnt = Counter()
    with open(stage1_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cnt.update(hash(ln) for ln in set(rec["t"].split("\n")))
    boiler = {h for h, c in cnt.items() if c >= min_docs}
    del cnt
    return boiler


# =============================================================================
# 7. STAGE 3: remove boilerplate, split by document, shuffle train, write files
# =============================================================================
def stage3(stage1_path, boiler, work, out_dir, args, stats):
    rng = random.Random(args.seed)
    n_buckets = 64
    bucket_files = [open(os.path.join(work, f"train_bucket_{i:02d}.jsonl"), "w",
                         encoding="utf-8") for i in range(n_buckets)]
    val_f = open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8", newline="\n")
    test_f = open(os.path.join(out_dir, "test.txt"), "w", encoding="utf-8", newline="\n")
    uniq = set()
    split_stats = {k: {"docs": 0, "words": 0, "bytes": 0} for k in ("train", "val", "test")}
    with open(stage1_path, "r", encoding="utf-8") as f:
        it = tqdm(f, desc="stage3", unit="doc") if tqdm else f
        for line in it:
            rec = json.loads(line)
            si, text = rec["s"], rec["t"]
            before = blen(text)
            if boiler:
                text = "\n".join(ln for ln in text.split("\n") if hash(ln) not in boiler)
            after = blen(text)
            stats["steps"]["4b_boilerplate_lines"]["bytes"] += before - after
            words = text.split()
            if len(words) < args.min_words:
                stats["steps"]["5_too_short_documents"]["bytes"] += after
                stats["steps"]["5_too_short_documents"]["docs"] += 1
                continue
            r = rng.random()
            split = "train" if r < 0.98 else ("val" if r < 0.99 else "test")
            st = stats["sources"][si]
            st["final_docs"] += 1
            st["final_bytes"] += after
            st["final_words"] += len(words)
            ss = split_stats[split]
            ss["docs"] += 1
            ss["words"] += len(words)
            ss["bytes"] += after
            uniq.update(words)
            if split == "train":
                bucket_files[rng.randrange(n_buckets)].write(
                    json.dumps(text, ensure_ascii=False) + "\n")
            elif split == "val":
                val_f.write(text + EOT)
            else:
                test_f.write(text + EOT)
    for bf in bucket_files:
        bf.close()
    val_f.close()
    test_f.close()

    # 7.1 shuffle train documents (bucket by bucket) and write train.txt
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8", newline="\n") as tf:
        for i in range(n_buckets):
            p = os.path.join(work, f"train_bucket_{i:02d}.jsonl")
            with open(p, "r", encoding="utf-8") as bf:
                docs = [json.loads(ln) for ln in bf]
            rng.shuffle(docs)
            for d in docs:
                tf.write(d + EOT)
            os.remove(p)
    return split_stats, len(uniq)


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--stats", default=None, help="default: <out-dir>/corpus_stats.json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-urdu-ratio", type=float, default=0.5)
    ap.add_argument("--boilerplate-min-docs", type=int, default=5,
                    help="a line present in at least this many documents is boilerplate "
                         "(0 disables the step)")
    ap.add_argument("--min-words", type=int, default=10, help="drop shorter documents")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    with open(args.sources, "r", encoding="utf-8") as f:
        sources = json.load(f)

    stats = {
        "seed": args.seed,
        "params": {"min_urdu_ratio": args.min_urdu_ratio,
                   "boilerplate_min_docs": args.boilerplate_min_docs,
                   "min_words": args.min_words},
        "steps": {s: {"bytes": 0, "docs": 0} for s in STEPS},
        "sources": [{"name": s["name"], "raw_bytes": 0, "raw_docs": 0,
                     "final_docs": 0, "final_bytes": 0, "final_words": 0} for s in sources],
    }
    t0 = time.time()
    s1 = stage1(sources, args.work_dir, args, stats)
    boiler = set()
    if args.boilerplate_min_docs > 0:
        print("finding boilerplate lines ...")
        boiler = find_boilerplate(s1, args.boilerplate_min_docs)
        print(f"  {len(boiler):,} boilerplate line hashes")
    split_stats, n_unique = stage3(s1, boiler, args.work_dir, args.out_dir, args, stats)
    os.remove(s1)

    total_bytes = sum(v["bytes"] for v in split_stats.values())
    stats["splits"] = split_stats
    stats["totals"] = {
        "bytes": total_bytes, "mb": round(total_bytes / 2**20, 1),
        "docs": sum(v["docs"] for v in split_stats.values()),
        "words": sum(v["words"] for v in split_stats.values()),
        "unique_words": n_unique,
        "raw_bytes": sum(s["raw_bytes"] for s in stats["sources"]),
    }
    sp = args.stats or os.path.join(args.out_dir, "corpus_stats.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats["totals"], indent=2))
    for s, v in stats["steps"].items():
        print(f"  {s:26s} removed {v['bytes'] / 2**20:10.2f} MB   {v['docs']:>9,} docs")
    print(f"done in {(time.time() - t0) / 60:.1f} min; stats -> {sp}")
    if total_bytes < 500 * 2**20:
        print(f"WARNING: corpus is only {total_bytes / 2**20:.0f} MB (< 500 MB required)")
    for src, st in zip(sources, stats["sources"]):
        if src.get("own_collection") and st["raw_bytes"] < 50 * 2**20:
            print(f"WARNING: own collection '{src['name']}' has only "
                  f"{st['raw_bytes'] / 2**20:.0f} MB raw text")


if __name__ == "__main__":
    main()
