#!/usr/bin/env python3
"""
make_data_card.py
=================
Generate DATA_CARD.md from sources.json + data/corpus_stats.json.
Numbers come from the pipeline; the text under "Known gaps and biases" is
generated from those numbers - READ IT AND ADD YOUR OWN OBSERVATIONS.

    python make_data_card.py --sources sources.json --stats data/corpus_stats.json
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import json
import os

MB = 2**20


def mb(x):
    return f"{x / MB:,.1f}"


# =============================================================================
# 2. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="sources.json")
    ap.add_argument("--stats", default="data/corpus_stats.json")
    ap.add_argument("--out", default="DATA_CARD.md")
    args = ap.parse_args()

    sources = json.load(open(args.sources, encoding="utf-8"))
    st = json.load(open(args.stats, encoding="utf-8"))
    tot, steps = st["totals"], st["steps"]

    L = ["# DATA CARD - Urdu corpus for the BPE tokenizer", ""]
    L += ["## 1. Sources", "",
          "| Source | Genre | URL | Licence / terms | Collected | Own? | Raw size (MB) | Cleaned size (MB) |",
          "|---|---|---|---|---|---|---|---|"]
    for s, x in zip(sources, st["sources"]):
        L.append(f"| {s['name']} | {s.get('genre', '?')} | {s.get('url', '?')} | {s.get('licence', 'TODO')} | "
                 f"{s.get('collection_date', 'TODO')} | {'yes' if s.get('own_collection') else 'no'} | "
                 f"{mb(x['raw_bytes'])} | {mb(x['final_bytes'])} |")
    own = sum(x["raw_bytes"] for s, x in zip(sources, st["sources"]) if s.get("own_collection"))
    L += ["", f"Own collection (scraped / API): **{mb(own)} MB** raw text. Evidence (URL, UTC time, status, "
          "size, SHA-256 per page / article):"]
    for s in sources:
        if s.get("own_collection"):
            L.append(f"- `{s.get('evidence', 'TODO: path of evidence csv')}` (source `{s['name']}`)")

    L += ["", "## 2. Corpus totals", "",
          "| Split | Documents | Words | MB |", "|---|---|---|---|"]
    for k, v in st["splits"].items():
        L.append(f"| {k} | {v['docs']:,} | {v['words']:,} | {mb(v['bytes'])} |")
    L.append(f"| **total** | **{tot['docs']:,}** | **{tot['words']:,}** | **{tot['mb']:,}** |")
    L += ["", f"Unique words (whitespace-separated): **{tot['unique_words']:,}**. "
          f"Split by document, 98/1/1, seed {st['seed']}; `<|endoftext|>` follows every document "
          "(no newline after it); train documents are shuffled.", ""]

    L += ["## 3. Cleaning steps", "", "| Step | Removed (MB) | % of raw | Documents removed |", "|---|---|---|---|"]
    raw = max(tot["raw_bytes"], 1)
    desc = {"1_extract_text": "1. strip HTML / wiki markup / URLs / e-mails / menus",
            "2_normalize": "2. Unicode normalisation (net change)",
            "3_language_filter": f"3. drop lines that are mostly not Urdu (>= {st['params']['min_urdu_ratio']:.0%} "
                                 "Urdu-script letters kept)",
            "4a_duplicate_documents": "4a. exact duplicate documents",
            "4b_boilerplate_lines": f"4b. boilerplate lines in >= {st['params']['boilerplate_min_docs']} documents",
            "5_too_short_documents": f"5. documents shorter than {st['params']['min_words']} words"}
    for k, v in steps.items():
        L.append(f"| {desc[k]} | {mb(v['bytes'])} | {100 * v['bytes'] / raw:.2f}% | {v['docs']:,} |")
    L += ["", "Normalisation choices: Arabic yeh/alef maksura -> U+06CC, Arabic kaf -> U+06A9, Arabic heh -> U+06C1 "
          "(U+06BE kept separate), tatweel and invisible characters removed, ZWNJ removed (typed inconsistently; "
          "splits one word into several vocabulary entries), digits -> ASCII, NFC.", ""]

    L += ["## 4. Known gaps and biases", ""]
    by_genre = {}
    for s, x in zip(sources, st["sources"]):
        by_genre[s.get("genre", "?")] = by_genre.get(s.get("genre", "?"), 0) + x["final_bytes"]
    total_final = max(sum(by_genre.values()), 1)
    for g, b in sorted(by_genre.items(), key=lambda kv: -kv[1]):
        L.append(f"- Genre **{g}** is {100 * b / total_final:.1f}% of the cleaned text.")
    L += ["- Roman Urdu and Latin-script text are removed by the language filter; only natural code-mixing "
          "inside Urdu lines survives, so the tokenizer is weak on Roman Urdu / English.",
          "- Diacritics are rare in everyday text, so they are poorly merged.",
          "- Language filter is a heuristic (script share + common function words); some Persian/Arabic/"
          "Sindhi lines may survive and some short Urdu lines may be lost.",
          "- TODO (edit): topical, regional or time-period biases you noticed in your own sources "
          "(e.g. politics-heavy news, Pakistani vs Indian usage).", ""]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
