#!/usr/bin/env python3
"""
analyze_tokenizer.py
====================
Part C: what did the tokenizer learn, and how does it compare?

Produces  analysis/results.json  and  analysis/*.png  which build_report.py
turns into report.html / report.pdf.

Example:
    python analyze_tokenizer.py --tokenizer tokenizer.json --test data/test.txt

Comparison tokenizers (Part C ONLY - tiktoken/tokenizers are NOT used anywhere
in the implementation):  tiktoken 'gpt2' and 'o200k_base'; optional extra
Hugging Face tokenizer via --hf <name> (needs the `tokenizers` package + internet).
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import json
import os
import sys
import unicodedata

from bpe import BPETokenizer, FIRST_MERGE_ID, SPECIAL_TOKEN, count_words
from normalize import normalize

# =============================================================================
# 2. HAND-PICKED INPUTS (seven required kinds + three extra probes for Q3)
# =============================================================================
HAND_INPUTS = {
    "news": "وزیر اعظم نے آج اسلام آباد میں اعلان کیا کہ حکومت آئندہ مالی سال کے بجٹ میں "
            "تعلیم اور صحت کے شعبوں کے لیے اضافی فنڈز مختص کرے گی۔",
    "poetry": "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے\n"
              "بہت نکلے مرے ارمان لیکن پھر بھی کم نکلے",
    "urdu_with_english": "میں نے اپنے laptop پر نیا software install کیا مگر Wi-Fi کنکشن بار بار "
                         "disconnect ہو جاتا ہے۔",
    "roman_urdu": "main kal office nahi aa sakta, meri tabiyat theek nahi hai aur doctor ne "
                  "aaraam karne ko kaha hai.",
    "numbers_dates": "۱۴ اگست ۱۹۴۷ کو پاکستان آزاد ہوا، اور 2025-26 کے بجٹ کا حجم 17,500 ارب روپے ہے۔",
    "diacritics": "اُردُو زبان میں زَبَر، زِیر اور پیش کا استعمال شِعر کو خُوبصُورت بَناتا ہے۔",
    "personal_names": "محمد علی جناح، علامہ محمد اقبال، فیض احمد فیض، عبدالستار ایدھی اور ملالہ یوسفزئی",
}
EXTRA_PROBES = {  # used only to pick the three worst cases for question 3
    "unnormalised_variants": "\u0643\u062a\u0627\u0628 \u064a\u0627\u062f \u0647\u0645 "
                             "\u0643\u0640\u062a\u0627\u0628 \u0628\u0647\u062a \u0627\u0644\u0644\u0647",
    "emoji_and_hashtag": "آج موسم بہت اچھا ہے 😀🌧️ #پاکستان",
    "long_digit_strings": "1234567890123456789 0987654321 3.14159265358979",
}
RAW_KEYS = {"unnormalised_variants"}  # deliberately NOT normalised

DRAFT_REASONS = {
    "roman_urdu": "The corpus is Urdu-script only (the cleaning step drops Latin-script lines), so "
                  "Roman Urdu has no learned merges and falls back to short English-like byte pieces.",
    "urdu_with_english": "Latin words are rare in the training data, so English/code-mixed words "
                         "break into 1-3 letter pieces; each Latin letter costs one byte-token.",
    "unnormalised_variants": "Normalisation is deliberately outside the tokenizer. Arabic yeh/kaf/heh "
                             "and tatweel map to byte sequences never seen in training, so the same "
                             "words fragment into raw bytes unless normalize() is applied first.",
    "emoji_and_hashtag": "Emoji are 4 UTF-8 bytes (plus variation selectors) and almost never occur "
                         "in the cleaned corpus, so each costs several byte tokens.",
    "long_digit_strings": "Digits were normalised to ASCII, but long or rare digit strings were seen "
                          "only in short common runs, so they split into many 1-3 digit tokens.",
    "diacritics": "Diacritics (zabar, zer, pesh) are rare in everyday text, so base letter and mark "
                  "rarely merge and each mark costs its own 2-byte token.",
    "personal_names": "Rare personal names are split at arbitrary points because their letter "
                      "sequences are infrequent in the corpus.",
    "numbers_dates": "Mixed digits, dates and punctuation break across the digit/punctuation "
                     "pre-token boundaries, so every group becomes at least one separate token.",
    "poetry": "Poetic vocabulary and rarer words are less frequent than news vocabulary.",
    "news": "News text is the best case for a vocabulary trained on this kind of text.",
}


# =============================================================================
# 3. HELPERS
# =============================================================================
def show(b):
    """Readable form of token bytes + whether they are valid UTF-8 on their own."""
    try:
        s = b.decode("utf-8")
        return s.replace("\n", "\\n").replace(" ", "\u2423"), True  # \u2423 = visible space
    except UnicodeDecodeError:
        return "<" + " ".join(f"{x:02x}" for x in b) + ">", False


def is_letter_token(b):
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return len(s) == 1 and unicodedata.category(s)[0] in "LM"


def head_of_file(path, max_mb):
    """Text of the first ~max_mb MB of `path`, cut at a line boundary."""
    out, n = [], 0
    with open(path, "rb") as f:
        for line in f:
            out.append(line)
            n += len(line)
            if max_mb and n >= max_mb * 2**20:
                break
    return b"".join(out).decode("utf-8", "replace")


# =============================================================================
# 4. QUESTION 1 - WHAT DID THE MERGES LEARN?
# =============================================================================
def question1(tok, test_counts):
    n = len(tok.merges)
    ranks = list(range(1, min(50, n) + 1)) + list(range(5000, n + 1, 5000))
    rows = []
    for r in ranks:
        tid = FIRST_MERGE_ID + r - 1
        a, b = tok.merges[r - 1]
        text, valid = show(tok.vocab[tid])
        rows.append({"merge": r, "id": tid, "pair": [a, b], "token": text,
                     "bytes": len(tok.vocab[tid]), "valid_utf8": valid,
                     "two_byte_letter": len(tok.vocab[tid]) == 2 and is_letter_token(tok.vocab[tid])})

    # two-byte letters rebuilt / tokens that are not valid UTF-8 (first 300 merges)
    first = min(300, n)
    letters = [r + 1 for r in range(first)
               if len(tok.vocab[FIRST_MERGE_ID + r]) == 2 and is_letter_token(tok.vocab[FIRST_MERGE_ID + r])]
    invalid = [r + 1 for r in range(first) if not show(tok.vocab[FIRST_MERGE_ID + r])[1]]

    # first whole word = first token that is itself a complete pre-token word of test.txt
    def whole_word_rank(min_chars):
        for r in range(n):
            try:
                s = tok.vocab[FIRST_MERGE_ID + r].decode("utf-8")
            except UnicodeDecodeError:
                continue
            core = s[1:] if s.startswith(" ") else s
            if len(core) >= min_chars and core.isalpha() and (s in test_counts or (" " + core) in test_counts
                                                              or core in test_counts):
                return r + 1, s
        return None, None

    w2 = whole_word_rank(2)
    w3 = whole_word_rank(3)

    longest = sorted(range(n), key=lambda r: (-len(tok.vocab[FIRST_MERGE_ID + r]), r))[:20]
    long_rows = []
    for r in longest:
        b = tok.vocab[FIRST_MERGE_ID + r]
        text, valid = show(b)
        long_rows.append({"merge": r + 1, "token": text, "bytes": len(b),
                          "chars": len(b.decode("utf-8", "replace")), "valid_utf8": valid})
    return {"rows": rows,
            "two_byte_letter_merges_first300": letters[:12],
            "num_two_byte_letters_first300": len(letters),
            "invalid_utf8_merges_first300": invalid[:12],
            "num_invalid_utf8_first300": len(invalid),
            "invalid_among_first50": sum(1 for x in invalid if x <= 50),
            "first_whole_word_2plus_chars": {"merge": w2[0], "token": show(w2[1].encode())[0] if w2[1] else None},
            "first_whole_word_3plus_chars": {"merge": w3[0], "token": show(w3[1].encode())[0] if w3[1] else None},
            "longest": long_rows}


# =============================================================================
# 5. QUESTION 2 - COMPARISON WITH EXISTING TOKENIZERS
# =============================================================================
def build_comparison_encoders(tok, hf_names):
    encs = {"ours (50k Urdu BPE)": lambda t: tok.encode(t, allow_special=False)}
    try:
        import tiktoken
        for label, name in (("GPT-2 (tiktoken gpt2)", "gpt2"),
                            ("o200k_base (tiktoken)", "o200k_base")):
            try:
                e = tiktoken.get_encoding(name)
                encs[label] = (lambda e: lambda t: e.encode(t, disallowed_special=()))(e)
            except Exception as ex:  # no internet for the vocab download, etc.
                print(f"WARNING: could not load tiktoken '{name}': {ex}")
    except ImportError:
        print("WARNING: tiktoken not installed (pip install tiktoken) - comparison skipped")
    for name in hf_names:
        try:
            from tokenizers import Tokenizer
            t = Tokenizer.from_pretrained(name)
            encs[f"{name} (HF)"] = (lambda t: lambda s: t.encode(s).ids)(t)
        except Exception as ex:
            print(f"WARNING: could not load HF tokenizer {name}: {ex}")
    return encs


def tokens_per_word(encs, text):
    words = max(len(text.split()), 1)
    return {name: round(len(fn(text)) / words, 3) for name, fn in encs.items()}


def question2(tok, encs, test_text):
    out = {"tokenizers": list(encs), "rows": []}
    test_plain = test_text.replace(SPECIAL_TOKEN, "\n")
    out["rows"].append({"input": "test.txt (sample)", "words": len(test_plain.split()),
                        "tpw": tokens_per_word(encs, test_plain)})
    for key, text in {**HAND_INPUTS, **EXTRA_PROBES}.items():
        t = text if key in RAW_KEYS else normalize(text)
        out["rows"].append({"input": key, "words": len(t.split()), "tpw": tokens_per_word(encs, t),
                            "extra_probe": key in EXTRA_PROBES})
    return out


def plot_tpw(q2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in q2["rows"] if not r.get("extra_probe")]
    names = q2["tokenizers"]
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    for i, nm in enumerate(names):
        ax.bar([x + i * width for x in range(len(rows))], [r["tpw"][nm] for r in rows],
               width, label=nm)
    ax.set_xticks([x + width * (len(names) - 1) / 2 for x in range(len(rows))])
    ax.set_xticklabels([r["input"] for r in rows], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("tokens per word (lower = better)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_lengths(tok, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lens = [len(tok.vocab[FIRST_MERGE_ID + r]) for r in range(len(tok.merges))]
    fig, ax = plt.subplots(figsize=(8.2, 1.9))
    ax.scatter(range(1, len(lens) + 1), lens, s=1, alpha=0.4)
    ax.set_xlabel("merge number", fontsize=8)
    ax.set_ylabel("token length (bytes)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# =============================================================================
# 6. QUESTION 3 - THREE INPUTS HANDLED BADLY
# =============================================================================
def question3(q2, test_counts, tok):
    ours = "ours (50k Urdu BPE)"
    base = q2["rows"][0]["tpw"][ours]
    ranked = sorted((r for r in q2["rows"][1:]), key=lambda r: -r["tpw"][ours])[:3]
    worst = [{"input": r["input"], "tpw": r["tpw"][ours], "baseline_test_tpw": base,
              "reason": DRAFT_REASONS.get(r["input"], "")} for r in ranked]
    # most fragmented FREQUENT words in the test set
    frequent = [(w, c) for w, c in test_counts.items() if c >= 20 and len(w.strip()) >= 3]
    frag = sorted(((len(tok._word_ids(w)), w, c) for w, c in frequent), reverse=True)[:10]
    return {"worst": worst,
            "most_fragmented_frequent_words": [{"word": show(w.encode())[0], "tokens": n, "count": c}
                                               for n, w, c in frag]}


# =============================================================================
# 7. DRAFT INTERPRETATIONS (build_report.py lets you edit them)
# =============================================================================
def draft_text(res, ours="ours (50k Urdu BPE)"):
    q1, q2, q3 = res["q1"], res["q2"], res["q3"]
    t = q2["rows"][0]["tpw"]
    others = ", ".join(f"{k}: {v}" for k, v in t.items() if k != ours)
    return {
        "q1": (f"The first merges rebuild frequent two-byte Urdu letters ({q1['num_two_byte_letters_first300']} of "
               f"the first 300 merges are single letters) and common letter pairs; "
               f"{q1['num_invalid_utf8_first300']} of the first 300 tokens are not valid UTF-8 on their own "
               f"(a lead byte plus a partial character). The first whole word of 2+ letters appears at merge "
               f"{q1['first_whole_word_2plus_chars']['merge']} and of 3+ letters at merge "
               f"{q1['first_whole_word_3plus_chars']['merge']}. Later merges are whole words and word stems, "
               f"the longest tokens being frequent words with a leading space."),
        "q2": (f"On test.txt our tokenizer needs {t.get(ours)} tokens per word versus {others}. "
               f"It wins clearly on Urdu-script text (news, poetry, names) because its merges were learned "
               f"on Urdu, while the English-trained GPT-2 vocabulary spends mostly byte fragments per letter. "
               f"Our advantage shrinks or reverses on Roman Urdu, English-mixed text and digits."),
        "q3": ("Worst cases by tokens per word: " + "; ".join(
            f"{w['input']} ({w['tpw']} vs {w['baseline_test_tpw']} on test.txt)" for w in q3["worst"]) +
            ". " + " ".join(w["reason"] for w in q3["worst"])),
    }


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--test", required=True, help="test.txt")
    ap.add_argument("--test-mb", type=float, default=5, help="use first N MB of test.txt (0 = all)")
    ap.add_argument("--hf", action="append", default=[], help="extra Hugging Face tokenizer name")
    ap.add_argument("--out-dir", default="analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = BPETokenizer.load(args.tokenizer)
    test_text = head_of_file(args.test, args.test_mb)
    tmp = os.path.join(args.out_dir, "_test_sample.txt")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(test_text)
    test_counts = count_words(tmp, progress=False)
    os.remove(tmp)

    encs = build_comparison_encoders(tok, args.hf)
    res = {"vocab_size": tok.vocab_size, "num_merges": len(tok.merges),
           "q1": question1(tok, test_counts)}
    res["q2"] = question2(tok, encs, test_text)
    res["q3"] = question3(res["q2"], test_counts, tok)
    res["draft_text"] = draft_text(res)
    plot_tpw(res["q2"], os.path.join(args.out_dir, "tokens_per_word.png"))
    plot_lengths(tok, os.path.join(args.out_dir, "token_lengths.png"))
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res["q2"]["rows"][:3], ensure_ascii=False, indent=1))
    print(f"wrote {args.out_dir}/results.json, tokens_per_word.png, token_lengths.png")


if __name__ == "__main__":
    sys.exit(main())
