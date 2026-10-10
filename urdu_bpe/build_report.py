#!/usr/bin/env python3
"""
build_report.py
===============
Turn analysis/results.json (+ your own wording) into report.html and, if
possible, report.pdf (max 4 pages).

1. python build_report.py          -> writes analysis/report_text.json (drafts)
2. EDIT analysis/report_text.json  -> write YOUR interpretation (2-3 sentences each)
3. python build_report.py          -> report.html (+ report.pdf if weasyprint or a
                                      headless Chrome/Edge is available)

Urdu text needs correct Arabic shaping, so the PDF is rendered by a browser
engine (Chrome/Edge/Chromium or WeasyPrint) instead of a PDF library.
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import base64
import html
import json
import os
import shutil
import subprocess
import sys

CSS = """
@page { size: A4; margin: 11mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 8.6pt; line-height: 1.28; color:#111; }
h1 { font-size: 14pt; margin: 0 0 2mm 0; } h2 { font-size: 10.5pt; margin: 3mm 0 1mm 0; border-bottom:1px solid #999; }
.ur { font-family: 'Noto Nastaliq Urdu','Jameel Noori Nastaleeq','Noto Naskh Arabic','Segoe UI',serif; direction: rtl; unicode-bidi: isolate; font-size: 10pt; }
table { border-collapse: collapse; margin: 1mm 0; } td, th { border: 1px solid #bbb; padding: 0.4mm 1.6mm; font-size: 7.4pt; }
th { background:#eee; } .grid { display:flex; gap:3mm; align-items:flex-start; } img { max-width:100%; }
.small { font-size: 7.4pt; color:#333; } .bad { color:#b00; }
"""


PATTERN_EXPLANATION = (
    "GPT-2's pattern treats combining marks as punctuation and so cuts a word such as \u0627\u064f\u0631\u062f\u064f\u0648 "
    "at every diacritic. Here a word is one or more of \\p{L} (letters) or \\p{M} (combining marks: zabar, zer, pesh, "
    "shadda ...), plus ZWNJ/ZWJ so that an invisible joiner never cuts a word. The optional leading space belongs to the "
    "following word (as in GPT-2), digits (\\p{N}, any script) form their own runs, punctuation/symbols/emoji (including the "
    "Urdu full stop and comma) form separate runs, and whitespace keeps the GPT-2 rule so the last space prefixes the next "
    "word. Every character belongs to exactly one class, so the split is lossless and merges never cross a word boundary."
)


def esc(s):
    return html.escape(str(s))


def table(headers, rows):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(c if isinstance(c, Raw) else f"<td>{esc(c)}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


class Raw(str):
    pass


def ur_cell(text, bad=False):
    cls = "ur bad" if bad else "ur"
    return Raw(f'<td><span class="{cls}">{esc(text)}</span></td>')


def img(path):
    with open(path, "rb") as f:
        return f'<img src="data:image/png;base64,{base64.b64encode(f.read()).decode()}">'


# =============================================================================
# 2. HTML ASSEMBLY
# =============================================================================
def build_html(res, text, info):
    q1, q2, q3 = res["q1"], res["q2"], res["q3"]
    parts = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>",
             f"<h1>Assignment 1 - Urdu BPE Tokenizer: Analysis</h1>",
             f"<div class='small'>{esc(info)} | vocabulary {res['vocab_size']:,} "
             f"({res['num_merges']:,} merges)</div>"]

    # ---- pre-tokenisation pattern (required: "explain your pattern in the report") ----
    from bpe import PATTERN
    parts.append("<h2>0. Pre-tokenisation pattern</h2><div class='small'><code>" + esc(PATTERN) + "</code></div>")
    parts.append("<p>" + esc(text.get("pattern", PATTERN_EXPLANATION)) + "</p>")

    # ---- Q1 ------------------------------------------------------------------
    parts.append("<h2>1. What did the merges learn?</h2>")
    rows = q1["rows"]
    third = (len(rows) + 2) // 3
    grid = []
    for i in range(0, len(rows), third):
        chunk = rows[i:i + third]
        grid.append(table(["#", "token", "B", "UTF-8"], [
            [r["merge"], ur_cell(r["token"], not r["valid_utf8"]), r["bytes"],
             "yes" if r["valid_utf8"] else "NO"] for r in chunk]))
    parts.append("<div class='grid'>" + "".join(grid) + "</div>")
    lg = q1["longest"]
    half = (len(lg) + 1) // 2
    parts.append("<div class='small'>20 longest tokens (merge #, token, bytes/chars):</div><div class='grid'>" +
                 "".join(table(["#", "token", "B", "chars"], [[r["merge"], ur_cell(r["token"]), r["bytes"], r["chars"]]
                                                           for r in lg[i:i + half]]) for i in (0, half)) + "</div>")
    parts.append(img(os.path.join(info_dir(), "token_lengths.png")))
    parts.append(f"<p>{esc(text['q1'])}</p>")

    # ---- Q2 ------------------------------------------------------------------
    parts.append("<h2>2. Comparison with existing tokenizers (tokens per word)</h2>")
    names = q2["tokenizers"]
    parts.append(table(["input", "words"] + names, [
        [r["input"] + (" *" if r.get("extra_probe") else ""), r["words"]] + [r["tpw"][n] for n in names]
        for r in q2["rows"]]))
    parts.append("<div class='small'>* extra probe used for question 3 only. Hand-picked inputs are "
                 "normalised with normalize() for every tokenizer (except unnormalised_variants).</div>")
    parts.append(img(os.path.join(info_dir(), "tokens_per_word.png")))
    parts.append(f"<p>{esc(text['q2'])}</p>")

    # ---- Q3 ------------------------------------------------------------------
    parts.append("<h2>3. Three inputs the tokenizer handles badly</h2>")
    parts.append(table(["input", "our tokens/word", "test.txt baseline"],
                       [[w["input"], w["tpw"], w["baseline_test_tpw"]] for w in q3["worst"]]))
    parts.append("<div class='small'>Most fragmented frequent test words: " + ", ".join(
        f"<span class='ur'>{esc(w['word'])}</span> ({w['tokens']})" for w in q3["most_fragmented_frequent_words"][:6])
        + "</div>")
    parts.append(f"<p>{esc(text['q3'])}</p></body></html>")
    return "".join(parts)


_INFO_DIR = "analysis"


def info_dir():
    return _INFO_DIR


# =============================================================================
# 3. PDF RENDERING (browser engine, for correct Urdu shaping)
# =============================================================================
def find_browser():
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge"):
        p = shutil.which(name)
        if p:
            return p
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if os.path.exists(p):
            return p
    return None


def render_pdf(html_path, pdf_path):
    try:
        import weasyprint
        doc = weasyprint.HTML(filename=html_path).render()
        doc.write_pdf(pdf_path)
        return len(doc.pages)
    except ImportError:
        pass
    b = find_browser()
    if not b:
        return None
    subprocess.run([b, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={os.path.abspath(pdf_path)}", "file://" + os.path.abspath(html_path)],
                   check=True, capture_output=True)
    try:  # count pages without extra libraries: count /Type /Page objects
        data = open(pdf_path, "rb").read()
        return data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") + data.count(b"/Type/Page>") \
            + data.count(b"/Type/Page/")
    except OSError:
        return None


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    global _INFO_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", default="analysis")
    ap.add_argument("--log", default="training_log.json")
    ap.add_argument("--html", default="report.html")
    ap.add_argument("--pdf", default="report.pdf")
    args = ap.parse_args()
    _INFO_DIR = args.analysis_dir

    res = json.load(open(os.path.join(args.analysis_dir, "results.json"), encoding="utf-8"))
    tp = os.path.join(args.analysis_dir, "report_text.json")
    if not os.path.exists(tp):
        json.dump(res["draft_text"], open(tp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"Wrote DRAFT interpretations to {tp}. Edit them in your own words, then run again.")
    text = json.load(open(tp, encoding="utf-8"))

    info = "training details unavailable"
    if os.path.exists(args.log):
        lg = json.load(open(args.log, encoding="utf-8"))
        m = lg["machine"]
        info = (f"Training wall-clock {lg['seconds_total'] / 3600:.2f} h ({lg['method']} trainer) on "
                f"{m['processor']}, {m['cpu_count']} CPUs, {m['ram_gb']} GB RAM, {m['platform']}, "
                f"Python {m['python']}")
    with open(args.html, "w", encoding="utf-8") as f:
        f.write(build_html(res, text, info))
    pages = render_pdf(args.html, args.pdf)
    if pages is None:
        print(f"Wrote {args.html}. Open it in Chrome/Edge -> Print -> Save as PDF "
              f"(A4, margins 'default', background graphics on) and name it report.pdf.")
    else:
        print(f"Wrote {args.pdf} ({pages} page(s)). Must be at most 4.")
        if pages > 4:
            print("TOO LONG: shorten report_text.json or reduce CSS font sizes.")


if __name__ == "__main__":
    sys.exit(main())
