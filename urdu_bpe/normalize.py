"""
normalize.py
============
Urdu Unicode normalisation for Assignment 1 (Part A).

Own implementation, Python standard library only (re, unicodedata).
Normalisation is a SEPARATE step: it is applied to the corpus *before*
tokenizer training and is never called from inside bpe.py.

Design decisions (justified)
----------------------------
* Arabic yeh (U+064A) and alef maksura (U+0649) -> Urdu yeh (U+06CC)
* Arabic kaf (U+0643)                           -> Urdu kaf  (U+06A9)
* Arabic heh (U+0647)                           -> heh goal  (U+06C1)
  U+06C1 (heh goal) and U+06BE (do-chashmi heh) are DIFFERENT letters and
  are kept apart (do-chashmi is used for aspirated sounds: بھ، کھ ...).
* Tatweel (U+0640) is removed: purely typographic stretching.
* ZWNJ (U+200C) is REMOVED (KEEP_ZWNJ = False).  Justification: on the web
  it is typed inconsistently (the same word appears with and without it),
  so keeping it would split one word into several vocabulary entries; and
  it can cut a word in two for naive pre-tokenisers.  The cost is only a
  cosmetic one (letter shapes may join).  Flip KEEP_ZWNJ to keep it.
* Other invisible characters (ZWJ, ZWSP, LRM/RLM, bidi controls, BOM,
  soft hyphen) are removed.
* Digits: Arabic-Indic (U+0660-0669) and Extended Arabic-Indic
  (U+06F0-06F9) are mapped to ASCII 0-9.  Justification: one byte per
  digit, one convention, and mixed-script text already uses ASCII digits.
* Unicode form: NFC (composes alef+madda -> آ, heh goal+hamza -> ۂ, etc.,
  so that visually identical sequences share one code point sequence).
  Arabic presentation forms (U+FB50-FDFF, U+FE70-FEFE) are expanded with
  NFKC, but ONLY those ranges (NFKC on everything would damage other text).
* Diacritics (zabar, zer, pesh ...) are KEPT: they are real content.
* Whitespace: all exotic spaces -> " ", runs of spaces -> one space,
  all line-break variants -> "\\n", runs of newlines -> one "\\n",
  spaces around newlines removed, text stripped.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import re
import unicodedata

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
KEEP_ZWNJ = False  # see module docstring for the justification

# =============================================================================
# 3. CHARACTER TABLES
# =============================================================================
# 3.1 Letter variants that look identical on screen
_CHAR_MAP = {
    "\u064A": "\u06CC",  # Arabic yeh        -> Urdu yeh
    "\u0649": "\u06CC",  # alef maksura      -> Urdu yeh
    "\u0643": "\u06A9",  # Arabic kaf        -> Urdu kaf (keheh)
    "\u0647": "\u06C1",  # Arabic heh        -> heh goal
    "\u0640": "",        # tatweel           -> removed
}

# 3.2 Invisible / formatting characters that are deleted
_INVISIBLE = [
    "\u200B",  # zero-width space
    "\u200D",  # zero-width joiner
    "\u200E", "\u200F",  # LRM, RLM
    "\u202A", "\u202B", "\u202C", "\u202D", "\u202E",  # bidi embeddings
    "\u2060",  # word joiner
    "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolates
    "\u061C",  # Arabic letter mark
    "\u180E",  # Mongolian vowel separator
    "\uFEFF",  # byte-order mark
    "\u00AD",  # soft hyphen
]
if not KEEP_ZWNJ:
    _INVISIBLE.append("\u200C")

# 3.3 Whitespace variants
_SPACE_VARIANTS = (
    ["\u00A0", "\u1680", "\u202F", "\u205F", "\u3000", "\t", "\x0b", "\x0c"]
    + [chr(c) for c in range(0x2000, 0x200B)]
)
_NEWLINE_VARIANTS = ["\r", "\u2028", "\u2029", "\x85"]

# 3.4 Build one translation table (fast: a single str.translate call)
_TABLE = {}
for _k, _v in _CHAR_MAP.items():
    _TABLE[ord(_k)] = _v
for _c in _INVISIBLE:
    _TABLE[ord(_c)] = ""
for _c in _SPACE_VARIANTS:
    _TABLE[ord(_c)] = " "
for _c in _NEWLINE_VARIANTS:
    _TABLE[ord(_c)] = "\n"
for _d in range(10):
    _TABLE[0x0660 + _d] = str(_d)  # Arabic-Indic digits
    _TABLE[0x06F0 + _d] = str(_d)  # Extended Arabic-Indic digits

# =============================================================================
# 4. REGULAR EXPRESSIONS
# =============================================================================
# Arabic presentation forms A and B (excluding U+FEFF, the BOM)
_PRESENTATION_RE = re.compile("[\ufb50-\ufdff\ufe70-\ufefe]")
# C0/C1 control characters (except \n), replacement char, private use
_CONTROL_RE = re.compile("[\x00-\x09\x0b-\x1f\x7f-\x9f\ufffd\ue000-\uf8ff]")
_SPACES_RE = re.compile(r" {2,}")
_NEWLINES_RE = re.compile(r" ?\n[ \n]*")


# =============================================================================
# 5. PUBLIC API
# =============================================================================
def normalize(text: str) -> str:
    """Return the normalised version of `text` (see module docstring)."""
    if not text:
        return ""
    # 5.1 Expand Arabic presentation forms (compatibility decomposition)
    text = _PRESENTATION_RE.sub(
        lambda m: unicodedata.normalize("NFKC", m.group()), text
    )
    # 5.2 Canonical composition (before letter mapping, so that yeh+hamza
    #     becomes U+0626 and is NOT turned into plain Urdu yeh)
    text = unicodedata.normalize("NFC", text)
    # 5.3 Letter variants, invisibles, digits, whitespace variants
    text = text.translate(_TABLE)
    # 5.4 Mapping may create new composable pairs (heh + hamza -> U+06C2)
    text = unicodedata.normalize("NFC", text)
    # 5.5 Remove control characters
    text = _CONTROL_RE.sub("", text)
    # 5.6 Whitespace clean-up
    text = _SPACES_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n", text)
    return text.strip()


if __name__ == "__main__":
    import sys

    sys.stdout.write(normalize(sys.stdin.read()) + "\n")
