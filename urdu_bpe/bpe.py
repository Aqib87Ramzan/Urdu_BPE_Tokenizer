"""
bpe.py
======
Byte-level BPE tokenizer written from scratch (Assignment 1, Part B).

Allowed libraries only: Python standard library, regex, tqdm (progress bar).
No tokenizer library and no published BPE code is used.

Vocabulary layout
-----------------
    ids 0..255        the 256 byte values
    id  256           <|endoftext|>  (special: never split, never merged)
    ids 257..49,999   merges, in the order they were learned (49,743 merges)

Two trainers are provided
-------------------------
    train_reference : recounts every pair after every merge (simple, slow)
    train_fast      : weighted distinct words + pair->words index + heap
Both use the SAME tie-breaking rule, so they learn identical merges.

Tie-breaking
------------
Highest count wins; ties go to the GREATER pair when the pair is compared as a
tuple of bytes, i.e. (bytes_of_first_token, bytes_of_second_token).
(In the astronomically rare case that two different id pairs have identical
byte tuples, the smaller (id_a, id_b) wins - applied identically in both
trainers.)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import heapq
import json
import time
from collections import Counter

import regex

try:  # progress bars are optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# =============================================================================
# 2. CONSTANTS
# =============================================================================
SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_ID = 256
FIRST_MERGE_ID = 257
FORMAT_NAME = "urdu-bpe-v1"

# -----------------------------------------------------------------------------
# Pre-tokenisation pattern (explained in the report)
# -----------------------------------------------------------------------------
#   's|'t|'re|'ve|'m|'ll|'d      English contractions (as in GPT-2)
#   ?[\p{L}\p{M}\u200c\u200d]+   a WORD: letters AND combining marks (Urdu
#                                diacritics are \p{M}); ZWNJ/ZWJ are allowed
#                                inside a word so they never cut it in two.
#                                The optional leading space belongs to the word.
#   ?\p{N}+                      a run of digits (any script)
#   ?[^\s\p{L}\p{M}\p{N}\u200c\u200d]+   a run of punctuation / symbols / emoji
#   \s+(?!\S)|\s+                whitespace (all but the last space of a run
#                                stay separate so the last one can prefix the
#                                next word - same trick as GPT-2)
# Every character falls in exactly one class, so the split is lossless.
PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[\p{L}\p{M}\u200c\u200d]+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{M}\p{N}\u200c\u200d]+"
    r"|\s+(?!\S)|\s+"
)


# =============================================================================
# 3. SMALL HELPERS
# =============================================================================
def _tie_key(b: bytes):
    """Key whose ASCENDING order is the DESCENDING order of `b` (bytes order).

    Used so a min-heap pops the GREATER byte string first.  The terminator 256
    makes a shorter string (a prefix) sort after the longer one in this key,
    which matches 'prefix < longer' in normal bytes ordering.
    """
    return tuple(255 - x for x in b) + (256,)


def _merge_ids(ids, a, b, new_id):
    """Replace every non-overlapping (a, b) in `ids`, scanning left to right."""
    out = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _base_vocab():
    vocab = [bytes([i]) for i in range(256)]
    vocab.append(SPECIAL_TOKEN.encode("utf-8"))  # id 256
    return vocab


def _to_bytes(word: str) -> bytes:
    # surrogatepass => even lone surrogates (invalid Unicode) round-trip
    return word.encode("utf-8", "surrogatepass")


# =============================================================================
# 4. WORD COUNTING (pre-tokenisation of the training corpus)
# =============================================================================
def count_words(path, max_bytes=None, progress=True):
    """Stream `path`, pre-tokenise with PATTERN, return Counter{word_str: n}.

    The special token is removed first (it is never part of a word/merge).
    """
    pat = regex.compile(PATTERN)
    counts = Counter()
    done = 0
    import os

    total = os.path.getsize(path)
    if max_bytes is not None:
        total = min(total, max_bytes)
    bar = tqdm(total=total, unit="B", unit_scale=True, desc="counting words",
               disable=not progress) if tqdm else None
    with open(path, "rb") as f:
        for raw in f:
            if max_bytes is not None and done >= max_bytes:
                break
            line = raw.decode("utf-8", "replace")
            for part in line.split(SPECIAL_TOKEN):
                if part:
                    counts.update(pat.findall(part))
            done += len(raw)
            if bar:
                bar.update(len(raw))
    if bar:
        bar.close()
    return counts


# =============================================================================
# 5. TRAINERS
# =============================================================================
def train_reference(items, num_merges, verbose=False):
    """REFERENCE trainer: recount every pair from scratch after every merge.

    items: list of (word_bytes, frequency).  Returns list of (id_a, id_b).
    """
    words = [list(b) for b, c in items if len(b) >= 2]
    freqs = [c for b, c in items if len(b) >= 2]
    vocab = _base_vocab()
    merges = []
    for step in range(num_merges):
        counts = {}
        for w, f in zip(words, freqs):
            for p in zip(w, w[1:]):
                counts[p] = counts.get(p, 0) + f
        if not counts:
            break
        best = max(
            counts,
            key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]],
                           -((p[0] << 16) | p[1])),
        )
        new_id = len(vocab)
        vocab.append(vocab[best[0]] + vocab[best[1]])
        merges.append(best)
        for i, w in enumerate(words):
            words[i] = _merge_ids(w, best[0], best[1], new_id)
        if verbose and (step + 1) % 100 == 0:
            print(f"[reference] merge {step + 1}/{num_merges}")
    return merges


def train_fast(items, num_merges, verbose=True):
    """FAST trainer with the three optimisations of the assignment.

    1. every distinct word is stored once, pairs are weighted by its frequency
    2. after a merge only the words that contained the pair are touched
       (pair -> list-of-word-indices index)
    3. (encoding-time cache lives in BPETokenizer._word_ids)
    A lazy-deletion max-heap finds the best pair in O(log P).
    Pair (a, b) is stored as the int (a << 16) | b.
    """
    # ---- 5.1 build word table ------------------------------------------------
    words, freqs = [], []
    for b, c in items:
        if len(b) >= 2:  # one-byte words have no pairs
            words.append(list(b))
            freqs.append(c)

    vocab = _base_vocab()
    tkeys = [_tie_key(b) for b in vocab]

    # ---- 5.2 initial pair counts and pair -> words index ---------------------
    pair_counts = {}
    pair_index = {}
    for idx, w in enumerate(words):
        f = freqs[idx]
        for i in range(len(w) - 1):
            k = (w[i] << 16) | w[i + 1]
            pair_counts[k] = pair_counts.get(k, 0) + f
            lst = pair_index.get(k)
            if lst is None:
                pair_index[k] = [idx]
            elif lst[-1] != idx:  # words are processed in order -> dedupe
                lst.append(idx)

    heap = [(-c, tkeys[k >> 16], tkeys[k & 0xFFFF], k)
            for k, c in pair_counts.items()]
    heapq.heapify(heap)

    # ---- 5.3 merge loop ------------------------------------------------------
    merges = []
    t0 = time.time()
    for step in range(num_merges):
        best = None
        while heap:
            negc, _, _, k = heapq.heappop(heap)
            if negc < 0 and pair_counts.get(k, 0) == -negc:  # not stale
                best = k
                break
        if best is None:
            break
        a, b = best >> 16, best & 0xFFFF
        new_id = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        tkeys.append(_tie_key(vocab[new_id]))
        merges.append((a, b))

        changed = set()
        for idx in pair_index.pop(best, ()):
            w = words[idx]
            new_w = _merge_ids(w, a, b, new_id)
            if len(new_w) == len(w):  # pair no longer in this word
                continue
            f = freqs[idx]
            for i in range(len(w) - 1):  # remove old pair counts
                k = (w[i] << 16) | w[i + 1]
                pair_counts[k] -= f
                changed.add(k)
            seen_new = set()
            for i in range(len(new_w) - 1):  # add new pair counts
                x, y = new_w[i], new_w[i + 1]
                k = (x << 16) | y
                pair_counts[k] = pair_counts.get(k, 0) + f
                changed.add(k)
                if (x == new_id or y == new_id) and k not in seen_new:
                    seen_new.add(k)  # only brand-new pairs need indexing
                    lst = pair_index.get(k)
                    if lst is None:
                        pair_index[k] = [idx]
                    else:
                        lst.append(idx)
            words[idx] = new_w

        for k in changed:
            c = pair_counts.get(k, 0)
            if c > 0:
                heapq.heappush(heap, (-c, tkeys[k >> 16], tkeys[k & 0xFFFF], k))
            else:
                pair_counts.pop(k, None)

        if verbose and ((step + 1) % 500 == 0 or step + 1 == num_merges):
            el = time.time() - t0
            eta = el / (step + 1) * (num_merges - step - 1)
            print(f"[fast] merge {step + 1}/{num_merges}  "
                  f"elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min", flush=True)
    return merges


# =============================================================================
# 6. THE TOKENIZER CLASS
# =============================================================================
class BPETokenizer:
    """Byte-level BPE tokenizer: train / encode / decode / save / load."""

    def __init__(self, merges=None, cache_limit=2_000_000):
        self.pattern = PATTERN
        self._re = regex.compile(self.pattern)
        self.cache_limit = cache_limit
        self.set_merges(merges or [])

    # ---- 6.1 vocabulary ------------------------------------------------------
    def set_merges(self, merges):
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab[SPECIAL_ID] = SPECIAL_TOKEN.encode("utf-8")
        self.ranks = {}
        for r, (a, b) in enumerate(self.merges):
            self.vocab[FIRST_MERGE_ID + r] = self.vocab[a] + self.vocab[b]
            self.ranks[(a, b)] = r
        self._cache = {}

    @property
    def vocab_size(self):
        return FIRST_MERGE_ID + len(self.merges)

    def token_bytes(self, token_id):
        return self.vocab[token_id]

    # ---- 6.2 training --------------------------------------------------------
    def train(self, word_counts, vocab_size=50_000, method="fast",
              min_word_freq=1, verbose=True):
        """Learn `vocab_size - 257` merges from a Counter{word: count}."""
        if vocab_size < FIRST_MERGE_ID:
            raise ValueError("vocab_size must be at least 257")
        num_merges = vocab_size - FIRST_MERGE_ID
        items = sorted((_to_bytes(w), c) for w, c in word_counts.items()
                       if c >= min_word_freq)
        if verbose:
            print(f"training on {len(items):,} distinct words, "
                  f"{num_merges:,} merges, method={method}")
        if method == "fast":
            merges = train_fast(items, num_merges, verbose)
        elif method == "reference":
            merges = train_reference(items, num_merges, verbose)
        else:
            raise ValueError("method must be 'fast' or 'reference'")
        self.set_merges(merges)
        return merges

    def train_from_file(self, path, vocab_size=50_000, method="fast",
                        min_word_freq=1, max_bytes=None, verbose=True):
        counts = count_words(path, max_bytes=max_bytes, progress=verbose)
        return self.train(counts, vocab_size, method, min_word_freq, verbose)

    # ---- 6.3 encoding --------------------------------------------------------
    def _word_ids(self, word):
        """Token ids of one pre-token (str), with a cache of finished words."""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(_to_bytes(word))
        ranks = self.ranks
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            ids = _merge_ids(ids, best_pair[0], best_pair[1],
                             FIRST_MERGE_ID + best_rank)
        ids = tuple(ids)
        if len(self._cache) < self.cache_limit:
            self._cache[word] = ids
        return ids

    def encode(self, text, allow_special=True):
        """str -> list[int].  '<|endoftext|>' becomes id 256 (never split)."""
        if not text:
            return []
        out = []
        parts = text.split(SPECIAL_TOKEN) if allow_special else [text]
        for i, part in enumerate(parts):
            if i > 0:
                out.append(SPECIAL_ID)
            if part:
                for w in self._re.findall(part):
                    out.extend(self._word_ids(w))
        return out

    # ---- 6.4 decoding --------------------------------------------------------
    def decode(self, ids):
        """list[int] -> str.  Exact inverse of encode for any Unicode string."""
        try:
            data = b"".join(self.vocab[int(i)] for i in ids)
        except KeyError as e:
            raise ValueError(f"unknown token id {e}") from None
        try:
            return data.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:  # arbitrary id sequences may cut a character
            return data.decode("utf-8", "replace")

    # ---- 6.5 saving / loading ------------------------------------------------
    def save(self, path, readable_path=None):
        data = {
            "format": FORMAT_NAME,
            "vocab_size": self.vocab_size,
            "special_tokens": {SPECIAL_TOKEN: SPECIAL_ID},
            "pattern": self.pattern,
            "merges": [[a, b] for a, b in self.merges],  # in learned order
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        if readable_path:  # one line per merge: id, parent ids, token bytes (hex)
            with open(readable_path, "w", encoding="utf-8") as f:
                for r, (a, b) in enumerate(self.merges):
                    tid = FIRST_MERGE_ID + r
                    f.write(f"{tid}\t{a}\t{b}\t{self.vocab[tid].hex()}\n")

    @classmethod
    def load(cls, path, cache_limit=2_000_000):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != FORMAT_NAME:
            raise ValueError("not a tokenizer file written by this module")
        tok = cls(data["merges"], cache_limit=cache_limit)
        tok.pattern = data["pattern"]
        tok._re = regex.compile(tok.pattern)
        return tok
