"""Unit tests for bpe.py  (run: python -m unittest discover -s tests -v)."""
# =============================================================================
# 1. IMPORTS / PATHS
# =============================================================================
import os
import random
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import regex  # noqa: E402

from bpe import (BPETokenizer, PATTERN, SPECIAL_ID, SPECIAL_TOKEN,  # noqa: E402
                 _to_bytes, train_fast, train_reference)

URDU = ("پاکستان حکومت وزیر اعظم کتاب علم تعلیم اُردُو زبان شاعری محبت دل "
        "خواب رات دن صبح شام سفر خبر اخبار بہت اچھا ہے کا کی کے میں اور").split()


def make_counts(n_words=4000, seed=1):
    rng = random.Random(seed)
    c = Counter()
    for _ in range(n_words):
        w = " " + rng.choice(URDU) + (rng.choice(URDU) if rng.random() < .3 else "")
        c[w] += 1
    c["۔"] += 50
    c[" English"] += 30
    return c


class TestPretokenizer(unittest.TestCase):
    def setUp(self):
        self.pat = regex.compile(PATTERN)

    def test_diacritics_stay_whole(self):
        self.assertEqual(self.pat.findall("اُردُو"), ["اُردُو"])

    def test_zwnj_does_not_split_word(self):
        w = "ہائے\u200cجی"
        self.assertEqual(self.pat.findall(w), [w])

    def test_leading_space_belongs_to_next_word(self):
        self.assertEqual(self.pat.findall("میں کتاب"), ["میں", " کتاب"])

    def test_no_merge_across_punctuation(self):
        self.assertEqual(self.pat.findall("کتاب۔"), ["کتاب", "۔"])

    def test_split_is_lossless(self):
        rng = random.Random(0)
        alphabet = list("aبپ ُ\n\t 1۲،۔!😀\u200c\u200d'sx") + ["\ud800"]
        for _ in range(500):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
            self.assertEqual("".join(self.pat.findall(s)), s)


class TestTrainers(unittest.TestCase):
    def test_classic_example_and_tie_break(self):
        items = [(b"aaabdaaabac", 1)]
        m = train_fast(items, 2, verbose=False)
        self.assertEqual(m[0], (97, 97))
        # (aa,a) and (a,b) both have count 2: the greater byte-tuple wins
        self.assertEqual(m[1], (257, 97))
        self.assertEqual(m, train_reference(items, 2))

    def test_fast_equals_reference(self):
        counts = make_counts()
        items = sorted((_to_bytes(w), c) for w, c in counts.items())
        n = 300
        self.assertEqual(train_fast(items, n, verbose=False),
                         train_reference(items, n))

    def test_fast_equals_reference_with_repeated_chars(self):
        items = [(b"aaaa", 5), (b"aaaaa", 3), (b"abab", 4), (b"baaab", 2)]
        self.assertEqual(train_fast(items, 40, verbose=False),
                         train_reference(items, 40))

    def test_special_token_never_merged(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w", encoding="utf-8") as f:
                for _ in range(200):
                    f.write("اردو کتاب " + SPECIAL_TOKEN + "\n")
            tok = BPETokenizer()
            tok.train_from_file(p, vocab_size=300, verbose=False)
        for a, b in tok.merges:
            self.assertNotEqual(a, SPECIAL_ID)
            self.assertNotEqual(b, SPECIAL_ID)
        for i, b in tok.vocab.items():
            if i != SPECIAL_ID:
                self.assertNotIn(b"<|endoftext|>", b)

    def test_ids_and_vocab_size(self):
        tok = BPETokenizer()
        tok.train(make_counts(), vocab_size=400, verbose=False)
        self.assertEqual(tok.vocab_size, 400)
        self.assertEqual(len(tok.merges), 400 - 257)
        self.assertEqual(tok.vocab[256], b"<|endoftext|>")
        self.assertEqual(tok.encode("کتاب", allow_special=False)[0] < 400, True)


class TestRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer()
        cls.tok.train(make_counts(), vocab_size=500, verbose=False)

    def test_samples(self):
        samples = ["", " ", "\n", "\n\n  \t", "hello world", "کتاب", "اُردُو زبان",
                   "ہائے\u200cجی", "😀🌧️ emoji", "a\u0000b", "mixed میں English 123 ۱۲۳",
                   "  leading and trailing  ", "line1\nline2\r\nline3",
                   "\ud800", "x\udfffy", SPECIAL_TOKEN, "a" + SPECIAL_TOKEN + "b",
                   "'s 're 'll", "٠١٢٣٤٥٦٧٨٩", "﷽ ﻻ"]
        for s in samples:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s, repr(s))
            self.assertEqual(self.tok.decode(self.tok.encode(s, allow_special=False)), s)

    def test_random_unicode(self):
        rng = random.Random(7)
        for _ in range(300):
            s = "".join(chr(rng.choice([rng.randint(0, 0x2FF), rng.randint(0x600, 0x6FF),
                                       rng.randint(0x1F300, 0x1F64F), 32, 10]))
                        for _ in range(rng.randint(0, 40)))
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_special_token_is_one_id(self):
        self.assertEqual(self.tok.encode(SPECIAL_TOKEN), [SPECIAL_ID])
        self.assertEqual(self.tok.encode("a" + SPECIAL_TOKEN + "b").count(SPECIAL_ID), 1)

    def test_four_letter_word_starts_as_eight_bytes(self):
        self.assertEqual(len(_to_bytes("کتاب")), 8)

    def test_cache_does_not_change_result(self):
        s = "پاکستان کتاب علم " * 20
        a = self.tok.encode(s)
        b = BPETokenizer(self.tok.merges, cache_limit=0).encode(s)
        self.assertEqual(a, b)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tok.json")
            self.tok.save(p, os.path.join(d, "m.txt"))
            t2 = BPETokenizer.load(p)
        self.assertEqual(t2.merges, self.tok.merges)
        s = "پاکستان اور اُردُو 123 😀"
        self.assertEqual(t2.encode(s), self.tok.encode(s))


if __name__ == "__main__":
    unittest.main()
