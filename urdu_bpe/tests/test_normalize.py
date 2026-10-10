"""Unit tests for normalize.py."""
# =============================================================================
# 1. IMPORTS / PATHS
# =============================================================================
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from normalize import normalize  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_yeh_kaf_heh(self):
        self.assertEqual(normalize("\u064a\u0649"), "\u06cc\u06cc")
        self.assertEqual(normalize("\u0643"), "\u06a9")
        self.assertEqual(normalize("\u0647"), "\u06c1")

    def test_do_chashmi_and_heh_goal_kept_apart(self):
        self.assertEqual(normalize("\u06be"), "\u06be")
        self.assertEqual(normalize("\u06c1"), "\u06c1")

    def test_tatweel_and_invisibles(self):
        self.assertEqual(normalize("ک\u0640ت\u200dا\u200eب\ufeff"), "کتاب")
        self.assertEqual(normalize("ہ\u200cم"), "ہم")  # ZWNJ removed

    def test_digits_ascii(self):
        self.assertEqual(normalize("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹0123456789"), "0123456789" * 3)

    def test_whitespace(self):
        self.assertEqual(normalize("a\u00a0 \u00a0b\r\n\r\n \n c\t"), "a b\nc")

    def test_hamza_composition_not_destroyed(self):
        # yeh + hamza above composes to U+0626 and must NOT become plain yeh
        self.assertEqual(normalize("\u064a\u0654"), "\u0626")
        self.assertEqual(normalize("\u06c1\u0654"), "\u06c2")
        self.assertEqual(normalize("\u0627\u0653"), "\u0622")

    def test_diacritics_kept(self):
        self.assertEqual(normalize("اُردُو"), "اُردُو")

    def test_presentation_forms(self):
        self.assertEqual(normalize("\ufedf\ufef0"), "\u0644\u06cc")  # lam, alef maksura

    def test_idempotent(self):
        s = "كـتاب ي  ه\u200c\n\n۱۲٣"
        self.assertEqual(normalize(normalize(s)), normalize(s))

    def test_empty(self):
        self.assertEqual(normalize(""), "")


if __name__ == "__main__":
    unittest.main()
