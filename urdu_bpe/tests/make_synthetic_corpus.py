"""Create a SYNTHETIC Urdu-like corpus to smoke-test the whole pipeline offline.
(This is test data only - it is NOT a substitute for the real corpus.)"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import json
import os
import random
import sys

WORDS = ("پاکستان حکومت وزیر اعظم نے آج کہا کہ ملک میں عوام کی خدمت کے لیے ہر ممکن اقدامات کیے "
         "جائیں گے اور کا ہے ہیں سے کو پر یہ وہ ایک بھی تھا تھی لوگ شہر لاہور کراچی اسلام آباد "
         "کرکٹ ٹیم میچ جیت سیریز اپنے نام کیا وقت دنیا کتاب علم تعلیم طلبہ سکول یونیورسٹی معیشت "
         "بجٹ روپے قیمت بازار موسم بارش سردی گرمی صحت ہسپتال ڈاکٹر مریض علاج زبان اردو شاعری غزل "
         "محبت دل آنکھ خواب رات دن صبح شام سڑک گاڑی ریل سفر خبر اخبار انٹرنیٹ موبائل کمپیوٹر نیا "
         "پرانا بڑا چھوٹا اچھا برا بہت کم زیادہ کل پھر لیکن اگر تو کیونکہ جب کوئی کچھ سب ہم آپ "
         "اُردُو زَبَر زِیر بھائی کھانا پھول ٹھنڈا ڈاکٹر ٹیکنالوجی").split()
ARABIC_VARIANTS = {"ی": "ي", "ک": "ك", "ہ": "ه"}


def sentence(rng):
    ws = [rng.choice(WORDS) for _ in range(rng.randint(6, 18))]
    s = " ".join(ws)
    r = rng.random()
    if r < 0.1:  # Arabic-letter typing variants
        s = "".join(ARABIC_VARIANTS.get(c, c) for c in s)
    elif r < 0.15:
        s = s.replace("ک", "ک\u0640", 1)
    elif r < 0.2:
        s = s + " " + str(rng.randint(1, 3000))
    elif r < 0.25:
        s = s.replace("0", "٠")
    return s + "۔"


def main(out_dir="raw", n_docs=2500, seed=5):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    boiler = ["شیئر کریں", "Copyright 2026 All rights reserved", "Home | About | Contact",
              "ہمارے بارے میں", "مزید پڑھیں"]
    persian = "این یک جمله فارسی است که در اینجا نوشته شده است و هیچ ربطی ندارد"
    with open(os.path.join(out_dir, "synth_news.jsonl"), "w", encoding="utf-8") as f1, \
            open(os.path.join(out_dir, "synth_wiki.jsonl"), "w", encoding="utf-8") as f2:
        for i in range(n_docs):
            paras = [sentence(rng) + " " + sentence(rng) for _ in range(rng.randint(2, 6))]
            if i % 7 == 0:
                paras.insert(0, "<div class='x'><b>Breaking news</b> read more at https://example.com/x?id=1</div>")
            if i % 5 == 0:
                paras.append(rng.choice(boiler))
            if i % 13 == 0:
                paras.append(persian)
            if i % 17 == 0:
                paras.append("== حوالہ جات ==")
            doc = "\n\n".join(paras)
            rec = {"text": doc}
            (f1 if i % 3 else f2).write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:  # exact duplicate document
                (f1 if i % 3 else f2).write(json.dumps(rec, ensure_ascii=False) + "\n")
    sources = [
        {"name": "synth_news", "path": os.path.join(out_dir, "synth_news.jsonl"),
         "format": "jsonl", "field": "text", "genre": "news", "url": "synthetic",
         "licence": "n/a", "collection_date": "2026-10-11", "own_collection": True},
        {"name": "synth_wiki", "path": os.path.join(out_dir, "synth_wiki.jsonl"),
         "format": "jsonl", "field": "text", "genre": "encyclopedia", "url": "synthetic",
         "licence": "n/a", "collection_date": "2026-10-11", "own_collection": False},
    ]
    with open(os.path.join(out_dir, "synth_sources.json"), "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["raw"]))
