# CS 417 - Assignment 1: Urdu BPE tokenizer from scratch

Everything below is run from this folder. Python 3.10+.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v          # 26 tests: tokenizer + normaliser
```

## Part A - corpus
```bash
# 1. own collection >= 50 MB (keeps evidence CSVs in raw/)
python collect_wikipedia.py --contact YOUR_EMAIL --target-mb 120
python collect_web.py --name my_news_site --sitemap https://SITE/sitemap.xml --contact YOUR_EMAIL --max-mb 60
 python collect_cc100.py
# 2. download published corpora (CC-100, UFAL, ...) into published/, then list ALL sources
cp sources.example.json sources.json             # edit paths, licences, collection dates
# 3. clean + dedupe + split 98/1/1 (writes data/train.txt, val.txt, test.txt, corpus_stats.json)
python prepare_corpus.py --sources sources.json --out-dir data
# 4. data card (then read it and add your own notes under "Known gaps and biases")
python make_data_card.py --sources sources.json --stats data/corpus_stats.json
```

## Part B - tokenizer
```bash
# fast trainer == reference trainer on a 1 MB sample (required check)
python verify_equivalence.py --train data/train.txt --sample-mb 1 --merges 1000
# train the real tokenizer: 50,000 tokens = 256 bytes + <|endoftext|> + 49,743 merges
python train_tokenizer.py --train data/train.txt --out tokenizer.json
# encode the corpus to flat uint16 arrays
python encode_corpus.py --tokenizer tokenizer.json --input data/train.txt --output data/train.bin
python encode_corpus.py --tokenizer tokenizer.json --input data/val.txt   --output data/val.bin
```
Memory: if training runs out of RAM, add `--min-word-freq 2` (ignores words seen once); report that you did.
Training prints progress/ETA and writes `training_log.json` (wall-clock time + machine for the report).

## Part C - report
```bash
python analyze_tokenizer.py --tokenizer tokenizer.json --test data/test.txt
python build_report.py            # first run writes analysis/report_text.json (DRAFT wording)
#  -> edit analysis/report_text.json in your own words, then:
python build_report.py            # report.html + report.pdf (Chrome/Edge/WeasyPrint); check <= 4 pages
```

## Data format notes
* Each document is followed directly by `<|endoftext|>` (no newline after it), documents are normalised first.
* `tokenizer.json`: `merges` = list of `[id_a, id_b]` in learned order; merge *k* creates id `256 + k`.
* `.bin` files: raw uint16, read with `np.fromfile(path, dtype=np.uint16)`.
* Normalisation (`normalize.py`) is applied to the corpus BEFORE training and must also be applied to any
  text before it is encoded with the tokenizer; the tokenizer itself is lossless on raw strings.
