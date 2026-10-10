#!/usr/bin/env bash
# Full rebuild from raw sources (see README.md). Stops at the first error.
set -euo pipefail
python -m unittest discover -s tests
python prepare_corpus.py --sources sources.json --out-dir data
python make_data_card.py --sources sources.json --stats data/corpus_stats.json
python verify_equivalence.py --train data/train.txt --sample-mb 1 --merges 1000
python train_tokenizer.py --train data/train.txt --out tokenizer.json
python encode_corpus.py --tokenizer tokenizer.json --input data/train.txt --output data/train.bin
python encode_corpus.py --tokenizer tokenizer.json --input data/val.txt --output data/val.bin
python analyze_tokenizer.py --tokenizer tokenizer.json --test data/test.txt
python build_report.py
