#!/usr/bin/env python3
"""
collect_web.py
==============
OWN COLLECTION #2: polite scraper for Urdu news sites / blogs / public-domain
books given a sitemap (or a plain list of URLs).

Rules enforced by the script
  * robots.txt is fetched for every host; disallowed URLs are skipped and a
    Crawl-delay larger than --delay is honoured
  * identifying User-Agent (--contact is mandatory)
  * sequential requests with a delay; no login, no paywall bypass
    (HTTP 401/402/403 pages are skipped)
  * every request is logged in an evidence CSV (URL, UTC time, status, sizes, SHA-256)

YOU must still read the site's terms of use and record the licence in
sources.json - this script cannot do that for you.

Example:
    python collect_web.py --name example_news --sitemap https://example.com/sitemap.xml \
        --contact you@university.edu --max-mb 60
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import csv
import datetime
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

_LOC_RE = re.compile(r"<loc>\s*(?:<!\[CDATA\[)?\s*([^<\]\s]+)\s*(?:\]\]>)?\s*</loc>", re.I)
_DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside",
              "form", "iframe", "svg", "button", "select"]


# =============================================================================
# 2. ROBOTS.TXT
# =============================================================================
class Robots:
    def __init__(self, session, ua):
        self.session, self.ua, self.cache = session, ua, {}

    def _get(self, url):
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.session.get(base + "/robots.txt", timeout=30)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                elif r.status_code in (401, 403):
                    rp.disallow_all = True
                else:
                    rp.allow_all = True
            except requests.RequestException:
                rp.disallow_all = True  # be conservative if it cannot be read
            self.cache[base] = rp
        return self.cache[base]

    def allowed(self, url):
        return self._get(url).can_fetch(self.ua, url)

    def crawl_delay(self, url):
        return self._get(url).crawl_delay(self.ua) or 0


# =============================================================================
# 3. SITEMAP + PAGE EXTRACTION
# =============================================================================
def fetch_text(session, url):
    r = session.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    if url.endswith(".gz") or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", "replace")


def sitemap_urls(session, robots, url, depth=0, limit=200000):
    """Return page URLs found in a sitemap (recursing into sitemap indexes)."""
    out = []
    if depth > 3 or not robots.allowed(url):
        return out
    try:
        locs = _LOC_RE.findall(fetch_text(session, url))
    except (requests.RequestException, OSError):
        return out
    for loc in locs:
        if re.search(r"\.xml(\.gz)?$", loc, re.I) or "sitemap" in loc.lower():
            out.extend(sitemap_urls(session, robots, loc, depth + 1, limit))
        else:
            out.append(loc)
        if len(out) >= limit:
            break
    return out


def extract_article(html_text):
    """Return (title, plain text) from an HTML page using only <p>/<h1-3> text."""
    soup = BeautifulSoup(html_text, "html.parser")
    for t in soup(_DROP_TAGS):
        t.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    box = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in box.find_all(["h1", "h2", "h3", "p"])]
    return title, "\n".join(p for p in paras if len(p) > 1)


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="short source name, used for file names")
    ap.add_argument("--sitemap", action="append", default=[], help="sitemap URL (repeatable)")
    ap.add_argument("--urls-file", help="text file with one URL per line")
    ap.add_argument("--contact", required=True)
    ap.add_argument("--max-mb", type=float, default=60)
    ap.add_argument("--max-urls", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--url-contains", default="", help="keep only URLs containing this text")
    ap.add_argument("--out-dir", default="raw")
    args = ap.parse_args()

    ua = f"UrduBPEAssignment/1.0 (student coursework; {args.contact})"
    session = requests.Session()
    session.headers["User-Agent"] = ua
    robots = Robots(session, ua)

    urls = []
    for sm in args.sitemap:
        urls.extend(sitemap_urls(session, robots, sm))
    if args.urls_file:
        urls.extend(u.strip() for u in open(args.urls_file, encoding="utf-8") if u.strip())
    if args.url_contains:
        urls = [u for u in urls if args.url_contains in u]
    urls = list(dict.fromkeys(urls))
    if args.max_urls:
        urls = urls[:args.max_urls]
    print(f"{len(urls):,} candidate URLs")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.jsonl")
    ev_path = os.path.join(args.out_dir, f"{args.name}.evidence.csv")
    done = set()
    new_ev = not os.path.exists(ev_path)
    if not new_ev:  # resume: skip URLs already requested
        with open(ev_path, encoding="utf-8", newline="") as f:
            done = {row[0] for row in csv.reader(f)}
    total = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            total = sum(len(json.loads(l)["text"].encode("utf-8")) for l in f if l.strip())
    target = args.max_mb * 2**20

    with open(out_path, "a", encoding="utf-8") as fout, \
            open(ev_path, "a", encoding="utf-8", newline="") as fev:
        ev = csv.writer(fev)
        if new_ev:
            ev.writerow(["url", "fetched_utc", "http_status", "html_bytes",
                         "text_bytes", "text_sha256"])
        it = tqdm(urls, unit="page", desc=args.name) if tqdm else urls
        for url in it:
            if total >= target:
                break
            if url in done or not robots.allowed(url):
                continue
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            try:
                r = session.get(url, timeout=60)
            except requests.RequestException:
                ev.writerow([url, now, "error", 0, 0, ""])
                continue
            text, title = "", ""
            ctype = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "html" in ctype:
                title, text = extract_article(r.text)
            if len(text) >= args.min_chars:
                nbytes = len(text.encode("utf-8"))
                fout.write(json.dumps({"url": url, "title": title, "text": text},
                                      ensure_ascii=False) + "\n")
                total += nbytes
                ev.writerow([url, now, r.status_code, len(r.content), nbytes,
                             hashlib.sha256(text.encode("utf-8")).hexdigest()])
            else:
                ev.writerow([url, now, r.status_code, len(r.content), 0, ""])
            fout.flush()
            fev.flush()
            time.sleep(max(args.delay, robots.crawl_delay(url)))
    print(f"collected {total / 2**20:.1f} MB -> {out_path}; evidence -> {ev_path}")


if __name__ == "__main__":
    sys.exit(main())
