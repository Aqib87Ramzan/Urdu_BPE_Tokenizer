#!/usr/bin/env python3
"""
collect_wikipedia.py
====================
OWN COLLECTION #1: Urdu Wikipedia through the official MediaWiki API
(action=query, generator=allpages, prop=extracts, explaintext).

Wikipedia text is CC BY-SA 4.0 (record this in the data card).  The Wikimedia
API etiquette requires an identifying User-Agent with contact information, so
--contact is mandatory.  Requests are sequential with a delay and `maxlag`.

Outputs
    raw/urdu_wikipedia.jsonl         one {"pageid","title","text"} per article
    raw/urdu_wikipedia.evidence.csv  one row per article fetched (proof of collection)
    raw/urdu_wikipedia.state.json    resume point

Example:
    python collect_wikipedia.py --contact you@university.edu --target-mb 120
"""
# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import csv
import datetime
import json
import os
import sys
import time

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# =============================================================================
# 2. API CALL WITH RETRIES
# =============================================================================
def api_get(session, url, params, retries=6):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            err = data.get("error")
            if err:  # maxlag, ratelimited, internal errors ...: wait and retry, never ignore
                print(f"\nAPI error '{err.get('code')}': {err.get('info', '')[:120]} - retrying", flush=True)
                time.sleep(10 + 5 * attempt)
                continue
            return data
        except (requests.RequestException, ValueError):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("API request kept failing; try again later")


# =============================================================================
# 3. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ur")
    ap.add_argument("--contact", required=True, help="e-mail/URL put in the User-Agent")
    ap.add_argument("--out", default="raw/urdu_wikipedia.jsonl")
    ap.add_argument("--evidence", default="raw/urdu_wikipedia.evidence.csv")
    ap.add_argument("--state", default="raw/urdu_wikipedia.state.json")
    ap.add_argument("--target-mb", type=float, default=0, help="stop after N MB (0 = all)")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--full-text", action="store_true",
                    help="whole articles (the API then returns ONE article per request: very slow). "
                         "Default: article introductions, up to 20 per request")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    url = f"https://{args.lang}.wikipedia.org/w/api.php"
    session = requests.Session()
    session.headers["User-Agent"] = f"UrduBPEAssignment/1.0 (student coursework; {args.contact})"

    params = {"action": "query", "format": "json", "formatversion": 2,
              "generator": "allpages", "gapnamespace": 0, "gaplimit": 20,
              "gapfilterredir": "nonredirects", "prop": "extracts|info",
              "explaintext": 1, "exlimit": "max", "maxlag": 5}
    if not args.full_text:
        params["exintro"] = 1
    total_bytes, seen = 0, set()
    mode = "w"
    if args.resume and os.path.exists(args.state):
        st = json.load(open(args.state, encoding="utf-8"))
        params.update(st["continue"])
        total_bytes = st["bytes"]
        mode = "a"
        print(f"resuming at {st['continue']} with {total_bytes / 2**20:.1f} MB")
    target = args.target_mb * 2**20

    new_evidence = mode == "w" or not os.path.exists(args.evidence)
    with open(args.out, mode, encoding="utf-8") as fout, \
            open(args.evidence, mode, encoding="utf-8", newline="") as fev:
        ev = csv.writer(fev)
        if new_evidence:
            ev.writerow(["pageid", "title", "revid", "fetched_utc", "text_bytes"])
        bar = tqdm(total=int(target) or None, unit="B", unit_scale=True,
                   initial=total_bytes, desc="wikipedia") if tqdm else None
        while True:
            data = api_get(session, url, params)
            for p in data.get("query", {}).get("pages", []):
                text = p.get("extract") or ""
                pid = p.get("pageid")
                if not text or pid in seen:
                    continue
                seen.add(pid)
                nbytes = len(text.encode("utf-8"))
                fout.write(json.dumps({"pageid": pid, "title": p.get("title"),
                                       "text": text}, ensure_ascii=False) + "\n")
                ev.writerow([pid, p.get("title"), p.get("lastrevid"),
                             datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                             nbytes])
                total_bytes += nbytes
                if bar is not None:
                    bar.update(nbytes)
            fout.flush()
            fev.flush()
            cont = data.get("continue")
            if cont:
                with open(args.state, "w", encoding="utf-8") as fs:
                    json.dump({"continue": cont, "bytes": total_bytes}, fs)
            if not cont:
                print("\nreached the end of the page list (no more results)")
                break
            if target and total_bytes >= target:
                print("\nreached the target size")
                break
            params.update(cont)
            time.sleep(args.delay)
        if bar is not None:
            bar.close()
    print(f"collected {total_bytes / 2**20:.1f} MB from {len(seen):,} articles")


if __name__ == "__main__":
    sys.exit(main())
