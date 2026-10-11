
#!/usr/bin/env python3
import argparse
import os
import requests
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Direct corpus download URL")
    parser.add_argument(
        "--out", default="published/urdu_monolingual.txt.gz"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with requests.get(args.url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(args.out, "wb") as f:
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc="UFAL Urdu"
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
