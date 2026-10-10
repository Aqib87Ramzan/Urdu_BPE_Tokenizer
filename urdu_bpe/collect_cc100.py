
#!/usr/bin/env python3
import argparse
import os
import requests
from tqdm import tqdm

URL = "https://data.statmt.org/cc-100/ur.txt.xz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="published/ur.txt.xz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with requests.get(URL, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(args.out, "wb") as f:
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc="CC-100 Urdu"
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    print(f"Downloaded CC-100 Urdu to {args.out}")


if __name__ == "__main__":
    main()
