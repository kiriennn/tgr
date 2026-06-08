"""Download Amazon Product Reviews (2014, McAuley) for a given category.

Default URLs point at the classic 5-core review files + metadata used by SASRec
and TIGER. For leakage-clean temporal preprocessing, pass ``--review-set all``
to download the unfiltered category reviews and then run
``preprocess.py --kcore-scope train_temporal``. If a mirror is down, pass
--reviews-url / --meta-url, or download manually and place the .json.gz files in
data/raw/.

Mirrors that have hosted these files:
  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon/  (2014)
  https://jmcauley.ucsd.edu/data/amazon/

Note: this requires internet access to the hosts above, which is why it cannot
run inside the offline build sandbox -- run it on your own machine / Colab.
"""
from __future__ import annotations

import argparse
import os
import urllib.request

BASE = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"


def download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"[skip] {dest} exists")
        return
    print(f"[get ] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Beauty",
                    help="Beauty | Sports_and_Outdoors | Toys_and_Games")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--review-set", choices=["5core", "all"], default="5core",
                    help="5core downloads reviews_<category>_5.json.gz; all "
                         "downloads reviews_<category>.json.gz for clean "
                         "temporal preprocessing")
    ap.add_argument("--reviews-url", default=None)
    ap.add_argument("--meta-url", default=None)
    args = ap.parse_args()

    suffix = "_5" if args.review_set == "5core" else ""
    reviews_url = args.reviews_url or f"{BASE}/reviews_{args.category}{suffix}.json.gz"
    meta_url = args.meta_url or f"{BASE}/meta_{args.category}.json.gz"
    download(reviews_url, f"{args.out}/reviews_{args.category}{suffix}.json.gz")
    download(meta_url, f"{args.out}/meta_{args.category}.json.gz")
    print("done.")


if __name__ == "__main__":
    main()
