#!/usr/bin/env python3
"""Download open PhysioNet records without committing datasets to Git."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://physionet.org/files"
VERSIONS = {"mitdb": "1.0.0", "nstdb": "1.0.0", "incartdb": "1.0.0"}


def get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "iot-ecg-research/0.1"})
    with urlopen(req) as response:
        return response.read().decode("utf-8")


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists  {dest}")
        return
    req = Request(url, headers={"User-Agent": "iot-ecg-research/0.1"})
    with urlopen(req) as response, dest.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"saved   {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", choices=VERSIONS, default="mitdb")
    ap.add_argument("--records", nargs="*")
    ap.add_argument("--all-records", action="store_true")
    ap.add_argument("--out", default="data/physionet")
    args = ap.parse_args()

    db, version = args.database, VERSIONS[args.database]
    root = f"{BASE}/{db}/{version}"
    if args.all_records:
        records = [x.strip() for x in get_text(f"{root}/RECORDS").splitlines() if x.strip()]
    elif args.records:
        records = args.records
    else:
        records = ["100"] if db == "mitdb" else []
        if not records:
            raise SystemExit("Specify --records or --all-records for this database")

    dest_root = Path(args.out) / db
    for record in records:
        for ext in ("hea", "dat", "atr"):
            url = f"{root}/{record}.{ext}"
            try:
                download(url, dest_root / f"{record}.{ext}")
            except Exception as exc:
                print(f"skip    {url}: {exc}")


if __name__ == "__main__":
    main()
