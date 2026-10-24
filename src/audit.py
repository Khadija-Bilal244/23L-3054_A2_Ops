"""Task 1.2 - dataset audit. Builds one row per image + prints the report tables.

python -m src.audit --root data/raw --out reports/audit.csv
Class = which folder the image sits in (path part starting with --ai-key / --real-key).
"""
import argparse
import hashlib
import os

import pandas as pd
from PIL import Image

from src.config import IMG_EXT


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def infer_class(rel_path, ai_key, real_key):
    for part in rel_path.lower().split(os.sep)[:-1]:
        if part.startswith(ai_key):
            return "ai"
        if part.startswith(real_key):
            return "real"
    return "unknown"


def audit(root, ai_key="ai", real_key="real"):
    rows = []
    for dp, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            row = dict(path=p, cls=infer_class(rel, ai_key, real_key),
                       file_size=os.path.getsize(p), width=None, height=None,
                       mode=None, format=None, sha256=sha256(p), corrupt=False)
            if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                row["corrupt"] = True  # not an image extension -> treat as unusable
            else:
                try:
                    with Image.open(p) as im:
                        im.verify()  # detects truncated / damaged files
                    with Image.open(p) as im:  # verify() invalidates the handle -> reopen
                        row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                        im.load()
                except Exception:
                    row["corrupt"] = True
            rows.append(row)
    df = pd.DataFrame(rows)
    df["is_duplicate"] = df.duplicated("sha256", keep="first")  # exact byte-identical copies
    return df


def report(df):
    print("\n== images per class ==")
    print(df.groupby("cls").size())
    ok = df[~df.corrupt]
    print("\n== min / median / max per class ==")
    print(ok.groupby("cls")[["width", "height", "file_size"]].agg(["min", "median", "max"]).T)
    print("\ncolour modes:\n", ok.groupby(["cls", "mode"]).size())
    print("formats:\n", ok.groupby(["cls", "format"]).size())
    print(f"\ncorrupt/unreadable files: {int(df.corrupt.sum())}")
    print(f"exact duplicates (extra copies): {int(df.is_duplicate.sum())}")
    # duplicates that sit in BOTH classes are label noise - flag them
    g = df.groupby("sha256").cls.nunique()
    print(f"hashes appearing in more than one class: {int((g > 1).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="reports/audit.csv")
    ap.add_argument("--ai-key", default="ai")
    ap.add_argument("--real-key", default="real")
    a = ap.parse_args()
    d = audit(a.root, a.ai_key, a.real_key)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d.to_csv(a.out, index=False)
    report(d)
