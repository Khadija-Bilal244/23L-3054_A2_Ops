"""Task 1.3 - visual EDA: sample grids + histograms per class.  python -m src.eda"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def grid(df, cls, n=12, out="reports/figures"):
    s = df[(df.cls == cls) & ~df.corrupt].sample(n, random_state=0)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, p in zip(axes.ravel(), s.path):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.axis("off")
    fig.suptitle(f"Random samples - class '{cls}'")
    fig.tight_layout()
    fig.savefig(f"{out}/samples_{cls}.png", dpi=120)
    plt.close(fig)


def hists(df, out):
    ok = df[~df.corrupt]
    for col in ["width", "height", "file_size"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for cls, g in ok.groupby("cls"):
            ax.hist(g[col], bins=40, alpha=0.55, label=cls)
        ax.set_xlabel(col)
        ax.set_ylabel("images")
        ax.legend()
        ax.set_title(f"{col} by class")
        fig.tight_layout()
        fig.savefig(f"{out}/hist_{col}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="reports/audit.csv")
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.audit)
    for c in ["ai", "real"]:
        grid(df, c, out=a.out)
    hists(df, a.out)
    print("figures written to", a.out)
