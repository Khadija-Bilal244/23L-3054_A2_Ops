"""Task 2.4-2.6 helpers: threshold choice, metrics, figures, error grid."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score, roc_curve)


def choose_threshold(y_val, p_val, beta=1.0):
    """Pick the threshold on VALIDATION data only (never on test).
    beta>1 favours recall of 'AI' (catch more fakes, flag more real photos);
    beta<1 favours precision (rarely accuse a real photo of being fake)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y_val, p_val >= t, beta=beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p), threshold=thr)


def save_confusion(y, p, thr, path):
    cm = confusion_matrix(y, (p >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["real", "ai"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix (thr={thr:.2f})")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return cm


def save_roc(y, p, path, label=""):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"{label} AUC={roc_auc_score(y, p):.3f}")
    plt.plot([0, 1], [0, 1], "--", c="grey")
    plt.xlabel("False positive rate (real called AI)")
    plt.ylabel("True positive rate (AI caught)")
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_curves(hist, best_epoch, path, stage_boundary=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k in zip(ax, ["loss", "accuracy"]):
        a.plot(hist[k], label="train")
        a.plot(hist[f"val_{k}"], label="val")
        a.axvline(best_epoch, c="r", ls="--", label=f"best epoch {best_epoch}")
        if stage_boundary:
            a.axvline(stage_boundary - 0.5, c="grey", ls=":", label="fine-tune starts")
        a.set_xlabel("epoch")
        a.set_title(k)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_error_grid(test_csv, p, thr, path, n=10):
    df = pd.read_csv(test_csv)
    df["p"] = p
    df["pred"] = (p >= thr).astype(int)
    wrong = df[df.pred != df.label].copy()
    wrong["conf"] = (wrong.p - thr).abs()
    wrong = wrong.sort_values("conf", ascending=False).head(n)  # most confidently wrong first
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        ax.imshow(Image.open(r.path).convert("RGB"))
        ax.set_title(f"true={'ai' if r.label else 'real'} p_ai={r.p:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    wrong[["path", "label", "p"]].to_csv(path.replace(".png", ".csv"), index=False)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
