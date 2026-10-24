"""Splits (stratified, de-duplicated) + tf.data pipeline (Task 2.1).

python -m src.data --audit reports/audit.csv --out splits
"""
import argparse
import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import SEED


def make_splits(audit_csv, out_dir, seed=SEED):
    df = pd.read_csv(audit_csv)
    df = df[(~df.corrupt) & (~df.is_duplicate) & (df.cls.isin(["ai", "real"]))].copy()
    df["label"] = (df.cls == "ai").astype(int)  # 1 = AI-generated
    # one hash -> one split, so identical images can never leak across splits
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=seed)
    os.makedirs(out_dir, exist_ok=True)
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        part[["path", "label"]].to_csv(f"{out_dir}/{name}.csv", index=False)
        print(name, len(part), part.label.value_counts().to_dict())


def _augmenter(seed):
    # Random, label-preserving changes. Applied to TRAIN only.
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, seed=seed),
        tf.keras.layers.RandomZoom(0.1, seed=seed),
        tf.keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augment")


def make_ds(csv, size=224, batch=32, train=False, augment=True, seed=SEED):
    """csv -> shuffle (train) -> decode+resize -> augment (train) -> batch -> prefetch.

    Output pixels are float32 in [0, 255]; each model carries its OWN preprocessing
    layer inside the graph, so serving only has to decode+resize identically.
    """
    df = pd.read_csv(csv)
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values.astype("float32")))
    if train:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        # Plain resize to size x size: aspect ratio is NOT preserved (squashed/stretched).
        img = tf.image.resize(img, (size, size))
        return img, y

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train and augment:
        aug = _augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def save_aug_batch(csv, path, size=224, seed=SEED):
    import matplotlib.pyplot as plt

    x, y = next(iter(make_ds(csv, size, 16, train=True, seed=seed)))
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, im, lab in zip(axes.ravel(), x.numpy(), y.numpy()):
        ax.imshow(im.clip(0, 255).astype("uint8"))
        ax.set_title("ai" if lab else "real", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="reports/audit.csv")
    ap.add_argument("--out", default="splits")
    a = ap.parse_args()
    make_splits(a.audit, a.out)
    save_aug_batch(f"{a.out}/train.csv", "reports/figures/augmented_batch.png")
