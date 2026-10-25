"""Bonus - Grad-CAM for the transfer (MobileNetV2) model.
python -m src.gradcam --model models/champion/model.keras --csv splits/test.csv
Saves 4 correct + 4 wrong predictions to reports/figures/gradcam.png
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


def gradcam(model, x, last_conv="out_relu"):
    base = model.get_layer("backbone")
    conv_model = tf.keras.Model(base.input, [base.get_layer(last_conv).output, base.output])
    with tf.GradientTape() as tape:
        h = model.get_layer("preprocess")(x)
        conv, feats = conv_model(h)
        tape.watch(conv)
        z = model.get_layer("gap")(feats)
        p = model.get_layer("p_ai")(z)
    grads = tape.gradient(p, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, -1))[0].numpy()
    return cam / (cam.max() + 1e-8), float(p[0, 0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/champion/model.keras")
    ap.add_argument("--cfg", default="models/champion/threshold.json")
    ap.add_argument("--csv", default="splits/test.csv")
    a = ap.parse_args()
    model = tf.keras.models.load_model(a.model)
    cfg = json.load(open(a.cfg))
    size, thr = cfg["size"], cfg["threshold"]
    _df = pd.read_csv(a.csv)
    df = _df.sample(min(300, len(_df)), random_state=0)
    good, bad = [], []
    for _, r in df.iterrows():
        img = tf.image.resize(tf.io.decode_image(tf.io.read_file(r.path), channels=3, expand_animations=False), (size, size))[None]
        cam, p = gradcam(model, img)
        (good if (p >= thr) == bool(r.label) else bad).append((img[0].numpy(), cam, p, r.label))
        if len(good) >= 4 and len(bad) >= 4:
            break
    items = good[:4] + bad[:4]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (im, cam, p, lab), tag in zip(axes.ravel(), items, ["correct"] * 4 + ["WRONG"] * 4):
        ax.imshow(im.astype("uint8"))
        heat = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((size, size)))
        ax.imshow(heat, cmap="jet", alpha=0.4)
        ax.set_title(f"{tag} true={'ai' if lab else 'real'} p_ai={p:.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("reports/figures/gradcam.png", dpi=110)
