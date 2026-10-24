"""Task 3.6 - model registry.

python -m src.registry best                       # list runs sorted by val AUC
python -m src.registry register --run-id <id>     # new version of the registered model
python -m src.registry alias --version 2          # give alias 'champion' to a version
python -m src.registry load-test path/to/img.jpg  # separate demo: load by alias + classify
"""
import argparse
import glob
import json
import os

import mlflow
from mlflow.tracking import MlflowClient

from src import config

NAME = f"{config.PROJECT}_visionguard"


def download_champion(dest="models/champion"):
    """Fetch the champion model ONCE (deploy time). The web server then reads a local file."""
    config.setup_mlflow()
    c = MlflowClient()
    mv = c.get_model_version_by_alias(NAME, "champion")
    local = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=dest + "_dl")
    os.makedirs(dest, exist_ok=True)
    for f in glob.glob(f"{local}/**/*", recursive=True):
        if os.path.isfile(f):
            os.replace(f, os.path.join(dest, os.path.basename(f)))
    print(f"champion = version {mv.version} -> {dest}")
    return dest, mv.version


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["best", "register", "alias", "download", "load-test"])
    ap.add_argument("image", nargs="?")
    ap.add_argument("--run-id")
    ap.add_argument("--version")
    ap.add_argument("--metric", default="metrics.val_best_roc_auc")
    a = ap.parse_args()
    config.setup_mlflow()
    client = MlflowClient()

    if a.cmd == "best":
        runs = mlflow.search_runs(order_by=[f"{a.metric} DESC"])  # select on VALIDATION, not test
        cols = ["run_id", "tags.mlflow.runName", a.metric, "metrics.val_best_f1", "metrics.test_f1", "metrics.test_roc_auc"]
        print(runs[[c for c in cols if c in runs.columns]].head(15).to_string())
    elif a.cmd == "register":
        mv = mlflow.register_model(f"runs:/{a.run_id}/model", NAME)
        print("registered version", mv.version)
    elif a.cmd == "alias":
        client.set_registered_model_alias(NAME, "champion", a.version)
        print("alias champion ->", a.version)
    elif a.cmd == "download":
        download_champion()
    else:  # load-test: independent of training code, uses only registry + keras
        import tensorflow as tf

        d, ver = download_champion("models/champion_test")
        model = tf.keras.models.load_model(f"{d}/model.keras")
        cfg = json.load(open(f"{d}/threshold.json"))
        raw = tf.io.decode_image(tf.io.read_file(a.image), channels=3, expand_animations=False)
        x = tf.image.resize(raw, (cfg["size"], cfg["size"]))[None]
        p = float(model.predict(x, verbose=0)[0, 0])
        print(f"champion v{ver}: p_ai={p:.4f} -> {'AI-generated' if p >= cfg['threshold'] else 'real'}")
