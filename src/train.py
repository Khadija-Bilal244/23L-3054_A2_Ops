"""Training for VisionGuard (Tasks 2.2-2.6, 3.3, 3.4).

Baseline CNN:   python -m src.train --model baseline --run-name baseline_lr1e-3
Transfer:       python -m src.train --model transfer --backbone mobilenetv2 --run-name tl_mnv2
Autolog mode:   python -m src.train --model transfer --tracking autolog --run-name tl_autolog
"""
import argparse
import os
import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src import evaluate as ev
from src.data import make_ds
from src.models import (build_baseline, build_transfer, compile_model, trainable_layer_names,
                        unfreeze_last_n)


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """MANUAL tracking: we decide exactly what is logged (all per-epoch metrics incl. lr)."""

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.offset + epoch)


def fit_stage(model, train_ds, val_ds, lr, epochs, ckpt, offset, manual, patience):
    compile_model(model, lr)
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, patience // 2), min_lr=1e-7),
    ]
    if manual:
        cbs.append(MlflowEpochLogger(offset))
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2).history
    model.load_weights(ckpt)  # guarantee we hold the best-val-loss weights
    return h, float(min(h["val_loss"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    ap.add_argument("--backbone", default="mobilenetv2")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-5, help="much smaller LR for fine-tuning")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--unfreeze-n", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0, help="F-beta used to pick the threshold on val")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tracking", choices=["manual", "autolog"], default="manual")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--deterministic", action="store_true")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(a.seed)  # python, numpy and TF seeds in one call
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()
    print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

    train_ds = make_ds(f"{a.splits}/train.csv", a.size, a.batch, train=True, augment=not a.no_aug, seed=a.seed)
    val_ds = make_ds(f"{a.splits}/val.csv", a.size, a.batch)
    test_ds = make_ds(f"{a.splits}/test.csv", a.size, a.batch)
    y_val = pd.read_csv(f"{a.splits}/val.csv").label.values
    y_test = pd.read_csv(f"{a.splits}/test.csv").label.values

    out = f"reports/runs/{a.run_name}"
    os.makedirs(out, exist_ok=True)
    mdir = f"models/{a.run_name}"
    os.makedirs(mdir, exist_ok=True)

    config.setup_mlflow()
    if a.tracking == "autolog":
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)
    else:
        mlflow.tensorflow.autolog(disable=True)
    manual = a.tracking == "manual"

    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.log_params({f"hp_{k}": v for k, v in vars(a).items()})
        mlflow.set_tags({"git_commit": config.git_commit(), "dvc_data_version": config.dvc_data_version(),
                         "seed": a.seed, "tracking_mode": a.tracking, "tf_version": tf.__version__,
                         "arch": a.model if a.model == "baseline" else a.backbone})

        # ---------------- stage 1 ----------------
        model = build_baseline(a.size, a.dropout) if a.model == "baseline" else build_transfer(a.size, a.backbone, a.dropout)
        model.summary()
        c1 = f"{mdir}/stage1.keras"
        h1, v1 = fit_stage(model, train_ds, val_ds, a.lr, a.epochs, c1, 0, manual, a.patience)
        hist, ckpt_best, v_best, boundary = dict(h1), c1, v1, None
        mlflow.log_text("\n".join(trainable_layer_names(model)), "trainable_stage1.txt")

        # ---------------- stage 2 (fine-tune) ----------------
        if a.model == "transfer" and a.unfreeze_n > 0 and a.ft_epochs > 0:
            names = unfreeze_last_n(model, a.unfreeze_n)
            mlflow.log_text("\n".join(names), "trainable_stage2.txt")
            c2 = f"{mdir}/stage2.keras"
            h2, v2 = fit_stage(model, train_ds, val_ds, a.ft_lr, a.ft_epochs, c2, len(h1["loss"]), manual, a.patience)
            boundary = len(h1["loss"])
            hist = {k: h1[k] + h2.get(k, []) for k in h1}
            if v2 < v1:
                ckpt_best, v_best = c2, v2
            model.load_weights(ckpt_best)

        best_epoch = int(np.argmin(hist["val_loss"]))
        ev.save_curves(hist, best_epoch, f"{out}/curves.png", boundary)

        # ---------------- evaluation ----------------
        p_val = model.predict(val_ds, verbose=0).ravel()
        thr = ev.choose_threshold(y_val, p_val, a.beta)  # threshold from VALIDATION only
        p_test = model.predict(test_ds, verbose=0).ravel()  # test set touched once
        m_val, m_test = ev.metrics(y_val, p_val, thr), ev.metrics(y_test, p_test, thr)
        cm = ev.save_confusion(y_test, p_test, thr, f"{out}/confusion_matrix.png")
        ev.save_roc(y_test, p_test, f"{out}/roc_curve.png", a.run_name)
        ev.save_error_grid(f"{a.splits}/test.csv", p_test, thr, f"{out}/misclassified.png")
        tn, fp, fn, tp = cm.ravel()
        print("TEST", m_test, "\nreal->fake (FP):", fp, "| fake->real (FN):", fn)

        mlflow.log_metrics({f"val_best_{k}": v for k, v in m_val.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_loss": v_best,
                            "test_real_called_fake": int(fp), "test_fake_called_real": int(fn)})

        # ---------------- artifacts ----------------
        model.save(f"{mdir}/model.keras")
        config_json = dict(threshold=thr, size=a.size, arch=a.model, backbone=a.backbone,
                           labels=config.LABELS, beta=a.beta, run_id=run.info.run_id)
        ev.dump_json(config_json, f"{mdir}/threshold.json")
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        with open(f"{out}/requirements_freeze.txt", "w") as f:
            f.write(freeze)
        mlflow.log_artifacts(out)  # curves, confusion matrix, ROC, error grid, pip list
        mlflow.log_artifacts(a.splits, "splits")
        mlflow.log_artifact(f"{mdir}/model.keras", "model")
        mlflow.log_artifact(f"{mdir}/threshold.json", "model")
        os.makedirs("reports", exist_ok=True)
        with open("reports/last_run_id.txt", "w") as f:
            f.write(run.info.run_id)
        print("RUN_ID", run.info.run_id)


if __name__ == "__main__":
    main()
