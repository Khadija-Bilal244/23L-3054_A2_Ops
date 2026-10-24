"""Task 2.7 (apply part) - Keras Tuner on VisionGuard: lr, dropout, number of unfrozen layers.
python -m src.tune --trials 8
Each trial's result is also logged to MLflow as a run (child of the tuning run).
"""
import argparse

import keras_tuner as kt
import mlflow
import pandas as pd
import tensorflow as tf

from src import config
from src.data import make_ds
from src.models import build_transfer, compile_model, unfreeze_last_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--splits", default="splits")
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(config.SEED)
    train = make_ds(f"{a.splits}/train.csv", a.size, 32, train=True)
    val = make_ds(f"{a.splits}/val.csv", a.size, 32)

    def build(hp):
        m = build_transfer(a.size, "mobilenetv2", hp.Float("dropout", 0.1, 0.5, step=0.1))
        unfreeze_last_n(m, hp.Choice("unfreeze_n", [0, 10, 20, 40]))
        return compile_model(m, hp.Float("lr", 1e-5, 1e-3, sampling="log"))

    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=a.trials, seed=config.SEED,
                            directory="reports/tuner", project_name="visionguard", overwrite=True)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    tuner.search(train, validation_data=val, epochs=a.epochs, callbacks=[es])
    tuner.results_summary()

    config.setup_mlflow()
    rows = []
    with mlflow.start_run(run_name="keras_tuner"):
        for t in tuner.oracle.get_best_trials(a.trials):
            with mlflow.start_run(run_name=f"trial_{t.trial_id}", nested=True):
                mlflow.log_params(t.hyperparameters.values)
                mlflow.log_metric("val_loss", t.score)
            rows.append({**t.hyperparameters.values, "val_loss": t.score})
    df = pd.DataFrame(rows)
    df.to_csv("reports/tuner_results.csv", index=False)
    print(df.to_string())
    print("\nNOTE: the best val score is the max over many noisy trials -> optimistic. Re-check on test ONCE.")


if __name__ == "__main__":
    main()
