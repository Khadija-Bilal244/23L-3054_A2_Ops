"""Task 3.1 / 3.2 - toy MLflow run on MNIST.
Local:    python labs/mlflow_toy.py --lr 0.001 ; python labs/mlflow_toy.py --lr 0.01 ; mlflow ui
DagsHub:  put MLFLOW_TRACKING_URI / DAGSHUB_USER / DAGSHUB_TOKEN in .env and run the same commands.
"""
import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=4)
a = ap.parse_args()

if os.getenv("DAGSHUB_USER") and os.getenv("DAGSHUB_TOKEN"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USER"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("toy_mnist")

(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr, xte = xtr / 255.0, xte / 255.0
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(64, activation="relu"),
                             tf.keras.layers.Dense(10, activation="softmax")])
model.compile(tf.keras.optimizers.Adam(a.lr), "sparse_categorical_crossentropy", ["accuracy"])

with mlflow.start_run(run_name=f"toy_lr_{a.lr}"):
    mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "units": 64})  # parameters: inputs we chose
    mlflow.set_tag("purpose", "tool-lab")  # tag: free-form label
    h = model.fit(xtr, ytr, validation_split=0.1, epochs=a.epochs, verbose=2).history
    for i in range(a.epochs):  # metrics: numbers measured, per step
        mlflow.log_metrics({"loss": h["loss"][i], "val_loss": h["val_loss"][i],
                            "accuracy": h["accuracy"][i], "val_accuracy": h["val_accuracy"][i]}, step=i)
    mlflow.log_metric("test_accuracy", model.evaluate(xte, yte, verbose=0)[1])
    plt.plot(h["loss"], label="train")
    plt.plot(h["val_loss"], label="val")
    plt.legend()
    plt.savefig("toy_loss.png")
    mlflow.log_artifact("toy_loss.png")  # artifact: any file attached to the run
    model.save("toy_model.keras")
    mlflow.log_artifact("toy_model.keras")
