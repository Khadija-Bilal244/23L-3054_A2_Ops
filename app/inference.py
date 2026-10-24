"""Task 4.1 - inference module. The model is loaded ONCE when the object is created.

Loading per request would cost seconds of disk IO + graph building each time (and RAM
spikes under concurrency). Preprocessing must equal training: same decoder, same resize,
same pixel range, otherwise the model sees a different input distribution and accuracy
silently drops.
"""
import json
import time

import numpy as np
import tensorflow as tf

from app.settings import Settings


class Predictor:
    def __init__(self, model_path=None, threshold_path=None):
        model_path = model_path or Settings.MODEL_PATH
        threshold_path = threshold_path or Settings.THRESHOLD_PATH
        self.model = tf.keras.models.load_model(model_path)
        cfg = json.load(open(threshold_path))
        self.size = int(cfg["size"])
        self.threshold = float(Settings.THRESHOLD or cfg["threshold"])
        self.model_name = Settings.MODEL_NAME
        self.model_version = Settings.MODEL_VERSION
        self.predict_bytes(tf.io.encode_png(np.zeros((8, 8, 3), np.uint8)).numpy())  # warm-up

    def predict_bytes(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        # identical to src/data.py: decode (3 channels) -> bilinear resize -> raw 0-255 floats;
        # the model's first layer does the model-specific scaling.
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        img = tf.image.resize(img, (self.size, self.size))[None]
        p = float(self.model(img, training=False).numpy()[0, 0])
        is_ai = p >= self.threshold
        return {
            "label": "ai_generated" if is_ai else "real",
            "confidence": p if is_ai else 1 - p,
            "prob_ai": p,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


if __name__ == "__main__":  # python -m app.inference img1.jpg img2.png   (Task 4.1 console output)
    import sys

    pr = Predictor()
    for path in sys.argv[1:]:
        print(path, pr.predict_bytes(open(path, "rb").read()))
