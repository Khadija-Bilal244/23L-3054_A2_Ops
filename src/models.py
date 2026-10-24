"""Task 2.2 (baseline CNN) and 2.3 (transfer learning)."""
import tensorflow as tf
from tensorflow.keras import layers

BACKBONES = {
    # name: (constructor, preprocessing layer factory)
    # MobileNetV2 was trained on pixels scaled to [-1, 1]  -> x/127.5 - 1
    "mobilenetv2": (tf.keras.applications.MobileNetV2, lambda: layers.Rescaling(1 / 127.5, offset=-1, name="preprocess")),
    # Keras EfficientNet has its own normalisation inside; it expects raw 0-255 pixels
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, lambda: layers.Identity(name="preprocess")),
}


def build_baseline(size=224, dropout=0.3):
    inp = layers.Input((size, size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="preprocess")(inp)  # 0-255 -> 0-1, keeps gradients tame
    for i, f in enumerate([32, 64, 128, 128], 1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)  # local patterns/textures
        x = layers.BatchNormalization(name=f"bn{i}")(x)  # stabilises/speeds training
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)  # shrink maps, add translation tolerance
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # far fewer params than Flatten+Dense
    x = layers.Dropout(dropout, name="dropout")(x)  # regularisation
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)  # P(image is AI-generated)
    return tf.keras.Model(inp, out, name="baseline_cnn")


def build_transfer(size=224, backbone="mobilenetv2", dropout=0.3):
    ctor, prep = BACKBONES[backbone]
    base = ctor(include_top=False, weights="imagenet", input_shape=(size, size, 3), name="backbone")
    base.trainable = False  # stage 1: frozen feature extractor
    inp = layers.Input((size, size, 3), name="image")
    x = prep()(inp)
    x = base(x, training=False)  # training=False keeps BatchNorm statistics frozen
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return tf.keras.Model(inp, out, name=f"transfer_{backbone}")


def unfreeze_last_n(model, n=20):
    """Stage 2: unfreeze the last n backbone layers (n~20 = last block of MobileNetV2).
    BatchNorm layers stay frozen (standard fine-tuning advice)."""
    base = model.get_layer("backbone")
    base.trainable = n > 0
    for lyr in base.layers:
        lyr.trainable = False
    if n > 0:
        for lyr in base.layers[-n:]:
            if not isinstance(lyr, layers.BatchNormalization):
                lyr.trainable = True
    return trainable_layer_names(model)


def trainable_layer_names(model):
    names = [lyr.name for lyr in model.layers if lyr.trainable and lyr.weights]
    base = model.get_layer("backbone") if "backbone" in [lyr.name for lyr in model.layers] else None
    if base is not None:
        names = [n for n in names if n != "backbone"] + [f"backbone/{lyr.name}" for lyr in base.layers if lyr.trainable and lyr.weights]
    return names


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",  # matches the single sigmoid output
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model
