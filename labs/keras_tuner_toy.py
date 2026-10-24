"""Task 2.7 (toy part) - random search on Fashion-MNIST over learning rate and units."""
import keras_tuner as kt
import tensorflow as tf

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x = x / 255.0


def build(hp):
    m = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
              "sparse_categorical_crossentropy", ["accuracy"])
    return m


t = kt.RandomSearch(build, "val_accuracy", max_trials=6, directory="reports/tuner_toy",
                    project_name="fmnist", overwrite=True)
t.search(x, y, validation_split=0.2, epochs=3)
t.results_summary()
print("best:", t.get_best_hyperparameters(1)[0].values)
