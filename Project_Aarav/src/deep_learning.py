"""Deep learning models for CAN bus intrusion detection.

Provides MLP, LSTM, and 1D-CNN architectures built with
TensorFlow / Keras.  Each builder returns a compiled Keras model.
A thin sklearn-compatible wrapper (``KerasClassifierWrapper``) is
provided so that the models integrate seamlessly with the existing
train / predict / evaluate pipeline.
"""

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

# Suppress TF info logs unless user opts in
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning(
        "TensorFlow not installed — deep learning models will be unavailable. "
        "Install with: pip install tensorflow"
    )


def _check_tf():
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for deep learning models. "
            "Install with: pip install tensorflow"
        )


# -----------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------

def build_mlp(
    input_dim: int,
    n_classes: int,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.3,
) -> "keras.Model":
    """Build a Multi-Layer Perceptron classifier.

    Args:
        input_dim: Number of input features.
        n_classes: Number of output classes.
        hidden_layers: List of neurons per hidden layer.
        dropout: Dropout rate between layers.

    Returns:
        Compiled Keras model.
    """
    _check_tf()
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]

    model = keras.Sequential(name="MLP")
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(
    window_size: int,
    n_features: int,
    n_classes: int,
    lstm_units: list[int] | None = None,
    dropout: float = 0.3,
) -> "keras.Model":
    """Build an LSTM model for sequential CAN frame analysis.

    Args:
        window_size: Number of time steps (frames per window).
        n_features: Number of features per frame.
        n_classes: Number of output classes.
        lstm_units: List of LSTM layer sizes.
        dropout: Dropout rate.

    Returns:
        Compiled Keras model.
    """
    _check_tf()
    if lstm_units is None:
        lstm_units = [64, 32]

    model = keras.Sequential(name="LSTM")
    model.add(layers.Input(shape=(window_size, n_features)))

    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn1d(
    window_size: int,
    n_features: int,
    n_classes: int,
    filters: list[int] | None = None,
    kernel_size: int = 3,
    dropout: float = 0.3,
) -> "keras.Model":
    """Build a 1D-CNN model for CAN bus traffic pattern detection.

    Args:
        window_size: Number of time steps.
        n_features: Number of features per step.
        n_classes: Number of output classes.
        filters: List of filter counts per Conv1D layer.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.

    Returns:
        Compiled Keras model.
    """
    _check_tf()
    if filters is None:
        filters = [64, 128]

    model = keras.Sequential(name="CNN1D")
    model.add(layers.Input(shape=(window_size, n_features)))

    for f in filters:
        model.add(layers.Conv1D(f, kernel_size, activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2, padding="same"))
        model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------------------------------------------------
# Sklearn-compatible wrapper
# -----------------------------------------------------------------------

class KerasClassifierWrapper:
    """Thin wrapper that gives Keras models the sklearn fit/predict API.

    This allows deep learning models to be used interchangeably with
    sklearn classifiers in the evaluation pipeline.
    """

    def __init__(
        self,
        build_fn: str,
        build_kwargs: dict,
        epochs: int = 20,
        batch_size: int = 256,
        validation_split: float = 0.1,
        seed: int = 42,
    ):
        self.build_fn_name = build_fn
        self.build_kwargs = build_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed
        self.model_ = None
        self.classes_ = None
        self.is_sequential = build_fn in ("lstm", "cnn1d")

    def fit(self, X: np.ndarray, y: np.ndarray):
        _check_tf()
        tf.random.set_seed(self.seed)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Build model based on type
        if self.build_fn_name == "mlp":
            self.model_ = build_mlp(
                input_dim=X.shape[1],
                n_classes=n_classes,
                **self.build_kwargs,
            )
        elif self.build_fn_name == "lstm":
            window_size, n_features = X.shape[1], X.shape[2]
            self.model_ = build_lstm(
                window_size=window_size,
                n_features=n_features,
                n_classes=n_classes,
                **self.build_kwargs,
            )
        elif self.build_fn_name == "cnn1d":
            window_size, n_features = X.shape[1], X.shape[2]
            self.model_ = build_cnn1d(
                window_size=window_size,
                n_features=n_features,
                n_classes=n_classes,
                **self.build_kwargs,
            )
        else:
            raise ValueError(f"Unknown deep learning model: {self.build_fn_name}")

        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=0,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.model_.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X, verbose=0)


def get_deep_learning_model(
    model_name: str,
    params: dict,
    seed: int = 42,
) -> KerasClassifierWrapper:
    """Factory function for deep learning models.

    Args:
        model_name: One of ``"mlp"``, ``"lstm"``, ``"cnn1d"``.
        params: Model parameters (passed to ``build_kwargs`` after
            extracting training params).
        seed: Random seed.

    Returns:
        ``KerasClassifierWrapper`` instance.
    """
    _check_tf()

    # Separate training params from architecture params
    training_keys = {"epochs", "batch_size", "validation_split"}
    training_params = {k: params[k] for k in training_keys if k in params}
    build_kwargs = {k: v for k, v in params.items() if k not in training_keys}

    return KerasClassifierWrapper(
        build_fn=model_name,
        build_kwargs=build_kwargs,
        seed=seed,
        **training_params,
    )


# Deep learning model names for identification
DL_MODEL_NAMES = {"mlp", "lstm", "cnn1d"}


def is_deep_learning_model(model_name: str) -> bool:
    """Check whether a model name refers to a deep learning model."""
    return model_name in DL_MODEL_NAMES
