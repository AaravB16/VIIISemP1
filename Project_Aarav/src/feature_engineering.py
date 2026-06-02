"""Feature engineering for CAN bus intrusion detection."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_frame_features(
    df: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frame-level features and labels.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (feature array X, label array y).
    """
    feature_cols = list(config["features"]["frame_level"])

    # Include inter-arrival index if available
    if "inter_arrival_idx" in df.columns:
        feature_cols.append("inter_arrival_idx")

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.astype(np.float32)
    y = df["specific_class"].values

    logger.info(f"Extracted frame features: X={X.shape}, y={y.shape}")
    return X, y


def build_sliding_windows(
    df: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding window sequences for temporal modeling.

    Creates overlapping windows of consecutive CAN frames. Each window
    is flattened into a single feature vector for use with traditional
    ML classifiers.

    The label for each window is taken from the last frame in the window.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary with window settings.

    Returns:
        Tuple of (windowed feature array X, label array y).
    """
    feat_config = config["features"]
    window_size = feat_config.get("window_size", 10)
    stride = feat_config.get("window_stride", 1)

    feature_cols = list(feat_config["frame_level"])
    if "inter_arrival_idx" in df.columns:
        feature_cols.append("inter_arrival_idx")

    available = [c for c in feature_cols if c in df.columns]
    values = df[available].values.astype(np.float32)
    labels = df["specific_class"].values

    n_frames = len(values)
    n_features = values.shape[1]

    if n_frames < window_size:
        logger.warning(
            f"Dataset has {n_frames} frames, less than window_size={window_size}. "
            "Falling back to frame-level features."
        )
        return values, labels

    # Build windows
    windows = []
    window_labels = []

    for i in range(0, n_frames - window_size + 1, stride):
        window = values[i : i + window_size].flatten()
        windows.append(window)
        # Label from the last frame in the window
        window_labels.append(labels[i + window_size - 1])

    X = np.array(windows, dtype=np.float32)
    y = np.array(window_labels)

    logger.info(
        f"Built sliding windows: {len(X)} windows, "
        f"window_size={window_size}, stride={stride}, "
        f"features_per_window={window_size * n_features}"
    )
    return X, y


def prepare_features(
    df: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features based on configuration.

    Dispatches to either frame-level or sliding-window feature extraction
    depending on config settings.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (feature array X, label array y).
    """
    use_window = config["features"].get("use_sliding_window", False)

    if use_window:
        return build_sliding_windows(df, config)
    else:
        return extract_frame_features(df, config)
