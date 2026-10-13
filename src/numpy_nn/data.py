"""MNIST loader; sklearn fetches the public dataset and NumPy handles arrays."""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist(data_dir: str, seed: int = 42):
    cache = Path(data_dir) / "mnist.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache) as data:
            images = data["images"]
            labels = data["labels"]
    else:
        dataset = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(cache.parent))
        images = np.asarray(dataset.data, dtype=np.uint8)
        labels = np.asarray(dataset.target, dtype=np.uint8)
        np.savez_compressed(cache, images=images, labels=labels)
    images = images.astype(np.float64) / 255.0
    labels = labels.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=10_000, random_state=seed, stratify=labels
    )
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train, y_train, test_size=1 / 7, random_state=seed, stratify=y_train
    )
    return x_fit, x_validation, x_test, y_fit, y_validation, y_test
