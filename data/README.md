# Dataset

The training command downloads MNIST through scikit-learn's OpenML fetcher on the first run and caches a compressed copy here. The raw data cache is ignored by Git.

Source: [OpenML MNIST](https://www.openml.org/d/554)

MNIST contains 70,000 28×28 grayscale handwritten digit images. This project flattens images to 784 normalized inputs and uses a stratified 60,000/10,000 train/test split, with validation carved only from the training portion.
