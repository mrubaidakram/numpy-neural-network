# NumPy Neural Network From Scratch

A fully connected MNIST classifier built from NumPy array operations. The project implements dense layers, ReLU, stable softmax, cross-entropy, manual backpropagation, mini-batch SGD, validation-based checkpoint selection, and held-out evaluation without a deep learning framework.

## What is implemented

- He initialization for ReLU hidden layers
- Vectorized forward pass and cached intermediates
- Numerically stable softmax and cross-entropy loss
- Explicit gradients for every weight and bias
- Mini-batch SGD with seeded shuffling
- Stratified train/validation/test separation
- Per-class report and confusion matrix
- Compressed NumPy checkpoint and JSON training reports

NumPy implements all model computation and optimization. scikit-learn is used only to download and split MNIST and to calculate reporting metrics.

## Dataset

The first run retrieves the public MNIST dataset from OpenML and stores the ignored cache under `data/`. Dataset details and source are in [data/README.md](data/README.md).

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m numpy_nn.train --epochs 15 --batch-size 128 --learning-rate 0.08
```

Data is read from `data/`; checkpoints and metrics are saved to `artifacts/`. Override these paths with `--data-dir` and `--output-dir`.

## Architecture

`784 → 256 ReLU → 128 ReLU → 10 logits`. Weights use float64 for readability and to make numerical gradient inspection straightforward. There are no pretrained weights or reported scores until a real training run is performed.

## Scope

An educational implementation intended to make the mechanics of a small neural network inspectable. It is not optimized for large datasets or GPU execution. No benchmark claim is made before a real run.
