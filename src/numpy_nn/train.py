"""Train and evaluate the NumPy MLP on MNIST."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from .data import load_mnist
from .model import MLPClassifier


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> dict:
    probabilities = model.predict_proba(x)
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "cross_entropy": float(log_loss(y, probabilities, labels=np.arange(10))),
        "confusion_matrix": confusion_matrix(y, predictions, labels=np.arange(10)).tolist(),
        "classification_report": classification_report(y, predictions, labels=np.arange(10), output_dict=True, zero_division=0),
    }


def iter_batches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        indices = order[start:start + batch_size]
        yield x[indices], y[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy-only MLP with manual backpropagation on MNIST.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")

    x_train, x_val, x_test, y_train, y_val, y_test = load_mnist(args.data_dir, args.seed)
    model = MLPClassifier((784, 256, 128, 10), seed=args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_weights = best_biases = None
    history = []

    for epoch in range(1, args.epochs + 1):
        losses = [model.train_batch(x, y, args.learning_rate) for x, y in iter_batches(x_train, y_train, args.batch_size, rng)]
        validation = evaluate(model, x_val, y_val)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(record)
        print(json.dumps({"epoch": epoch, "train_loss": record["train_loss"], "validation_accuracy": validation["accuracy"]}), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_weights = [weight.copy() for weight in model.weights]
            best_biases = [bias.copy() for bias in model.biases]

    assert best_weights is not None and best_biases is not None
    model.weights, model.biases = best_weights, best_biases
    result = {
        "selected_by": "validation_accuracy",
        "validation": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
        "seed": args.seed,
        "architecture": model.layer_sizes,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate},
    }
    np.savez_compressed(output / "model.npz", **{f"weight_{i}": w for i, w in enumerate(model.weights)}, **{f"bias_{i}": b for i, b in enumerate(model.biases)})
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_accuracy": result["validation"]["accuracy"], "test_accuracy": result["test"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
