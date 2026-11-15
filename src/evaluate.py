"""Evaluate a trained model once on the untouched test split."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODELS_DIR, RESULTS_DIR
from src.preprocessing import load_image_datasets


def _classification_metrics(truth, predictions, probabilities, class_names):
    number_of_classes = len(class_names)
    matrix = [[0 for _ in class_names] for _ in class_names]
    for actual, predicted in zip(truth, predictions):
        matrix[int(actual)][int(predicted)] += 1

    report = {}
    supports = []
    for index, name in enumerate(class_names):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        supports.append(support)

    total = sum(supports)
    accuracy = sum(matrix[index][index] for index in range(number_of_classes)) / total
    macro = {
        key: sum(report[name][key] for name in class_names) / number_of_classes
        for key in ("precision", "recall", "f1-score")
    }
    weighted = {
        key: (
            sum(report[name][key] * report[name]["support"] for name in class_names)
            / total
        )
        for key in ("precision", "recall", "f1-score")
    }
    report["accuracy"] = accuracy
    report["macro avg"] = {**macro, "support": total}
    report["weighted avg"] = {**weighted, "support": total}

    auc_scores = []
    auc_note = None
    for class_index, class_name in enumerate(class_names):
        positive = [
            float(probabilities[row][class_index])
            for row, actual in enumerate(truth)
            if int(actual) == class_index
        ]
        negative = [
            float(probabilities[row][class_index])
            for row, actual in enumerate(truth)
            if int(actual) != class_index
        ]
        if not positive or not negative:
            auc_note = (
                "Macro one-vs-rest ROC-AUC is undefined because the test split "
                f"does not contain positive and negative examples for {class_name}."
            )
            break
        wins = sum(
            1.0 if positive_score > negative_score else
            0.5 if positive_score == negative_score else 0.0
            for positive_score in positive
            for negative_score in negative
        )
        auc_scores.append(wins / (len(positive) * len(negative)))

    auc = sum(auc_scores) / len(auc_scores) if len(auc_scores) == number_of_classes else None
    metrics = {
        "accuracy": accuracy,
        "precision": macro["precision"],
        "recall": macro["recall"],
        "precision_macro": macro["precision"],
        "recall_macro": macro["recall"],
        "f1": macro["f1-score"],
        "f1_macro": macro["f1-score"],
        "f1_weighted": weighted["f1-score"],
        "roc_auc_ovr_macro": auc,
        "roc_auc_note": auc_note,
        "test_images": total,
    }
    return matrix, report, metrics


def evaluate(model_path: Path = None, batch_size: int = 16):
    model_path = model_path or MODELS_DIR / "pest_resnet152v2_final.keras"
    labels_path = MODELS_DIR / "class_names.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Class labels not found: {labels_path}")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Evaluation requires TensorFlow in the active Python environment. "
            "Install the dependencies from requirements.txt."
        ) from error

    data = load_image_datasets(batch_size=batch_size)
    with labels_path.open(encoding="utf-8") as file:
        class_names = json.load(file)
    if class_names != data["class_names"]:
        raise ValueError(
            "Model class labels do not match test folder classes. "
            "Use the training dataset that belongs to this model."
        )

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(data["test"], verbose=1)
    predictions = np.argmax(probabilities, axis=1).tolist()
    truth = np.asarray(data["test_labels"])
    matrix, report, metrics = _classification_metrics(
        truth, predictions, probabilities, class_names
    )
    leakage_path = RESULTS_DIR / "train_test_duplicate_leakage.csv"
    if leakage_path.is_file():
        with leakage_path.open(newline="", encoding="utf-8-sig") as file:
            leakage_rows = sum(1 for _ in csv.DictReader(file))
        metrics["train_test_duplicate_leakage_records"] = leakage_rows
        if leakage_rows:
            metrics["test_set_warning"] = (
                "Exact train/test duplicates were detected during dataset "
                "inspection; test metrics may be optimistic and should not be "
                "reported as independent generalization performance until the "
                "dataset split is corrected."
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "evaluation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)
    with (RESULTS_DIR / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)
    with (RESULTS_DIR / "classification_report.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class_or_average", "precision", "recall", "f1_score", "support"]
        )
        for name, values in report.items():
            if isinstance(values, dict):
                writer.writerow(
                    [
                        name,
                        values["precision"],
                        values["recall"],
                        values["f1-score"],
                        values["support"],
                    ]
                )

    with (RESULTS_DIR / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        csv.writer(file).writerows(matrix)
    figure, axis = plt.subplots(
        figsize=(max(8, len(class_names)), max(7, len(class_names) * 0.8))
    )
    display = axis.imshow(np.asarray(matrix), interpolation="nearest", cmap="Blues")
    figure.colorbar(display, ax=axis)
    labels = list(range(len(class_names)))
    axis.set(
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Test-set confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            axis.text(
                column,
                row,
                str(matrix[row][column]),
                ha="center",
                va="center",
                color="white" if matrix[row][column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    class_metrics = {name: report[name] for name in class_names}
    figure, axis = plt.subplots(figsize=(max(10, len(class_names) * 1.2), 6))
    positions = np.arange(len(class_names))
    width = 0.25
    for offset, key, label in (
        (-width, "precision", "Precision"),
        (0, "recall", "Recall"),
        (width, "f1-score", "F1-score"),
    ):
        axis.bar(
            positions + offset,
            [class_metrics[name][key] for name in class_names],
            width,
            label=label,
        )
    axis.set_xticks(positions, class_names, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Per-class test performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "per_class_performance.png", dpi=160)
    plt.close(figure)

    history_path = RESULTS_DIR / "training_history.json"
    if history_path.is_file():
        with history_path.open(encoding="utf-8") as file:
            history = json.load(file)
        for key, title, filename in (
            ("accuracy", "Training accuracy", "training_accuracy.png"),
            ("val_accuracy", "Validation accuracy", "validation_accuracy.png"),
            ("loss", "Training loss", "training_loss.png"),
            ("val_loss", "Validation loss", "validation_loss.png"),
        ):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(
                [row["epoch"] for row in history],
                [row[key] for row in history],
                marker=".",
            )
            axis.set(xlabel="Epoch", ylabel=title.split()[-1], title=title)
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(RESULTS_DIR / filename, dpi=160)
            plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(f"Evaluation reports saved to {RESULTS_DIR}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    return args


if __name__ == "__main__":
    try:
        arguments = parse_args()
        evaluate(arguments.model, arguments.batch_size)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
