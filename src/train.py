"""Train the classification head, fine-tune, and save the best Keras model."""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone

from src.config import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.model import build_model, configure_fine_tuning
from src.preprocessing import load_image_datasets


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Training requires TensorFlow in the active Python environment. "
            "Install the dependencies from requirements.txt and select that "
            "environment in VS Code. No training was started."
        ) from error
    return tf


def _plot_history(history_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open(encoding="utf-8") as file:
        records = json.load(file)
    epochs = [record["epoch"] for record in records]
    for key, title, filename in (
        ("accuracy", "Training accuracy", "training_accuracy.png"),
        ("val_accuracy", "Validation accuracy", "validation_accuracy.png"),
        ("loss", "Training loss", "training_loss.png"),
        ("val_loss", "Validation loss", "validation_loss.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, [record[key] for record in records], marker=".")
        axis.set(xlabel="Epoch", ylabel=title.split()[-1], title=title)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(RESULTS_DIR / filename, dpi=160)
        plt.close(figure)


def train(args):
    tf = _tensorflow()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOGS_DIR / "training.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    try:
        data = load_image_datasets(
            validation_fraction=args.validation_split,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model, base_model, _ = build_model(
            len(data["class_names"]), args.fine_tune_layers
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise RuntimeError(f"Training setup failed: {error}") from error

    def compile_model(learning_rate):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    stage_histories = []
    best_loss = float("inf")
    best_checkpoint = None

    def run_stage(name, epochs, learning_rate):
        nonlocal best_loss, best_checkpoint
        checkpoint = CHECKPOINTS_DIR / f"{name}_best.keras"
        compile_model(learning_rate)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=max(1, args.patience // 2),
                min_lr=1e-7,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                str(LOGS_DIR / "epoch_metrics.csv"),
                append=(name != "head"),
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(LOGS_DIR / "tensorboard" / name)
            ),
        ]
        history = model.fit(
            data["train"],
            validation_data=data["validation"],
            epochs=epochs,
            callbacks=callbacks,
            shuffle=False,
            verbose=1,
        )
        values = history.history.get("val_loss", [])
        if not values:
            raise RuntimeError(f"{name} training produced no validation metrics.")
        stage_best = min(values)
        stage_histories.append(
            {
                "name": name,
                "history": history.history,
                "best_validation_loss": float(stage_best),
            }
        )
        if stage_best < best_loss:
            best_loss = float(stage_best)
            best_checkpoint = checkpoint

    try:
        run_stage("head", args.head_epochs, args.head_learning_rate)
        unfrozen_layers = configure_fine_tuning(
            base_model, args.fine_tune_layers
        )
        logging.info("Unfrozen %d ResNet layers", unfrozen_layers)
        run_stage("fine_tune", args.fine_tune_epochs, args.fine_tune_learning_rate)
    except tf.errors.ResourceExhaustedError as error:
        raise RuntimeError(
            "GPU/CPU memory was exhausted during training. Reduce --batch-size "
            "(for example, --batch-size 8 or 4) and retry."
        ) from error

    if best_checkpoint is None or not best_checkpoint.is_file():
        raise RuntimeError("Training completed without saving a best checkpoint.")
    final_model_path = MODELS_DIR / "pest_resnet152v2_final.keras"
    best_model_path = MODELS_DIR / "pest_resnet152v2_best.keras"
    shutil.copy2(best_checkpoint, best_model_path)
    shutil.copy2(best_checkpoint, final_model_path)
    with (MODELS_DIR / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(data["class_names"], file, indent=2)

    combined_history = []
    epoch_number = 0
    for stage in stage_histories:
        history = stage["history"]
        for index in range(len(history["loss"])):
            epoch_number += 1
            combined_history.append(
                {
                    "epoch": epoch_number,
                    "stage": stage["name"],
                    **{
                        key: float(values[index])
                        for key, values in history.items()
                    },
                }
            )
    history_path = RESULTS_DIR / "training_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(combined_history, file, indent=2)
    _plot_history(history_path)

    model_config = {
        "architecture": "ResNet152V2",
        "input_size": [224, 224, 3],
        "number_of_classes": len(data["class_names"]),
        "class_names": data["class_names"],
        "validation_split": args.validation_split,
        "seed": args.seed,
        "fine_tune_layers": args.fine_tune_layers,
        "pretrained_weights": "ImageNet",
        "best_validation_loss": best_loss,
    }
    with (MODELS_DIR / "model_config.json").open("w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=2)

    metadata = {
        **model_config,
        "dataset_root": data["dataset_root"],
        "train_images": len(data["train_paths"]),
        "validation_images": len(data["validation_paths"]),
        "test_images": len(data["test_paths"]),
        "batch_size": args.batch_size,
        "head_epochs_requested": args.head_epochs,
        "fine_tune_epochs_requested": args.fine_tune_epochs,
        "head_learning_rate": args.head_learning_rate,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__,
        "best_model": str(final_model_path.relative_to(MODELS_DIR)),
    }
    with (MODELS_DIR / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)
    logging.info("Saved final model to %s", final_model_path)
    print(f"Best model saved to {final_model_path}")
    print(f"Class labels saved to {MODELS_DIR / 'class_names.json'}")
    print(f"Metadata saved to {MODELS_DIR / 'training_metadata.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.head_epochs,
        args.fine_tune_epochs,
        args.fine_tune_layers,
        args.patience,
    ) < 1:
        parser.error("batch size, epochs, fine-tune layers, and patience must be positive.")
    if args.head_learning_rate <= 0 or args.fine_tune_learning_rate <= 0:
        parser.error("learning rates must be positive.")
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
