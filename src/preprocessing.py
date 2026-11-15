"""Create reproducible, stratified TensorFlow datasets from image folders."""

from pathlib import Path
import random

from PIL import Image, UnidentifiedImageError

from src.config import (
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    dataset_directories,
)


def _list_images(split_dir: Path, class_names: list):
    paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise ValueError(f"Class folder is missing: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    raise ValueError(
                        f"Unreadable image found: {path}. See the inspection "
                        f"report and manually replace or exclude it. Details: {error}"
                    ) from error
                paths.append(str(path))
                labels.append(label)
    return paths, labels


def _stratified_split(paths, labels, validation_fraction, seed):
    indices_by_class = {}
    for index, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(index)
    randomizer = random.Random(seed)
    train_indices = []
    validation_indices = []
    for label, indices in sorted(indices_by_class.items()):
        if len(indices) < 2:
            raise ValueError(
                "Each class needs at least two training images for a "
                "stratified train/validation split."
            )
        randomizer.shuffle(indices)
        validation_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * validation_fraction)),
        )
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    randomizer.shuffle(train_indices)
    randomizer.shuffle(validation_indices)
    return (
        [paths[index] for index in train_indices],
        [paths[index] for index in validation_indices],
        [labels[index] for index in train_indices],
        [labels[index] for index in validation_indices],
    )


def load_image_datasets(validation_fraction=0.2, batch_size=16, seed=42):
    """Return train, validation, and untouched test tf.data datasets and labels."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    dataset_root, train_dir, test_dir = dataset_directories()
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"Expected train and test folders in {dataset_root}")

    class_names = sorted(
        folder.name for folder in train_dir.iterdir() if folder.is_dir()
    )
    test_class_names = sorted(
        folder.name for folder in test_dir.iterdir() if folder.is_dir()
    )
    if not class_names or class_names != test_class_names:
        raise ValueError(
            "Train and test class folders must be non-empty and match exactly. "
            f"Train={class_names}, test={test_class_names}"
        )

    train_paths, train_labels = _list_images(train_dir, class_names)
    test_paths, test_labels = _list_images(test_dir, class_names)
    if not train_paths or not test_paths:
        raise ValueError("No readable training or test images were found.")
    (
        train_paths,
        validation_paths,
        train_labels,
        validation_labels,
    ) = _stratified_split(
        train_paths, train_labels, validation_fraction, seed
    )

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for preprocessing/training. Install the "
            "project requirements in a Python environment supported by the "
            "selected TensorFlow release."
        ) from error

    def decode_and_resize(path, label):
        contents = tf.io.read_file(path)
        image = tf.io.decode_image(contents, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32), label

    def make_dataset(paths, labels, training):
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            dataset = dataset.shuffle(
                buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
            )
        dataset = dataset.map(
            decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    return {
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "train": make_dataset(train_paths, train_labels, training=True),
        "validation": make_dataset(
            validation_paths, validation_labels, training=False
        ),
        "test": make_dataset(test_paths, test_labels, training=False),
        "train_labels": train_labels,
        "validation_labels": validation_labels,
        "test_labels": test_labels,
        "train_paths": train_paths,
        "validation_paths": validation_paths,
        "test_paths": test_paths,
    }
