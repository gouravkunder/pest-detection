"""Inspect images and save inventory, integrity, and distribution reports."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from src.config import (
    IMAGE_EXTENSIONS,
    RESULTS_DIR,
    dataset_directories,
)


INVENTORY_FIELDS = [
    "split",
    "class_name",
    "image_path",
    "width",
    "height",
    "mode",
    "size_bytes",
    "sha256",
]


def _image_files(class_dir: Path):
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _class_folders(split_dir: Path) -> dict:
    return {
        folder.name: folder
        for folder in sorted(split_dir.iterdir())
        if folder.is_dir()
    }


def _plot_distribution(classes, counts, output_path: Path) -> None:
    positions = range(len(classes))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(10, len(classes) * 1.2), 6))
    axis.bar(
        [position - width / 2 for position in positions],
        [counts[name]["train_images"] for name in classes],
        width,
        label="Train",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [counts[name]["test_images"] for name in classes],
        width,
        label="Test",
    )
    axis.set_xticks(list(positions), classes, rotation=35, ha="right")
    axis.set_ylabel("Image files")
    axis.set_title("Dataset distribution by pest class")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def inspect_dataset() -> dict:
    """Scan all classes without modifying or removing any dataset files."""
    dataset_root, train_dir, test_dir = dataset_directories()
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected both train and test directories under {dataset_root}"
        )

    train_folders = _class_folders(train_dir)
    test_folders = _class_folders(test_dir)
    classes = sorted(set(train_folders) | set(test_folders))
    if not classes:
        raise RuntimeError(f"No class folders found under {EXTRACTED_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = {
        name: {
            "train_images": 0,
            "test_images": 0,
            "train_readable": 0,
            "test_readable": 0,
        }
        for name in classes
    }
    inventory = []
    corrupted = []
    dimensions = Counter()
    hashes = defaultdict(list)

    for split, folders in (("train", train_folders), ("test", test_folders)):
        for class_name, class_dir in folders.items():
            for image_path in _image_files(class_dir):
                counts[class_name][f"{split}_images"] += 1
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        mode = image.mode
                        image.verify()
                    digest = _sha256(image_path)
                    row = {
                        "split": split,
                        "class_name": class_name,
                        "image_path": image_path.relative_to(
                            dataset_root
                        ).as_posix(),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "size_bytes": image_path.stat().st_size,
                        "sha256": digest,
                    }
                    inventory.append(row)
                    hashes[digest].append(row)
                    dimensions[(width, height)] += 1
                    counts[class_name][f"{split}_readable"] += 1
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    corrupted.append(
                        {
                            "split": split,
                            "class_name": class_name,
                            "image_path": image_path.relative_to(
                                dataset_root
                            ).as_posix(),
                            "error": str(error),
                        }
                    )

    duplicate_rows = [
        row for records in hashes.values() if len(records) > 1 for row in records
    ]
    duplicate_groups = [
        records for records in hashes.values() if len(records) > 1
    ]
    leakage_groups = [
        records
        for records in hashes.values()
        if len({record["split"] for record in records}) > 1
    ]
    leakage_rows = [
        row
        for records in hashes.values()
        if len({record["split"] for record in records}) > 1
        for row in records
    ]
    class_rows = [{"class_name": name, **counts[name]} for name in classes]

    _write_csv(
        RESULTS_DIR / "image_inventory.csv", INVENTORY_FIELDS, inventory
    )
    _write_csv(
        RESULTS_DIR / "class_counts.csv",
        [
            "class_name",
            "train_images",
            "test_images",
            "train_readable",
            "test_readable",
        ],
        class_rows,
    )
    _write_csv(
        RESULTS_DIR / "corrupted_images.csv",
        ["split", "class_name", "image_path", "error"],
        corrupted,
    )
    _write_csv(
        RESULTS_DIR / "exact_duplicates.csv",
        INVENTORY_FIELDS,
        duplicate_rows,
    )
    _write_csv(
        RESULTS_DIR / "train_test_duplicate_leakage.csv",
        INVENTORY_FIELDS,
        leakage_rows,
    )
    _plot_distribution(classes, counts, RESULTS_DIR / "dataset_distribution.png")

    per_split_dimensions = {}
    for split in ("train", "test"):
        split_records = [row for row in inventory if row["split"] == split]
        widths = [row["width"] for row in split_records]
        heights = [row["height"] for row in split_records]
        per_split_dimensions[split] = {
            "readable_images": len(split_records),
            "distinct_dimensions": len(
                {(row["width"], row["height"]) for row in split_records}
            ),
            "mean_width": mean(widths) if widths else None,
            "mean_height": mean(heights) if heights else None,
        }

    train_counts = [counts[name]["train_images"] for name in classes]
    minimum_train = min(train_counts)
    maximum_train = max(train_counts)
    summary = {
        "dataset_root": str(dataset_root),
        "classes": classes,
        "number_of_classes": len(classes),
        "class_folder_mismatches": {
            "missing_from_train": sorted(set(test_folders) - set(train_folders)),
            "missing_from_test": sorted(set(train_folders) - set(test_folders)),
        },
        "total_image_files": sum(
            item[f"{split}_images"]
            for item in counts.values()
            for split in ("train", "test")
        ),
        "readable_image_files": len(inventory),
        "corrupted_image_count": len(corrupted),
        "duplicate_image_record_count": len(duplicate_rows),
        "duplicate_hash_group_count": len(duplicate_groups),
        "train_test_duplicate_record_count": len(leakage_rows),
        "train_test_duplicate_hash_group_count": len(leakage_groups),
        "cross_class_duplicate_group_count": sum(
            len({record["class_name"] for record in records}) > 1
            for records in duplicate_groups
        ),
        "class_imbalance": {
            "train_minimum": minimum_train,
            "train_maximum": maximum_train,
            "train_max_to_min_ratio": (
                maximum_train / minimum_train if minimum_train else None
            ),
        },
        "image_dimensions": {
            f"{width}x{height}": amount
            for (width, height), amount in sorted(dimensions.items())
        },
        "split_dimension_summary": per_split_dimensions,
        "recommended_handling": [
            "Review corrupted_images.csv and manually replace or exclude unreadable files; this project never deletes dataset files.",
            "Review exact_duplicates.csv; duplicate files are reported but not removed.",
            "Resolve train/test matches listed in train_test_duplicate_leakage.csv before reporting final test metrics, because test leakage inflates evaluation.",
        ],
        "class_counts": class_rows,
    }
    with (RESULTS_DIR / "dataset_summary.json").open(
        "w", encoding="utf-8"
    ) as json_file:
        json.dump(summary, json_file, indent=2)

    print(f"Dataset root: {dataset_root}")
    print(
        f"Classes: {len(classes)} | image files: {summary['total_image_files']} "
        f"| readable: {len(inventory)} | corrupted: {len(corrupted)}"
    )
    print(
        f"Duplicate records: {len(duplicate_rows)} | "
        f"duplicate groups: {len(duplicate_groups)} | "
        f"train/test leakage records: {len(leakage_rows)} "
        f"({len(leakage_groups)} shared hashes)"
    )
    for row in class_rows:
        print(
            f"{row['class_name']}: train={row['train_images']}, "
            f"test={row['test_images']}"
        )
    print(f"Reports saved to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    inspect_dataset()
