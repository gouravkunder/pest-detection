"""Shared project paths and dataset discovery."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SEARCH_ROOT = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _has_class_folders(split_dir: Path) -> bool:
    return split_dir.is_dir() and any(
        child.is_dir() for child in split_dir.iterdir()
    )


def discover_dataset_root(search_root: Path = DATASET_SEARCH_ROOT) -> Path:
    """Find a unique directory containing train/ and test/ class folders."""
    if not search_root.exists():
        raise FileNotFoundError(f"Dataset search folder not found: {search_root}")

    candidates = []
    for train_dir in search_root.rglob("train"):
        if not train_dir.is_dir():
            continue
        test_dir = train_dir.parent / "test"
        if _has_class_folders(train_dir) and _has_class_folders(test_dir):
            candidates.append(train_dir.parent.resolve())

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "Could not find train/ and test/ folders with class subfolders "
            f"under {search_root}. Expected e.g. dataset/extracted/train and "
            "dataset/extracted/test."
        )
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            "More than one dataset root was found. Keep one dataset under "
            f"{search_root} or configure discovery explicitly: {paths}"
        )
    return candidates[0]


def dataset_directories():
    """Return the discovered dataset root and its train/test folders."""
    dataset_root = discover_dataset_root()
    return dataset_root, dataset_root / "train", dataset_root / "test"
