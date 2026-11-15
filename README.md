# Agricultural Pest Classification with ResNet152V2

This project uses transfer learning with an ImageNet-pretrained ResNet152V2
model to classify agricultural pests. It uses ordinary Python scripts (no
notebooks), discovers class names from the dataset folders, and never modifies
or deletes the original images.

## Dataset layout

The project detects one dataset root under `dataset/` that contains paired
`train/` and `test/` folders. Each split contains one subfolder per class. For
example:

```text
dataset/
└── pest/
    ├── train/
    │   ├── aphids/
    │   └── ...
    └── test/
        ├── aphids/
        └── ...
```

Class folders must match between train and test. All image class names are
detected automatically; the scripts do not assume a fixed number of classes.

## Environment setup (Windows PowerShell)

The existing `.venv` is configured with **64-bit Python 3.13** and TensorFlow
2.21.0. Use this environment in VS Code and install/update the project packages
there:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a new machine, install 64-bit Python 3.13, then create and populate an
environment (use a different directory name if `.venv` already exists):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the VS Code interpreter. The tested
TensorFlow setup runs on CPU on native Windows; GPU acceleration requires a
separately configured supported environment such as WSL2. Training ResNet152V2
on CPU may be slow.

## Execution workflow

Run commands from the project root using the selected environment's Python:

```powershell
python -m src.dataset_inspection
python -m src.train --batch-size 16 --head-epochs 20 --fine-tune-epochs 15
python -m src.evaluate --batch-size 16
python -m src.predict
```

The last command opens a file picker. Choose a local image to classify. You can
also pass an image path directly:

```powershell
python -m src.predict "path\to\new_image.jpg"
```

1. **Inspection:** verifies images and writes `image_inventory.csv`,
   `class_counts.csv`, `corrupted_images.csv`, `exact_duplicates.csv`,
   `train_test_duplicate_leakage.csv`, `dataset_distribution.png`, and
   `dataset_summary.json` into `results/`. Duplicates and corrupt files are
   reported, not silently deleted. Manually resolve corruption and train/test
   leakage before treating test results as final.
2. **Preprocessing:** resizes to 224×224 RGB, uses a seeded stratified split of
   `train/` into training and validation subsets, and leaves `test/` untouched.
   The model applies ResNetV2's `[-1, 1]` input scaling. Random augmentation
   runs only while training.
3. **Training:** first freezes the ResNet base and trains the classifier, then
   fine-tunes only the requested final base layers with Batch Normalization
   layers frozen. Early stopping, best-only model checkpoints, learning-rate
   reduction, dropout, augmentation, and TensorBoard/CSV logs are enabled.
   Adjust `--batch-size` (e.g. 8 or 4) if memory is insufficient.
4. **Evaluation:** evaluates only the untouched `test/` split and writes
   accuracy, macro/weighted precision-recall/F1, macro F1, ROC-AUC when
   computable, reports, confusion matrix, and plots to `results/`. Do not use
   test results for model selection.
5. **Prediction:** loads the saved model and labels; it does not train again.
   Run without an image argument to select one using the file picker.

Training settings can be changed with `--validation-split`, `--patience`,
`--fine-tune-layers`, `--head-learning-rate`, `--fine-tune-learning-rate`,
`--seed`, and the epoch/batch-size flags shown by:

```powershell
python -m src.train --help
```

## Model files and sharing

Share these files together for prediction without retraining:

- `models/pest_resnet152v2_final.keras` — complete architecture and learned
  weights in Keras format.
- `models/class_names.json` — output index-to-class mapping.
- `requirements.txt` — runtime dependencies.
- `src/` — prediction code and project configuration (the recipient does not
  need the dataset, training scripts to be executed, or the training computer).

The recipient installs a compatible TensorFlow environment, puts the model and
label files under `models/`, then runs `python -m src.predict "image.jpg"`.
`training_metadata.json` and `model_config.json` are useful provenance files but
are not required for prediction.

## Outputs

- `models/`: final and best models, stage checkpoints, class labels, model
  configuration, and training metadata.
- `results/`: dataset inspection, training history/plots, and test evaluation.
- `logs/`: training log, epoch CSV metrics, and TensorBoard event files.
