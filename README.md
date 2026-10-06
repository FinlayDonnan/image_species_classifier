# Kruger Camera-Trap Species Classifier

A ResNet-18 image classifier that identifies wildlife species from camera-trap images collected in Kruger National Park (2015–2016). The project is split into two stages: a preprocessing/data-splitting script (`preprocess.py`) and a training/evaluation script (`train.py`).

This work was carried out as part of an EPSRC-funded undergraduate research project.

## Overview

Camera-trap annotations are cleaned, filtered down to species with enough images to train on, and split into a **fixed test set** and a **training pool**. The fixed test set is built once and reused across every experiment, so that results from different training configurations (i.e. different mixes of 2015 vs 2016 images) are directly comparable. A ResNet-18, pretrained on ImageNet, is then fine-tuned on the resulting training set and evaluated on the fixed test set, with accuracy broken down overall, per species, per year, and per day/night.

## Repository structure

```
.
├── preprocess.py         # cleans annotations, builds train/test label CSVs
├── train.py              # trains and evaluates the ResNet-18 classifier
├── requirements.txt
├── data/                 # not included- see "Data setup" below
│   ├── Kruger_2015_16_annotations_dupes_removed_final4.csv
│   ├── 2015/
│   └── 2016/
└── output/               # created automatically by preprocess.py / train.py
```

## Setup

1. Clone the repository and install dependencies (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. A GPU with CUDA is strongly recommended for `train.py` but not required- it will fall back to CPU automatically.

## Data setup

This repository does not include the raw camera-trap images or annotations (too large / not public). To run the pipeline, place your data in a `data/` folder next to the scripts, matching this layout:

```
data/
├── Kruger_2015_16_annotations_dupes_removed_final4.csv
├── 2015/
│   └── <site>/<date_range>/<camera>/<folder>/<filename>
└── 2016/
    └── <site>/<date_range>/<camera>/<folder>/<filename>
```

The annotation CSV is expected to have (at least) the following columns: `site`, `camera`, `folder`, `filename`, `date_range`, `timestamp`, `binomial`, `species`.

If your data is laid out differently, adjust `DATA_DIR`, `input_csv`, `image_root_2015`, `image_root_2016`, and `build_path()` at the top of `preprocess.py`.

## Usage

### 1. Preprocess the data

```bash
python preprocess.py
```

This will:
- Drop exact-duplicate annotation rows, `unknown`/`rare` binomials, and the `Nwaswitshaka` site.
- Collapse annotation rows into one row per image, keeping only images with a single labeled species (discarding multi-species images and unlabeled/empty frames).
- Add `year` (from the timestamp) and `day_night` (from average Kruger sunrise/sunset times) columns.
- Keep only species with at least `threshold` (default 300) images in total.
- Build a **fixed test set**: for each species and year, a fraction (`test_proportion`, default 25%, with a floor of `min_test_images_per_year` images) is reserved for testing, sampled evenly across camera sites. This test set and the remaining training pool are cached to `output/test_pool_raw.csv` / `output/train_pool_raw.csv` so that every experiment is evaluated on identical test images.
- Build a training set for the selected `EXPERIMENT_NAME`, drawing a target number of 2015 and 2016 images per species from the training pool (site-balanced), duplicating images where a species doesn't have enough unique images to reach its target.
- Write `output/train_labels.csv`, `output/test_labels.csv`, and `output/label_map.json`.

**Key configuration options** (top of `preprocess.py`):

| Variable | Purpose |
|---|---|
| `threshold` | Minimum total images for a species to be kept |
| `test_proportion` / `min_test_images_per_year` | Controls the size of the fixed test set |
| `EXPERIMENTS` / `EXPERIMENT_NAME` | Defines the available training-ratio configurations (2015 vs 2016 image counts per species) and which one to build this run |
| `FORCE_REBUILD_TEST_SET` | Set to `True` to discard and rebuild the fixed test set from scratch. Leave `False` so repeated runs reuse the same test images |
| `random_state` | Random seed for all sampling/duplication steps |

To compare training configurations, run `preprocess.py` once per entry in `EXPERIMENTS` (changing `EXPERIMENT_NAME` each time, leaving `FORCE_REBUILD_TEST_SET = False`), then run `train.py` on each resulting `output/train_labels.csv`.

### 2. Train and evaluate the model

```bash
python train.py
```

This will:
- Load `output/train_labels.csv`, `output/test_labels.csv`, and `output/label_map.json`.
- Fine-tune an ImageNet-pretrained ResNet-18 (cropping off the 100px timestamp strip at the bottom of each image, with standard augmentation on the training set).
- After each epoch, evaluate on the fixed test set and print overall accuracy plus breakdowns by species, year, and day/night.
- Save the best-performing checkpoint (by test accuracy) to `resnet18_species.pt`.
- Run a final evaluation of the best checkpoint, printing a confusion matrix and a full classification report (precision/recall/F1 per species).

**Key configuration options** (top of `train.py`):

| Variable | Purpose |
|---|---|
| `batch_size`, `num_epochs`, `learning_rate` | Standard training hyperparameters |
| `feature_extract_only` | `True` freezes all pretrained layers and trains only the final classification layer, `False` fine-tunes the whole network |
| `crop_bottom_px` | Pixels cropped from the bottom of each image (default 100, to remove the camera's timestamp overlay) |
| `seed` | Random seed- the whole pipeline (model init, data shuffling, augmentation) is seeded for reproducibility |

## Reproducibility

Both scripts are fully seeded: `torch`, `numpy`, and `random` are seeded, cuDNN is forced into deterministic mode, and `DataLoader` workers use a seeded generator. Running the same configuration again with the same `random_state`/`seed` produces identical splits.

## Outputs

All outputs are written to `./output/`, created automatically if it doesn't exist:

| File | Description |
|---|---|
| `multi_species_summary.csv` | Images excluded for containing more than one annotation row (multi-species or repeated-species frames) with species lists, for review |
| `multi_species_exclude_list.csv` | Minimal key list of the same excluded images |
| `test_pool_raw.csv` | The fixed test set (same across every experiment) |
| `train_pool_raw.csv` | The pool of images available for training, after reserving the fixed test set |
| `train_labels.csv` | Final training image list for the selected experiment, with labels and file paths |
| `test_labels.csv` | Final test image list, with labels and file paths |
| `label_map.json` | Mapping from species name (`binomial`) to integer class ID |
| `resnet18_species.pt` | Best model checkpoint (produced by `train.py`), containing `model_state_dict` and `label_map` |

## Notes

- Species are identified by `binomial` name. Images annotated only with a common `species` name but no `binomial`, or with more than one distinct species in frame, are excluded from training (see `multi_species_summary.csv` for what was dropped and why).
- Day/night labeling uses Kruger's approximate average sunrise (06:22) and sunset (17:26) times applied to each image's timestamp. It is not used as a training signal, only as an evaluation breakdown.
- `year` and `day_night` are likewise not used as model inputs- they are returned alongside each image purely so that test accuracy can be broken down along those dimensions.
