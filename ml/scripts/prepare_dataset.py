"""
PestCast — IP102 Dataset Preparation

Downloads the IP102 pest image dataset, selects priority species,
cleans/balances, and splits into train/val/test.
"""

import os
import shutil
import random
import json
import yaml
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# IP102 full class labels (102 classes)
IP102_CLASSES = {
    0: "rice leafroller", 1: "rice leaf caterpillar", 2: "paddy stem maggot",
    3: "asiatic rice borer", 4: "yellow rice borer", 5: "rice shell pest",
    6: "rice water weevil", 7: "rice leafhopper", 8: "corn borer",
    9: "grain spreader thrips", 10: "rice gall midge", 11: "rice stemfly",
    12: "armyworm", 13: "large cutworm", 14: "corn earworm",
    15: "yellow cutworm", 16: "wireworm", 17: "cereal leaf beetle",
    18: "flea beetle", 19: "red spider", 20: "rice leaf roller",
    21: "rice stem borer", 22: "green rice leafhopper", 23: "brown planthopper",
    24: "peach borer", 25: "beet fly", 26: "beet armyworm",
    27: "beet weevil", 28: "fall webworm", 29: "cabbage army worm",
    30: "beet army worm", 31: "english grain aphid", 32: "green peach aphid",
    33: "aphids", 34: "longlegged spider mite", 35: "wheat blossom midge",
    36: "pentatomidae", 37: "wheat sawfly", 38: "whitefly",
    39: "wheat phloeothrips", 40: "wheat aphid", 41: "cotton bollworm",
    42: "mites", 43: "lytta polita", 44: "legume blister beetle",
    45: "blister beetle", 46: "therioaphis maculata", 47: "alfalfa weevil",
    48: "flax budworm", 49: "alfalfa plant bug", 50: "tarnished plant bug",
    51: "locust", 52: "looper", 53: "bollworm",
    54: "cotton aphid", 55: "thrips", 56: "fall armyworm",
    57: "tobacco cutworm", 58: "grub", 59: "mole cricket",
    60: "cabbage worm", 61: "japanese beetle", 62: "colorado potato beetle",
    63: "leaf miner", 64: "stink bug", 65: "ants",
    66: "aphid", 67: "sawfly larva", 68: "slug",
    69: "snail", 70: "sawfly", 71: "stem borer",
    72: "diamondback moth", 73: "cucumber beetle", 74: "flea beetle larvae",
    75: "grasshopper", 76: "leafhopper", 77: "psyllid",
    78: "scale insect", 79: "spider mite", 80: "squash bug",
    81: "tomato hornworm", 82: "rice grasshopper", 83: "weevil",
    84: "plant bug", 85: "cicada", 86: "leaf beetle",
    87: "bark beetle", 88: "chafer", 89: "fruit fly",
    90: "codling moth", 91: "tent caterpillar", 92: "gypsy moth",
    93: "tussock moth", 94: "bagworm", 95: "webworm",
    96: "leaf roller", 97: "cutworm", 98: "moth",
    99: "beetle", 100: "cricket", 101: "earwig",
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_image(img_path: Path) -> bool:
    """Check if an image file is valid and not corrupted."""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def discover_dataset(root_dir: Path) -> dict[int, list[Path]]:
    """
    Discover images organized by class in the IP102 directory structure.
    Supports both flat (class_id/img.jpg) and split (train/class_id/img.jpg) layouts.
    """
    class_images = {}

    # Check for flat layout: root/class_id/images
    flat_dirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if flat_dirs:
        for class_dir in sorted(flat_dirs):
            class_id = int(class_dir.name)
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            if images:
                class_images[class_id] = images
        return class_images

    # Check for split layout: root/train/class_id/images (merge all splits)
    for split in ["train", "val", "test"]:
        split_dir = root_dir / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name.isdigit():
                class_id = int(class_dir.name)
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
                if class_id not in class_images:
                    class_images[class_id] = []
                class_images[class_id].extend(images)

    return class_images


def curate_subset(
    class_images: dict[int, list[Path]],
    target_classes: dict[int, str],
) -> dict[int, list[Path]]:
    """Select only the target pest species."""
    curated = {}
    for class_id, class_name in target_classes.items():
        if class_id in class_images:
            curated[class_id] = class_images[class_id]
            print(f"  Class {class_id:3d} ({class_name:25s}): {len(class_images[class_id])} images")
        else:
            print(f"  Class {class_id:3d} ({class_name:25s}): NOT FOUND in dataset")
    return curated


def clean_dataset(class_images: dict[int, list[Path]]) -> dict[int, list[Path]]:
    """Remove corrupted/invalid images."""
    cleaned = {}
    removed = 0
    for class_id, images in tqdm(class_images.items(), desc="Validating images"):
        valid = []
        for img_path in images:
            if validate_image(img_path):
                valid.append(img_path)
            else:
                removed += 1
        cleaned[class_id] = valid
    print(f"  Removed {removed} corrupted images")
    return cleaned


def balance_dataset(
    class_images: dict[int, list[Path]],
    strategy: str = "oversample",
    max_per_class: int | None = None,
) -> dict[int, list[Path]]:
    """
    Balance class distribution.
    - oversample: duplicate minority class images to match median count
    - undersample: cap each class at min count
    - max_per_class: hard cap per class (for faster training)
    """
    counts = {cid: len(imgs) for cid, imgs in class_images.items()}
    median_count = int(np.median(list(counts.values())))
    target_count = max_per_class or median_count

    print(f"  Class counts before balancing: min={min(counts.values())}, "
          f"max={max(counts.values())}, median={median_count}")
    print(f"  Target per class: {target_count}")

    balanced = {}
    for class_id, images in class_images.items():
        if len(images) >= target_count:
            balanced[class_id] = random.sample(images, target_count)
        elif strategy == "oversample":
            # Oversample by repeating images
            oversampled = list(images)
            while len(oversampled) < target_count:
                oversampled.append(random.choice(images))
            balanced[class_id] = oversampled
        else:
            balanced[class_id] = images

    return balanced


def create_splits(
    class_images: dict[int, list[Path]],
    target_classes: dict[int, str],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Split dataset and copy images into train/val/test directories."""
    random.seed(seed)
    np.random.seed(seed)

    # Create new class index mapping (0 to N-1)
    old_to_new = {}
    labels = {}
    for new_idx, (old_idx, name) in enumerate(sorted(target_classes.items())):
        old_to_new[old_idx] = new_idx
        labels[new_idx] = name

    # Save label mapping
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\n  Saved label mapping to {labels_path}")

    split_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for old_class_id, images in tqdm(sorted(class_images.items()), desc="Creating splits"):
        new_class_id = old_to_new[old_class_id]
        class_name = target_classes[old_class_id]

        # Shuffle and split
        img_list = list(images)
        random.shuffle(img_list)

        # Stratified split
        n_total = len(img_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = img_list[:n_train]
        val_imgs = img_list[n_train:n_train + n_val]
        test_imgs = img_list[n_train + n_val:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_class_dir = output_dir / split_name / str(new_class_id)
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for i, src_path in enumerate(split_imgs):
                dst_path = split_class_dir / f"{class_name.replace(' ', '_')}_{i:04d}{src_path.suffix}"
                shutil.copy2(src_path, dst_path)

            split_counts[split_name][class_name] = len(split_imgs)

    # Print summary
    print("\n  Split Summary:")
    print(f"  {'Class':<25s} {'Train':>6s} {'Val':>6s} {'Test':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for class_name in sorted(split_counts["train"].keys()):
        print(f"  {class_name:<25s} {split_counts['train'][class_name]:>6d} "
              f"{split_counts['val'][class_name]:>6d} {split_counts['test'][class_name]:>6d}")
    total_train = sum(split_counts["train"].values())
    total_val = sum(split_counts["val"].values())
    total_test = sum(split_counts["test"].values())
    print(f"  {'TOTAL':<25s} {total_train:>6d} {total_val:>6d} {total_test:>6d}")

    return labels


def main():
    parser = argparse.ArgumentParser(description="Prepare IP102 dataset for PestCast")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to training config")
    parser.add_argument("--ip102-dir", type=str, required=True,
                        help="Path to downloaded IP102 dataset root")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for curated dataset (default: data/ip102_curated)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Max images per class (for faster iteration)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip image corruption checks")
    args = parser.parse_args()

    config = load_config(args.config)
    target_classes = {int(k): v for k, v in config["dataset"]["target_classes"].items()}
    output_dir = Path(args.output_dir or "data/ip102_curated")
    ip102_root = Path(args.ip102_dir)

    print("=" * 60)
    print("PestCast — IP102 Dataset Preparation")
    print("=" * 60)

    # Step 1: Discover dataset
    print(f"\n[1/5] Discovering images in {ip102_root}...")
    all_classes = discover_dataset(ip102_root)
    print(f"  Found {len(all_classes)} classes, "
          f"{sum(len(v) for v in all_classes.values())} total images")

    # Step 2: Select target species
    print(f"\n[2/5] Selecting {len(target_classes)} target species...")
    curated = curate_subset(all_classes, target_classes)
    print(f"  Selected {sum(len(v) for v in curated.values())} images across {len(curated)} classes")

    # Step 3: Clean
    if not args.skip_validation:
        print("\n[3/5] Validating images (removing corrupted)...")
        curated = clean_dataset(curated)
    else:
        print("\n[3/5] Skipping image validation")

    # Step 4: Balance
    print("\n[4/5] Balancing dataset...")
    balanced = balance_dataset(curated, strategy="oversample", max_per_class=args.max_per_class)

    # Step 5: Split and copy
    print(f"\n[5/5] Creating train/val/test splits in {output_dir}...")
    if output_dir.exists():
        print(f"  Removing existing output dir: {output_dir}")
        shutil.rmtree(output_dir)

    labels = create_splits(
        balanced,
        target_classes,
        output_dir,
        train_ratio=config["splits"]["train"],
        val_ratio=config["splits"]["val"],
        seed=config["splits"]["seed"],
    )

    print(f"\n{'=' * 60}")
    print("Done! Dataset ready at:", output_dir)
    print(f"Labels file: {output_dir / 'labels.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
