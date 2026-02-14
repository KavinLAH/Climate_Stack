#!/usr/bin/env python3
"""
Data Preparation for Pest Classification Model
================================================
Curates a balanced 18-species subset from the IP102 dataset.
- Filters to selected high-priority pest species
- Re-maps class IDs to contiguous 0-17 range
- Oversamples rare classes via augmentation (flip, rotation, color jitter)
- Caps large classes at 1500 images
- Copies images into curated_data/{train,val,test}/{0..17}/
- Exports labels.json and dataset_stats.json
"""

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_DIR = BASE_DIR / "classification"
OUTPUT_DIR = BASE_DIR / "curated_data"

# IP102 class ID → human-readable name
SELECTED_CLASSES = {
    23: "corn borer",
    24: "army worm",
    22: "red spider mite",
    8:  "brown plant hopper",
    9:  "white backed plant hopper",
    72: "whitefly",
    39: "cabbage army worm",
    40: "beet army worm",
    58: "plant bug (Apolygus)",
    48: "tarnished plant bug",
    87: "tobacco cutworm",
    15: "grub",
    38: "flea beetle",
    49: "locust",
    0:  "rice leaf roller",
    68: "spotted lanternfly",
    70: "leafhopper",
    25: "aphids",
}

# Sorted IP102 IDs for deterministic new-ID mapping
SORTED_IP102_IDS = sorted(SELECTED_CLASSES.keys())
IP102_TO_NEW = {ip_id: new_id for new_id, ip_id in enumerate(SORTED_IP102_IDS)}

OVERSAMPLE_TARGET = 300   # Minimum per class (train)
DOWNSAMPLE_CAP = 1500     # Maximum per class (train)

RANDOM_SEED = 42

# ── Augmentation helpers ─────────────────────────────────────────────────────

random.seed(RANDOM_SEED)


def augment_image(img: Image.Image) -> Image.Image:
    """Apply random augmentation: flip, rotation, brightness/contrast jitter."""
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    # Brightness jitter
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    # Contrast jitter
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img


# ── Parsing helpers ──────────────────────────────────────────────────────────

def parse_annotation_file(filepath: Path):
    """Parse train.txt / val.txt / test.txt → list of (filename, class_id)."""
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], int(parts[1])))
    return entries


def build_labels_json():
    """Create labels.json: new_id → pest name."""
    labels = {}
    for ip_id in SORTED_IP102_IDS:
        new_id = IP102_TO_NEW[ip_id]
        labels[str(new_id)] = SELECTED_CLASSES[ip_id]
    return labels


# ── Main pipeline ────────────────────────────────────────────────────────────

def curate_split(split_name: str, entries: list, balance: bool = False):
    """
    Filter entries to selected classes, copy images, optionally balance.
    Returns per-class counts (before and after balancing).
    """
    selected = [(fn, cls_id) for fn, cls_id in entries if cls_id in IP102_TO_NEW]
    print(f"  [{split_name}] Selected {len(selected)} images from {len(entries)} total")

    # Group by new class ID
    class_images: dict[int, list[str]] = {}
    for fn, cls_id in selected:
        new_id = IP102_TO_NEW[cls_id]
        class_images.setdefault(new_id, []).append((fn, cls_id))

    counts_before = {cid: len(imgs) for cid, imgs in class_images.items()}
    counts_after = {}

    for new_id in sorted(class_images.keys()):
        out_dir = OUTPUT_DIR / split_name / str(new_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        images = class_images[new_id]
        ip102_id = SORTED_IP102_IDS[new_id]
        src_dir = CLASSIFICATION_DIR / split_name / str(ip102_id)

        copied = 0

        if balance:
            # Downsample if over cap
            if len(images) > DOWNSAMPLE_CAP:
                random.shuffle(images)
                images = images[:DOWNSAMPLE_CAP]

            # Copy original images
            for fn, _ in images:
                src = src_dir / fn
                if src.exists():
                    dst = out_dir / fn
                    shutil.copy2(src, dst)
                    copied += 1

            # Oversample if under target
            if copied < OVERSAMPLE_TARGET:
                needed = OVERSAMPLE_TARGET - copied
                print(f"    Class {new_id} ({SELECTED_CLASSES[ip102_id]}): "
                      f"{copied} originals, generating {needed} augmented copies")
                source_files = [out_dir / fn for fn, _ in images if (out_dir / fn).exists()]
                if source_files:
                    for i in range(needed):
                        src_file = random.choice(source_files)
                        try:
                            img = Image.open(src_file).convert("RGB")
                            aug_img = augment_image(img)
                            aug_name = f"aug_{i:04d}_{src_file.name}"
                            aug_img.save(out_dir / aug_name, "JPEG", quality=95)
                            copied += 1
                        except Exception as e:
                            print(f"    Warning: could not augment {src_file.name}: {e}")
        else:
            # Val / Test: just copy, no balancing
            for fn, _ in images:
                src = src_dir / fn
                if src.exists():
                    dst = out_dir / fn
                    shutil.copy2(src, dst)
                    copied += 1

        counts_after[new_id] = copied

    return counts_before, counts_after


def main():
    print("=" * 60)
    print("Pest Classification — Data Preparation")
    print("=" * 60)

    # Clean output directory
    if OUTPUT_DIR.exists():
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Parse annotation files
    train_entries = parse_annotation_file(BASE_DIR / "train.txt")
    val_entries = parse_annotation_file(BASE_DIR / "val.txt")
    test_entries = parse_annotation_file(BASE_DIR / "test.txt")

    print(f"\nTotal entries: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")
    print(f"Selected {len(SELECTED_CLASSES)} species, new IDs 0-{len(SELECTED_CLASSES)-1}\n")

    # Curate each split
    stats = {}

    print("Processing train split (with balancing)...")
    before, after = curate_split("train", train_entries, balance=True)
    stats["train"] = {"before": before, "after": after}

    print("\nProcessing val split...")
    before, after = curate_split("val", val_entries, balance=False)
    stats["val"] = {"before": before, "after": after}

    print("\nProcessing test split...")
    before, after = curate_split("test", test_entries, balance=False)
    stats["test"] = {"before": before, "after": after}

    # Save labels.json
    labels = build_labels_json()
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved labels.json → {labels_path}")

    # Also save to base dir for deployment
    shutil.copy2(labels_path, BASE_DIR / "labels.json")

    # Save dataset stats
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset_stats.json → {stats_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"{'ID':>4}  {'Species':<30}  {'Train':>6}  {'Val':>5}  {'Test':>5}")
    print("-" * 60)
    total_train = total_val = total_test = 0
    for new_id in range(len(SELECTED_CLASSES)):
        ip_id = SORTED_IP102_IDS[new_id]
        name = SELECTED_CLASSES[ip_id]
        tr = stats["train"]["after"].get(new_id, 0)
        va = stats["val"]["after"].get(new_id, 0)
        te = stats["test"]["after"].get(new_id, 0)
        total_train += tr
        total_val += va
        total_test += te
        print(f"{new_id:>4}  {name:<30}  {tr:>6}  {va:>5}  {te:>5}")
    print("-" * 60)
    print(f"{'':>4}  {'TOTAL':<30}  {total_train:>6}  {total_val:>5}  {total_test:>5}")
    print(f"\nDone! Curated dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
