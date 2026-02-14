#!/usr/bin/env python3
"""
Training Script for Pest Classification Model
==============================================
Fine-tunes EfficientNetB0 on the curated 18-class pest dataset.
- Phase 1: Train classifier head (base frozen) for N epochs
- Phase 2: Unfreeze last 20 layers, fine-tune at low LR
- Exports pest_model.keras and training history plots
"""

import argparse
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "curated_data"
MODEL_PATH = BASE_DIR / "pest_model.keras"
PLOT_PATH = BASE_DIR / "training_curves.png"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 18

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train pest classification model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs for head training (default: 20)")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="Epochs for fine-tuning (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip fine-tuning phase")
    return parser.parse_args()

# ── Data loading ─────────────────────────────────────────────────────────────

def load_datasets():
    """Load train/val datasets from curated_data directory."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


def compute_class_weights(train_ds):
    """Compute class weights inversely proportional to class frequency."""
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy().tolist())

    counts = np.bincount(all_labels, minlength=NUM_CLASSES)
    total = len(all_labels)
    weights = {}
    for i in range(NUM_CLASSES):
        if counts[i] > 0:
            weights[i] = total / (NUM_CLASSES * counts[i])
        else:
            weights[i] = 1.0
    return weights

# ── Model building ───────────────────────────────────────────────────────────

def build_model():
    """Build EfficientNetB0 with frozen base + classification head."""
    # EfficientNetB0 expects inputs in [0, 255] range and does its own preprocessing
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    return model, base_model


def compile_model(model, lr):
    """Compile with Adam optimizer and sparse categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

# ── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(phase="head"):
    """Standard callbacks for training."""
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(histories, save_path):
    """Plot accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epoch_offset = 0
    colors = ["#2196F3", "#FF5722"]

    for idx, (phase_name, hist) in enumerate(histories):
        epochs = range(epoch_offset + 1, epoch_offset + len(hist.history["accuracy"]) + 1)

        ax1.plot(epochs, hist.history["accuracy"], "-o", color=colors[idx],
                 label=f"{phase_name} train", markersize=3)
        ax1.plot(epochs, hist.history["val_accuracy"], "--s", color=colors[idx],
                 label=f"{phase_name} val", markersize=3, alpha=0.7)

        ax2.plot(epochs, hist.history["loss"], "-o", color=colors[idx],
                 label=f"{phase_name} train", markersize=3)
        ax2.plot(epochs, hist.history["val_loss"], "--s", color=colors[idx],
                 label=f"{phase_name} val", markersize=3, alpha=0.7)

        epoch_offset += len(hist.history["accuracy"])

    ax1.set_title("Accuracy", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Loss", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved → {save_path}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("Pest Classification — Training")
    print("=" * 60)

    # Load data
    print("\nLoading datasets...")
    train_ds, val_ds = load_datasets()

    # Class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(train_ds)

    # Build model
    print("Building EfficientNetB0 model...")
    model, base_model = build_model()
    compile_model(model, lr=args.lr)
    model.summary()

    histories = []

    # ── Phase 1: Head training ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Training classifier head ({args.epochs} epochs, lr={args.lr})")
    print(f"{'='*60}")

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=get_callbacks("head"),
    )
    histories.append(("Head", hist1))

    # ── Phase 2: Fine-tuning ─────────────────────────────────────────────
    if not args.no_finetune and args.finetune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 2: Fine-tuning last 20 layers ({args.finetune_epochs} epochs)")
        print(f"{'='*60}")

        # Unfreeze last 20 layers of the base
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        fine_lr = args.lr * 0.01  # 1e-5
        compile_model(model, lr=fine_lr)

        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.finetune_epochs,
            class_weight=class_weights,
            callbacks=get_callbacks("finetune"),
        )
        histories.append(("Fine-tune", hist2))

    # Save final plots
    plot_history(histories, PLOT_PATH)

    # Final val metrics
    print(f"\n{'='*60}")
    print("Final Evaluation on Validation Set")
    print(f"{'='*60}")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
