#!/usr/bin/env python3
"""
Evaluation Script for Pest Classification Model
================================================
- Computes top-1 and top-3 accuracy on the test set
- Generates per-class precision/recall/F1
- Produces confusion matrix heatmap
"""

import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "curated_data"
MODEL_PATH = BASE_DIR / "pest_model.keras"
LABELS_PATH = DATA_DIR / "labels.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 18


def load_test_data():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    return test_ds


def top_k_accuracy(y_true, y_probs, k=3):
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = sum(1 for i, true_label in enumerate(y_true)
                  if true_label in top_k_preds[i])
    return correct / len(y_true)


def plot_confusion_matrix(cm, class_names, save_path):
    """Generate and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=16, pad=20)
    fig.colorbar(im, ax=ax, shrink=0.8)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved → {save_path}")


def main():
    print("=" * 60)
    print("Pest Classification — Evaluation")
    print("=" * 60)

    # Load labels
    with open(LABELS_PATH) as f:
        labels_map = json.load(f)
    class_names = [labels_map[str(i)] for i in range(NUM_CLASSES)]

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load test data
    print("Loading test dataset...")
    test_ds = load_test_data()

    # Collect all predictions and ground truth
    all_labels = []
    all_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        all_probs.append(probs)
        all_labels.extend(labels.numpy().tolist())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.argmax(all_probs, axis=1)

    # Metrics
    top1 = np.mean(all_preds == all_labels)
    top3 = top_k_accuracy(all_labels, all_probs, k=3)

    print(f"\n{'='*60}")
    print(f"Test Results ({len(all_labels)} images)")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {top1:.4f} ({top1*100:.1f}%)")
    print(f"Top-3 Accuracy: {top3:.4f} ({top3*100:.1f}%)")
    target = "✅ PASSED" if top3 > 0.85 else "❌ BELOW TARGET"
    print(f"Top-3 Target (>85%): {target}")

    # Classification report
    print(f"\n{'='*60}")
    print("Per-Class Classification Report")
    print(f"{'='*60}")
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=3)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)

    # Save results JSON
    results = {
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "num_test_images": len(all_labels),
        "target_met": top3 > 0.85,
    }
    results_path = BASE_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
