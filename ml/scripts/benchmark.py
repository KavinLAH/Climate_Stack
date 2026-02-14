"""
PestCast — Model Benchmarking & Evaluation

Measures inference latency, accuracy metrics, per-class performance,
and generates confusion matrix visualizations.
"""

import json
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import numpy as np
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix, top_k_accuracy_score,
)
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> nn.Module:
    arch = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    dropout = config["model"]["dropout"]

    model = timm.create_model(arch, pretrained=False)
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    return model


def benchmark_pytorch_latency(model: nn.Module, input_size: int, device: torch.device,
                               num_runs: int = 100, warmup: int = 10) -> dict:
    """Measure PyTorch inference latency."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "device": str(device),
        "num_runs": num_runs,
    }


def benchmark_tflite_latency(tflite_path: Path, input_size: int,
                              num_runs: int = 100, warmup: int = 10) -> dict:
    """Measure TFLite inference latency."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]["dtype"]
    dummy = np.random.randn(1, input_size, input_size, 3).astype(input_dtype)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    latencies = []
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        start = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
        "model_file": tflite_path.name,
        "model_size_mb": tflite_path.stat().st_size / 1024 / 1024,
        "num_runs": num_runs,
    }


def evaluate_accuracy(model: nn.Module, data_dir: Path, config: dict,
                       device: torch.device) -> dict:
    """Full accuracy evaluation on test set."""
    input_size = config["model"]["input_size"]
    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4,
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top1 = (all_preds == all_labels).mean()
    top3 = top_k_accuracy_score(all_labels, all_probs, k=min(3, all_probs.shape[1]))

    return {
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray,
                           class_names: list[str], output_path: Path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {output_path}")


def plot_training_history(history_path: Path, output_path: Path):
    """Plot training curves from history JSON."""
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(len(history["train_loss"]))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history["train_top1"], label="Train Top-1")
    axes[1].plot(epochs, history["val_top1"], label="Val Top-1")
    axes[1].plot(epochs, history["val_top3"], label="Val Top-3")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].axhline(y=0.85, color="r", linestyle="--", alpha=0.5, label="Target (85%)")
    axes[1].legend()
    axes[1].grid(True)

    # Learning Rate
    axes[2].plot(epochs, history["lr"])
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PestCast model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pth")
    parser.add_argument("--data-dir", type=str, default="data/ip102_curated")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tflite-dir", type=str, default="models/export",
                        help="Directory containing TFLite models to benchmark")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    logs_dir = Path(config["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load labels
    with open(data_dir / "labels.json") as f:
        labels_map = json.load(f)
    class_names = [labels_map[str(i)] for i in range(len(labels_map))]

    print("=" * 60)
    print("PestCast — Model Benchmarking")
    print("=" * 60)

    results = {}

    # --- PyTorch Model ---
    print(f"\n[1/4] Loading PyTorch model on {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Latency
    print("\n[2/4] Benchmarking PyTorch inference latency...")
    pytorch_latency = benchmark_pytorch_latency(
        model, config["model"]["input_size"], device
    )
    results["pytorch_latency"] = pytorch_latency
    print(f"  Mean: {pytorch_latency['mean_ms']:.1f}ms | "
          f"P95: {pytorch_latency['p95_ms']:.1f}ms | "
          f"Device: {device}")

    # Accuracy
    print("\n[3/4] Evaluating accuracy on test set...")
    accuracy = evaluate_accuracy(model, data_dir, config, device)
    results["pytorch_accuracy"] = {
        "top1": accuracy["top1_accuracy"],
        "top3": accuracy["top3_accuracy"],
    }
    print(f"  Top-1: {accuracy['top1_accuracy']:.4f}")
    print(f"  Top-3: {accuracy['top3_accuracy']:.4f}")

    # Per-class report
    report = classification_report(
        accuracy["labels"], accuracy["predictions"],
        target_names=class_names, output_dict=True,
    )
    results["per_class"] = report
    print(f"\n  Per-class report:")
    print(classification_report(
        accuracy["labels"], accuracy["predictions"],
        target_names=class_names,
    ))

    # Confusion matrix
    plot_confusion_matrix(
        accuracy["labels"], accuracy["predictions"],
        class_names, logs_dir / "confusion_matrix.png"
    )

    # Training curves
    history_path = logs_dir / "training_history.json"
    if history_path.exists():
        plot_training_history(history_path, logs_dir / "training_curves.png")

    # --- TFLite Models ---
    tflite_dir = Path(args.tflite_dir)
    if tflite_dir.exists():
        print("\n[4/4] Benchmarking TFLite models...")
        tflite_results = {}
        for tflite_file in sorted(tflite_dir.glob("pest_model*.tflite")):
            try:
                latency = benchmark_tflite_latency(
                    tflite_file, config["model"]["input_size"]
                )
                tflite_results[tflite_file.name] = latency
                print(f"  {tflite_file.name}: mean={latency['mean_ms']:.1f}ms, "
                      f"size={latency['model_size_mb']:.1f}MB")
            except Exception as e:
                print(f"  {tflite_file.name}: FAILED ({e})")
        results["tflite_latency"] = tflite_results
    else:
        print("\n[4/4] No TFLite models found, skipping")

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    results["model_info"] = {
        "architecture": config["model"]["architecture"],
        "num_classes": config["model"]["num_classes"],
        "input_size": config["model"]["input_size"],
        "total_params": param_count,
        "total_params_millions": round(param_count / 1e6, 2),
    }

    # Save results
    results_path = logs_dir / "benchmark_results.json"
    # Remove non-serializable numpy arrays
    serializable = {k: v for k, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {config['model']['architecture']}")
    print(f"  Params: {param_count/1e6:.2f}M")
    print(f"  Top-1 Acc: {accuracy['top1_accuracy']:.4f}")
    print(f"  Top-3 Acc: {accuracy['top3_accuracy']:.4f}")
    print(f"  PyTorch Latency: {pytorch_latency['mean_ms']:.1f}ms ({device})")
    target_met = accuracy["top3_accuracy"] >= 0.85
    print(f"  Target (>85% top-3): {'MET' if target_met else 'NOT MET'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
