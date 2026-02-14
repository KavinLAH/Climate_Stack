#!/usr/bin/env python3
"""
Benchmarking Script for Pest Classification Model
==================================================
- Measures TFLite inference latency (mean, median, p95)
- Reports model file size
- Tests reduced resolution (160x160) if latency > 500ms
- Generates benchmark_report.json
"""

import json
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "curated_data"
TFLITE_PATH = BASE_DIR / "pest_model.tflite"

NUM_BENCHMARK_IMAGES = 100
WARMUP_RUNS = 5


def load_test_images(img_size=224, num_images=NUM_BENCHMARK_IMAGES):
    """Load test images for benchmarking."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=(img_size, img_size),
        batch_size=1,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    images = []
    labels = []
    for img, lbl in test_ds.take(num_images):
        images.append(img.numpy())
        labels.append(lbl.numpy()[0])
    return images, labels


def benchmark_tflite(tflite_path, images, input_size=224):
    """Run inference benchmark and return latency stats + accuracy."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]

    # Check if we need to resize input
    expected_shape = input_details[0]["shape"]
    if expected_shape[1] != input_size:
        interpreter.resize_tensor_input(input_details[0]["index"],
                                         [1, input_size, input_size, 3])
        interpreter.allocate_tensors()

    # Warmup
    for i in range(min(WARMUP_RUNS, len(images))):
        img = images[i]
        if input_dtype == np.uint8:
            input_scale = input_details[0]["quantization"][0]
            input_zero_point = input_details[0]["quantization"][1]
            img = (img / input_scale + input_zero_point).astype(np.uint8)
        else:
            img = img.astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

    # Benchmark
    latencies = []
    predictions = []

    for img in images:
        if input_dtype == np.uint8:
            input_scale = input_details[0]["quantization"][0]
            input_zero_point = input_details[0]["quantization"][1]
            img_input = (img / input_scale + input_zero_point).astype(np.uint8)
        else:
            img_input = img.astype(np.float32)

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        elapsed = (time.perf_counter() - start) * 1000  # ms

        latencies.append(elapsed)
        predictions.append(np.argmax(output[0]))

    return latencies, predictions


def compute_stats(latencies):
    """Compute latency statistics."""
    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
    }


def main():
    print("=" * 60)
    print("Pest Classification — Benchmark")
    print("=" * 60)

    model_size = TFLITE_PATH.stat().st_size / (1024 * 1024)
    print(f"\nModel: {TFLITE_PATH}")
    print(f"Model size: {model_size:.2f} MB")

    # Load test images at 224x224
    print(f"\nLoading {NUM_BENCHMARK_IMAGES} test images (224×224)...")
    images_224, labels = load_test_images(img_size=224, num_images=NUM_BENCHMARK_IMAGES)

    # Benchmark at 224x224
    print("Running inference benchmark (224×224)...")
    latencies_224, preds_224 = benchmark_tflite(TFLITE_PATH, images_224, input_size=224)
    stats_224 = compute_stats(latencies_224)
    acc_224 = np.mean(np.array(preds_224) == np.array(labels))

    print(f"\n{'='*40}")
    print(f"Results @ 224×224")
    print(f"{'='*40}")
    print(f"  Mean latency:   {stats_224['mean_ms']:.1f} ms")
    print(f"  Median latency: {stats_224['median_ms']:.1f} ms")
    print(f"  P95 latency:    {stats_224['p95_ms']:.1f} ms")
    print(f"  Accuracy:       {acc_224:.4f} ({acc_224*100:.1f}%)")

    target_met = stats_224["mean_ms"] < 500
    print(f"  Latency target (<500ms): {'✅ MET' if target_met else '⚠️ EXCEEDED'}")

    report = {
        "model_path": str(TFLITE_PATH),
        "model_size_mb": model_size,
        "num_test_images": NUM_BENCHMARK_IMAGES,
        "results_224": {
            **stats_224,
            "accuracy": float(acc_224),
            "target_met": target_met,
        },
    }

    # If latency too high, also benchmark at 160x160
    if not target_met:
        print(f"\n{'='*40}")
        print(f"Latency too high — testing 160×160 resolution")
        print(f"{'='*40}")

        images_160, _ = load_test_images(img_size=160, num_images=NUM_BENCHMARK_IMAGES)
        latencies_160, preds_160 = benchmark_tflite(TFLITE_PATH, images_160, input_size=160)
        stats_160 = compute_stats(latencies_160)
        acc_160 = np.mean(np.array(preds_160) == np.array(labels))

        print(f"  Mean latency:   {stats_160['mean_ms']:.1f} ms")
        print(f"  Median latency: {stats_160['median_ms']:.1f} ms")
        print(f"  P95 latency:    {stats_160['p95_ms']:.1f} ms")
        print(f"  Accuracy:       {acc_160:.4f} ({acc_160*100:.1f}%)")

        target_met_160 = stats_160["mean_ms"] < 500
        print(f"  Latency target (<500ms): {'✅ MET' if target_met_160 else '⚠️ EXCEEDED'}")

        report["results_160"] = {
            **stats_160,
            "accuracy": float(acc_160),
            "target_met": target_met_160,
        }

    # Tradeoff summary
    print(f"\n{'='*60}")
    print("Tradeoff Summary")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Size (MB)':>10} {'Latency (ms)':>13} {'Accuracy':>10}")
    print("-" * 52)
    print(f"{'INT8 224×224':<15} {model_size:>10.2f} {stats_224['mean_ms']:>13.1f} {acc_224:>10.1%}")
    if "results_160" in report:
        print(f"{'INT8 160×160':<15} {model_size:>10.2f} {stats_160['mean_ms']:>13.1f} {acc_160:>10.1%}")

    # Save report
    report_path = BASE_DIR / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark report saved → {report_path}")


if __name__ == "__main__":
    main()
