"""
PestCast — TFLite Export & Quantization

Converts the trained PyTorch model to TFLite with INT8 quantization
for on-device inference.

Pipeline: PyTorch → ONNX → TFLite (INT8)
"""

import os
import json
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import numpy as np
import timm
from PIL import Image
from torchvision import transforms, datasets


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> nn.Module:
    """Rebuild the model architecture (same as train.py)."""
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


def export_to_onnx(model: nn.Module, input_size: int, onnx_path: Path):
    """Export PyTorch model to ONNX."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"  ONNX model saved: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")


def convert_onnx_to_tflite(onnx_path: Path, tflite_path: Path, quantize: bool = False,
                            calibration_data: np.ndarray | None = None):
    """Convert ONNX model to TFLite, optionally with INT8 quantization."""
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    # ONNX -> TF SavedModel
    print("  Converting ONNX -> TF SavedModel...")
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    saved_model_dir = onnx_path.parent / "tf_saved_model"
    tf_rep.export_graph(str(saved_model_dir))

    # TF SavedModel -> TFLite
    print("  Converting TF SavedModel -> TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quantize and calibration_data is not None:
        print("  Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_representative_dataset(calibration_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"  TFLite model saved: {tflite_path} ({tflite_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return tflite_path


def convert_onnx_to_tflite_via_onnx2tf(onnx_path: Path, export_dir: Path,
                                         quantize: bool = False,
                                         calibration_data: np.ndarray | None = None):
    """
    Alternative: use onnx2tf which is more reliable for many model architectures.
    Falls back to this if onnx-tf fails.
    """
    import subprocess
    import tensorflow as tf

    saved_model_dir = export_dir / "tf_saved_model"

    # onnx2tf conversion
    print("  Converting ONNX -> TF SavedModel via onnx2tf...")
    cmd = [
        "onnx2tf", "-i", str(onnx_path),
        "-o", str(saved_model_dir),
        "-osd",  # output saved model
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # TF SavedModel -> TFLite
    print("  Converting TF SavedModel -> TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # Float16 quantization (good balance of size reduction + compatibility)
    tflite_f16_path = export_dir / "pest_model_f16.tflite"
    converter_f16 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter_f16.convert()
    with open(tflite_f16_path, "wb") as f:
        f.write(tflite_f16)
    print(f"  Float16 TFLite: {tflite_f16_path} ({tflite_f16_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Full INT8 quantization
    if quantize and calibration_data is not None:
        tflite_int8_path = export_dir / "pest_model_int8.tflite"
        converter_int8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = _make_representative_dataset(calibration_data)
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.uint8
        converter_int8.inference_output_type = tf.uint8

        try:
            tflite_int8 = converter_int8.convert()
            with open(tflite_int8_path, "wb") as f:
                f.write(tflite_int8)
            print(f"  INT8 TFLite: {tflite_int8_path} ({tflite_int8_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  INT8 quantization failed: {e}")
            print("  Using Float16 model as fallback")

    # Also export a dynamic range quantized version (simplest, no calibration data needed)
    tflite_dynq_path = export_dir / "pest_model_dynq.tflite"
    converter_dyn = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dyn = converter_dyn.convert()
    with open(tflite_dynq_path, "wb") as f:
        f.write(tflite_dyn)
    print(f"  Dynamic range TFLite: {tflite_dynq_path} ({tflite_dynq_path.stat().st_size / 1024 / 1024:.1f} MB)")


def _make_representative_dataset(calibration_data: np.ndarray):
    """Generator for TFLite INT8 calibration."""
    def representative_dataset():
        for i in range(len(calibration_data)):
            yield [calibration_data[i:i+1].astype(np.float32)]
    return representative_dataset


def gather_calibration_data(data_dir: Path, input_size: int, num_samples: int = 200) -> np.ndarray:
    """Load a subset of training images as calibration data for quantization."""
    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_dir / "train", transform=transform)

    # Sample evenly from dataset
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    samples = []
    for idx in indices:
        img, _ = dataset[idx]
        samples.append(img.numpy())

    return np.array(samples)


def validate_tflite(tflite_path: Path, data_dir: Path, input_size: int, labels: dict):
    """Run inference on test set with TFLite model and report accuracy."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"]

    print(f"\n  Validating TFLite model: {tflite_path.name}")
    print(f"  Input: {input_shape}, dtype: {input_dtype}")

    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform)

    correct_top1 = 0
    correct_top3 = 0
    total = 0
    latencies = []

    for img, label in test_dataset:
        img_np = img.numpy()[np.newaxis, ...]  # (1, 3, H, W)

        # TFLite expects NHWC
        img_np = np.transpose(img_np, (0, 2, 3, 1))

        if input_dtype == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            img_np = (img_np / input_scale + input_zero_point).astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], img_np.astype(input_dtype))

        import time
        start = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000)

        output = interpreter.get_tensor(output_details[0]["index"])[0]
        pred = np.argmax(output)
        top3 = np.argsort(output)[-3:]

        if pred == label:
            correct_top1 += 1
        if label in top3:
            correct_top3 += 1
        total += 1

    print(f"  Top-1 Accuracy: {correct_top1/total:.4f}")
    print(f"  Top-3 Accuracy: {correct_top3/total:.4f}")
    print(f"  Avg Latency: {np.mean(latencies):.1f}ms (on this machine)")
    print(f"  P95 Latency: {np.percentile(latencies, 95):.1f}ms")

    return {
        "top1_acc": correct_top1 / total,
        "top3_acc": correct_top3 / total,
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Export PestCast model to TFLite")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/ip102_curated",
                        help="Dataset dir for calibration + validation")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    export_dir = Path(config["paths"]["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    input_size = config["model"]["input_size"]

    print("=" * 60)
    print("PestCast — TFLite Export")
    print("=" * 60)

    # Load trained model
    print("\n[1/4] Loading trained model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded from epoch {checkpoint['epoch']}, "
          f"best val top3: {checkpoint['best_val_acc']:.4f}")

    # Copy labels.json to export dir
    labels_src = data_dir / "labels.json"
    labels_dst = export_dir / "labels.json"
    if labels_src.exists():
        import shutil
        shutil.copy2(labels_src, labels_dst)
        print(f"  Copied labels.json to {labels_dst}")

    # Export to ONNX
    print("\n[2/4] Exporting to ONNX...")
    onnx_path = export_dir / "pest_model.onnx"
    export_to_onnx(model, input_size, onnx_path)

    # Gather calibration data
    print(f"\n[3/4] Gathering calibration data ({config['quantization']['calibration_samples']} samples)...")
    cal_data = gather_calibration_data(
        data_dir, input_size, config["quantization"]["calibration_samples"]
    )
    print(f"  Calibration data shape: {cal_data.shape}")

    # Convert to TFLite
    print("\n[4/4] Converting to TFLite...")
    try:
        convert_onnx_to_tflite_via_onnx2tf(
            onnx_path, export_dir, quantize=True, calibration_data=cal_data
        )
    except Exception as e:
        print(f"  onnx2tf failed: {e}")
        print("  Trying onnx-tf fallback...")
        tflite_path = export_dir / "pest_model.tflite"
        convert_onnx_to_tflite(
            onnx_path, tflite_path, quantize=True, calibration_data=cal_data
        )

    # Validate
    if not args.skip_validation:
        print("\n" + "=" * 60)
        print("Validating exported models")
        print("=" * 60)

        with open(labels_dst) as f:
            labels = json.load(f)

        results = {}
        for tflite_file in export_dir.glob("pest_model*.tflite"):
            try:
                r = validate_tflite(tflite_file, data_dir, input_size, labels)
                results[tflite_file.name] = r
            except Exception as e:
                print(f"  Validation failed for {tflite_file.name}: {e}")

        with open(export_dir / "export_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Export complete! Models in: {export_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
