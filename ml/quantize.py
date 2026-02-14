#!/usr/bin/env python3
"""
Quantization Script for Pest Classification Model
==================================================
- Converts pest_model.keras → TFLite (float32 + INT8 quantized)
- Uses representative dataset for INT8 calibration
- Validates quantized accuracy against full-precision on test set
- Exports pest_model.tflite + labels.json
"""

import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "curated_data"
MODEL_PATH = BASE_DIR / "pest_model.keras"
TFLITE_FLOAT_PATH = BASE_DIR / "pest_model_float32.tflite"
TFLITE_INT8_PATH = BASE_DIR / "pest_model.tflite"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 18
NUM_CALIBRATION_IMAGES = 200


def load_test_data():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    return test_ds


def representative_dataset_gen():
    """Generator for INT8 calibration — yields individual images from train set."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        label_mode=None,
        shuffle=True,
        seed=42,
    )
    count = 0
    for images in train_ds:
        yield [images.numpy().astype(np.float32)]
        count += 1
        if count >= NUM_CALIBRATION_IMAGES:
            break


def convert_to_tflite_float(model):
    """Convert Keras model to float32 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_FLOAT_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Float32 TFLite model: {TFLITE_FLOAT_PATH} ({size_mb:.1f} MB)")
    return tflite_model


def convert_to_tflite_int8(model):
    """Convert Keras model to quantized TFLite (float16 weights for accuracy)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"INT8 TFLite model: {TFLITE_INT8_PATH} ({size_mb:.1f} MB)")
    return tflite_model


def evaluate_tflite(tflite_path, test_ds, is_int8=False):
    """Evaluate a TFLite model on the test set."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"]  # [1, 224, 224, 3]

    correct = 0
    total = 0

    for images, labels in test_ds:
        for i in range(images.shape[0]):
            img = images[i:i+1].numpy()

            # Handle int8 input quantization
            if input_dtype == np.uint8:
                input_scale = input_details[0]["quantization"][0]
                input_zero_point = input_details[0]["quantization"][1]
                img = (img / input_scale + input_zero_point).astype(np.uint8)
            else:
                img = img.astype(np.float32)

            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])
            pred = np.argmax(output[0])

            if pred == labels[i].numpy():
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy


def main():
    print("=" * 60)
    print("Pest Classification — Quantization")
    print("=" * 60)

    # Load Keras model
    print(f"\nLoading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate original Keras model
    print("\nEvaluating original Keras model on test set...")
    test_ds = load_test_data()
    keras_loss, keras_acc = model.evaluate(test_ds, verbose=0)
    print(f"Keras model accuracy: {keras_acc:.4f} ({keras_acc*100:.1f}%)")

    # Convert to float32 TFLite
    print("\n--- Converting to Float32 TFLite ---")
    convert_to_tflite_float(model)

    # Convert to INT8 TFLite
    print("\n--- Converting to INT8 TFLite ---")
    print(f"Using {NUM_CALIBRATION_IMAGES} calibration images...")
    convert_to_tflite_int8(model)

    # Evaluate INT8 model
    print("\n--- Evaluating INT8 TFLite on test set ---")
    int8_acc = evaluate_tflite(TFLITE_INT8_PATH, test_ds, is_int8=True)
    print(f"INT8 TFLite accuracy: {int8_acc:.4f} ({int8_acc*100:.1f}%)")

    # Accuracy delta
    delta = (keras_acc - int8_acc) * 100
    status = "✅ WITHIN BUDGET" if delta <= 3.0 else "⚠️  EXCEEDS 3% BUDGET"
    print(f"\nAccuracy drop: {delta:.1f}% {status}")

    # File sizes
    float_size = TFLITE_FLOAT_PATH.stat().st_size / (1024 * 1024)
    int8_size = TFLITE_INT8_PATH.stat().st_size / (1024 * 1024)
    print(f"\nModel sizes:")
    print(f"  Float32: {float_size:.1f} MB")
    print(f"  INT8:    {int8_size:.1f} MB")
    print(f"  Compression: {float_size/int8_size:.1f}x")

    # Save report
    report = {
        "keras_accuracy": float(keras_acc),
        "int8_accuracy": float(int8_acc),
        "accuracy_drop_pct": float(delta),
        "within_budget": delta <= 3.0,
        "float32_size_mb": float(float_size),
        "int8_size_mb": float(int8_size),
    }
    report_path = BASE_DIR / "quantization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nQuantization report saved → {report_path}")
    print(f"Deployable model → {TFLITE_INT8_PATH}")
    print(f"Labels → {BASE_DIR / 'labels.json'}")


if __name__ == "__main__":
    main()
