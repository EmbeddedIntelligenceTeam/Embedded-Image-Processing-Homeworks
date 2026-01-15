"""
EE4065 - Final Project - Question 2
TFLite Export for Real YOLO-Tiny
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# ==================== CONFIGURATION ====================
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "yolo_tiny_weights.h5")
ARCH_PATH = os.path.join(os.path.dirname(__file__), "yolo_tiny_architecture.json")
MODEL_H5_PATH = os.path.join(os.path.dirname(__file__), "yolo_tiny_model.h5")
OUTPUT_DIR = os.path.dirname(__file__)
ESP32_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "esp32_cam", "ESP32_YOLO_Detection")

# Model params (must match training)
IMG_SIZE = 96
GRID_SIZE = 12
NUM_ANCHORS = 3
NUM_CLASSES = 4
BOX_PARAMS = 5 + NUM_CLASSES


def load_model_from_files():
    """Load model from architecture JSON and weights"""
    from tensorflow.keras.models import model_from_json
    
    # Try loading H5 model first
    if os.path.exists(MODEL_H5_PATH):
        try:
            model = keras.models.load_model(MODEL_H5_PATH, compile=False)
            print(f"Loaded model from: {MODEL_H5_PATH}")
            return model
        except Exception as e:
            print(f"Could not load H5 model: {e}")
    
    # Load from architecture + weights
    if os.path.exists(ARCH_PATH) and os.path.exists(WEIGHTS_PATH):
        with open(ARCH_PATH, 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        model.load_weights(WEIGHTS_PATH)
        print(f"Loaded model from architecture + weights")
        return model
    
    raise FileNotFoundError("No model files found. Run train_yolo_real.py first!")


def generate_calibration_data(num_samples=200):
    """Generate calibration data for quantization"""
    (x_train, _), _ = keras.datasets.mnist.load_data()
    
    X = []
    for _ in range(num_samples):
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        # Random digit
        idx = np.random.randint(0, len(x_train))
        digit = x_train[idx]
        
        # Random scale and position
        scale = np.random.uniform(0.6, 1.4)
        new_size = int(28 * scale)
        new_size = max(16, min(new_size, IMG_SIZE // 2))
        
        resized = cv2.resize(digit, (new_size, new_size))
        
        x = np.random.randint(0, IMG_SIZE - new_size)
        y = np.random.randint(0, IMG_SIZE - new_size)
        
        canvas[y:y+new_size, x:x+new_size] = resized
        canvas = canvas / 255.0
        
        X.append(canvas)
    
    return np.array(X)[..., np.newaxis].astype(np.float32)


def convert_to_tflite(model, output_path, X_calibration):
    """Convert to int8 quantized TFLite"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Representative dataset
    def representative_dataset():
        for i in range(min(200, len(X_calibration))):
            yield [X_calibration[i:i+1]]
    
    converter.representative_dataset = representative_dataset
    
    # Convert
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model


def convert_to_c_header(tflite_model, output_path):
    """Convert to C header for Arduino"""
    hex_lines = []
    bytes_per_line = 12
    
    for i in range(0, len(tflite_model), bytes_per_line):
        chunk = tflite_model[i:i+bytes_per_line]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        hex_lines.append(f'    {hex_str}')
    
    hex_array = ',\n'.join(hex_lines)
    
    c_code = f'''/*
 * EE4065 Final Project - Question 2
 * Real YOLO-Tiny Digit Detection Model
 * 
 * Model: YOLO-Tiny with anchors
 * Classes: 0, 3, 5, 8
 * Input: {IMG_SIZE}x{IMG_SIZE}x1 (int8)
 * Output: {GRID_SIZE}x{GRID_SIZE}x{NUM_ANCHORS}x{BOX_PARAMS} (int8)
 * Grid: {GRID_SIZE}x{GRID_SIZE}
 * Anchors: {NUM_ANCHORS}
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <Arduino.h>

const unsigned int model_data_len = {len(tflite_model)};

alignas(8) const unsigned char model_data[] = {{
{hex_array}
}};

#endif // MODEL_DATA_H
'''
    
    with open(output_path, 'w') as f:
        f.write(c_code)
    
    print(f"C header saved: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Real YOLO-Tiny TFLite Export")
    print("=" * 60)
    
    print("\n[1/4] Loading model...")
    try:
        model = load_model_from_files()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    print("\n[2/4] Generating calibration data...")
    X_cal = generate_calibration_data(200)
    print(f"  Calibration shape: {X_cal.shape}")
    
    print("\n[3/4] Converting to TFLite...")
    tflite_path = os.path.join(OUTPUT_DIR, "yolo_tiny_real.tflite")
    tflite_model = convert_to_tflite(model, tflite_path, X_cal)
    
    print("\n[4/4] Generating C header...")
    os.makedirs(ESP32_OUTPUT_DIR, exist_ok=True)
    header_path = os.path.join(ESP32_OUTPUT_DIR, "model_data.h")
    convert_to_c_header(tflite_model, header_path)
    
    print("\n" + "=" * 60)
    print("  Export complete!")
    print(f"  TFLite: {tflite_path}")
    print(f"  C header: {header_path}")
    print("=" * 60)
