"""
EE4065 - Final Project - Question 2
TFLite Export Script for ESP32-CAM
----------------------------------
Converts trained YOLO-tiny model to int8 quantized TFLite format
and generates C header file for Arduino.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ==================== CONFIGURATION ====================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo_tiny_digit.keras")
OUTPUT_DIR = os.path.dirname(__file__)
ESP32_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "esp32_cam")

# ==================== TFLITE CONVERSION ====================
def convert_to_tflite(model, output_path, X_calibration=None):
    """
    Convert Keras model to int8 quantized TFLite.
    
    Args:
        model: Keras model
        output_path: Path to save .tflite file
        X_calibration: Calibration data for quantization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Full integer quantization for ESP32
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Representative dataset for quantization calibration
    if X_calibration is not None:
        def representative_dataset():
            for i in range(min(200, len(X_calibration))):
                sample = X_calibration[i:i+1].astype(np.float32)
                yield [sample]
        converter.representative_dataset = representative_dataset
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model


def convert_to_c_header(tflite_model, output_path):
    """
    Convert TFLite model to C header file for Arduino.
    """
    # Format as hex array
    hex_lines = []
    bytes_per_line = 12
    
    for i in range(0, len(tflite_model), bytes_per_line):
        chunk = tflite_model[i:i+bytes_per_line]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        hex_lines.append(f'    {hex_str}')
    
    hex_array = ',\n'.join(hex_lines)
    
    c_code = f'''/*
 * EE4065 Final Project - Question 2
 * YOLO-Tiny Digit Detection Model for ESP32-CAM
 * 
 * Auto-generated TFLite model (int8 quantized)
 * Classes: 0, 3, 5, 8
 * Input: 96x96x1 grayscale
 * Output: 6x6x9 detection grid
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


# ==================== GENERATE CALIBRATION DATA ====================
def generate_calibration_data(num_samples=200):
    """Generate calibration data from MNIST for quantization."""
    import cv2
    
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    
    TARGET_DIGITS = [0, 3, 5, 8]
    IMG_SIZE = 96
    
    X = []
    for digit in TARGET_DIGITS:
        mask = y_train == digit
        digit_images = x_train[mask][:num_samples // len(TARGET_DIGITS)]
        
        for img in digit_images:
            canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            
            # Random placement
            scale = np.random.uniform(0.7, 1.3)
            new_size = int(28 * scale)
            new_size = max(16, min(new_size, IMG_SIZE - 10))
            
            resized = cv2.resize(img, (new_size, new_size))
            
            max_pos = IMG_SIZE - new_size - 1
            x_pos = np.random.randint(0, max(1, max_pos))
            y_pos = np.random.randint(0, max(1, max_pos))
            
            canvas[y_pos:y_pos+new_size, x_pos:x_pos+new_size] = resized
            canvas = canvas / 255.0
            
            X.append(canvas)
    
    return np.array(X)[..., np.newaxis].astype(np.float32)


# ==================== MAIN ====================
if __name__ == "__main__":
    import cv2
    
    print("=" * 60)
    print("  YOLO-Tiny TFLite Export for ESP32-CAM")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_yolo_tiny.py first!")
        exit(1)
    
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  Model loaded: {MODEL_PATH}")
    
    # Generate calibration data
    print("\n[2/4] Generating calibration data...")
    X_calibration = generate_calibration_data(200)
    print(f"  Calibration samples: {len(X_calibration)}")
    
    # Convert to TFLite
    print("\n[3/4] Converting to TFLite (int8 quantized)...")
    tflite_path = os.path.join(OUTPUT_DIR, "yolo_tiny_digit.tflite")
    tflite_model = convert_to_tflite(model, tflite_path, X_calibration)
    
    # Convert to C header
    print("\n[4/4] Generating C header file...")
    os.makedirs(ESP32_OUTPUT_DIR, exist_ok=True)
    header_path = os.path.join(ESP32_OUTPUT_DIR, "model_data.h")
    convert_to_c_header(tflite_model, header_path)
    
    print("\n" + "=" * 60)
    print("  Export complete!")
    print(f"  TFLite model: {tflite_path}")
    print(f"  C header: {header_path}")
    print("=" * 60)
