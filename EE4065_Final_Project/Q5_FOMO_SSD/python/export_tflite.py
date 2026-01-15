"""
EE4065 - Final Project - Question 5a
FOMO Model Export to TFLite

Converts trained FOMO model to TFLite format for ESP32 deployment.
Includes int8 quantization and C header generation.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 96


def create_representative_dataset(num_samples=200):
    """
    Create representative dataset for quantization calibration.
    Uses MNIST digits similar to training data.
    """
    (x_train, _), _ = keras.datasets.mnist.load_data()
    
    samples = []
    for _ in range(num_samples):
        # Create random composite image
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        noise = np.random.uniform(0, 0.1, (IMG_SIZE, IMG_SIZE))
        canvas = canvas + noise
        
        # Add 1-2 digits
        num_digits = np.random.randint(1, 3)
        for _ in range(num_digits):
            digit_idx = np.random.randint(0, len(x_train))
            digit_img = x_train[digit_idx].astype(np.float32) / 255.0
            
            scale = np.random.uniform(0.7, 1.5)
            new_size = int(28 * scale)
            new_size = max(16, min(new_size, IMG_SIZE // 2))
            
            digit_resized = tf.image.resize(
                digit_img[..., np.newaxis], 
                [new_size, new_size]
            ).numpy()[:, :, 0]
            
            x = np.random.randint(2, IMG_SIZE - new_size - 2)
            y = np.random.randint(2, IMG_SIZE - new_size - 2)
            
            canvas[y:y+new_size, x:x+new_size] = np.maximum(
                canvas[y:y+new_size, x:x+new_size],
                digit_resized
            )
        
        canvas = np.clip(canvas, 0, 1)
        samples.append(canvas[..., np.newaxis])
    
    return np.array(samples, dtype=np.float32)


def representative_dataset_gen():
    """Generator for TFLite converter representative dataset"""
    dataset = create_representative_dataset(200)
    for sample in dataset:
        yield [sample[np.newaxis, ...].astype(np.float32)]


def convert_to_tflite(model_path, output_name="fomo_digit", quantize="int8"):
    """
    Convert Keras model to TFLite with quantization.
    
    Args:
        model_path: Path to trained Keras model
        output_name: Base name for output files
        quantize: Quantization type ("none", "float16", "int8")
    """
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = keras.models.load_model(model_path, compile=False)
    model.summary()
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize == "int8":
        print("Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantize == "float16":
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        print("No quantization applied (float32)...")
    
    # Convert
    try:
        tflite_model = converter.convert()
        print(f"Conversion successful!")
    except Exception as e:
        print(f"Int8 conversion failed: {e}")
        print("Falling back to float16...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        quantize = "float16"
    
    # Save TFLite model
    tflite_path = os.path.join(OUTPUT_DIR, f"{output_name}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = len(tflite_model) / 1024
    print(f"TFLite model saved: {tflite_path}")
    print(f"Model size: {model_size:.1f} KB")
    
    # Verify the model
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Generate C header
    print("\nGenerating C header file...")
    generate_c_header(tflite_model, output_name, quantize)
    
    return tflite_path


def generate_c_header(tflite_model, output_name, quantize):
    """Generate C header file from TFLite model bytes"""
    header_path = os.path.join(OUTPUT_DIR, "..", "esp32_cam", "esp32_fomo_digit", "model_data.h")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    
    with open(header_path, 'w') as f:
        f.write("// Auto-generated FOMO digit detection model\n")
        f.write(f"// Model: {output_name}\n")
        f.write(f"// Quantization: {quantize}\n")
        f.write(f"// Size: {len(tflite_model)} bytes\n")
        f.write(f"// Input: 96x96x1 (grayscale)\n")
        f.write(f"// Output: 12x12x11 (background + 10 digits)\n\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write(f"const unsigned int model_data_len = {len(tflite_model)};\n")
        f.write("alignas(8) const unsigned char model_data[] = {\n")
        
        # Write bytes in rows of 12
        for i in range(0, len(tflite_model), 12):
            row = tflite_model[i:i+12]
            hex_str = ", ".join(f"0x{b:02x}" for b in row)
            if i + 12 < len(tflite_model):
                f.write(f"    {hex_str},\n")
            else:
                f.write(f"    {hex_str}\n")
        
        f.write("};\n\n")
        f.write("#endif // MODEL_DATA_H\n")
    
    print(f"C header saved: {header_path}")


def test_tflite_model(tflite_path):
    """Test TFLite model with sample data"""
    print(f"\nTesting TFLite model: {tflite_path}")
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create test input
    test_samples = create_representative_dataset(5)
    
    for i, sample in enumerate(test_samples):
        # Prepare input
        input_data = sample[np.newaxis, ...]
        
        # Handle quantized input
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Handle quantized output
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Find detections (non-background max)
        detections = np.argmax(output_data[0], axis=-1)
        detected_digits = np.unique(detections[detections > 0])
        
        if len(detected_digits) > 0:
            print(f"  Sample {i+1}: Detected digits: {[d-1 for d in detected_digits]}")
        else:
            print(f"  Sample {i+1}: No digits detected")
    
    print("TFLite model test complete!")


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FOMO model to TFLite")
    parser.add_argument("--model", type=str, default=None, help="Path to Keras model")
    parser.add_argument("--quantize", type=str, default="int8", 
                       choices=["none", "float16", "int8"], help="Quantization type")
    parser.add_argument("--test", action="store_true", help="Test the exported model")
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        # Look for trained model
        for name in ["fomo_digit_best.keras", "fomo_digit.keras", "fomo_digit.h5"]:
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.exists(path):
                model_path = path
                break
        else:
            print("Error: No trained model found!")
            print("Please run train_fomo.py first or specify --model path")
            exit(1)
    
    # Convert
    tflite_path = convert_to_tflite(model_path, quantize=args.quantize)
    
    # Test
    if args.test:
        test_tflite_model(tflite_path)
