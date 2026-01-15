"""
Q4 Multi-Model Accuracy Test
Tests all TFLite models on MNIST test dataset
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def test_tflite_model(model_path, x_test, y_test, num_samples=1000):
    """Test a TFLite model on MNIST test data"""
    print(f"\n{'='*50}")
    print(f"Testing: {model_path}")
    print(f"{'='*50}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    # Get quantization params if int8
    input_scale = input_details[0].get('quantization_parameters', {}).get('scales', [1.0])
    input_zp = input_details[0].get('quantization_parameters', {}).get('zero_points', [0])
    if len(input_scale) > 0:
        input_scale = input_scale[0]
    else:
        input_scale = 1.0
    if len(input_zp) > 0:
        input_zp = input_zp[0]
    else:
        input_zp = 0
        
    output_scale = output_details[0].get('quantization_parameters', {}).get('scales', [1.0])
    output_zp = output_details[0].get('quantization_parameters', {}).get('zero_points', [0])
    if len(output_scale) > 0:
        output_scale = output_scale[0]
    else:
        output_scale = 1.0
    if len(output_zp) > 0:
        output_zp = output_zp[0]
    else:
        output_zp = 0
    
    print(f"Input scale: {input_scale}, zp: {input_zp}")
    print(f"Output scale: {output_scale}, zp: {output_zp}")
    
    correct = 0
    total = min(num_samples, len(x_test))
    
    for i in range(total):
        # Prepare input
        img = x_test[i]
        
        # Resize to model input size if needed
        h, w = input_shape[1], input_shape[2]
        if h != 28 or w != 28:
            img_resized = tf.image.resize(img[..., np.newaxis], (h, w)).numpy().squeeze()
        else:
            img_resized = img
        
        # Normalize to [0, 1]
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Reshape for model
        if len(input_shape) == 4:
            if input_shape[3] == 1:
                img_input = img_norm.reshape(1, h, w, 1)
            else:
                img_input = np.stack([img_norm]*input_shape[3], axis=-1).reshape(1, h, w, input_shape[3])
        else:
            img_input = img_norm.reshape(input_shape)
        
        # Quantize if needed
        if input_dtype == np.int8:
            img_input = (img_input / input_scale + input_zp).astype(np.int8)
        elif input_dtype == np.uint8:
            img_input = (img_input / input_scale + input_zp).astype(np.uint8)
        else:
            img_input = img_input.astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize if needed
        if output_details[0]['dtype'] == np.int8:
            output = (output.astype(np.float32) - output_zp) * output_scale
        elif output_details[0]['dtype'] == np.uint8:
            output = (output.astype(np.float32) - output_zp) * output_scale
        
        # Get prediction
        pred = np.argmax(output)
        
        if pred == y_test[i]:
            correct += 1
    
    accuracy = 100.0 * correct / total
    print(f"\nResults: {correct}/{total} correct")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy


def main():
    print("="*60)
    print("Q4 Multi-Model TFLite Accuracy Test")
    print("="*60)
    
    # Load MNIST
    print("\nLoading MNIST test data...")
    (_, _), (x_test, y_test) = mnist.load_data()
    print(f"Test samples: {len(x_test)}")
    
    # Models to test
    models = [
        "squeezenet_mini.tflite",
        "mobilenet_mini.tflite", 
        "simple_cnn.tflite"
    ]
    
    results = {}
    
    for model_name in models:
        try:
            acc = test_tflite_model(model_name, x_test, y_test, num_samples=1000)
            results[model_name] = acc
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            results[model_name] = 0
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, acc in results.items():
        print(f"{model}: {acc:.2f}%")
    
    avg = sum(results.values()) / len(results)
    print(f"\nAverage: {avg:.2f}%")


if __name__ == "__main__":
    main()
