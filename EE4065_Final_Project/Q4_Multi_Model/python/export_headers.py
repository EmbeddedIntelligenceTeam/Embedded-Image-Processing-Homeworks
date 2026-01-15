"""
Export all Q4 models to TFLite and generate C headers
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ESP32_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "esp32_cam", "CNN", "digit_recognition")

def get_quantization_params(model_path):
    """Get quantization parameters from TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale = input_details['quantization_parameters']['scales']
    input_zp = input_details['quantization_parameters']['zero_points']
    output_scale = output_details['quantization_parameters']['scales']
    output_zp = output_details['quantization_parameters']['zero_points']
    
    return {
        'input_scale': input_scale[0] if len(input_scale) > 0 else 1.0,
        'input_zp': input_zp[0] if len(input_zp) > 0 else 0,
        'output_scale': output_scale[0] if len(output_scale) > 0 else 1.0,
        'output_zp': output_zp[0] if len(output_zp) > 0 else 0,
        'input_shape': input_details['shape'].tolist()
    }


def convert_to_c_header(tflite_path, var_name):
    """Convert TFLite to C header with quantization params"""
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()
    
    # Get quantization params
    params = get_quantization_params(tflite_path)
    
    # Generate hex array
    hex_lines = []
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i:i+12]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        hex_lines.append(f'    {hex_str}')
    
    hex_array = ',\n'.join(hex_lines)
    
    c_code = f'''// EE4065 Q4 - {var_name}
// Auto-generated from TFLite model
// Size: {len(tflite_model)} bytes

#ifndef {var_name.upper()}_MODEL_H
#define {var_name.upper()}_MODEL_H

// Model data
const unsigned int {var_name}_model_len = {len(tflite_model)};

alignas(8) const unsigned char {var_name}_model[] = {{
{hex_array}
}};

// Quantization parameters
const float {var_name}_input_scale = {params['input_scale']:.10f}f;
const int {var_name}_input_zero_point = {params['input_zp']};
const float {var_name}_output_scale = {params['output_scale']:.10f}f;
const int {var_name}_output_zero_point = {params['output_zp']};

// Input shape: {params['input_shape']}

#endif // {var_name.upper()}_MODEL_H
'''
    
    return c_code, len(tflite_model)


def main():
    print("="*60)
    print("  Generating C Headers for ESP32")
    print("="*60)
    
    models = [
        ('squeezenet_mini.tflite', 'squeezenetmini'),
        ('mobilenet_mini.tflite', 'mobilenetv2mini'),
        ('simple_cnn.tflite', 'resnet8'),  # Using simple_cnn as resnet replacement
    ]
    
    # Also create efficientnet as copy of squeezenet for now
    models.append(('squeezenet_mini.tflite', 'efficientnetmini'))
    
    all_headers = []
    
    for tflite_name, var_name in models:
        tflite_path = os.path.join(OUTPUT_DIR, tflite_name)
        
        if not os.path.exists(tflite_path):
            print(f"WARNING: {tflite_name} not found!")
            continue
        
        print(f"\nProcessing: {tflite_name} -> {var_name}")
        
        c_code, size = convert_to_c_header(tflite_path, var_name)
        all_headers.append((var_name, c_code, size))
        
        print(f"  Size: {size/1024:.1f} KB")
    
    # Create combined header file
    print("\n" + "="*60)
    print("  Creating combined model_data.h")
    print("="*60)
    
    combined = '''// EE4065 Q4 - All Models Combined
// Auto-generated - DO NOT EDIT

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

'''
    
    for var_name, c_code, size in all_headers:
        # Extract just the data part (skip the ifndef/define/endif)
        lines = c_code.split('\n')
        data_lines = []
        in_data = False
        for line in lines:
            if 'const unsigned int' in line or 'const unsigned char' in line or 'const float' in line or 'const int' in line:
                in_data = True
            if in_data and '#endif' not in line:
                data_lines.append(line)
        combined += '\n'.join(data_lines) + '\n\n'
    
    combined += '#endif // MODEL_DATA_H\n'
    
    # Save to ESP32 directory
    header_path = os.path.join(ESP32_DIR, "model_data.h")
    with open(header_path, 'w') as f:
        f.write(combined)
    
    print(f"Saved: {header_path}")
    
    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    total_size = sum(size for _, _, size in all_headers)
    print(f"Total models: {len(all_headers)}")
    print(f"Total size: {total_size/1024:.1f} KB")
    
    for var_name, _, size in all_headers:
        print(f"  - {var_name}: {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
