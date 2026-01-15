"""
Retrain MobileNet-Mini with better architecture and proper quantization
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

INPUT_SIZE = 32  # Smaller input for better ESP32 compatibility
NUM_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 64

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_mobilenet_mini_v2():
    """Simplified MobileNet that works well with TFLite int8"""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    
    # Initial conv
    x = layers.Conv2D(16, 3, strides=2, padding='same')(inputs)
    x = layers.ReLU()(x)
    
    # Block 1
    x = layers.DepthwiseConv2D(3, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(24, 1, padding='same')(x)
    x = layers.ReLU()(x)
    
    # Block 2 (stride)
    x = layers.DepthwiseConv2D(3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 1, padding='same')(x)
    x = layers.ReLU()(x)
    
    # Block 3
    x = layers.DepthwiseConv2D(3, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 1, padding='same')(x)
    x = layers.ReLU()(x)
    
    # Block 4 (stride)
    x = layers.DepthwiseConv2D(3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 1, padding='same')(x)
    x = layers.ReLU()(x)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name='MobileNetMiniV2')


def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Resize to model input size
    x_train_resized = np.array([cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) for img in x_test])
    
    x_train = x_train_resized.astype(np.float32) / 255.0
    x_test = x_test_resized.astype(np.float32) / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return (x_train, y_train), (x_test, y_test)


def convert_to_tflite(model, x_calibration, output_name):
    """Convert with int8 quantization"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    def representative_dataset():
        for i in range(min(1000, len(x_calibration))):
            sample = x_calibration[i:i+1].astype(np.float32)
            yield [sample]
    
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(OUTPUT_DIR, f"{output_name}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved: {output_name}.tflite ({len(tflite_model) / 1024:.1f} KB)")
    return tflite_model


def convert_to_c_header(tflite_model, output_name, model_var_name):
    """Convert TFLite to C header"""
    hex_lines = []
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i:i+12]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        hex_lines.append(f'    {hex_str}')
    
    hex_array = ',\n'.join(hex_lines)
    
    c_code = f'''// EE4065 Q4 - {output_name} (Retrained)
// Size: {len(tflite_model)} bytes

#ifndef {model_var_name.upper()}_H
#define {model_var_name.upper()}_H

const unsigned int {model_var_name}_len = {len(tflite_model)};

alignas(8) const unsigned char {model_var_name}[] = {{
{hex_array}
}};

#endif
'''
    
    header_path = os.path.join(OUTPUT_DIR, f"{output_name}.h")
    with open(header_path, 'w') as f:
        f.write(c_code)
    
    print(f"Saved: {output_name}.h")


def test_tflite(model_path, x_test, y_test, num_samples=1000):
    """Test TFLite accuracy"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    
    correct = 0
    for i in range(min(num_samples, len(x_test))):
        img = x_test[i:i+1]
        
        # Quantize input
        img_q = (img / input_scale + input_zp).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], img_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize
        output = (output.astype(np.float32) - output_zp) * output_scale
        
        if np.argmax(output) == y_test[i]:
            correct += 1
    
    return 100.0 * correct / min(num_samples, len(x_test))


if __name__ == "__main__":
    print("="*60)
    print("  Retraining MobileNet-Mini V2")
    print("="*60)
    
    # Load data
    print("\n[1] Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = prepare_dataset()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    
    # Create model
    print("\n[2] Creating model...")
    model = create_mobilenet_mini_v2()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\n[3] Training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ],
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[4] Keras Model Accuracy: {acc*100:.2f}%")
    
    # Save weights
    model.save_weights(os.path.join(OUTPUT_DIR, "mobilenet_mini_weights.h5"))
    
    # Convert to TFLite
    print("\n[5] Converting to TFLite...")
    tflite_model = convert_to_tflite(model, x_test, "mobilenet_mini")
    convert_to_c_header(tflite_model, "mobilenet_mini", "mobilenet_mini_data")
    
    # Test TFLite
    print("\n[6] Testing TFLite model...")
    tflite_acc = test_tflite(os.path.join(OUTPUT_DIR, "mobilenet_mini.tflite"), x_test, y_test)
    print(f"TFLite Accuracy: {tflite_acc:.2f}%")
    
    print("\n" + "="*60)
    print("  DONE!")
    print("="*60)
