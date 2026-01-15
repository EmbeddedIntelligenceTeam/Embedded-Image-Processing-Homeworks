"""
EE4065 - Q4 Multi-Model Training (Fixed - No BatchNorm)

Models work better with TFLite int8 without BatchNormalization.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_SIZE = 48
NUM_CLASSES = 10
EPOCHS = 15
BATCH_SIZE = 64

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== MODEL 1: Simple CNN ====================
def create_simple_cnn():
    """Simple CNN baseline"""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name='SimpleCNN')


# ==================== MODEL 2: MobileNet-Mini (No BatchNorm) ====================
def create_mobilenet_mini():
    """MobileNet-style without BatchNorm for TFLite compatibility"""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    
    # Initial conv
    x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)
    
    # Depthwise separable blocks (simplified without BatchNorm)
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 1, padding='same', activation='relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name='MobileNetMini')


# ==================== MODEL 3: SqueezeNet-Mini ====================
def create_squeezenet_mini():
    """SqueezeNet-style with fire modules"""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    
    # Initial conv
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    
    # Fire module 1
    squeeze = layers.Conv2D(8, 1, activation='relu')(x)
    expand1 = layers.Conv2D(16, 1, activation='relu')(squeeze)
    expand3 = layers.Conv2D(16, 3, padding='same', activation='relu')(squeeze)
    x = layers.Concatenate()([expand1, expand3])
    
    # Fire module 2
    squeeze = layers.Conv2D(8, 1, activation='relu')(x)
    expand1 = layers.Conv2D(16, 1, activation='relu')(squeeze)
    expand3 = layers.Conv2D(16, 3, padding='same', activation='relu')(squeeze)
    x = layers.Concatenate()([expand1, expand3])
    x = layers.MaxPooling2D(2)(x)
    
    # Fire module 3
    squeeze = layers.Conv2D(16, 1, activation='relu')(x)
    expand1 = layers.Conv2D(32, 1, activation='relu')(squeeze)
    expand3 = layers.Conv2D(32, 3, padding='same', activation='relu')(squeeze)
    x = layers.Concatenate()([expand1, expand3])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name='SqueezeNetMini')


# ==================== DATASET ====================
def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    import cv2
    x_train_resized = np.array([cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) for img in x_test])
    
    x_train = x_train_resized.astype(np.float32) / 255.0
    x_test = x_test_resized.astype(np.float32) / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return (x_train, y_train), (x_test, y_test)


# ==================== TRAINING ====================
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# ==================== TFLITE CONVERSION (INT8) ====================
def convert_to_tflite(model, x_calibration, output_name):
    """Convert to TFLite with int8 quantization"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Full integer quantization for ESP32
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Representative dataset for quantization calibration
    def representative_dataset():
        for i in range(min(500, len(x_calibration))):
            sample = x_calibration[i:i+1].astype(np.float32)
            yield [sample]
    
    converter.representative_dataset = representative_dataset
    
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"  Int8 failed ({e}), trying dynamic range...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
    
    tflite_path = os.path.join(OUTPUT_DIR, f"{output_name}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"  {output_name}.tflite: {len(tflite_model) / 1024:.1f} KB")
    return tflite_model


def convert_to_c_header(tflite_model, output_name, model_var_name):
    """Convert TFLite to C header"""
    hex_lines = []
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i:i+12]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        hex_lines.append(f'    {hex_str}')
    
    hex_array = ',\n'.join(hex_lines)
    
    c_code = f'''// EE4065 Q4 - {output_name}
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
    
    print(f"  {output_name}.h generated")


# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  EE4065 Q4: Multi-Model Training (Fixed)")
    print("=" * 60)
    
    # Prepare data
    print("\n[1] Loading dataset...")
    (x_train, y_train), (x_test, y_test) = prepare_dataset()
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")
    
    # Create models
    models = {
        'simple_cnn': create_simple_cnn(),
        'mobilenet_mini': create_mobilenet_mini(),
        'squeezenet_mini': create_squeezenet_mini()
    }
    
    results = {}
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\n[{i+2}] Training {name}...")
        print(f"  Parameters: {model.count_params():,}")
        
        history = train_model(model, x_train, y_train, x_test, y_test)
        
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        results[name] = {'accuracy': acc, 'params': model.count_params()}
        print(f"  Final Accuracy: {acc*100:.2f}%")
        
        # Convert to TFLite
        print(f"  Converting to TFLite (float16)...")
        tflite_model = convert_to_tflite(model, x_test, name)
        convert_to_c_header(tflite_model, name, f"{name}_data")
    
    # Summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Parameters':<15}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']*100:.2f}%      {res['params']:,}")
