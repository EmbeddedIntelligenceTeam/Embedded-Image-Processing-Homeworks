"""
EE4065 - Embedded Digital Image Processing
Final Project - Question 2a

YOLO-Tiny Digit Detection Model Training
-----------------------------------------
- Uses MNIST dataset
- 4 classes: 0, 3, 5, 8
- YOLO-style grid-based detection
- Optimized for ESP32-CAM (~150KB model)

Model outputs bounding boxes + class predictions
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

# ==================== CONFIGURATION ====================
# Classes to detect (indices in MNIST)
TARGET_DIGITS = [0, 3, 5, 8]
NUM_CLASSES = len(TARGET_DIGITS)
CLASS_NAMES = [str(d) for d in TARGET_DIGITS]

# Model parameters - optimized for ESP32
IMG_SIZE = 96         # Input image size (larger for detection)
GRID_SIZE = 6         # 6x6 grid cells
BBOX_PARAMS = 5       # x, y, w, h, confidence
OUTPUT_CHANNELS = BBOX_PARAMS + NUM_CLASSES  # 5 + 4 = 9

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Output paths
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== DATA PREPARATION ====================
def create_detection_dataset(num_samples_per_class=1000, split='train'):
    """
    Create YOLO-format detection dataset from MNIST.
    Places MNIST digits randomly on larger canvas with bounding boxes.
    
    Returns:
        X: Images (N, 96, 96, 1)
        y: Labels (N, 6, 6, 9) - grid cells with [x, y, w, h, conf, class0, class3, class5, class8]
    """
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    if split == 'train':
        x_data, y_data = x_train, y_train
    else:
        x_data, y_data = x_test, y_test
    
    # Filter for target digits only
    images = []
    labels = []
    
    for digit_idx, digit in enumerate(TARGET_DIGITS):
        mask = y_data == digit
        digit_images = x_data[mask][:num_samples_per_class]
        
        for img in digit_images:
            images.append(img)
            labels.append(digit_idx)  # Class index (0-3 for our 4 classes)
    
    # Shuffle
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    
    # Create detection dataset
    X = []
    Y = []
    
    for img, label in zip(images, labels):
        # Create canvas
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        # Random scale (0.5 to 1.5)
        scale = random.uniform(0.7, 1.3)
        new_size = int(28 * scale)
        new_size = max(16, min(new_size, IMG_SIZE - 10))
        
        # Resize digit
        resized = tf.image.resize(img[..., np.newaxis], (new_size, new_size)).numpy()[:, :, 0]
        
        # Random position
        max_x = IMG_SIZE - new_size - 1
        max_y = IMG_SIZE - new_size - 1
        x_pos = random.randint(0, max_x) if max_x > 0 else 0
        y_pos = random.randint(0, max_y) if max_y > 0 else 0
        
        # Place digit on canvas
        canvas[y_pos:y_pos+new_size, x_pos:x_pos+new_size] = resized
        
        # Add some noise
        noise = np.random.randn(IMG_SIZE, IMG_SIZE) * 10
        canvas = np.clip(canvas + noise, 0, 255)
        
        # Random rotation (small)
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
        canvas = cv2.warpAffine(canvas, M, (IMG_SIZE, IMG_SIZE))
        
        # Normalize
        canvas = canvas.astype(np.float32) / 255.0
        
        # Calculate bounding box (normalized 0-1)
        # YOLO format: center_x, center_y, width, height
        cx = (x_pos + new_size / 2) / IMG_SIZE
        cy = (y_pos + new_size / 2) / IMG_SIZE
        w = new_size / IMG_SIZE
        h = new_size / IMG_SIZE
        
        # Create grid labels (6x6x9)
        grid_labels = np.zeros((GRID_SIZE, GRID_SIZE, OUTPUT_CHANNELS), dtype=np.float32)
        
        # Find which grid cell contains the center
        grid_x = int(cx * GRID_SIZE)
        grid_y = int(cy * GRID_SIZE)
        grid_x = min(grid_x, GRID_SIZE - 1)
        grid_y = min(grid_y, GRID_SIZE - 1)
        
        # Offset within grid cell (0-1)
        cell_x = cx * GRID_SIZE - grid_x
        cell_y = cy * GRID_SIZE - grid_y
        
        # Set grid cell values
        grid_labels[grid_y, grid_x, 0] = cell_x  # x offset
        grid_labels[grid_y, grid_x, 1] = cell_y  # y offset
        grid_labels[grid_y, grid_x, 2] = w       # width
        grid_labels[grid_y, grid_x, 3] = h       # height
        grid_labels[grid_y, grid_x, 4] = 1.0     # confidence (object present)
        grid_labels[grid_y, grid_x, 5 + label] = 1.0  # one-hot class
        
        X.append(canvas)
        Y.append(grid_labels)
    
    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    Y = np.array(Y)
    
    return X, Y


# ==================== MODEL ARCHITECTURE ====================
def depthwise_separable_conv(x, filters, stride=1):
    """Depthwise separable convolution - more efficient for ESP32"""
    x = layers.DepthwiseConv2D(3, padding='same', strides=stride, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)  # ReLU6 for better quantization
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    return x


def create_yolo_tiny_model():
    """
    Create tiny YOLO-like model for ESP32.
    
    Input: 96x96x1 grayscale
    Output: 6x6x9 grid (5 bbox params + 4 classes)
    
    Target size: ~150KB (int8 quantized)
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Initial conv
    x = layers.Conv2D(8, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    # 48x48x8
    
    # Depthwise separable blocks
    x = depthwise_separable_conv(x, 16)
    x = layers.MaxPooling2D(2)(x)
    # 24x24x16
    
    x = depthwise_separable_conv(x, 32)
    x = layers.MaxPooling2D(2)(x)
    # 12x12x32
    
    x = depthwise_separable_conv(x, 64)
    x = layers.MaxPooling2D(2)(x)
    # 6x6x64
    
    # Detection head
    x = layers.Conv2D(32, 1, padding='same')(x)
    x = layers.ReLU(6.0)(x)
    
    # Output layer - 9 channels per grid cell
    # [x, y, w, h, conf, class0, class3, class5, class8]
    outputs = layers.Conv2D(OUTPUT_CHANNELS, 1, padding='same', name='detection')(x)
    
    model = keras.Model(inputs, outputs)
    return model


# ==================== CUSTOM LOSS FUNCTION ====================
def yolo_loss(y_true, y_pred):
    """
    Custom YOLO loss function.
    
    y_true, y_pred: (batch, 6, 6, 9)
    Channels: [x, y, w, h, conf, c0, c1, c2, c3]
    """
    # Object mask - where confidence = 1
    obj_mask = y_true[..., 4:5]  # (batch, 6, 6, 1)
    
    # Coordinate loss (only where object exists)
    coord_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., :4] - y_pred[..., :4]))
    
    # Confidence loss
    conf_pred = tf.sigmoid(y_pred[..., 4:5])
    conf_loss = tf.reduce_sum(
        obj_mask * tf.square(y_true[..., 4:5] - conf_pred) +
        0.5 * (1 - obj_mask) * tf.square(y_true[..., 4:5] - conf_pred)  # Less weight for no-object
    )
    
    # Class loss (only where object exists)
    class_pred = tf.nn.softmax(y_pred[..., 5:])
    class_loss = tf.reduce_sum(
        obj_mask * tf.reduce_sum(tf.square(y_true[..., 5:] - class_pred), axis=-1, keepdims=True)
    )
    
    # Total loss
    lambda_coord = 5.0
    lambda_conf = 1.0
    lambda_class = 1.0
    
    total_loss = lambda_coord * coord_loss + lambda_conf * conf_loss + lambda_class * class_loss
    
    return total_loss / tf.cast(tf.shape(y_true)[0], tf.float32)


# ==================== TRAINING ====================
def train_model():
    """Train the YOLO-tiny model"""
    import cv2  # Import here for compatibility
    global cv2
    
    print("=" * 60)
    print("  YOLO-Tiny Digit Detection Training")
    print("  Classes:", CLASS_NAMES)
    print("=" * 60)
    
    # Create dataset
    print("\n[1/5] Creating training dataset from MNIST...")
    X_train, Y_train = create_detection_dataset(num_samples_per_class=1500, split='train')
    print(f"  Training samples: {len(X_train)}")
    
    print("\n[2/5] Creating validation dataset...")
    X_val, Y_val = create_detection_dataset(num_samples_per_class=300, split='test')
    print(f"  Validation samples: {len(X_val)}")
    
    # Create model
    print("\n[3/5] Creating model...")
    model = create_yolo_tiny_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=yolo_loss
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
    
    # Train
    print("\n[4/5] Training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save Keras model
    print("\n[5/5] Saving model...")
    model_path = os.path.join(OUTPUT_DIR, "yolo_tiny_digit.keras")
    model.save(model_path)
    print(f"  Keras model saved: {model_path}")
    
    # Also save as H5 for compatibility
    h5_path = os.path.join(OUTPUT_DIR, "yolo_tiny_digit.h5")
    model.save(h5_path, save_format='h5')
    print(f"  H5 model saved: {h5_path}")
    
    return model, X_val, Y_val


# ==================== MAIN ====================
if __name__ == "__main__":
    import cv2
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train model
    model, X_val, Y_val = train_model()
    
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Next step: Run export_tflite.py to convert for ESP32")
    print("=" * 60)
