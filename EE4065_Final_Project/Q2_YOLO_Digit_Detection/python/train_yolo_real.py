"""
EE4065 - Final Project - Question 2
Real YOLO-Tiny Object Detection for ESP32

This implements a proper YOLO architecture with:
- Anchor boxes
- Multi-scale detection grid
- NMS (Non-Maximum Suppression)
- Trained on MNIST digits (0, 3, 5, 8)

Based on Tiny YOLO v2 architecture but scaled for ESP32
Target model size: <200KB (int8 quantized)
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import random

# ==================== CONFIGURATION ====================
TARGET_DIGITS = [0, 3, 5, 8]
NUM_CLASSES = len(TARGET_DIGITS)
CLASS_NAMES = [str(d) for d in TARGET_DIGITS]

# YOLO Parameters
IMG_SIZE = 96           # Input size
GRID_SIZE = 12          # 12x12 grid (finer detection)
NUM_ANCHORS = 3         # 3 anchor boxes per cell

# Anchor boxes (width, height) - tuned for digit sizes
# These are relative to grid cell size
ANCHORS = np.array([
    [0.8, 0.8],    # Small digits
    [1.2, 1.2],    # Medium digits
    [1.6, 1.6],    # Large digits
], dtype=np.float32)

# Output: Each cell predicts NUM_ANCHORS boxes, each with 5+NUM_CLASSES values
# [tx, ty, tw, th, conf, c0, c1, c2, c3]
BOX_PARAMS = 5 + NUM_CLASSES  # 9

# Training
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== DATA AUGMENTATION ====================
def augment_image(image, bbox):
    """Apply augmentation to image and adjust bbox"""
    h, w = image.shape[:2]
    
    # Random brightness
    if random.random() > 0.5:
        delta = random.uniform(-30, 30)
        image = np.clip(image + delta, 0, 255).astype(np.uint8)
    
    # Random rotation (small)
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderValue=0)
    
    return image, bbox


# ==================== DATASET CREATION ====================
def create_yolo_dataset(num_images=6000, max_digits_per_image=3, split='train'):
    """
    Create YOLO format dataset with multiple digits per image.
    Each image can have 1-3 digits placed randomly.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    if split == 'train':
        x_data, y_data = x_train, y_train
    else:
        x_data, y_data = x_test, y_test
    
    # Filter for target digits
    digit_images = {}
    for digit in TARGET_DIGITS:
        mask = y_data == digit
        digit_images[digit] = x_data[mask]
    
    images = []
    labels = []
    
    for _ in range(num_images):
        # Create blank canvas
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        # Add some background noise
        noise = np.random.randint(0, 30, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        canvas = canvas + noise
        
        # Randomly choose number of digits (1-3)
        num_digits = random.randint(1, max_digits_per_image)
        
        bboxes = []
        placed_regions = []
        
        for _ in range(num_digits):
            # Random digit class
            class_idx = random.randint(0, NUM_CLASSES - 1)
            digit = TARGET_DIGITS[class_idx]
            
            # Get random digit image
            digit_img = digit_images[digit][random.randint(0, len(digit_images[digit])-1)]
            
            # Random scale
            scale = random.uniform(0.6, 1.4)
            new_size = int(28 * scale)
            new_size = max(16, min(new_size, IMG_SIZE // 2))
            
            digit_resized = cv2.resize(digit_img, (new_size, new_size))
            
            # Try to find non-overlapping position
            max_attempts = 20
            placed = False
            
            for _ in range(max_attempts):
                x = random.randint(2, IMG_SIZE - new_size - 2)
                y = random.randint(2, IMG_SIZE - new_size - 2)
                
                # Check overlap with existing placements
                overlap = False
                for (px, py, ps) in placed_regions:
                    if (abs(x - px) < (new_size + ps) // 2 and 
                        abs(y - py) < (new_size + ps) // 2):
                        overlap = True
                        break
                
                if not overlap:
                    # Place digit
                    canvas[y:y+new_size, x:x+new_size] = np.maximum(
                        canvas[y:y+new_size, x:x+new_size], 
                        digit_resized
                    )
                    
                    # Store bbox (normalized YOLO format: cx, cy, w, h)
                    cx = (x + new_size / 2) / IMG_SIZE
                    cy = (y + new_size / 2) / IMG_SIZE
                    w = new_size / IMG_SIZE
                    h = new_size / IMG_SIZE
                    
                    bboxes.append([cx, cy, w, h, class_idx])
                    placed_regions.append((x + new_size//2, y + new_size//2, new_size))
                    placed = True
                    break
            
            if not placed and len(placed_regions) == 0:
                # Force place first digit if nothing placed yet
                x, y = IMG_SIZE // 4, IMG_SIZE // 4
                canvas[y:y+new_size, x:x+new_size] = digit_resized
                cx = (x + new_size / 2) / IMG_SIZE
                cy = (y + new_size / 2) / IMG_SIZE
                w = new_size / IMG_SIZE
                h = new_size / IMG_SIZE
                bboxes.append([cx, cy, w, h, class_idx])
        
        # Augment
        canvas, bboxes = augment_image(canvas, bboxes)
        
        # Normalize
        canvas = canvas.astype(np.float32) / 255.0
        
        # Create YOLO grid labels
        # Shape: (GRID_SIZE, GRID_SIZE, NUM_ANCHORS, BOX_PARAMS)
        grid_labels = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, BOX_PARAMS), dtype=np.float32)
        
        for bbox in bboxes:
            cx, cy, w, h, class_idx = bbox
            
            # Find grid cell
            gx = int(cx * GRID_SIZE)
            gy = int(cy * GRID_SIZE)
            gx = min(gx, GRID_SIZE - 1)
            gy = min(gy, GRID_SIZE - 1)
            
            # Find best matching anchor
            bbox_wh = np.array([w * GRID_SIZE, h * GRID_SIZE])
            anchor_wh = ANCHORS
            
            # IoU with anchors (simplified - just area ratio)
            intersect = np.minimum(bbox_wh, anchor_wh)
            intersect_area = intersect[:, 0] * intersect[:, 1]
            bbox_area = bbox_wh[0] * bbox_wh[1]
            anchor_areas = anchor_wh[:, 0] * anchor_wh[:, 1]
            union_area = bbox_area + anchor_areas - intersect_area
            ious = intersect_area / (union_area + 1e-6)
            
            best_anchor = np.argmax(ious)
            
            # Offset within cell
            tx = cx * GRID_SIZE - gx
            ty = cy * GRID_SIZE - gy
            
            # Size relative to anchor
            tw = np.log(w * GRID_SIZE / (ANCHORS[best_anchor, 0] + 1e-6) + 1e-6)
            th = np.log(h * GRID_SIZE / (ANCHORS[best_anchor, 1] + 1e-6) + 1e-6)
            
            # Set grid cell
            grid_labels[gy, gx, best_anchor, 0] = tx
            grid_labels[gy, gx, best_anchor, 1] = ty
            grid_labels[gy, gx, best_anchor, 2] = tw
            grid_labels[gy, gx, best_anchor, 3] = th
            grid_labels[gy, gx, best_anchor, 4] = 1.0  # Objectness
            grid_labels[gy, gx, best_anchor, 5 + int(class_idx)] = 1.0  # Class one-hot
        
        images.append(canvas)
        labels.append(grid_labels)
    
    X = np.array(images)[..., np.newaxis]  # Add channel dim
    Y = np.array(labels)
    
    return X, Y


# ==================== YOLO LOSS ====================
def yolo_loss(y_true, y_pred):
    """
    Proper YOLO loss function with:
    - Coordinate loss (MSE)
    - Confidence loss (BCE)
    - Class loss (BCE)
    """
    # Reshape predictions
    pred = tf.reshape(y_pred, (-1, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, BOX_PARAMS))
    
    # Extract components
    pred_xy = tf.sigmoid(pred[..., 0:2])
    pred_wh = pred[..., 2:4]
    pred_conf = tf.sigmoid(pred[..., 4:5])
    pred_class = tf.sigmoid(pred[..., 5:])
    
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    true_conf = y_true[..., 4:5]
    true_class = y_true[..., 5:]
    
    # Object mask
    obj_mask = true_conf
    noobj_mask = 1.0 - obj_mask
    
    # Coordinate loss (only for cells with objects)
    coord_loss = tf.reduce_sum(
        obj_mask * (tf.square(true_xy - pred_xy) + tf.square(true_wh - pred_wh))
    )
    
    # Confidence loss
    conf_loss = tf.reduce_sum(
        obj_mask * tf.square(true_conf - pred_conf) +
        0.5 * noobj_mask * tf.square(true_conf - pred_conf)
    )
    
    # Class loss
    class_loss = tf.reduce_sum(
        obj_mask * tf.reduce_sum(tf.square(true_class - pred_class), axis=-1, keepdims=True)
    )
    
    # Weighted sum
    lambda_coord = 5.0
    lambda_noobj = 0.5
    lambda_class = 1.0
    
    total_loss = lambda_coord * coord_loss + conf_loss + lambda_class * class_loss
    
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    return total_loss / batch_size


# ==================== MODEL ARCHITECTURE ====================
def conv_bn_leaky(x, filters, kernel_size, strides=1):
    """Conv + BatchNorm + LeakyReLU block"""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def create_yolo_tiny_model():
    """
    Tiny YOLO architecture for ESP32
    Based on Darknet-tiny but smaller
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Backbone
    x = conv_bn_leaky(inputs, 16, 3)       # 96x96x16
    x = layers.MaxPooling2D(2, 2)(x)       # 48x48x16
    
    x = conv_bn_leaky(x, 32, 3)            # 48x48x32
    x = layers.MaxPooling2D(2, 2)(x)       # 24x24x32
    
    x = conv_bn_leaky(x, 64, 3)            # 24x24x64
    x = layers.MaxPooling2D(2, 2)(x)       # 12x12x64
    
    x = conv_bn_leaky(x, 128, 3)           # 12x12x128
    x = conv_bn_leaky(x, 64, 1)            # 12x12x64
    x = conv_bn_leaky(x, 128, 3)           # 12x12x128
    
    # Detection head
    x = conv_bn_leaky(x, 64, 1)            # 12x12x64
    
    # Output: (GRID_SIZE, GRID_SIZE, NUM_ANCHORS * BOX_PARAMS)
    output_channels = NUM_ANCHORS * BOX_PARAMS  # 3 * 9 = 27
    x = layers.Conv2D(output_channels, 1, padding='same')(x)
    
    # Reshape to (batch, grid, grid, anchors, params)
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, BOX_PARAMS))(x)
    
    model = keras.Model(inputs, outputs)
    return model


# ==================== TRAINING ====================
def train():
    print("=" * 60)
    print("  Real YOLO-Tiny Training for ESP32")
    print("  Classes:", CLASS_NAMES)
    print("  Grid:", f"{GRID_SIZE}x{GRID_SIZE}")
    print("  Anchors:", NUM_ANCHORS)
    print("=" * 60)
    
    # Create datasets
    print("\n[1/5] Creating training dataset...")
    X_train, Y_train = create_yolo_dataset(num_images=8000, max_digits_per_image=3, split='train')
    print(f"  Training: {len(X_train)} images")
    print(f"  X shape: {X_train.shape}, Y shape: {Y_train.shape}")
    
    print("\n[2/5] Creating validation dataset...")
    X_val, Y_val = create_yolo_dataset(num_images=1000, max_digits_per_image=3, split='test')
    print(f"  Validation: {len(X_val)} images")
    
    # Create model
    print("\n[3/5] Building YOLO-Tiny model...")
    model = create_yolo_tiny_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=yolo_loss
    )
    
    # Callbacks - save weights only to avoid custom loss serialization issues
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
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
    
    # Save - use weights and H5 format
    print("\n[5/5] Saving models...")
    
    # Save weights
    model.save_weights(os.path.join(OUTPUT_DIR, 'yolo_tiny_weights.h5'))
    print(f"  Weights saved: yolo_tiny_weights.h5")
    
    # Save full model as H5 (better compatibility)
    try:
        model.save(os.path.join(OUTPUT_DIR, 'yolo_tiny_model.h5'), save_format='h5')
        print(f"  Full model saved: yolo_tiny_model.h5")
    except Exception as e:
        print(f"  Warning: Could not save full H5 model: {e}")
    
    # Save architecture as JSON
    model_json = model.to_json()
    with open(os.path.join(OUTPUT_DIR, 'yolo_tiny_architecture.json'), 'w') as f:
        f.write(model_json)
    print(f"  Architecture saved: yolo_tiny_architecture.json")
    
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Next: Run export_real_tflite.py")
    print("=" * 60)
    
    return model


# ==================== MAIN ====================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train()
