"""
EE4065 - Final Project - Question 5a
FOMO (Faster Objects, More Objects) Digit Detection

This implements the FOMO architecture based on Edge Impulse's approach:
- MobileNetV2 backbone (alpha=0.35 for lightweight)
- Heat map output for centroid-based detection
- Supports ESP32-CAM deployment

Based on: https://github.com/bhoke/FOMO
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import Model, layers, optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import random


# Custom IoU metric that serializes properly
class MeanIoU(keras.metrics.Metric):
    """Mean IoU for multi-class segmentation (excluding background)"""
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_true_classes = tf.argmax(y_true, axis=-1)
        
        # Calculate IoU for non-background classes (1-10)
        total_iou = 0.0
        valid_classes = 0
        
        for c in range(1, self.num_classes):  # Skip background (0)
            pred_mask = tf.cast(tf.equal(y_pred_classes, c), tf.float32)
            true_mask = tf.cast(tf.equal(y_true_classes, c), tf.float32)
            
            intersection = tf.reduce_sum(pred_mask * true_mask)
            union = tf.reduce_sum(pred_mask) + tf.reduce_sum(true_mask) - intersection
            
            # Only count classes that appear in ground truth
            class_present = tf.cast(tf.reduce_sum(true_mask) > 0, tf.float32)
            iou = tf.where(union > 0, intersection / union, 0.0)
            total_iou += iou * class_present
            valid_classes += class_present
        
        mean_iou = tf.where(valid_classes > 0, total_iou / valid_classes, 0.0)
        self.total_iou.assign_add(mean_iou)
        self.count.assign_add(1.0)
    
    def result(self):
        return tf.where(self.count > 0, self.total_iou / self.count, 0.0)
    
    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

# ==================== CONFIGURATION ====================
NUM_CLASSES = 11  # 10 digits (0-9) + background
CLASS_NAMES = ['background'] + [str(d) for d in range(10)]

# Model Parameters
IMG_SIZE = 96           # Input size (96x96 for ESP32)
GRID_SIZE = 12          # Output grid (96/8 = 12)
ALPHA = 0.35            # MobileNetV2 width multiplier (lightweight)

# Training
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== FOMO MODEL ====================
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """MobileNetV2 inverted residual block"""
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    prefix = f"block_{block_id}_"
    in_channels = inputs.shape[-1]
    x = inputs

    if block_id:
        # Expand
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand"
        )(inputs)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=prefix + "expand_BN")(x)
        x = layers.ReLU(6.0, name=prefix + "expand_relu")(x)
        
        # For FOMO, we cut at block_6 to get features
        if block_id == 6:
            return x
    else:
        prefix = "expanded_conv_"

    # Depthwise conv
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same",
        name=prefix + "depthwise"
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=prefix + "depthwise_BN")(x)
    x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # Project
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project"
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=prefix + "project_BN")(x)

    if pointwise_filters == in_channels and stride == 1:
        x = layers.Add(name=prefix + "add")([x, inputs])
    return x


def create_fomo_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES, alpha=ALPHA):
    """
    FOMO model with MobileNetV2 backbone.
    Output: (GRID_SIZE, GRID_SIZE, num_classes) heat map
    """
    first_block_filters = _make_divisible(32 * alpha, 8)
    
    inputs = keras.Input(shape=input_shape)
    
    # Convert grayscale to 3 channels if needed
    if input_shape[-1] == 1:
        x = layers.Concatenate()([inputs, inputs, inputs])
    else:
        x = inputs
    
    # Initial conv
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="Conv1"
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name="bn_Conv1")(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    # MobileNetV2 blocks (cut at block 6 for FOMO)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=6)
    
    # FOMO head - lightweight classification head
    x = layers.Conv2D(32, kernel_size=1, strides=1, activation='relu', name='head')(x)
    
    # Output layer - softmax for multi-class
    outputs = layers.Conv2D(
        num_classes, 
        kernel_size=1, 
        strides=1, 
        activation="softmax", 
        name='output'
    )(x)
    
    model = Model(inputs, outputs, name="FOMO_Digit")
    return model


# ==================== DATASET CREATION ====================
def create_fomo_dataset(num_images=5000, max_digits_per_image=3, split='train'):
    """
    Create FOMO format dataset with segmentation masks.
    Each pixel in the output grid corresponds to a class.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    if split == 'train':
        x_data, y_data = x_train, y_train
    else:
        x_data, y_data = x_test, y_test
    
    # Group by digit
    digit_images = {d: x_data[y_data == d] for d in range(10)}
    
    images = []
    labels = []
    
    for _ in range(num_images):
        # Create blank canvas
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        # Add background noise
        noise = np.random.randint(0, 25, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        canvas = canvas + noise
        
        # Segmentation mask (GRID_SIZE x GRID_SIZE x NUM_CLASSES)
        seg_mask = np.zeros((GRID_SIZE, GRID_SIZE, NUM_CLASSES), dtype=np.float32)
        seg_mask[..., 0] = 1.0  # All background initially
        
        # Random number of digits
        num_digits = random.randint(1, max_digits_per_image)
        placed_regions = []
        
        for _ in range(num_digits):
            # Random digit (0-9)
            digit = random.randint(0, 9)
            digit_img = digit_images[digit][random.randint(0, len(digit_images[digit])-1)]
            
            # Random scale
            scale = random.uniform(0.7, 1.5)
            new_size = int(28 * scale)
            new_size = max(16, min(new_size, IMG_SIZE // 2))
            
            digit_resized = cv2.resize(digit_img, (new_size, new_size))
            
            # Try to place without overlap
            max_attempts = 15
            placed = False
            
            for _ in range(max_attempts):
                x = random.randint(2, IMG_SIZE - new_size - 2)
                y = random.randint(2, IMG_SIZE - new_size - 2)
                
                # Check overlap
                overlap = False
                for (px, py, ps) in placed_regions:
                    if (abs(x - px) < (new_size + ps) // 2 + 5 and 
                        abs(y - py) < (new_size + ps) // 2 + 5):
                        overlap = True
                        break
                
                if not overlap:
                    # Place digit on canvas
                    canvas[y:y+new_size, x:x+new_size] = np.maximum(
                        canvas[y:y+new_size, x:x+new_size], 
                        digit_resized
                    )
                    
                    # Update segmentation mask
                    # Find grid cells covered by this digit
                    cell_size = IMG_SIZE // GRID_SIZE  # 8 pixels per cell
                    cx = x + new_size // 2
                    cy = y + new_size // 2
                    
                    gx = min(cx // cell_size, GRID_SIZE - 1)
                    gy = min(cy // cell_size, GRID_SIZE - 1)
                    
                    # Set center cell to digit class
                    seg_mask[gy, gx, 0] = 0.0  # Remove background
                    seg_mask[gy, gx, digit + 1] = 1.0  # Set digit class
                    
                    placed_regions.append((x + new_size//2, y + new_size//2, new_size))
                    placed = True
                    break
            
            if not placed and len(placed_regions) == 0:
                # Force place first digit
                x, y = IMG_SIZE // 4, IMG_SIZE // 4
                canvas[y:y+new_size, x:x+new_size] = digit_resized
                cell_size = IMG_SIZE // GRID_SIZE
                gx = min((x + new_size//2) // cell_size, GRID_SIZE - 1)
                gy = min((y + new_size//2) // cell_size, GRID_SIZE - 1)
                seg_mask[gy, gx, 0] = 0.0
                seg_mask[gy, gx, digit + 1] = 1.0
        
        # Data augmentation
        if random.random() > 0.5:
            delta = random.uniform(-25, 25)
            canvas = np.clip(canvas.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        
        # Normalize
        canvas = canvas.astype(np.float32) / 255.0
        
        images.append(canvas)
        labels.append(seg_mask)
    
    X = np.array(images)[..., np.newaxis]  # (N, 96, 96, 1)
    Y = np.array(labels)  # (N, 12, 12, 11)
    
    return X, Y


# ==================== LOSS FUNCTION ====================
def weighted_dice_loss(class_weights, smooth=1e-5):
    """
    Weighted Dice loss for segmentation.
    Handles class imbalance (many background pixels, few digit pixels).
    """
    weights = tf.constant(class_weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        axes = [0, 1, 2]  # sum over batch, height, width
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        union = tf.reduce_sum(y_true + y_pred, axis=axes)
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        weighted_dice = weights * dice_score
        loss_val = 1.0 - tf.reduce_sum(weighted_dice) / tf.reduce_sum(weights)
        return loss_val
    
    return loss


def cosine_annealing_with_warmup(epoch, lr, total_epochs, warmup_epochs=5, min_lr=1e-6, max_lr=0.01):
    """Learning rate schedule with warmup and cosine annealing"""
    if epoch < warmup_epochs:
        return min_lr + (max_lr - min_lr) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


# ==================== TRAINING ====================
def train(epochs=EPOCHS, test_mode=False):
    print("=" * 60)
    print("  FOMO Digit Detection Training")
    print("  Classes:", CLASS_NAMES)
    print("  Input size:", f"{IMG_SIZE}x{IMG_SIZE}")
    print("  Output grid:", f"{GRID_SIZE}x{GRID_SIZE}")
    print("=" * 60)
    
    # Reduced data for test mode
    if test_mode:
        train_size, val_size = 500, 100
        epochs = min(epochs, 5)
    else:
        train_size, val_size = 8000, 1000
    
    # Create datasets
    print(f"\n[1/5] Creating training dataset ({train_size} images)...")
    X_train, Y_train = create_fomo_dataset(num_images=train_size, max_digits_per_image=3, split='train')
    print(f"  X shape: {X_train.shape}, Y shape: {Y_train.shape}")
    
    print(f"\n[2/5] Creating validation dataset ({val_size} images)...")
    X_val, Y_val = create_fomo_dataset(num_images=val_size, max_digits_per_image=3, split='test')
    
    # Create model
    print("\n[3/5] Building FOMO model...")
    model = create_fomo_model()
    model.summary()
    
    # Class weights (background is very common, digits are rare)
    # Higher weight for digit classes to focus on detection
    class_weights = [0.1] + [1.0] * 10  # background=0.1, digits=1.0
    
    # Compile
    loss_fn = weighted_dice_loss(class_weights)
    model.compile(
        loss=loss_fn,
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
        metrics=[MeanIoU(NUM_CLASSES)]
    )
    
    # Callbacks - use weights-only checkpoint to avoid Keras format issues
    callbacks = [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "fomo_digit_best.weights.h5"),
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_with_warmup(
                epoch, lr, total_epochs=epochs, warmup_epochs=3, 
                min_lr=1e-6, max_lr=LEARNING_RATE
            )
        ),
        EarlyStopping(monitor='val_mean_iou', patience=10, mode='max', restore_best_weights=True)
    ]
    
    # Train
    print(f"\n[4/5] Training for {epochs} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    print("\n[5/5] Saving models...")
    model.save(os.path.join(OUTPUT_DIR, "fomo_digit.keras"))
    print(f"  Model saved: fomo_digit.keras")
    
    # Also save as H5 for compatibility
    try:
        model.save(os.path.join(OUTPUT_DIR, "fomo_digit.h5"), save_format='h5')
        print(f"  Model saved: fomo_digit.h5")
    except Exception as e:
        print(f"  Warning: Could not save H5 format: {e}")
    
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best Mean IoU: {max(history.history.get('val_mean_iou', [0])):.4f}")
    print("  Next: Run export_tflite.py")
    print("=" * 60)
    
    return model, history


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FOMO digit detection model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--test", action="store_true", help="Quick test mode with less data")
    args = parser.parse_args()
    
    train(epochs=args.epochs, test_mode=args.test)
