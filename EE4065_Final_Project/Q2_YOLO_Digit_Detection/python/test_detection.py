"""
EE4065 - Final Project - Question 2
Python Detection Test Script
-----------------------------
Test the trained YOLO-tiny model on images.
Draws bounding boxes and class labels.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# ==================== CONFIGURATION ====================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo_tiny_digit.keras")
IMG_SIZE = 96
GRID_SIZE = 6
CLASS_NAMES = ['0', '3', '5', '8']
CONFIDENCE_THRESHOLD = 0.3
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def preprocess_image(image_path):
    """Load and preprocess image for detection."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Keep original for drawing
    original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    
    # Resize for model
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized[np.newaxis, ..., np.newaxis], original


def decode_predictions(predictions, conf_threshold=0.3):
    """
    Decode YOLO grid predictions to bounding boxes.
    
    Args:
        predictions: (6, 6, 9) grid output
        conf_threshold: Minimum confidence to keep
    
    Returns:
        List of (x1, y1, x2, y2, confidence, class_id, class_name)
    """
    detections = []
    
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            cell = predictions[gy, gx]
            
            # Get confidence (apply sigmoid)
            conf = 1 / (1 + np.exp(-cell[4]))
            
            if conf < conf_threshold:
                continue
            
            # Get class (apply softmax)
            class_scores = cell[5:]
            class_scores = np.exp(class_scores) / np.sum(np.exp(class_scores))
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Combined confidence
            final_conf = conf * class_conf
            if final_conf < conf_threshold:
                continue
            
            # Decode bounding box
            # x, y are offsets within cell (0-1)
            # w, h are relative to whole image (0-1)
            cx = (gx + cell[0]) / GRID_SIZE
            cy = (gy + cell[1]) / GRID_SIZE
            w = cell[2]
            h = cell[3]
            
            # Convert to corner format
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = min(1, cx + w / 2)
            y2 = min(1, cy + h / 2)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(final_conf),
                'class_id': int(class_id),
                'class_name': CLASS_NAMES[class_id]
            })
    
    return detections


def draw_detections(image, detections):
    """Draw bounding boxes on image."""
    h, w = image.shape[:2]
    result = image.copy()
    
    for det in detections:
        x1 = int(det['bbox'][0] * w)
        y1 = int(det['bbox'][1] * h)
        x2 = int(det['bbox'][2] * w)
        y2 = int(det['bbox'][3] * h)
        
        color = COLORS[det['class_id']]
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(result, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result


def detect(model, image_path):
    """Run detection on an image."""
    # Preprocess
    input_data, original = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(input_data, verbose=0)[0]
    
    # Decode
    detections = decode_predictions(predictions, CONFIDENCE_THRESHOLD)
    
    # Draw
    result = draw_detections(original, detections)
    
    return result, detections


def create_test_image():
    """Create a test image with MNIST digit."""
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Find a digit from our classes
    TARGET_DIGITS = [0, 3, 5, 8]
    digit = np.random.choice(TARGET_DIGITS)
    mask = y_test == digit
    img = x_test[mask][0]
    
    # Create canvas
    canvas = np.zeros((200, 200), dtype=np.uint8)
    
    # Random placement
    scale = np.random.uniform(1.5, 3.0)
    new_size = int(28 * scale)
    resized = cv2.resize(img, (new_size, new_size))
    
    x_pos = np.random.randint(10, 200 - new_size - 10)
    y_pos = np.random.randint(10, 200 - new_size - 10)
    
    canvas[y_pos:y_pos+new_size, x_pos:x_pos+new_size] = resized
    
    test_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    cv2.imwrite(test_path, canvas)
    
    print(f"Created test image with digit {digit} at {test_path}")
    return test_path


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-Tiny Digit Detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--create-test', action='store_true', help='Create test image from MNIST')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  YOLO-Tiny Digit Detection Test")
    print("=" * 60)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_yolo_tiny.py first!")
        exit(1)
    
    print("\nLoading model...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    
    # Get or create image
    if args.create_test or args.image is None:
        print("Creating test image...")
        image_path = create_test_image()
    else:
        image_path = args.image
    
    # Run detection
    print(f"\nRunning detection on: {image_path}")
    result_img, detections = detect(model, image_path)
    
    # Print results
    print(f"\nDetections: {len(detections)}")
    for det in detections:
        print(f"  - Digit {det['class_name']}: conf={det['confidence']:.2f}, "
              f"bbox=({det['bbox'][0]:.2f}, {det['bbox'][1]:.2f}, "
              f"{det['bbox'][2]:.2f}, {det['bbox'][3]:.2f})")
    
    # Save result
    output_path = os.path.join(os.path.dirname(__file__), "detection_result.png")
    cv2.imwrite(output_path, result_img)
    print(f"\nResult saved: {output_path}")
    
    # Show result
    cv2.imshow("Detection Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
