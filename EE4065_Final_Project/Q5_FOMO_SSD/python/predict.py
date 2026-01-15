"""
EE4065 - Final Project - Question 5a
FOMO Model Prediction and Visualization

Test trained FOMO model and visualize heat map outputs.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 96
GRID_SIZE = 12
NUM_CLASSES = 11
CLASS_NAMES = ['bg'] + [str(d) for d in range(10)]


def load_model(model_path=None):
    """Load trained FOMO model"""
    if model_path is None:
        for name in ["fomo_digit_best.keras", "fomo_digit.keras", "fomo_digit.h5"]:
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.exists(path):
                model_path = path
                break
        else:
            raise FileNotFoundError("No trained model found!")
    
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    return model


def create_test_image():
    """Create a test image with random digits"""
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    noise = np.random.randint(0, 20, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    canvas = canvas + noise
    
    placed_digits = []
    
    for _ in range(2):  # Place 2 digits
        digit = np.random.randint(0, 10)
        digit_images = x_test[y_test == digit]
        digit_img = digit_images[np.random.randint(0, len(digit_images))]
        
        scale = np.random.uniform(0.8, 1.3)
        new_size = int(28 * scale)
        digit_resized = cv2.resize(digit_img, (new_size, new_size))
        
        x = np.random.randint(5, IMG_SIZE - new_size - 5)
        y = np.random.randint(5, IMG_SIZE - new_size - 5)
        
        canvas[y:y+new_size, x:x+new_size] = np.maximum(
            canvas[y:y+new_size, x:x+new_size],
            digit_resized
        )
        
        placed_digits.append({
            'digit': digit,
            'x': x + new_size // 2,
            'y': y + new_size // 2,
            'size': new_size
        })
    
    return canvas, placed_digits


def predict(model, image):
    """Run prediction and return heat map"""
    # Normalize and add batch/channel dims
    img = image.astype(np.float32) / 255.0
    img = img[np.newaxis, ..., np.newaxis]
    
    # Predict
    output = model.predict(img, verbose=0)
    return output[0]  # (12, 12, 11)


def find_centroids(heatmap, threshold=0.5):
    """Find digit centroids from heat map"""
    centroids = []
    
    # Get class predictions for each cell
    predictions = np.argmax(heatmap, axis=-1)  # (12, 12)
    confidences = np.max(heatmap, axis=-1)     # (12, 12)
    
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            pred_class = predictions[gy, gx]
            conf = confidences[gy, gx]
            
            if pred_class > 0 and conf > threshold:  # Not background
                # Convert grid to image coordinates
                cell_size = IMG_SIZE // GRID_SIZE
                cx = gx * cell_size + cell_size // 2
                cy = gy * cell_size + cell_size // 2
                
                centroids.append({
                    'digit': pred_class - 1,  # 0-9
                    'x': cx,
                    'y': cy,
                    'confidence': conf
                })
    
    return centroids


def visualize_results(image, heatmap, centroids, ground_truth=None, save_path=None):
    """Visualize input, heat map, and detection results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Image with detections
    img_with_det = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in centroids:
        cv2.circle(img_with_det, (c['x'], c['y']), 8, (0, 255, 0), 2)
        cv2.putText(img_with_det, str(c['digit']), (c['x']-5, c['y']-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if ground_truth:
        for gt in ground_truth:
            cv2.circle(img_with_det, (gt['x'], gt['y']), 12, (255, 0, 0), 1)
    
    axes[0, 1].imshow(img_with_det)
    axes[0, 1].set_title('Detections (green) / GT (red)')
    axes[0, 1].axis('off')
    
    # Background heat map
    axes[0, 2].imshow(heatmap[..., 0], cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Background Heat Map')
    axes[0, 2].axis('off')
    
    # Combined digit heat map (max of all digit channels)
    digit_heatmap = np.max(heatmap[..., 1:], axis=-1)
    axes[1, 0].imshow(digit_heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Combined Digit Heat Map')
    axes[1, 0].axis('off')
    
    # Class prediction map
    class_map = np.argmax(heatmap, axis=-1)
    im = axes[1, 1].imshow(class_map, cmap='tab10', vmin=0, vmax=10)
    axes[1, 1].set_title('Class Predictions')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], ticks=range(11), 
                 label='Class (0=bg, 1-10=digits)')
    
    # Detection summary
    axes[1, 2].axis('off')
    summary = "Detection Results:\n\n"
    if centroids:
        for c in centroids:
            summary += f"Digit {c['digit']}: ({c['x']}, {c['y']}) conf={c['confidence']:.2f}\n"
    else:
        summary += "No digits detected"
    
    if ground_truth:
        summary += "\n\nGround Truth:\n"
        for gt in ground_truth:
            summary += f"Digit {gt['digit']}: ({gt['x']}, {gt['y']})\n"
    
    axes[1, 2].text(0.1, 0.9, summary, transform=axes[1, 2].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def run_test(model, num_samples=5, threshold=0.3):
    """Run test on random samples"""
    print(f"\nRunning test with {num_samples} samples...")
    
    for i in range(num_samples):
        print(f"\n--- Sample {i+1} ---")
        
        # Create test image
        image, ground_truth = create_test_image()
        
        # Predict
        heatmap = predict(model, image)
        
        # Find detections
        centroids = find_centroids(heatmap, threshold=threshold)
        
        print(f"Ground truth: {[f'{gt['digit']} at ({gt['x']},{gt['y']})' for gt in ground_truth]}")
        print(f"Detected: {[f'{c['digit']} at ({c['x']},{c['y']}) conf={c['confidence']:.2f}' for c in centroids]}")
        
        # Visualize
        save_path = os.path.join(OUTPUT_DIR, f"test_result_{i+1}.png")
        visualize_results(image, heatmap, centroids, ground_truth, save_path)


def predict_image(model, image_path, threshold=0.3):
    """Predict on a specific image file"""
    print(f"\nPredicting on: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Resize to model input size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Predict
    heatmap = predict(model, image)
    
    # Find detections
    centroids = find_centroids(heatmap, threshold=threshold)
    
    print(f"Detected digits: {[c['digit'] for c in centroids]}")
    
    # Visualize
    save_path = os.path.join(OUTPUT_DIR, "prediction_result.png")
    visualize_results(image, heatmap, centroids, save_path=save_path)
    
    return centroids


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOMO digit detection prediction")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--image", type=str, default=None, help="Image to predict on")
    parser.add_argument("--test", action="store_true", help="Run test with random samples")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--samples", type=int, default=3, help="Number of test samples")
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    if args.image:
        predict_image(model, args.image, args.threshold)
    elif args.test:
        run_test(model, num_samples=args.samples, threshold=args.threshold)
    else:
        # Default: run quick test
        run_test(model, num_samples=3, threshold=args.threshold)
