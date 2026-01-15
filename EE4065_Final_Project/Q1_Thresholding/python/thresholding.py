"""
EE4065 - Embedded Digital Image Processing
Final Project - Question 1a

Thresholding Function for Object Extraction
-------------------------------------------
Scenario:
- Single bright object in the image
- Background pixels are darker than object pixels
- Object size: 1000 pixels

Method: Extract exactly 1000 brightest pixels by sorting histogram
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def find_threshold_for_object_size(image: np.ndarray, target_pixels: int = 1000) -> int:
    """
    Find the threshold value that extracts exactly target_pixels bright pixels.
    
    Algorithm:
    1. Compute histogram of grayscale image
    2. Count pixels from brightest (255) to darkest (0)
    3. Find threshold where cumulative count >= target_pixels
    
    Args:
        image: Grayscale input image
        target_pixels: Number of pixels the object should have (default: 1000)
    
    Returns:
        Optimal threshold value (0-255)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate histogram (256 bins for 0-255 intensity values)
    histogram, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    
    # Count from brightest to darkest
    cumulative_count = 0
    threshold = 255
    
    for intensity in range(255, -1, -1):
        cumulative_count += histogram[intensity]
        if cumulative_count >= target_pixels:
            threshold = intensity
            break
    
    return threshold


def apply_thresholding(image: np.ndarray, target_pixels: int = 1000) -> Tuple[np.ndarray, int, int]:
    """
    Apply thresholding to extract the bright object.
    
    Args:
        image: Input image (grayscale or BGR)
        target_pixels: Expected object size in pixels
    
    Returns:
        Tuple of (binary_mask, threshold_value, actual_extracted_pixels)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Find optimal threshold
    threshold = find_threshold_for_object_size(gray, target_pixels)
    
    # Apply thresholding - pixels >= threshold become white (255)
    _, binary_mask = cv2.threshold(gray, threshold - 1, 255, cv2.THRESH_BINARY)
    
    # Count actual extracted pixels
    actual_pixels = np.sum(binary_mask == 255)
    
    return binary_mask, threshold, actual_pixels


def visualize_results(original: np.ndarray, mask: np.ndarray, threshold: int, 
                      target_pixels: int, actual_pixels: int):
    """
    Display original image, binary mask, and histogram analysis.
    """
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title(f'Binary Mask (Threshold = {threshold})')
    axes[0, 1].axis('off')
    
    # Histogram with threshold line
    histogram, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    axes[1, 0].bar(range(256), histogram, color='blue', alpha=0.7)
    axes[1, 0].axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold = {threshold}')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram with Threshold')
    axes[1, 0].legend()
    
    # Extracted object overlay
    if len(original.shape) == 3:
        overlay = original.copy()
    else:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Highlight extracted pixels in green
    overlay[mask == 255] = [0, 255, 0]
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Extracted Object: {actual_pixels} pixels (Target: {target_pixels})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('thresholding_result.png', dpi=150)
    plt.show()


def create_test_image(width: int = 320, height: int = 240, 
                      object_pixels: int = 1000) -> np.ndarray:
    """
    Create a synthetic test image with a bright object on dark background.
    
    Args:
        width, height: Image dimensions
        object_pixels: Number of pixels for the bright object
    
    Returns:
        Synthetic grayscale image
    """
    # Dark background (intensity 30-50)
    image = np.random.randint(30, 50, (height, width), dtype=np.uint8)
    
    # Calculate object dimensions (approximate circle)
    radius = int(np.sqrt(object_pixels / np.pi))
    center_x, center_y = width // 2, height // 2
    
    # Create bright circular object (intensity 200-230)
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = np.random.randint(200, 230, np.sum(mask), dtype=np.uint8)
    
    return image


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    TARGET_OBJECT_SIZE = 1000  # pixels
    
    print("=" * 60)
    print("EE4065 - Question 1a: Thresholding for Object Extraction")
    print("=" * 60)
    print(f"\nTarget object size: {TARGET_OBJECT_SIZE} pixels")
    
    # Load Lena grayscale image
    print("\nLoading Lena_gray.png...")
    image = cv2.imread("Lena_gray.png", cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("ERROR: Could not load Lena_gray.png!")
        print("Creating synthetic test image as fallback...")
        image = create_test_image(320, 240, object_pixels=TARGET_OBJECT_SIZE)
    
    # Apply thresholding
    print("Applying adaptive thresholding...")
    binary_mask, threshold, actual_pixels = apply_thresholding(image, TARGET_OBJECT_SIZE)
    
    # Print results
    print(f"\n--- RESULTS ---")
    print(f"Computed threshold value: {threshold}")
    print(f"Target pixels: {TARGET_OBJECT_SIZE}")
    print(f"Extracted pixels: {actual_pixels}")
    print(f"Accuracy: {100 * min(actual_pixels, TARGET_OBJECT_SIZE) / max(actual_pixels, TARGET_OBJECT_SIZE):.1f}%")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_results(image, binary_mask, threshold, TARGET_OBJECT_SIZE, actual_pixels)
    
    # Save binary mask
    cv2.imwrite("binary_mask.png", binary_mask)
    print("\nResults saved to: thresholding_result.png, binary_mask.png")
