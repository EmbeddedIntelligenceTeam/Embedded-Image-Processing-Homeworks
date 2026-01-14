"""
EE4065 - Embedded Digital Image Processing
Final Project - Question 3

Upsampling and Downsampling Operations
---------------------------------------
- Q3a: Upsampling with any scale factor (including non-integer like 1.5)
- Q3b: Downsampling with any scale factor (including non-integer like 2/3)

Implements bilinear interpolation for smooth scaling.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def upsample(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Upsample an image by a given scale factor.
    
    Uses bilinear interpolation for smooth results.
    Supports non-integer scale factors like 1.5, 2.5, etc.
    
    Args:
        image: Input grayscale or color image
        scale: Scale factor (>1 for upsampling) e.g., 1.5, 2, 2.5
    
    Returns:
        Upsampled image
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")
    
    h, w = image.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    if len(image.shape) == 2:
        # Grayscale
        result = np.zeros((new_h, new_w), dtype=image.dtype)
    else:
        # Color
        result = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    
    # Bilinear interpolation
    for y in range(new_h):
        for x in range(new_w):
            # Map to source coordinates
            src_x = x / scale
            src_y = y / scale
            
            # Get integer and fractional parts
            x0 = int(src_x)
            y0 = int(src_y)
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            
            # Fractional parts for interpolation
            fx = src_x - x0
            fy = src_y - y0
            
            # Bilinear interpolation formula
            if len(image.shape) == 2:
                value = (1 - fx) * (1 - fy) * image[y0, x0] + \
                        fx * (1 - fy) * image[y0, x1] + \
                        (1 - fx) * fy * image[y1, x0] + \
                        fx * fy * image[y1, x1]
                result[y, x] = np.clip(value, 0, 255).astype(image.dtype)
            else:
                for c in range(image.shape[2]):
                    value = (1 - fx) * (1 - fy) * image[y0, x0, c] + \
                            fx * (1 - fy) * image[y0, x1, c] + \
                            (1 - fx) * fy * image[y1, x0, c] + \
                            fx * fy * image[y1, x1, c]
                    result[y, x, c] = np.clip(value, 0, 255).astype(image.dtype)
    
    return result


def downsample(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Downsample an image by a given scale factor.
    
    Uses area averaging for anti-aliasing (better quality than simple subsampling).
    Supports non-integer scale factors like 2/3, 0.5, 0.75, etc.
    
    Args:
        image: Input grayscale or color image
        scale: Scale factor (<1 for downsampling) e.g., 0.5, 0.667, 0.75
               OR factor to divide by (e.g., 2 means 1/2 size)
    
    Returns:
        Downsampled image
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")
    
    # If scale > 1, interpret as divisor (e.g., scale=2 means 1/2 size)
    if scale > 1:
        actual_scale = 1.0 / scale
    else:
        actual_scale = scale
    
    h, w = image.shape[:2]
    new_h = max(1, int(h * actual_scale))
    new_w = max(1, int(w * actual_scale))
    
    if len(image.shape) == 2:
        result = np.zeros((new_h, new_w), dtype=np.float32)
    else:
        result = np.zeros((new_h, new_w, image.shape[2]), dtype=np.float32)
    
    # Area averaging for anti-aliasing
    # Each output pixel is the average of the corresponding area in input
    src_h_step = h / new_h
    src_w_step = w / new_w
    
    for y in range(new_h):
        for x in range(new_w):
            # Source region boundaries
            y_start = y * src_h_step
            y_end = (y + 1) * src_h_step
            x_start = x * src_w_step
            x_end = (x + 1) * src_w_step
            
            # Integer boundaries
            y0 = int(y_start)
            y1 = min(int(np.ceil(y_end)), h)
            x0 = int(x_start)
            x1 = min(int(np.ceil(x_end)), w)
            
            # Extract region and average
            if len(image.shape) == 2:
                region = image[y0:y1, x0:x1].astype(np.float32)
                result[y, x] = np.mean(region)
            else:
                for c in range(image.shape[2]):
                    region = image[y0:y1, x0:x1, c].astype(np.float32)
                    result[y, x, c] = np.mean(region)
    
    return result.astype(image.dtype)


def visualize_scaling(original: np.ndarray, upsampled: np.ndarray, 
                      downsampled: np.ndarray, up_scale: float, down_scale: float):
    """Visualize original, upsampled, and downsampled images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title(f'Original\n{original.shape[1]}x{original.shape[0]}')
    axes[0].axis('off')
    
    # Upsampled
    axes[1].imshow(upsampled, cmap='gray' if len(upsampled.shape) == 2 else None)
    axes[1].set_title(f'Upsampled (x{up_scale})\n{upsampled.shape[1]}x{upsampled.shape[0]}')
    axes[1].axis('off')
    
    # Downsampled
    axes[2].imshow(downsampled, cmap='gray' if len(downsampled.shape) == 2 else None)
    down_label = f'1/{down_scale}' if down_scale > 1 else f'x{down_scale}'
    axes[2].set_title(f'Downsampled ({down_label})\n{downsampled.shape[1]}x{downsampled.shape[0]}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('scaling_result.png', dpi=150)
    plt.show()


# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  EE4065 - Question 3: Upsampling & Downsampling")
    print("=" * 60)
    
    # Create or load test image
    print("\nCreating test image (MNIST digit)...")
    from tensorflow import keras
    (x_train, _), _ = keras.datasets.mnist.load_data()
    original = x_train[0]  # 28x28 digit
    
    print(f"Original size: {original.shape}")
    
    # Test upsampling with non-integer scale
    up_scale = 1.5
    print(f"\n[1] Upsampling by {up_scale}x...")
    upsampled = upsample(original, up_scale)
    print(f"  Result size: {upsampled.shape}")
    print(f"  Expected: ({int(28*up_scale)}, {int(28*up_scale)})")
    
    # Test downsampling with non-integer scale (2/3)
    down_scale = 2/3  # Results in 2/3 of original size
    print(f"\n[2] Downsampling to {down_scale:.3f}x (2/3)...")
    downsampled = downsample(original, down_scale)
    print(f"  Result size: {downsampled.shape}")
    
    # Test with divisor notation (1.5 means 1/1.5 size)
    down_scale2 = 1.5  # Interpreted as divisor: 1/1.5 size
    print(f"\n[3] Downsampling by 1/{down_scale2}...")
    downsampled2 = downsample(original, down_scale2)
    print(f"  Result size: {downsampled2.shape}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_scaling(original, upsampled, downsampled2, up_scale, down_scale2)
    
    print("\n" + "=" * 60)
    print("  Complete! Results saved to scaling_result.png")
    print("=" * 60)
