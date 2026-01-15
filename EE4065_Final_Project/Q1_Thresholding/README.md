# Question 1: Thresholding Function (20 points)

## Problem Statement
Form a thresholding function for the following scenario:
- Single bright object in the image acquired by ESP32-CAM
- Background pixels are darker than object pixels
- Object size: **1000 pixels**
- Extract the object based on its size

## Solution Approach

### Algorithm: Histogram-Based Adaptive Thresholding
1. Compute histogram of the grayscale image (256 bins)
2. Count pixels from brightest intensity (255) down to darkest (0)
3. Find the threshold where cumulative count reaches target pixels (1000)
4. Apply binary thresholding at computed value

```
Threshold = argmin{T : Σ(count[i]) >= 1000, for i = 255 to T}
```

## Files

### Q1a: Python Implementation (5 points)
📁 `python/thresholding.py`

**Features:**
- `find_threshold_for_object_size()` - Computes optimal threshold
- `apply_thresholding()` - Creates binary mask
- `visualize_results()` - Generates histogram and result visualization
- `create_test_image()` - Creates synthetic test image

**Usage:**
```bash
cd python
python thresholding.py
```

### Q1b: ESP32-CAM Implementation (15 points)
📁 `esp32_cam/esp32_thresholding.ino`

**Features:**
- Camera capture at QVGA (320x240) grayscale
- Real-time histogram computation
- Adaptive threshold calculation
- Serial output with results

**Upload to ESP32-CAM:**
1. Open in Arduino IDE
2. Select Board: "AI Thinker ESP32-CAM"
3. Upload and open Serial Monitor (115200 baud)

## Output Example
```
========================================
       THRESHOLDING RESULTS
========================================
Target object pixels: 1000
Computed threshold:   187
Extracted pixels:     1003
Extraction accuracy:  99.7%
========================================
```
