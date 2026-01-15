# Question 3: Upsampling and Downsampling (20 points)

## Problem Statement
Implement upsampling and downsampling operations on ESP32-CAM.
- **Q3a (10 points)**: Upsampling with any scale factor
- **Q3b (10 points)**: Downsampling with any scale factor
- Must handle **non-integer values** like 1.5, 2/3, etc.

## Solution Approach

### Upsampling (Bilinear Interpolation)
- Maps each destination pixel to a fractional source position
- Interpolates between 4 neighboring source pixels
- Produces smooth, anti-aliased results

```
dst[y,x] = (1-fx)(1-fy)*src[y0,x0] + fx(1-fy)*src[y0,x1] +
           (1-fx)*fy*src[y1,x0] + fx*fy*src[y1,x1]
```

### Downsampling (Area Averaging)
- Averages all source pixels that map to each destination pixel
- Provides anti-aliasing to prevent moiré patterns
- Better quality than simple subsampling

## Files

### Python Implementation
📁 `python/scaling.py`
- `upsample(image, scale)` - Upsample by any factor
- `downsample(image, scale)` - Downsample by any factor
- Visualization included

### ESP32-CAM Implementation
📁 `esp32_cam/ESP32_Scaling.ino`
- `upsample()` - C implementation with bilinear interpolation
- `downsample()` - C implementation with area averaging
- Tests: 1.5x upsample, 2/3 downsample, 1/1.5 downsample, 0.5x downsample

## Usage

### Python Test
```bash
cd python
python scaling.py
```

### ESP32-CAM Test
1. Upload `ESP32_Scaling.ino` to ESP32-CAM
2. Open Serial Monitor (115200 baud)
3. Press any key to capture and scale
4. View scaling results in Serial output

## Example Output
```
Captured: 320x240
Original: 320x240, min=12, max=245, mean=98.3

--- TEST 1: Upsample by 1.5x ---
Upsampling: 320x240 -> 480x360 (scale=1.50)
Upsampled: 480x360, min=12, max=245, mean=98.3

--- TEST 2: Downsample by 2/3 (scale=0.667) ---
Downsampling: 320x240 -> 213x160 (scale=0.667)
Downsampled (2/3): 213x160, min=14, max=243, mean=98.5
```

## Non-Integer Scale Support

| Input Scale | Operation | Result |
|-------------|-----------|--------|
| 1.5 (up) | 320x240 → 480x360 | Upsample 1.5x |
| 2/3 = 0.667 | 320x240 → 213x160 | Downsample to 2/3 |
| 1.5 (down) | 320x240 → 213x160 | Downsample by 1.5 divisor |
| 0.5 | 320x240 → 160x120 | Downsample to half |
