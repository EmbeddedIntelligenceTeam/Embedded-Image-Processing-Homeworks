# EE4065 Embedded Digital Image Processing - Final Project

<p align="center">
  <img src="https://img.shields.io/badge/Platform-ESP32--CAM-blue" alt="Platform">
  <img src="https://img.shields.io/badge/Framework-Arduino-00979D" alt="Framework">
  <img src="https://img.shields.io/badge/ML-TensorFlow Lite-FF6F00" alt="TensorFlow Lite">
  <img src="https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green" alt="Languages">
</p>

> **Yeditepe University - Electrical and Electronics Engineering Department**  
> **Embedded Digital Image Processing Final Project**

This project implements various image processing and machine learning techniques on the ESP32-CAM module for handwritten digit recognition and detection systems.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Hardware Requirements](#-hardware-requirements)
- [Software Requirements](#-software-requirements)
- [Question 1: Thresholding](#-question-1-thresholding)
- [Question 2: YOLO Digit Detection](#-question-2-yolo-digit-detection)
- [Question 3: Upsampling and Downsampling](#-question-3-upsampling-and-downsampling)
- [Question 4: Multi-Model Digit Recognition](#-question-4-multi-model-digit-recognition)
- [Question 5: FOMO Digit Detection](#-question-5-fomo-digit-detection-bonus)
- [Installation and Usage](#-installation-and-usage)
- [References](#-references)

---

## 🎯 Project Overview

| Question | Topic | Points | Status |
|----------|-------|--------|--------|
| Q1 | Thresholding | 20 | ✅ Completed |
| Q2 | YOLO Digit Detection | 40 | ✅ Completed |
| Q3 | Upsampling/Downsampling | 20 | ✅ Completed |
| Q4 | Multi-Model CNN | 20 | ✅ Completed |
| Q5 | FOMO Digit Detection (Bonus) | 20 | ✅ Completed |

---

## 🔧 Hardware Requirements

| Component | Description |
|-----------|-------------|
| ESP32-CAM | AI-Thinker module (4MB Flash, PSRAM) |
| USB-TTL Converter | FTDI FT232RL or CH340G |
| Power Supply | 5V, min 500mA |

### ESP32-CAM Pin Configuration (AI-Thinker)

```cpp
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
```

---

## 💻 Software Requirements

### Python
```bash
pip install tensorflow numpy opencv-python matplotlib pillow
```

### Arduino IDE
- Board: `AI Thinker ESP32-CAM`
- Flash Mode: `QIO`
- Partition Scheme: `Huge APP (3MB No OTA/1MB SPIFFS)`

---

## 🔍 Question 1: Thresholding

### Problem Statement
Extract exactly **1000 pixels** of a bright object from a darker background using adaptive thresholding.

### Algorithm
The algorithm uses **histogram-based threshold selection**:
1. Calculate the image histogram
2. Compute cumulative sum from brightest to darkest pixels
3. Find the threshold where exactly 1000 pixels are above it

```python
def find_threshold_by_size(image, target_pixels=1000):
    histogram = np.histogram(image.flatten(), bins=256)[0]
    cumsum = 0
    for intensity in range(255, -1, -1):
        cumsum += histogram[intensity]
        if cumsum >= target_pixels:
            return intensity
    return 128
```

### Results

#### Python Implementation
![Q1 Python Thresholding](images/q1_python_thresholding.png)

**Key Observations:**
- **Input**: Lena grayscale image
- **Target**: 1000 pixels
- **Computed Threshold**: 213
- **Extracted Pixels**: 1066 (93.8% accuracy)
- The histogram clearly shows the threshold separating bright pixels (hat highlights)

#### ESP32-CAM Implementation
![Q1 ESP32 Thresholding](images/q1_esp32_thresholding.png)

**Key Observations:**
- Real-time thresholding on ESP32-CAM with web interface
- **Computed Threshold**: 252 (very high due to selfie lighting)
- **Extracted Pixels**: 1121 (87.9% accuracy)
- Binary mask shows extracted bright regions (face highlights)
- The algorithm adapts to different lighting conditions

### Files
- `Q1_Thresholding/python/thresholding.py` - PC implementation
- `Q1_Thresholding/esp32_cam/esp32_thresholding/` - ESP32 code

---

## 🎯 Question 2: YOLO Digit Detection

### Problem Statement
Detect handwritten digits (0-9) using a YOLO-style architecture on ESP32-CAM with bounding boxes.

### Architecture
Custom ultra-lightweight YOLO-Tiny designed for ESP32 constraints:

```
Input: 96×96×1 (Grayscale)
├── Conv2D (32, 3×3, stride=2)  → 48×48×32
├── Conv2D (64, 3×3, stride=2)  → 24×24×64
├── Conv2D (128, 3×3, stride=2) → 12×12×128
├── Conv2D (256, 3×3, stride=2) → 6×6×256
└── Detection Head (6×6×15)

Output: 6×6 grid × 15 values per cell
        (tx, ty, tw, th, conf, class[0-9])
```

### Key Features
- **Adaptive Thresholding**: Converts camera images to MNIST-like format
- **NMS (Non-Maximum Suppression)**: Removes duplicate detections
- **Real-time Web Interface**: Modern gradient UI with detection overlay

### Results

#### Detection Result 1 - Multiple Digits
![Q2 YOLO Detection 1](images/q2_yolo_detection1.png)

**Analysis:**
- **4 digits detected**: 3, 5, 8, 0
- **Inference time**: 33.582 seconds (includes preprocessing)
- **Confidence scores**: 64.9% - 99.4%
- Bounding boxes accurately surround each digit
- Web interface shows coordinates for each detection

#### Detection Result 2 - Clear Digits
![Q2 YOLO Detection 2](images/q2_yolo_detection2.png)

**Analysis:**
- **3 digits detected**: 0, 3, 5
- **Confidence scores**: 97.1% - 97.8% (very high)
- Clean bounding boxes with excellent localization
- Model correctly identifies digit at multiple positions

#### Detection Result 3 - Multiple Same Digits
![Q2 YOLO Detection 3](images/q2_yolo_detection3.png)

**Analysis:**
- **6 digits detected** including duplicates
- Successfully detects same digit (5) at different positions
- Shows model's ability to handle cluttered scenes
- Some lower confidence predictions (39.7%) for partially visible digits

### Technical Details

#### Preprocessing Pipeline
```cpp
void preprocessImage(uint8_t* src, int8_t* dst) {
    // 1. Calculate average brightness
    uint8_t avg = calculateAverage(src);
    
    // 2. Adaptive threshold (FOMO-style)
    uint8_t threshold = avg - 30;
    
    // 3. Convert to MNIST format (white digit on black)
    for (int i = 0; i < size; i++) {
        dst[i] = (src[i] < threshold) ? 127 : -128;
    }
}
```

#### Detection Decoding
```cpp
void decodeDetections() {
    for (int gy = 0; gy < 6; gy++) {
        for (int gx = 0; gx < 6; gx++) {
            float conf = sigmoid(output[offset + 4]);
            if (conf < THRESHOLD) continue;
            
            // Decode bounding box
            float bx = (sigmoid(tx) + gx) / 6;
            float by = (sigmoid(ty) + gy) / 6;
            float bw = tw;
            float bh = th;
            
            // Store detection
            detections[n++] = {digit, bx, by, bw, bh, conf};
        }
    }
    applyNMS();  // Remove duplicates
}
```

### Model Performance
| Metric | Value |
|--------|-------|
| Model Size | 18 KB (TFLite int8) |
| Inference Time | ~120 ms |
| Training Accuracy | 85% |

---

## 📐 Question 3: Upsampling and Downsampling

### Problem Statement
Implement image scaling with **non-integer factors** (e.g., 1.5×, 2/3×) using bilinear interpolation.

### Algorithm: Bilinear Interpolation

For each destination pixel (dx, dy):
```
sx = dx × (src_width / dst_width)
sy = dy × (src_height / dst_height)

Interpolated = (1-fx)×(1-fy)×P00 + fx×(1-fy)×P10
             + (1-fx)×fy×P01 + fx×fy×P11
```

### Results

#### Python Implementation
![Q3 Python Scaling](images/q3_python_scaling.png)

**Analysis:**
- **Original**: 28×28 (MNIST digit "5")
- **Upsampled (1.5×)**: 42×42 - Smooth interpolation preserves edges
- **Downsampled (1/1.5)**: 18×18 - Aliasing visible but digit recognizable
- Bilinear interpolation maintains visual quality

#### ESP32-CAM Implementation
![Q3 ESP32 Scaling](images/q3_esp32_scaling.png)

**Analysis:**
- **Original**: 320×240 (QVGA camera capture)
- **Upsampled (1.5×)**: 480×360 - Larger image with smooth interpolation
- **Downsampled (0.67×)**: 213×160 - Reduced size for faster processing
- Real-time scaling with color preservation (RGB565 format)
- Web interface shows side-by-side comparison

### Implementation
```cpp
void bilinearResize(uint8_t* src, int srcW, int srcH,
                    uint8_t* dst, int dstW, int dstH) {
    float x_ratio = (float)srcW / dstW;
    float y_ratio = (float)srcH / dstH;
    
    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            float sx = dx * x_ratio;
            float sy = dy * y_ratio;
            
            // Get 4 nearest pixels
            int x0 = (int)sx, y0 = (int)sy;
            float fx = sx - x0, fy = sy - y0;
            
            // Bilinear interpolation
            dst[dy*dstW + dx] = 
                (1-fx)*(1-fy)*src[y0*srcW + x0] +
                fx*(1-fy)*src[y0*srcW + x0+1] +
                (1-fx)*fy*src[(y0+1)*srcW + x0] +
                fx*fy*src[(y0+1)*srcW + x0+1];
        }
    }
}
```

---

## 🧠 Question 4: Multi-Model Digit Recognition

### Problem Statement
Implement handwritten digit recognition using **multiple CNN models** and fuse their results.

### Models Implemented

| Model | Architecture | Size | Accuracy |
|-------|--------------|------|----------|
| SqueezeNet-Mini | Fire modules | 24 KB | 98.4% |
| MobileNetV2-Mini | Depthwise separable conv | 21 KB | 97.4% |
| ResNet-8 | Residual connections | 33 KB | 98.9% |
| EfficientNet-Mini | Compound scaling | 24 KB | 98.4% |
| **Ensemble** | Weighted voting | - | **~99%** |

### Results

#### Recognition Result 1 - Digit "5"
![Q4 CNN Result 1](images/q4_cnn_result1.png)

**Analysis:**
- **Detected Digit**: 5
- **Ensemble Confidence**: 5.4%
- **Inference Time**: 1810 ms
- Left panel shows live camera and processed 32×32 input
- Right panel shows individual model results
- All 4 models available for individual testing

#### Recognition Result 2 - Digit "0"
![Q4 CNN Result 2](images/q4_cnn_result2.png)

**Analysis:**
- **Detected Digit**: 0
- **All Models Agree**: SqueezeNet, MobileNet, ResNet, EfficientNet all predict "0"
- **Individual Inference Times**: ~450 ms per model
- **Ensemble Time**: 1809 ms (runs all 4 models)
- Model fusion produces robust predictions

### Ensemble Method
```cpp
int runEnsemble() {
    float combined[10] = {0};
    float weights[] = {0.25, 0.25, 0.25, 0.25};
    
    for (int m = 0; m < 4; m++) {
        runSingleModel(m);
        for (int c = 0; c < 10; c++) {
            combined[c] += weights[m] * probabilities[m][c];
        }
    }
    
    return argmax(combined);
}
```

### Key Features
- **FOMO-style Preprocessing**: Adaptive thresholding for MNIST-like input
- **TFLite int8 Quantization**: All models <35KB for ESP32 flash
- **Modern Web UI**: Gradient design with model selection buttons
- **Live Camera Preview**: 32×32 processed input visualization

---

## 🔍 Question 5: FOMO Digit Detection (BONUS)

### Problem Statement
Implement **FOMO (Faster Objects, More Objects)** for lightweight object detection on ESP32-CAM.

### FOMO vs YOLO

| Feature | YOLO | FOMO |
|---------|------|------|
| Output | Bounding boxes | Centroids |
| Complexity | Higher | Lower |
| Speed | ~120 ms | ~100 ms |
| Model Size | 18 KB | 58 KB |
| NMS Required | Yes | No |

### Architecture: MobileNetV2 Backbone

```
Input: 96×96×1
├── Conv (stride=2)                 → 48×48
├── Inverted Residual (stride=1)    → 48×48
├── Inverted Residual (stride=2)    → 24×24
├── Inverted Residual (stride=2)    → 12×12 (8× downsampling)
└── Detection Head                  → 12×12×11

Output: 12×12 grid × 11 classes (background + 10 digits)
```

### Results

#### Detection Result 1 - Five Digits
![Q5 FOMO Detection 1](images/q5_fomo_detection1.png)

**Analysis:**
- **5 digits detected**: 0, 3, 5, 8, 9
- **All with 99.6% confidence!**
- **Inference Time**: 5030 ms
- Centroid-based detection (no bounding boxes)
- Modern web UI with gradient design

#### Detection Result 2 - Four Digits
![Q5 FOMO Detection 2](images/q5_fomo_detection2.png)

**Analysis:**
- **4 digits detected**: 2, 4, 7, 8
- **Confidence**: 91.0% - 99.6%
- Note: Digit 4 has lower confidence (91.0%) - slightly occluded
- Grid coordinates shown (e.g., "at (20,36)")

#### Detection Result 3 - Clear Detection
![Q5 FOMO Detection 3](images/q5_fomo_detection3.png)

**Analysis:**
- **3 digits detected**: 1, 4, 9
- **All 99.6% confidence**
- Clean detections with accurate localization
- 12×12 grid provides 8-pixel resolution

#### Detection Result 4 - Multiple Digits (7, 5, 3)
![Q5 FOMO Detection 4](images/q5_fomo_detection4.png)

**Analysis:**
- **4 digits detected**: 3, 5, 7 (two 7s detected at different positions)
- **Confidence**: 97.7% - 99.6%
- Model detects duplicate digits at different grid locations
- Inference time: 5032 ms

#### Detection Result 5 - Single Digit
![Q5 FOMO Detection 5](images/q5_fomo_detection5.png)

**Analysis:**
- **1 digit detected**: 5
- **Confidence**: 99.6%
- Clean single detection with no false positives
- Demonstrates model's precision with isolated digits

### Key Advantages

1. **No NMS Required**: Each grid cell independently predicts class
2. **High Confidence**: Most detections >99%
3. **Simple Decoding**: Just find argmax per cell
4. **Lightweight**: Only 58 KB model

### Implementation
```cpp
void decodeDetections() {
    for (int gy = 0; gy < 12; gy++) {
        for (int gx = 0; gx < 12; gx++) {
            // Find best class (skip background at index 0)
            int best_class = 0;
            float best_conf = 0;
            
            for (int c = 1; c < 11; c++) {
                float conf = output[gy][gx][c];
                if (conf > best_conf) {
                    best_conf = conf;
                    best_class = c - 1;  // Digit 0-9
                }
            }
            
            if (best_conf > THRESHOLD) {
                // Report centroid (center of grid cell)
                int cx = gx * 8 + 4;
                int cy = gy * 8 + 4;
                addDetection(best_class, cx, cy, best_conf);
            }
        }
    }
}
```

---

## 🚀 Installation and Usage

### 1. Clone Repository
```bash
git clone https://github.com/[username]/EE4065_Final_Project.git
cd EE4065_Final_Project
```

### 2. Python Setup
```bash
pip install tensorflow numpy opencv-python matplotlib pillow
```

### 3. Arduino IDE Setup
1. Add ESP32 board URL in Preferences
2. Install `AI Thinker ESP32-CAM` board
3. Install `TensorFlowLite_ESP32` library

### 4. Upload to ESP32-CAM
1. Open `.ino` file
2. Connect GPIO0 to GND
3. Upload code
4. Disconnect GPIO0, press Reset
5. Open Serial Monitor for IP address
6. Access web interface at `http://[IP]`

---

## 📊 Summary of Results

| Question | Model/Method | Key Metric | Result |
|----------|--------------|------------|--------|
| Q1 | Histogram Thresholding | Accuracy | 87.9-93.8% |
| Q2 | YOLO-Tiny | Detection Confidence | 65-99% |
| Q3 | Bilinear Interpolation | Scaling Factors | 1.5×, 0.67× |
| Q4 | Ensemble CNN | Classification Accuracy | 98.23% |
| Q5 | FOMO MobileNetV2 | Detection Confidence | 91-99.6% |

---

## 📚 References

- [bhoke/FOMO](https://github.com/bhoke/FOMO) - FOMO implementation reference
- [STMicroelectronics Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-CAM Pinout](https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/)

---

<p align="center">
  <strong>Yeditepe University - Electrical and Electronics Engineering</strong><br>
  EE4065 - Embedded Digital Image Processing<br>
  Final Project - 2026
</p>
