# Question 2: YOLO Handwritten Digit Detection (40 points)

## Problem Statement
Implement handwritten digit detection via YOLO on ESP32-CAM.
- Train on MNIST dataset
- 4 classes: **0, 3, 5, 8** (reduced for ESP32 RAM constraints)
- Real-time object detection with bounding boxes

## Solution: YOLO-Tiny Architecture

Custom lightweight YOLO-like detector optimized for ESP32:
- **Input**: 96x96x1 grayscale
- **Output**: 6x6x9 grid (5 bbox params + 4 classes)
- **Target size**: ~150KB (int8 quantized)

```
Input 96x96x1
  ├── Conv2D 3x3, 8ch, stride=2 → 48x48
  ├── DepthwiseSeparable → 16ch
  ├── MaxPool → 24x24
  ├── DepthwiseSeparable → 32ch
  ├── MaxPool → 12x12
  ├── DepthwiseSeparable → 64ch
  ├── MaxPool → 6x6
  └── Detection Head → 6x6x9
```

## Files

### Python Training Pipeline
📁 `python/`

| File | Description |
|------|-------------|
| `train_yolo_tiny.py` | MNIST-based training for 4 classes |
| `export_tflite.py` | Int8 quantized TFLite export |
| `test_detection.py` | Python detection test script |

### ESP32-CAM Deployment
📁 `esp32_cam/`

| File | Description |
|------|-------------|
| `ESP32_YOLO_Detection.ino` | Arduino sketch with TFLite Micro |
| `model_data.h` | Generated model (after training) |

## Usage

### Step 1: Train Model (PC)
```bash
cd python
pip install tensorflow opencv-python
python train_yolo_tiny.py
```

### Step 2: Export for ESP32
```bash
python export_tflite.py
```
This generates `esp32_cam/model_data.h`

### Step 3: Test on PC
```bash
python test_detection.py --create-test
```

### Step 4: Deploy to ESP32-CAM
1. Install TensorFlowLite_ESP32 library in Arduino IDE
2. Open `esp32_cam/ESP32_YOLO_Detection.ino`
3. Select Board: "AI Thinker ESP32-CAM"
4. Upload and open Serial Monitor (115200)

## Expected Output
```
========================================
  DETECTIONS: 1 objects found
========================================
  [1] Digit 3: conf=0.87
      BBox: (0.25, 0.30) - (0.65, 0.70)
      Pixel: (24, 29) - (62, 67)
========================================
```
