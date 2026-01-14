/**
 * EE4065 Final Project - Question 4
 * Handwritten Digit Recognition with Multiple CNN Models
 * 
 * Author: Yusuf - ESP32-CAM Implementation
 * 
 * Features:
 *   - 4 different CNN architectures comparison
 *   - Model fusion with weighted averaging
 *   - Real-time web interface
 *   - Adaptive image preprocessing
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// TFLite Micro Libraries
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Weights
#include "squeezenetmini_model.h"
#include "mobilenetv2mini_model.h"
#include "resnet8_model.h"
#include "efficientnetmini_model.h"

// ====================== Network Settings ======================
const char* WIFI_SSID = "Yusuf's Xiaomi";
const char* WIFI_PASS = "yusuf4418";
const char* AP_SSID = "YusufDigit_Q4";
const char* AP_PASS = "ee4065q4";

// ====================== Hardware Pins (AI-THINKER) ======================
#define CAM_PIN_PWDN     32
#define CAM_PIN_RESET    -1
#define CAM_PIN_XCLK      0
#define CAM_PIN_SIOD     26
#define CAM_PIN_SIOC     27
#define CAM_PIN_D7       35
#define CAM_PIN_D6       34
#define CAM_PIN_D5       39
#define CAM_PIN_D4       36
#define CAM_PIN_D3       21
#define CAM_PIN_D2       19
#define CAM_PIN_D1       18
#define CAM_PIN_D0        5
#define CAM_PIN_VSYNC    25
#define CAM_PIN_HREF     23
#define CAM_PIN_PCLK     22
#define LED_FLASH         4

// ====================== Model Configuration ======================
enum ModelType { 
    MDL_SQUEEZE = 0, 
    MDL_MOBILE = 1, 
    MDL_RESNET = 2, 
    MDL_EFFICIENT = 3,
    MDL_ENSEMBLE = 4 
};

#define IMG_INPUT_DIM 32
#define IMG_CHANNELS 3
#define DIGIT_CLASSES 10
#define ARENA_BYTES (92 * 1024)
#define PREPROCESS_DIM 160

// Model name strings
const char* MODEL_LABELS[] = {"SqueezeNet", "MobileNetV2", "ResNet-8", "EfficientNet", "Ensemble"};

// ====================== Global Variables ======================
httpd_handle_t webServer = NULL;
uint8_t* tensorMemory = nullptr;

const tflite::Model* activeModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* inputLayer = nullptr;
TfLiteTensor* outputLayer = nullptr;

int activeModelIdx = MDL_SQUEEZE;
float predictionProbs[5][DIGIT_CLASSES];
int predictedDigits[5];
float confidenceScores[5];
uint32_t inferenceMs[5];

uint8_t processedPreview[IMG_INPUT_DIM * IMG_INPUT_DIM];
uint8_t* grayBuffer = nullptr;
uint8_t* binaryBuffer = nullptr;
uint8_t* morphBuffer = nullptr;

// ====================== Model Data Accessors ======================
const unsigned char* getModelBytes(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_model;
        case MDL_MOBILE:    return mobilenetv2mini_model;
        case MDL_RESNET:    return resnet8_model;
        case MDL_EFFICIENT: return efficientnetmini_model;
        default: return squeezenetmini_model;
    }
}

unsigned int getModelSize(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_model_len;
        case MDL_MOBILE:    return mobilenetv2mini_model_len;
        case MDL_RESNET:    return resnet8_model_len;
        case MDL_EFFICIENT: return efficientnetmini_model_len;
        default: return squeezenetmini_model_len;
    }
}

float getInScale(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_input_scale;
        case MDL_MOBILE:    return mobilenetv2mini_input_scale;
        case MDL_RESNET:    return resnet8_input_scale;
        case MDL_EFFICIENT: return efficientnetmini_input_scale;
        default: return 0.003921569f;
    }
}

int getInZeroPoint(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_input_zero_point;
        case MDL_MOBILE:    return mobilenetv2mini_input_zero_point;
        case MDL_RESNET:    return resnet8_input_zero_point;
        case MDL_EFFICIENT: return efficientnetmini_input_zero_point;
        default: return 0;
    }
}

float getOutScale(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_output_scale;
        case MDL_MOBILE:    return mobilenetv2mini_output_scale;
        case MDL_RESNET:    return resnet8_output_scale;
        case MDL_EFFICIENT: return efficientnetmini_output_scale;
        default: return 0.00390625f;
    }
}

int getOutZeroPoint(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_output_zero_point;
        case MDL_MOBILE:    return mobilenetv2mini_output_zero_point;
        case MDL_RESNET:    return resnet8_output_zero_point;
        case MDL_EFFICIENT: return efficientnetmini_output_zero_point;
        default: return -128;
    }
}

// ====================== TFLite Initialization ======================
static tflite::MicroMutableOpResolver<20> opResolver;

bool setupTFLite(int modelIdx) {
    Serial.printf("[TFLite] Loading %s model...\n", MODEL_LABELS[modelIdx]);
    
    activeModel = tflite::GetModel(getModelBytes(modelIdx));
    if (activeModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[TFLite] Schema version mismatch!");
        return false;
    }
    
    // Register required operations (once)
    static bool opsReady = false;
    if (!opsReady) {
        opResolver.AddConv2D();
        opResolver.AddDepthwiseConv2D();
        opResolver.AddMaxPool2D();
        opResolver.AddAveragePool2D();
        opResolver.AddReshape();
        opResolver.AddSoftmax();
        opResolver.AddRelu();
        opResolver.AddRelu6();
        opResolver.AddAdd();
        opResolver.AddMul();
        opResolver.AddMean();
        opResolver.AddPad();
        opResolver.AddConcatenation();
        opResolver.AddQuantize();
        opResolver.AddDequantize();
        opResolver.AddLogistic();
        opResolver.AddFullyConnected();
        opsReady = true;
    }
    
    static tflite::MicroErrorReporter errReporter;
    static tflite::MicroInterpreter staticInterp(
        activeModel, opResolver, tensorMemory, ARENA_BYTES, &errReporter);
    tflInterpreter = &staticInterp;
    
    if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFLite] Tensor allocation failed!");
        return false;
    }
    
    inputLayer = tflInterpreter->input(0);
    outputLayer = tflInterpreter->output(0);
    
    Serial.printf("[TFLite] Input shape: [%d,%d,%d,%d]\n", 
        inputLayer->dims->data[0], inputLayer->dims->data[1],
        inputLayer->dims->data[2], inputLayer->dims->data[3]);
    Serial.printf("[TFLite] Arena usage: %d bytes\n", tflInterpreter->arena_used_bytes());
    
    activeModelIdx = modelIdx;
    return true;
}

// ====================== Camera Setup ======================
bool setupCamera() {
    camera_config_t camCfg;
    camCfg.ledc_channel = LEDC_CHANNEL_0;
    camCfg.ledc_timer = LEDC_TIMER_0;
    camCfg.pin_d0 = CAM_PIN_D0;
    camCfg.pin_d1 = CAM_PIN_D1;
    camCfg.pin_d2 = CAM_PIN_D2;
    camCfg.pin_d3 = CAM_PIN_D3;
    camCfg.pin_d4 = CAM_PIN_D4;
    camCfg.pin_d5 = CAM_PIN_D5;
    camCfg.pin_d6 = CAM_PIN_D6;
    camCfg.pin_d7 = CAM_PIN_D7;
    camCfg.pin_xclk = CAM_PIN_XCLK;
    camCfg.pin_pclk = CAM_PIN_PCLK;
    camCfg.pin_vsync = CAM_PIN_VSYNC;
    camCfg.pin_href = CAM_PIN_HREF;
    camCfg.pin_sscb_sda = CAM_PIN_SIOD;
    camCfg.pin_sscb_scl = CAM_PIN_SIOC;
    camCfg.pin_pwdn = CAM_PIN_PWDN;
    camCfg.pin_reset = CAM_PIN_RESET;
    camCfg.xclk_freq_hz = 20000000;
    camCfg.pixel_format = PIXFORMAT_RGB565;
    camCfg.frame_size = FRAMESIZE_QVGA;
    camCfg.jpeg_quality = 10;
    camCfg.fb_count = 2;
    camCfg.fb_location = CAMERA_FB_IN_PSRAM;
    camCfg.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    
    if (esp_camera_init(&camCfg) != ESP_OK) {
        Serial.println("[Camera] Initialization failed!");
        return false;
    }
    
    Serial.println("[Camera] Ready");
    return true;
}

// ====================== Image Preprocessing ======================
// Custom preprocessing: center crop, Otsu threshold, morphology, resize
void preprocessFrame(camera_fb_t* frame, int8_t* modelInput, int modelIdx) {
    uint16_t* rgb565 = (uint16_t*)frame->buf;
    int srcW = frame->width;
    int srcH = frame->height;
    
    float inScale = getInScale(modelIdx);
    int inZP = getInZeroPoint(modelIdx);
    
    // Step 1: Center crop and downsample to 160x160 grayscale
    int cropDim = min(srcW, srcH);
    int offsetX = (srcW - cropDim) / 2;
    int offsetY = (srcH - cropDim) / 2;
    float scaleRatio = (float)cropDim / PREPROCESS_DIM;
    
    uint8_t minPix = 255, maxPix = 0;
    
    for (int y = 0; y < PREPROCESS_DIM; y++) {
        for (int x = 0; x < PREPROCESS_DIM; x++) {
            int sx = offsetX + (int)(x * scaleRatio);
            int sy = offsetY + (int)(y * scaleRatio);
            
            uint16_t pix = rgb565[sy * srcW + sx];
            // Extract RGB565 and convert to grayscale
            uint8_t r = (pix >> 11) & 0x1F;
            uint8_t g = (pix >> 5) & 0x3F;
            uint8_t b = pix & 0x1F;
            uint8_t gray = (r * 77 + g * 150 + b * 29) >> 8;
            
            grayBuffer[y * PREPROCESS_DIM + x] = gray;
            if (gray < minPix) minPix = gray;
            if (gray > maxPix) maxPix = gray;
        }
    }
    
    // Step 2: Contrast stretching
    int range = maxPix - minPix;
    if (range < 10) range = 10;
    
    for (int i = 0; i < PREPROCESS_DIM * PREPROCESS_DIM; i++) {
        int stretched = ((grayBuffer[i] - minPix) * 255) / range;
        grayBuffer[i] = constrain(stretched, 0, 255);
    }
    
    // Step 3: Otsu's automatic thresholding
    int histogram[256] = {0};
    for (int i = 0; i < PREPROCESS_DIM * PREPROCESS_DIM; i++) {
        histogram[grayBuffer[i]]++;
    }
    
    int totalPixels = PREPROCESS_DIM * PREPROCESS_DIM;
    float sumAll = 0;
    for (int i = 0; i < 256; i++) sumAll += i * histogram[i];
    
    float sumBackground = 0;
    int weightBackground = 0;
    float maxVariance = 0;
    int bestThreshold = 128;
    
    for (int t = 0; t < 256; t++) {
        weightBackground += histogram[t];
        if (weightBackground == 0) continue;
        int weightForeground = totalPixels - weightBackground;
        if (weightForeground == 0) break;
        
        sumBackground += t * histogram[t];
        float meanBg = sumBackground / weightBackground;
        float meanFg = (sumAll - sumBackground) / weightForeground;
        float variance = (float)weightBackground * weightForeground * (meanBg - meanFg) * (meanBg - meanFg);
        
        if (variance > maxVariance) {
            maxVariance = variance;
            bestThreshold = t;
        }
    }
    
    // Adjust threshold for darker ink detection
    bestThreshold -= 60;
    if (bestThreshold < 15) bestThreshold = 15;
    
    // Step 4: Create binary image (white digit on black)
    for (int i = 0; i < PREPROCESS_DIM * PREPROCESS_DIM; i++) {
        binaryBuffer[i] = (grayBuffer[i] < bestThreshold) ? 255 : 0;
    }
    
    // Step 5: Dilation (thicken digit strokes)
    memcpy(morphBuffer, binaryBuffer, PREPROCESS_DIM * PREPROCESS_DIM);
    for (int y = 1; y < PREPROCESS_DIM - 1; y++) {
        for (int x = 1; x < PREPROCESS_DIM - 1; x++) {
            int idx = y * PREPROCESS_DIM + x;
            if (binaryBuffer[idx] || binaryBuffer[idx-1] || binaryBuffer[idx+1] ||
                binaryBuffer[idx-PREPROCESS_DIM] || binaryBuffer[idx+PREPROCESS_DIM]) {
                morphBuffer[idx] = 255;
            }
        }
    }
    memcpy(binaryBuffer, morphBuffer, PREPROCESS_DIM * PREPROCESS_DIM);
    
    // Step 6: Find bounding box
    int minX = PREPROCESS_DIM, minY = PREPROCESS_DIM, maxX = -1, maxY = -1;
    int digitPixels = 0;
    
    for (int y = 0; y < PREPROCESS_DIM; y++) {
        for (int x = 0; x < PREPROCESS_DIM; x++) {
            if (binaryBuffer[y * PREPROCESS_DIM + x]) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                digitPixels++;
            }
        }
    }
    
    // Fallback if no digit found
    if (digitPixels < 20) {
        minX = 0; minY = 0; 
        maxX = PREPROCESS_DIM - 1; 
        maxY = PREPROCESS_DIM - 1;
    }
    
    // Add 15% padding
    int padX = (maxX - minX + 1) * 15 / 100;
    int padY = (maxY - minY + 1) * 15 / 100;
    minX = max(0, minX - padX);
    minY = max(0, minY - padY);
    maxX = min(PREPROCESS_DIM - 1, maxX + padX);
    maxY = min(PREPROCESS_DIM - 1, maxY + padY);
    
    int boxW = maxX - minX + 1;
    int boxH = maxY - minY + 1;
    
    // Step 7: Resize bounding box to 32x32 and quantize
    for (int y = 0; y < IMG_INPUT_DIM; y++) {
        for (int x = 0; x < IMG_INPUT_DIM; x++) {
            int sx = minX + (x * boxW) / IMG_INPUT_DIM;
            int sy = minY + (y * boxH) / IMG_INPUT_DIM;
            sx = constrain(sx, 0, PREPROCESS_DIM - 1);
            sy = constrain(sy, 0, PREPROCESS_DIM - 1);
            
            uint8_t pixVal = binaryBuffer[sy * PREPROCESS_DIM + sx];
            processedPreview[y * IMG_INPUT_DIM + x] = pixVal;
            
            // Quantize: normalize to [0,1] then apply scale/zero_point
            float normalized = pixVal / 255.0f;
            int8_t quantized = (int8_t)(normalized / inScale + inZP);
            
            // Replicate grayscale to RGB channels
            modelInput[(y * IMG_INPUT_DIM + x) * 3 + 0] = quantized;
            modelInput[(y * IMG_INPUT_DIM + x) * 3 + 1] = quantized;
            modelInput[(y * IMG_INPUT_DIM + x) * 3 + 2] = quantized;
        }
    }
}

// ====================== Inference ======================
int runPrediction(int modelIdx) {
    // Discard stale frame
    camera_fb_t* oldFrame = esp_camera_fb_get();
    if (oldFrame) esp_camera_fb_return(oldFrame);
    
    // Capture fresh frame
    camera_fb_t* frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("[Inference] Camera capture failed!");
        return -1;
    }
    
    // Switch model if needed
    if (activeModelIdx != modelIdx) {
        if (!setupTFLite(modelIdx)) {
            esp_camera_fb_return(frame);
            return -1;
        }
    }
    
    // Preprocess
    preprocessFrame(frame, inputLayer->data.int8, modelIdx);
    esp_camera_fb_return(frame);
    
    // Run inference
    uint32_t startTime = millis();
    if (tflInterpreter->Invoke() != kTfLiteOk) {
        Serial.println("[Inference] Failed!");
        return -1;
    }
    inferenceMs[modelIdx] = millis() - startTime;
    
    // Decode output
    float outScale = getOutScale(modelIdx);
    int outZP = getOutZeroPoint(modelIdx);
    
    int bestClass = 0;
    float bestScore = -1000;
    
    for (int c = 0; c < DIGIT_CLASSES; c++) {
        float prob = (outputLayer->data.int8[c] - outZP) * outScale;
        predictionProbs[modelIdx][c] = prob;
        if (prob > bestScore) {
            bestScore = prob;
            bestClass = c;
        }
    }
    
    predictedDigits[modelIdx] = bestClass;
    confidenceScores[modelIdx] = bestScore;
    
    Serial.printf("[%s] Predicted: %d (%.1f%%) in %lu ms\n",
        MODEL_LABELS[modelIdx], bestClass, bestScore * 100, inferenceMs[modelIdx]);
    
    return bestClass;
}

// Ensemble: run all models and average probabilities
int runEnsemble() {
    Serial.println("\n[Ensemble] Running all models...");
    
    for (int m = 0; m < 4; m++) {
        runPrediction(m);
    }
    
    // Average probabilities
    float avgProbs[DIGIT_CLASSES] = {0};
    for (int c = 0; c < DIGIT_CLASSES; c++) {
        for (int m = 0; m < 4; m++) {
            avgProbs[c] += predictionProbs[m][c];
        }
        avgProbs[c] /= 4.0f;
    }
    
    // Find best
    int bestClass = 0;
    float bestProb = avgProbs[0];
    for (int c = 1; c < DIGIT_CLASSES; c++) {
        if (avgProbs[c] > bestProb) {
            bestProb = avgProbs[c];
            bestClass = c;
        }
    }
    
    for (int c = 0; c < DIGIT_CLASSES; c++) {
        predictionProbs[MDL_ENSEMBLE][c] = avgProbs[c];
    }
    predictedDigits[MDL_ENSEMBLE] = bestClass;
    confidenceScores[MDL_ENSEMBLE] = bestProb;
    inferenceMs[MDL_ENSEMBLE] = inferenceMs[0] + inferenceMs[1] + inferenceMs[2] + inferenceMs[3];
    
    Serial.printf("[Ensemble] Final: %d (%.1f%%) total %lu ms\n",
        bestClass, bestProb * 100, inferenceMs[MDL_ENSEMBLE]);
    
    return bestClass;
}

// ====================== Web Interface ======================
const char* WEB_PAGE = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Q4: Multi-Model Digit Recognition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #fff; 
            min-height: 100vh; 
            padding: 15px;
        }
        .header { text-align: center; margin-bottom: 25px; }
        .header h1 { 
            color: #00ffa3; 
            font-size: 28px;
            text-shadow: 0 0 20px rgba(0,255,163,0.4);
        }
        .header p { color: #aaa; margin-top: 5px; }
        .main-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            max-width: 1100px; 
            margin: 0 auto; 
        }
        .card {
            background: rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        .card-title { color: #00ffa3; margin-bottom: 15px; font-size: 18px; }
        .preview-row { display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; }
        .preview-box { text-align: center; }
        .preview-box img { 
            width: 140px; height: 140px; 
            border-radius: 12px; 
            border: 2px solid #444; 
            object-fit: cover;
        }
        .preview-box span { display: block; color: #888; font-size: 12px; margin-top: 5px; }
        .btn-row { display: flex; gap: 10px; justify-content: center; margin-top: 15px; flex-wrap: wrap; }
        .btn {
            padding: 10px 18px;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .btn-refresh { background: #555; color: #fff; }
        .btn-flash { background: #ff9500; color: #000; }
        .btn-reset { background: linear-gradient(135deg, #ff4444, #cc0000); color: #fff; }
        .model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }
        .model-btn {
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            color: #fff;
            transition: all 0.3s;
        }
        .model-btn:hover { transform: scale(1.03); }
        .btn-m0 { background: linear-gradient(135deg, #667eea, #764ba2); }
        .btn-m1 { background: linear-gradient(135deg, #11998e, #38ef7d); }
        .btn-m2 { background: linear-gradient(135deg, #ee0979, #ff6a00); }
        .btn-m3 { background: linear-gradient(135deg, #fc4a1a, #f7b733); }
        .btn-ensemble { 
            grid-column: span 2; 
            background: linear-gradient(135deg, #00ffa3, #00d4ff);
            color: #000;
            font-size: 16px;
        }
        .result-display {
            text-align: center;
            margin: 20px 0;
            padding: 25px;
            background: rgba(0,255,163,0.1);
            border-radius: 15px;
        }
        .digit-result { font-size: 80px; font-weight: bold; color: #00ffa3; }
        .confidence { font-size: 22px; color: #00d4ff; margin-top: 10px; }
        .model-label { color: #888; margin-top: 8px; }
        .prob-grid { 
            display: grid; 
            grid-template-columns: repeat(5, 1fr); 
            gap: 6px; 
            margin-top: 20px; 
        }
        .prob-cell { 
            text-align: center; 
            padding: 10px 5px; 
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        .prob-digit { font-size: 20px; }
        .prob-val { font-size: 11px; color: #888; }
        .all-results { margin-top: 15px; }
        .result-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 6px;
        }
        .result-row.highlight { background: rgba(0,255,163,0.15); }
        .status { text-align: center; color: #888; margin-top: 15px; font-size: 14px; }
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
            .model-grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔢 Multi-Model Digit Recognition</h1>
        <p>EE4065 Final Project - Question 4</p>
    </div>
    
    <div class="main-grid">
        <div class="card">
            <div class="card-title">📷 Camera & Controls</div>
            <div class="preview-row">
                <div class="preview-box">
                    <img id="camImg" src="" onclick="refreshCam()">
                    <span>Live Camera</span>
                </div>
                <div class="preview-box">
                    <img id="procImg" src="" style="border-color: #00ffa3;">
                    <span>Processed (32×32)</span>
                </div>
            </div>
            <div class="btn-row">
                <button class="btn btn-refresh" onclick="refreshCam()">🔄 Refresh</button>
                <button class="btn btn-flash" id="flashBtn" onclick="toggleFlash()">💡 Flash</button>
                <button class="btn btn-reset" onclick="resetDevice()">♻️ Reset</button>
            </div>
            
            <div class="card-title" style="margin-top:20px;">🤖 Select Model</div>
            <div class="model-grid">
                <button class="model-btn btn-m0" onclick="predict(0)">SqueezeNet</button>
                <button class="model-btn btn-m1" onclick="predict(1)">MobileNetV2</button>
                <button class="model-btn btn-m2" onclick="predict(2)">ResNet-8</button>
                <button class="model-btn btn-m3" onclick="predict(3)">EfficientNet</button>
                <button class="model-btn btn-ensemble" onclick="predict(4)">🔗 Run Ensemble</button>
            </div>
            
            <div class="result-display">
                <div class="digit-result" id="digitRes">-</div>
                <div class="confidence" id="confRes">Hold a digit in front of camera</div>
                <div class="model-label" id="modelRes"></div>
            </div>
            
            <div class="prob-grid" id="probGrid"></div>
        </div>
        
        <div class="card">
            <div class="card-title">📊 Individual Model Results</div>
            <div class="all-results">
                <div class="result-row"><span>SqueezeNet</span><span id="r0">-</span></div>
                <div class="result-row"><span>MobileNetV2</span><span id="r1">-</span></div>
                <div class="result-row"><span>ResNet-8</span><span id="r2">-</span></div>
                <div class="result-row"><span>EfficientNet</span><span id="r3">-</span></div>
                <div class="result-row highlight"><span><b>Ensemble</b></span><span id="r4">-</span></div>
            </div>
            <div class="status" id="statusMsg">Ready</div>
        </div>
    </div>
    
    <script>
        const modelLabels = ['SqueezeNet', 'MobileNetV2', 'ResNet-8', 'EfficientNet', 'Ensemble'];
        let flashState = false;
        
        // Initialize probability grid
        (function initProbGrid() {
            let html = '';
            for (let i = 0; i < 10; i++) {
                html += '<div class="prob-cell"><div class="prob-digit">'+i+'</div><div class="prob-val" id="p'+i+'">-</div></div>';
            }
            document.getElementById('probGrid').innerHTML = html;
        })();
        
        function refreshCam() {
            document.getElementById('camImg').src = '/snapshot?' + Date.now();
        }
        
        window.onload = refreshCam;
        
        async function toggleFlash() {
            flashState = !flashState;
            await fetch('/flash?on=' + (flashState ? '1' : '0'));
            document.getElementById('flashBtn').style.background = flashState ? '#0f0' : '#ff9500';
            refreshCam();
        }
        
        async function resetDevice() {
            document.getElementById('statusMsg').textContent = 'Resetting...';
            await fetch('/reset');
            setTimeout(refreshCam, 500);
        }
        
        async function predict(modelId) {
            document.getElementById('digitRes').textContent = '...';
            document.getElementById('confRes').textContent = 'Running ' + modelLabels[modelId] + '...';
            document.getElementById('statusMsg').textContent = 'Processing...';
            
            try {
                const resp = await fetch('/predict?model=' + modelId);
                const data = await resp.json();
                
                document.getElementById('digitRes').textContent = data.prediction;
                document.getElementById('confRes').textContent = (data.confidence * 100).toFixed(1) + '% confidence';
                document.getElementById('modelRes').textContent = modelLabels[modelId] + ' - ' + data.time + 'ms';
                
                // Update probability cells
                for (let i = 0; i < 10; i++) {
                    const pct = (data.probs[i] * 100).toFixed(1);
                    document.getElementById('p'+i).textContent = pct + '%';
                    document.getElementById('p'+i).parentElement.style.background = 
                        i == data.prediction ? 'rgba(0,255,163,0.25)' : 'rgba(255,255,255,0.05)';
                }
                
                // Update per-model results
                for (let m = 0; m <= modelId; m++) {
                    if (data.all && data.all[m]) {
                        document.getElementById('r'+m).textContent = 
                            data.all[m].digit + ' (' + (data.all[m].conf * 100).toFixed(0) + '%) ' + data.all[m].time + 'ms';
                    }
                }
                document.getElementById('r'+modelId).textContent = 
                    data.prediction + ' (' + (data.confidence * 100).toFixed(0) + '%) ' + data.time + 'ms';
                
                document.getElementById('procImg').src = '/debug_input?' + Date.now();
                document.getElementById('statusMsg').textContent = 'Done';
            } catch(e) {
                document.getElementById('statusMsg').textContent = 'Error: ' + e.message;
            }
        }
    </script>
</body>
</html>
)rawliteral";

// ====================== HTTP Handlers ======================
esp_err_t handleRoot(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, WEB_PAGE, strlen(WEB_PAGE));
}

esp_err_t handleSnapshot(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // Convert to JPEG for web display
    size_t jpgLen = 0;
    uint8_t *jpgBuf = NULL;
    bool converted = frame2jpg(fb, 80, &jpgBuf, &jpgLen);
    esp_camera_fb_return(fb);
    
    if (!converted) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    esp_err_t res = httpd_resp_send(req, (const char*)jpgBuf, jpgLen);
    free(jpgBuf);
    return res;
}

esp_err_t handleDebugImg(httpd_req_t *req) {
    // Send 32x32 BMP of processed input
    const int w = IMG_INPUT_DIM, h = IMG_INPUT_DIM;
    const int hdrSize = 54 + 256 * 4;
    const int imgSize = w * h;
    const int fileSize = hdrSize + imgSize;
    
    uint8_t* bmp = (uint8_t*)malloc(fileSize);
    if (!bmp) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    memset(bmp, 0, fileSize);
    bmp[0] = 'B'; bmp[1] = 'M';
    *(uint32_t*)(bmp + 2) = fileSize;
    *(uint32_t*)(bmp + 10) = hdrSize;
    *(uint32_t*)(bmp + 14) = 40;
    *(int32_t*)(bmp + 18) = w;
    *(int32_t*)(bmp + 22) = h;
    *(uint16_t*)(bmp + 26) = 1;
    *(uint16_t*)(bmp + 28) = 8;
    *(uint32_t*)(bmp + 34) = imgSize;
    *(uint32_t*)(bmp + 46) = 256;
    
    for (int i = 0; i < 256; i++) {
        bmp[54 + i * 4 + 0] = i;
        bmp[54 + i * 4 + 1] = i;
        bmp[54 + i * 4 + 2] = i;
    }
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bmp[hdrSize + (h - 1 - y) * w + x] = processedPreview[y * w + x];
        }
    }
    
    httpd_resp_set_type(req, "image/bmp");
    esp_err_t res = httpd_resp_send(req, (const char*)bmp, fileSize);
    free(bmp);
    return res;
}

esp_err_t handlePredict(httpd_req_t *req) {
    char query[32];
    int modelId = 0;
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[8];
        if (httpd_query_key_value(query, "model", val, sizeof(val)) == ESP_OK) {
            modelId = atoi(val);
        }
    }
    
    int result;
    if (modelId == MDL_ENSEMBLE) {
        result = runEnsemble();
    } else {
        result = runPrediction(modelId);
    }
    
    // Build JSON response
    char json[1024];
    char probStr[256] = "[";
    for (int i = 0; i < DIGIT_CLASSES; i++) {
        char tmp[16];
        sprintf(tmp, "%.4f%s", predictionProbs[modelId][i], i < 9 ? "," : "");
        strcat(probStr, tmp);
    }
    strcat(probStr, "]");
    
    sprintf(json, "{\"prediction\":%d,\"confidence\":%.4f,\"time\":%lu,\"probs\":%s}",
        predictedDigits[modelId], confidenceScores[modelId], inferenceMs[modelId], probStr);
    
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, json, strlen(json));
}

esp_err_t handleFlash(httpd_req_t *req) {
    char query[16];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[4];
        if (httpd_query_key_value(query, "on", val, sizeof(val)) == ESP_OK) {
            digitalWrite(LED_FLASH, atoi(val) ? HIGH : LOW);
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

esp_err_t handleReset(httpd_req_t *req) {
    // Clear cached predictions
    for (int m = 0; m < 5; m++) {
        predictedDigits[m] = -1;
        confidenceScores[m] = 0;
        for (int c = 0; c < DIGIT_CLASSES; c++) {
            predictionProbs[m][c] = 0;
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

void startWebServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 10;
    
    if (httpd_start(&webServer, &config) == ESP_OK) {
        httpd_uri_t uriRoot = { "/", HTTP_GET, handleRoot, NULL };
        httpd_uri_t uriSnap = { "/snapshot", HTTP_GET, handleSnapshot, NULL };
        httpd_uri_t uriDebug = { "/debug_input", HTTP_GET, handleDebugImg, NULL };
        httpd_uri_t uriPredict = { "/predict", HTTP_GET, handlePredict, NULL };
        httpd_uri_t uriFlash = { "/flash", HTTP_GET, handleFlash, NULL };
        httpd_uri_t uriReset = { "/reset", HTTP_GET, handleReset, NULL };
        
        httpd_register_uri_handler(webServer, &uriRoot);
        httpd_register_uri_handler(webServer, &uriSnap);
        httpd_register_uri_handler(webServer, &uriDebug);
        httpd_register_uri_handler(webServer, &uriPredict);
        httpd_register_uri_handler(webServer, &uriFlash);
        httpd_register_uri_handler(webServer, &uriReset);
        
        Serial.println("[Web] Server started");
    }
}

// ====================== Main Setup & Loop ======================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);  // Disable brownout
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("  EE4065 Q4: Multi-Model Digit Recognition");
    Serial.println("  Author: Yusuf");
    Serial.println("========================================\n");
    
    // Initialize LED
    pinMode(LED_FLASH, OUTPUT);
    digitalWrite(LED_FLASH, LOW);
    
    // Allocate PSRAM buffers
    if (psramFound()) {
        tensorMemory = (uint8_t*)ps_malloc(ARENA_BYTES);
        grayBuffer = (uint8_t*)ps_malloc(PREPROCESS_DIM * PREPROCESS_DIM);
        binaryBuffer = (uint8_t*)ps_malloc(PREPROCESS_DIM * PREPROCESS_DIM);
        morphBuffer = (uint8_t*)ps_malloc(PREPROCESS_DIM * PREPROCESS_DIM);
        Serial.printf("[Memory] PSRAM allocated: %d KB tensor + %d KB buffers\n", 
            ARENA_BYTES/1024, (PREPROCESS_DIM*PREPROCESS_DIM*3)/1024);
    } else {
        Serial.println("[Memory] ERROR: PSRAM not found!");
        while(1) delay(1000);
    }
    
    // Initialize camera
    if (!setupCamera()) {
        Serial.println("[Camera] FAILED - halting");
        while(1) delay(1000);
    }
    
    // Load initial model
    if (!setupTFLite(MDL_SQUEEZE)) {
        Serial.println("[TFLite] FAILED - halting");
        while(1) delay(1000);
    }
    
    // Connect to WiFi
    Serial.print("[WiFi] Connecting to ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    int wifiAttempts = 0;
    while (WiFi.status() != WL_CONNECTED && wifiAttempts++ < 30) {
        delay(500);
        Serial.print(".");
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected!");
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\n[WiFi] Failed - starting AP mode");
        WiFi.softAP(AP_SSID, AP_PASS);
        Serial.print("[WiFi] AP IP: ");
        Serial.println(WiFi.softAPIP());
    }
    
    // Start web server
    startWebServer();
    
    Serial.println("\n[System] Ready! Open browser to access web interface.");
}

void loop() {
    delay(10);
}
