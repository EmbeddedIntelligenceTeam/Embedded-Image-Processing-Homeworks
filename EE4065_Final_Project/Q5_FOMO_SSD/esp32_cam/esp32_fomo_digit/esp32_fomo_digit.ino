/*
 * EE4065 - Final Project - Question 5a
 * FOMO Digit Detection on ESP32-CAM
 * SIMPLIFIED DEBUG VERSION
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "model_data.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// WiFi
const char* ssid = "Yusuf's Xiaomi";
const char* password = "yusuf4418";

// Model
#define INPUT_SIZE 96
#define GRID_SIZE 12
#define NUM_CLASSES 11
#define THRESHOLD 0.4

// AI-Thinker ESP32-CAM pins
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
#define FLASH_GPIO_NUM     4

WebServer server(80);

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::MicroErrorReporter error_reporter;

constexpr int kArenaSize = 150 * 1024;
uint8_t* arena = nullptr;

struct Det { int digit, x, y; float conf; };
Det dets[10];
int numDets = 0;
unsigned long infTime = 0;

// Debug counters
uint32_t frameNum = 0;
uint32_t lastPixelSum = 0;

bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;  // Single buffer for simplicity
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    
    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 1);
    s->set_contrast(s, 2);
    
    Serial.println("Camera OK");
    return true;
}

bool initModel() {
    arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) arena = (uint8_t*)malloc(kArenaSize);
    if (!arena) { Serial.println("Arena failed!"); return false; }
    
    model = tflite::GetModel(model_data);
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter interp(model, resolver, arena, kArenaSize, &error_reporter);
    interpreter = &interp;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Allocate failed!");
        return false;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.printf("Model OK. Input type: %d\n", input->type);
    return true;
}

void doInference(uint8_t* img) {
    unsigned long t0 = millis();
    
    // Calculate average brightness for adaptive threshold
    uint32_t sum = 0;
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        sum += img[i];
    }
    uint8_t avg = sum / (INPUT_SIZE * INPUT_SIZE);
    uint8_t threshold = avg - 30;  // Pixels darker than avg-30 are "ink"
    
    Serial.printf("Avg brightness: %d, threshold: %d\n", avg, threshold);
    
    // Apply threshold + invert to make MNIST-like
    // MNIST: white digits (255) on black background (0)
    // Camera: dark ink on bright paper
    // After processing: bright paper -> 0 (black), dark ink -> 255 (white)
    
    if (input->type == kTfLiteUInt8) {
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
            // If pixel is darker than threshold -> it's ink -> make it white (255)
            // Otherwise -> it's paper -> make it black (0)
            input->data.uint8[i] = (img[i] < threshold) ? 255 : 0;
        }
    } else {
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
            input->data.f[i] = (img[i] < threshold) ? 1.0f : 0.0f;
        }
    }
    
    interpreter->Invoke();
    infTime = millis() - t0;
    
    // Find detections
    numDets = 0;
    float scale = (output->type == kTfLiteUInt8) ? output->params.scale : 1.0f;
    int zp = (output->type == kTfLiteUInt8) ? output->params.zero_point : 0;
    
    for (int gy = 0; gy < GRID_SIZE; gy++) {
        for (int gx = 0; gx < GRID_SIZE; gx++) {
            int bestC = 0;
            float bestConf = 0;
            
            for (int c = 0; c < NUM_CLASSES; c++) {
                int idx = (gy * GRID_SIZE + gx) * NUM_CLASSES + c;
                float conf;
                if (output->type == kTfLiteUInt8) {
                    conf = (output->data.uint8[idx] - zp) * scale;
                } else {
                    conf = output->data.f[idx];
                }
                if (conf > bestConf) {
                    bestConf = conf;
                    bestC = c;
                }
            }
            
            if (bestC > 0 && bestConf > THRESHOLD && numDets < 10) {
                dets[numDets].digit = bestC - 1;
                dets[numDets].x = gx * 8 + 4;
                dets[numDets].y = gy * 8 + 4;
                dets[numDets].conf = bestConf;
                numDets++;
            }
        }
    }
    
    Serial.printf("Inference: %dms, Detections: %d\n", infTime, numDets);
}

// Simple HTML
const char* html = R"HTML(
<!DOCTYPE html><html><head>
<title>FOMO Debug</title>
<style>body{font-family:sans-serif;background:#222;color:#fff;padding:20px}
img{border:2px solid #0ff;margin:10px 0}
button{padding:15px 30px;font-size:18px;background:#0f0;border:none;cursor:pointer}
pre{background:#333;padding:10px;margin:10px 0}</style>
</head><body>
<h1>FOMO Digit Detection - Debug</h1>
<img id="cam" width="288" height="288">
<br><button onclick="run()">DETECT</button>
<pre id="out">Click DETECT to run inference</pre>
<script>
var n=0;
function run(){
  n++;
  document.getElementById("cam").src="/img?"+n;
  fetch("/run").then(r=>r.text()).then(t=>{
    document.getElementById("out").innerText=t;
  });
}
setInterval(function(){
  n++;
  document.getElementById("cam").src="/img?"+n;
},1000);
</script>
</body></html>
)HTML";

void handleRoot() {
    server.send(200, "text/html", html);
}

void handleImg() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "No frame");
        return;
    }
    
    frameNum++;
    Serial.printf("Frame %d, len=%d, first4bytes: %02X %02X %02X %02X\n", 
        frameNum, fb->len, fb->buf[0], fb->buf[1], fb->buf[2], fb->buf[3]);
    
    // Convert to simple BMP
    const int w=96, h=96;
    const int hdrSize = 54+256*4;
    const int imgSize = w*h;
    const int fileSize = hdrSize + imgSize;
    
    uint8_t* bmp = (uint8_t*)malloc(fileSize);
    if (!bmp) {
        esp_camera_fb_return(fb);
        server.send(500, "text/plain", "malloc fail");
        return;
    }
    
    memset(bmp, 0, fileSize);
    bmp[0]='B'; bmp[1]='M';
    *(uint32_t*)(bmp+2) = fileSize;
    *(uint32_t*)(bmp+10) = hdrSize;
    *(uint32_t*)(bmp+14) = 40;
    *(int32_t*)(bmp+18) = w;
    *(int32_t*)(bmp+22) = h;
    *(uint16_t*)(bmp+26) = 1;
    *(uint16_t*)(bmp+28) = 8;
    *(uint32_t*)(bmp+34) = imgSize;
    *(uint32_t*)(bmp+46) = 256;
    
    for (int i=0;i<256;i++) {
        bmp[54+i*4+0]=i; bmp[54+i*4+1]=i; bmp[54+i*4+2]=i;
    }
    
    for (int y=0;y<h;y++) {
        for (int x=0;x<w;x++) {
            bmp[hdrSize + (h-1-y)*w + x] = fb->buf[y*w+x];
        }
    }
    
    esp_camera_fb_return(fb);
    
    server.send_P(200, "image/bmp", (const char*)bmp, fileSize);
    free(bmp);
}

void handleRun() {
    // Get fresh frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(200, "text/plain", "ERROR: No camera frame!");
        return;
    }
    
    Serial.printf("\n=== RUN INFERENCE ===\n");
    Serial.printf("Frame size: %d bytes\n", fb->len);
    Serial.printf("First pixels: %d %d %d %d\n", fb->buf[0], fb->buf[1], fb->buf[2], fb->buf[3]);
    Serial.printf("Center pixel: %d\n", fb->buf[48*96+48]);
    
    doInference(fb->buf);
    
    esp_camera_fb_return(fb);
    
    // Build result text
    String result = "Frame: " + String(frameNum) + "\n";
    result += "Inference time: " + String(infTime) + " ms\n";
    result += "Detections: " + String(numDets) + "\n\n";
    
    if (numDets == 0) {
        result += "No digits detected\n";
    } else {
        for (int i = 0; i < numDets; i++) {
            result += "Digit " + String(dets[i].digit);
            result += " at (" + String(dets[i].x) + "," + String(dets[i].y) + ")";
            result += " conf=" + String(dets[i].conf * 100, 1) + "%\n";
        }
    }
    
    server.send(200, "text/plain", result);
}

void setup() {
    Serial.begin(115200);
    Serial.println("\n\n=== FOMO DEBUG ===");
    
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    
    if (!initCamera()) while(1) delay(1000);
    if (!initModel()) while(1) delay(1000);
    
    WiFi.begin(ssid, password);
    Serial.print("WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nIP: " + WiFi.localIP().toString());
    
    server.on("/", handleRoot);
    server.on("/img", handleImg);
    server.on("/run", handleRun);
    server.begin();
    
    Serial.println("Ready!");
}

void loop() {
    server.handleClient();
}
