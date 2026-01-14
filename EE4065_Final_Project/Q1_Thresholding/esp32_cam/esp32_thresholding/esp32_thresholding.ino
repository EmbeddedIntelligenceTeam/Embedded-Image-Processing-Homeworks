/*
 * EE4065 - Final Project - Question 1b
 * Thresholding with Web Interface
 * 
 * Features:
 * - Capture image (QVGA)
 * - Adaptive thresholding to extract exactly 1000 pixels (approx)
 * - Web Interface to view Original vs Thresholded images
 * - Stats display
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ==================== WIFI CREDENTIALS ====================
const char* ssid = "Yusuf's Xiaomi";
const char* password = "yusuf4418";

WebServer server(80);

// ==================== CAMERA PINS ====================
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

// ==================== PARAMS ====================
#define TARGET_OBJECT_PIXELS 1000
#define IMG_W 320
#define IMG_H 240
#define HISTOGRAM_SIZE 256

// ==================== GLOBALS ====================
uint8_t* original_img = nullptr;
uint8_t* thresholded_img = nullptr;
size_t img_size = 0;

// Stats
uint8_t current_threshold = 0;
uint32_t extracted_count = 0;
uint32_t histogram[HISTOGRAM_SIZE];

// ==================== ALGORITHMS ====================

void computeHistogram(uint8_t* data, size_t len) {
    memset(histogram, 0, sizeof(histogram));
    for(size_t i=0; i<len; i++) {
        histogram[data[i]]++;
    }
}

uint8_t findThreshold(uint8_t* data, size_t len, uint32_t target) {
    computeHistogram(data, len);
    
    uint32_t sum = 0;
    for(int i=255; i>=0; i--) {
        sum += histogram[i];
        if(sum >= target) {
            return (uint8_t)i;
        }
    }
    return 0;
}

void applyThreshold(uint8_t* src, uint8_t* dst, size_t len, uint8_t th) {
    extracted_count = 0;
    for(size_t i=0; i<len; i++) {
        if(src[i] >= th) {
            dst[i] = 255; // White
            extracted_count++;
        } else {
            dst[i] = 0;   // Black
        }
    }
}

// ==================== CAMERA ====================
bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM; config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM; config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_QVGA; config.fb_count = 1;
    
    return esp_camera_init(&config) == ESP_OK;
}

void captureAndProcess() {
    // Warmup
    for(int i=0; i<3; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if(fb) esp_camera_fb_return(fb);
        delay(50);
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if(!fb) { Serial.println("Capture failed"); return; }
    
    img_size = fb->width * fb->height;
    
    // Allocate buffers if needed
    if(!original_img) original_img = (uint8_t*)malloc(img_size);
    if(!thresholded_img) thresholded_img = (uint8_t*)malloc(img_size);
    
    if(original_img && thresholded_img) {
        // Copy original
        memcpy(original_img, fb->buf, img_size);
        
        // Process
        current_threshold = findThreshold(original_img, img_size, TARGET_OBJECT_PIXELS);
        applyThreshold(original_img, thresholded_img, img_size, current_threshold);
        
        Serial.printf("Processed: Th=%d, Count=%d\n", current_threshold, extracted_count);
    }
    
    esp_camera_fb_return(fb);
}

// ==================== WEB HANDLERS ====================

// Send BMP helper
void sendBMP(uint8_t* data, int w, int h) {
    if (!data) { server.send(404, "text/plain", "No image"); return; }
    
    int rowSize = ((w + 3) / 4) * 4;
    int paletteSize = 256 * 4;
    int pixelDataSize = rowSize * h;
    int headerSize = 54;
    int fileSize = headerSize + paletteSize + pixelDataSize;
    int dataOffset = headerSize + paletteSize;

    uint8_t header[54];
    memset(header, 0, 54);
    
    header[0] = 'B'; header[1] = 'M';
    header[2] = fileSize & 0xFF; header[3] = (fileSize >> 8) & 0xFF;
    header[4] = (fileSize >> 16) & 0xFF; header[5] = (fileSize >> 24) & 0xFF;
    header[10] = dataOffset & 0xFF; header[11] = (dataOffset >> 8) & 0xFF;
    
    header[14] = 40;
    header[18] = w & 0xFF; header[19] = (w >> 8) & 0xFF;
    header[22] = h & 0xFF; header[23] = (h >> 8) & 0xFF;
    header[26] = 1; header[28] = 8;
    header[34] = pixelDataSize & 0xFF; header[35] = (pixelDataSize >> 8) & 0xFF;
    header[46] = 0; header[47] = 1; // 256 colors
    
    server.setContentLength(fileSize);
    server.send(200, "image/bmp", "");
    server.client().write(header, 54);
    
    // Palette (Grayscale)
    for (int i = 0; i < 256; i++) {
        uint8_t p[4] = {(uint8_t)i, (uint8_t)i, (uint8_t)i, 0};
        server.client().write(p, 4);
    }
    
    // Pixel Data (Bottom-up)
    uint8_t* row = (uint8_t*)malloc(rowSize);
    if(row) {
        for(int y = h-1; y>=0; y--) {
            memset(row, 0, rowSize);
            memcpy(row, data + y*w, w);
            server.client().write(row, rowSize);
        }
        free(row);
    }
}

void handleRoot() {
    String html = "<!DOCTYPE html><html><head>";
    html += "<title>Q1: Thresholding</title>";
    html += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<style>";
    html += "body{font-family:Arial;background:#1a1a1a;color:#fff;text-align:center;margin:0;padding:20px}";
    html += "h1{background:-webkit-linear-gradient(45deg,#00f260,#0575e6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:30px}";
    html += ".container{display:flex;flex-wrap:wrap;justify-content:center;gap:20px}";
    html += ".card{background:rgba(255,255,255,0.05);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);padding:15px;border-radius:15px;max-width:340px}";
    html += "img{width:100%;border-radius:8px;border:1px solid #333}";
    html += ".btn{background:linear-gradient(45deg,#00f260,#0575e6);border:none;color:white;padding:15px 40px;border-radius:25px;font-size:18px;cursor:pointer;margin:30px;transition:transform 0.2s}";
    html += ".btn:active{transform:scale(0.95)}";
    html += "table{width:100%;margin-top:10px;border-collapse:collapse}";
    html += "td,th{padding:8px;border-bottom:1px solid #333;text-align:left}th{color:#aaa}";
    html += ".val{text-align:right;font-family:monospace;font-size:1.1em;color:#0f0}";
    html += "</style></head><body>";
    
    html += "<h1>Q1: Adaptive Thresholding</h1>";
    
    html += "<button class='btn' onclick=\"location.href='/capture'\">Capture & Process</button>";
    
    html += "<div class='container'>";
    
    // Original Card
    html += "<div class='card'><h3>Original</h3>";
    html += "<img src='/original.bmp?t=" + String(millis()) + "'></div>";
    
    // Thresholded Card
    html += "<div class='card'><h3>Thresholded</h3>";
    html += "<img src='/thresholded.bmp?t=" + String(millis()) + "'>";
    
    // Stats
    html += "<table>";
    html += "<tr><th>Target Pixels</th><td class='val'>" + String(TARGET_OBJECT_PIXELS) + "</td></tr>";
    html += "<tr><th>Calculated Threshold</th><td class='val'>" + String(current_threshold) + "</td></tr>";
    html += "<tr><th>Extracted Pixels</th><td class='val'>" + String(extracted_count) + "</td></tr>";
    float acc = 0;
    if(TARGET_OBJECT_PIXELS > 0) {
        acc = (1.0 - abs((float)extracted_count - TARGET_OBJECT_PIXELS)/TARGET_OBJECT_PIXELS) * 100.0;
    }
    html += "<tr><th>Accuracy</th><td class='val'>" + String(acc, 1) + "%</td></tr>";
    html += "</table></div>";
    
    html += "</div></body></html>";
    server.send(200, "text/html", html);
}

void handleCapture() {
    captureAndProcess();
    server.sendHeader("Location", "/");
    server.send(302, "text/plain", "");
}

void handleOriginal() {
    sendBMP(original_img, IMG_W, IMG_H);
}

void handleThresholded() {
    sendBMP(thresholded_img, IMG_W, IMG_H);
}

void setup() {
    Serial.begin(115200);
    
    // PSRAM check (Thresholding needs buffers)
    if(psramFound()){
        Serial.println("PSRAM found");
    } else {
        Serial.println("No PSRAM - Buffers might fail if image large");
    }
    
    if(!initCamera()) {
        Serial.println("Camera Init Failed");
        while(1) delay(100);
    }
    
    WiFi.begin(ssid, password);
    while(WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.print("IP: "); Serial.println(WiFi.localIP());
    
    server.on("/", handleRoot);
    server.on("/capture", handleCapture);
    server.on("/original.bmp", handleOriginal);
    server.on("/thresholded.bmp", handleThresholded);
    
    server.begin();
    
    // Initial blank/black images
    original_img = (uint8_t*)calloc(IMG_W*IMG_H, 1);
    thresholded_img = (uint8_t*)calloc(IMG_W*IMG_H, 1);
}

void loop() {
    server.handleClient();
}
