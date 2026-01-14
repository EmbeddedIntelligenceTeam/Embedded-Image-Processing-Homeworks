/*
 * EE4065 - Final Project - Question 3
 * Upsampling and Downsampling - COLOR Version
 * 
 * Features:
 * - RGB color camera support
 * - Bilinear interpolation for upsampling
 * - Area averaging for downsampling
 * - Non-integer scale factors (1.5x, 2/3, etc.)
 * 
 * Board: AI Thinker ESP32-CAM
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <Arduino.h>

// ==================== WiFi ====================
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

// ==================== GLOBAL BUFFERS (RGB888) ====================
uint8_t* original_img = nullptr;     // RGB888 format: 3 bytes per pixel
uint8_t* upsampled_img = nullptr;
uint8_t* downsampled_img = nullptr;
int orig_w = 0, orig_h = 0;
int up_w = 0, up_h = 0;
int down_w = 0, down_h = 0;

float up_scale = 1.5f;
float down_scale = 0.667f;  // 2/3

// ==================== RGB565 to RGB888 ====================
void rgb565_to_rgb888(uint16_t rgb565, uint8_t* r, uint8_t* g, uint8_t* b) {
    *r = ((rgb565 >> 11) & 0x1F) << 3;
    *g = ((rgb565 >> 5) & 0x3F) << 2;
    *b = (rgb565 & 0x1F) << 3;
}

// ==================== BILINEAR INTERPOLATION (RGB) ====================
void bilinearInterpolateRGB(uint8_t* image, int width, int height, 
                            float src_x, float src_y, uint8_t* r, uint8_t* g, uint8_t* b) {
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    x0 = max(0, min(x0, width - 1));
    y0 = max(0, min(y0, height - 1));
    
    float fx = src_x - x0;
    float fy = src_y - y0;
    
    // Get 4 neighboring pixels (RGB888 = 3 bytes per pixel)
    uint8_t* p00 = &image[(y0 * width + x0) * 3];
    uint8_t* p10 = &image[(y0 * width + x1) * 3];
    uint8_t* p01 = &image[(y1 * width + x0) * 3];
    uint8_t* p11 = &image[(y1 * width + x1) * 3];
    
    // Interpolate each channel
    *r = (uint8_t)constrain((1-fx)*(1-fy)*p00[0] + fx*(1-fy)*p10[0] + (1-fx)*fy*p01[0] + fx*fy*p11[0], 0, 255);
    *g = (uint8_t)constrain((1-fx)*(1-fy)*p00[1] + fx*(1-fy)*p10[1] + (1-fx)*fy*p01[1] + fx*fy*p11[1], 0, 255);
    *b = (uint8_t)constrain((1-fx)*(1-fy)*p00[2] + fx*(1-fy)*p10[2] + (1-fx)*fy*p01[2] + fx*fy*p11[2], 0, 255);
}

// ==================== UPSAMPLE (RGB) ====================
void upsampleRGB(uint8_t* src, int src_w, int src_h, uint8_t* dst, float scale, int* dst_w, int* dst_h) {
    *dst_w = (int)(src_w * scale);
    *dst_h = (int)(src_h * scale);
    
    for (int y = 0; y < *dst_h; y++) {
        for (int x = 0; x < *dst_w; x++) {
            float src_x = x / scale;
            float src_y = y / scale;
            
            uint8_t r, g, b;
            bilinearInterpolateRGB(src, src_w, src_h, src_x, src_y, &r, &g, &b);
            
            int idx = (y * (*dst_w) + x) * 3;
            dst[idx] = r;
            dst[idx + 1] = g;
            dst[idx + 2] = b;
        }
    }
    Serial.printf("Upsampled: %dx%d -> %dx%d (%.2fx)\n", src_w, src_h, *dst_w, *dst_h, scale);
}

// ==================== DOWNSAMPLE (RGB) ====================
void downsampleRGB(uint8_t* src, int src_w, int src_h, uint8_t* dst, float scale, int* dst_w, int* dst_h) {
    float actual_scale = (scale > 1.0f) ? (1.0f / scale) : scale;
    
    *dst_w = max(1, (int)(src_w * actual_scale));
    *dst_h = max(1, (int)(src_h * actual_scale));
    
    float src_x_step = (float)src_w / *dst_w;
    float src_y_step = (float)src_h / *dst_h;
    
    for (int y = 0; y < *dst_h; y++) {
        for (int x = 0; x < *dst_w; x++) {
            int y0 = (int)(y * src_y_step);
            int y1 = min((int)ceil((y + 1) * src_y_step), src_h);
            int x0 = (int)(x * src_x_step);
            int x1 = min((int)ceil((x + 1) * src_x_step), src_w);
            
            float sumR = 0, sumG = 0, sumB = 0;
            int count = 0;
            
            for (int sy = y0; sy < y1; sy++) {
                for (int sx = x0; sx < x1; sx++) {
                    int idx = (sy * src_w + sx) * 3;
                    sumR += src[idx];
                    sumG += src[idx + 1];
                    sumB += src[idx + 2];
                    count++;
                }
            }
            
            int dst_idx = (y * (*dst_w) + x) * 3;
            dst[dst_idx] = (uint8_t)(sumR / count);
            dst[dst_idx + 1] = (uint8_t)(sumG / count);
            dst[dst_idx + 2] = (uint8_t)(sumB / count);
        }
    }
    Serial.printf("Downsampled: %dx%d -> %dx%d (%.3fx)\n", src_w, src_h, *dst_w, *dst_h, actual_scale);
}

// ==================== CAPTURE AND SCALE ====================
void captureAndScale() {
    // Warm up camera
    for (int i = 0; i < 5; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(50);
    }
    
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { Serial.println("Capture failed!"); return; }
    
    orig_w = fb->width;
    orig_h = fb->height;
    
    Serial.printf("Captured: %dx%d, format=%d\n", orig_w, orig_h, fb->format);
    
    // Free old buffers
    if (original_img) free(original_img);
    if (upsampled_img) free(upsampled_img);
    if (downsampled_img) free(downsampled_img);
    
    // Convert RGB565 to RGB888
    original_img = (uint8_t*)ps_malloc(orig_w * orig_h * 3);
    if (!original_img) { 
        Serial.println("Failed to allocate original buffer!"); 
        esp_camera_fb_return(fb);
        return;
    }
    
    uint16_t* src565 = (uint16_t*)fb->buf;
    for (int i = 0; i < orig_w * orig_h; i++) {
        // RGB565 is big-endian in camera buffer, need to swap
        uint16_t pixel = src565[i];
        pixel = (pixel >> 8) | (pixel << 8);  // Swap bytes
        uint8_t r, g, b;
        rgb565_to_rgb888(pixel, &r, &g, &b);
        original_img[i * 3] = r;
        original_img[i * 3 + 1] = g;
        original_img[i * 3 + 2] = b;
    }
    
    esp_camera_fb_return(fb);
    
    // Upsample
    int max_up = (int)(orig_w * up_scale) * (int)(orig_h * up_scale) * 3;
    upsampled_img = (uint8_t*)ps_malloc(max_up);
    if (upsampled_img) {
        upsampleRGB(original_img, orig_w, orig_h, upsampled_img, up_scale, &up_w, &up_h);
    } else {
        Serial.println("Failed to allocate upsample buffer!");
    }
    
    // Downsample
    int max_down = (int)(orig_w * down_scale + 1) * (int)(orig_h * down_scale + 1) * 3;
    downsampled_img = (uint8_t*)malloc(max_down);
    if (downsampled_img) {
        downsampleRGB(original_img, orig_w, orig_h, downsampled_img, down_scale, &down_w, &down_h);
    } else {
        Serial.println("Failed to allocate downsample buffer!");
    }
    
    Serial.println("Capture and scaling complete!");
}

// ==================== WEB HANDLERS ====================
void handleRoot() {
    String html = "<!DOCTYPE html><html><head>";
    html += "<title>Upsampling & Downsampling</title>";
    html += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<style>";
    html += "body{font-family:Arial;background:#222;color:#fff;text-align:center;padding:20px}";
    html += "h1{color:#0cf}";
    html += ".row{display:flex;justify-content:center;flex-wrap:wrap;gap:20px;margin:20px 0}";
    html += ".box{background:#333;padding:15px;border-radius:8px}";
    html += ".box h3{color:#0f0;margin:0 0 10px 0}";
    html += ".box p{color:#888;margin:5px 0}";
    html += "img{border:2px solid #0cf;border-radius:4px;max-width:100%}";
    html += "button{background:#0cf;color:#000;border:none;padding:15px 40px;font-size:18px;";
    html += "border-radius:8px;cursor:pointer;margin:15px}";
    html += "button:hover{background:#0ad}";
    html += "</style></head><body>";
    
    html += "<h1>Upsampling & Downsampling (Color)</h1>";
    html += "<p>EE4065 Question 3</p>";
    
    html += "<button onclick=\"location.href='/capture'\">Capture & Scale</button>";
    
    html += "<div class='row'>";
    html += "<div class='box'><h3>Original</h3>";
    html += "<p>" + String(orig_w) + " x " + String(orig_h) + "</p>";
    html += "<img src='/original.bmp'></div>";
    
    html += "<div class='box'><h3>Upsampled (" + String(up_scale) + "x)</h3>";
    html += "<p>" + String(up_w) + " x " + String(up_h) + "</p>";
    html += "<img src='/upsampled.bmp'></div>";
    
    html += "<div class='box'><h3>Downsampled (" + String(down_scale, 2) + "x)</h3>";
    html += "<p>" + String(down_w) + " x " + String(down_h) + "</p>";
    html += "<img src='/downsampled.bmp'></div>";
    
    html += "</div></body></html>";
    server.send(200, "text/html", html);
}

void handleCapture() {
    captureAndScale();
    server.sendHeader("Location", "/");
    server.send(302, "text/plain", "");
}

// ==================== 24-bit RGB BMP SENDER ====================
void sendBMP_RGB(uint8_t* data, int w, int h) {
    if (!data || w == 0 || h == 0) { 
        server.send(404, "text/plain", "No image"); 
        return; 
    }
    
    // 24-bit BMP (no palette)
    int rowSize = ((w * 3 + 3) / 4) * 4;  // Row must be 4-byte aligned
    int pixelDataSize = rowSize * h;
    int headerSize = 14 + 40;
    int fileSize = headerSize + pixelDataSize;

    uint8_t header[54];
    memset(header, 0, 54);
    
    // BMP File Header
    header[0] = 'B'; header[1] = 'M';
    header[2] = fileSize & 0xFF;
    header[3] = (fileSize >> 8) & 0xFF;
    header[4] = (fileSize >> 16) & 0xFF;
    header[5] = (fileSize >> 24) & 0xFF;
    header[10] = 54;  // Data offset (no palette for 24-bit)
    
    // DIB Header
    header[14] = 40;
    header[18] = w & 0xFF;
    header[19] = (w >> 8) & 0xFF;
    header[22] = h & 0xFF;
    header[23] = (h >> 8) & 0xFF;
    header[26] = 1;   // Color planes
    header[28] = 24;  // Bits per pixel
    header[34] = pixelDataSize & 0xFF;
    header[35] = (pixelDataSize >> 8) & 0xFF;
    header[36] = (pixelDataSize >> 16) & 0xFF;
    header[37] = (pixelDataSize >> 24) & 0xFF;
    
    server.setContentLength(fileSize);
    server.send(200, "image/bmp", "");
    server.client().write(header, 54);
    
    // Send pixel data (bottom-up, BGR format)
    uint8_t* row = (uint8_t*)malloc(rowSize);
    if (row) {
        for (int y = h - 1; y >= 0; y--) {
            memset(row, 0, rowSize);
            for (int x = 0; x < w; x++) {
                int src_idx = (y * w + x) * 3;
                // Convert RGB to BGR (BMP format)
                row[x * 3] = data[src_idx + 2];      // B
                row[x * 3 + 1] = data[src_idx + 1];  // G
                row[x * 3 + 2] = data[src_idx];      // R
            }
            server.client().write(row, rowSize);
        }
        free(row);
    }
}

void handleOriginal() { sendBMP_RGB(original_img, orig_w, orig_h); }
void handleUpsampled() { sendBMP_RGB(upsampled_img, up_w, up_h); }
void handleDownsampled() { sendBMP_RGB(downsampled_img, down_w, down_h); }

// ==================== CAMERA INIT (RGB565) ====================
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
    config.pixel_format = PIXFORMAT_RGB565;  // COLOR!
    config.frame_size = FRAMESIZE_QVGA;      // 320x240
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    
    return esp_camera_init(&config) == ESP_OK;
}

void setup() {
    Serial.begin(115200); delay(1000);
    Serial.println("\n=== EE4065 Q3: Upsampling & Downsampling (COLOR) ===\n");
    
    // Check PSRAM
    if (!psramFound()) {
        Serial.println("PSRAM not found! Color requires PSRAM.");
        while(1) delay(1000);
    }
    Serial.printf("PSRAM: %d bytes free\n", ESP.getFreePsram());
    
    if (!initCamera()) { Serial.println("Camera failed!"); while(1) delay(1000); }
    Serial.println("Camera OK (RGB565)!");
    
    WiFi.begin(ssid, password);
    Serial.print("WiFi");
    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print("."); }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nConnected: " + WiFi.localIP().toString());
    } else {
        WiFi.softAP("ESP32-Scaling", "12345678");
        Serial.println("\nAP Mode: " + WiFi.softAPIP().toString());
    }
    
    server.on("/", handleRoot);
    server.on("/capture", handleCapture);
    server.on("/original.bmp", handleOriginal);
    server.on("/upsampled.bmp", handleUpsampled);
    server.on("/downsampled.bmp", handleDownsampled);
    server.begin();
    
    captureAndScale();
    
    Serial.println("Ready! Open: http://" + (WiFi.status()==WL_CONNECTED ? WiFi.localIP().toString() : WiFi.softAPIP().toString()));
}

void loop() { server.handleClient(); delay(1); }
