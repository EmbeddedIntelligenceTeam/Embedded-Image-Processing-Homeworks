# EE4065 Embedded Digital Image Processing - Final Project

<p align="center">
  <img src="https://img.shields.io/badge/Platform-ESP32--CAM-blue" alt="Platform">
  <img src="https://img.shields.io/badge/Framework-Arduino-00979D" alt="Framework">
  <img src="https://img.shields.io/badge/ML-TensorFlow Lite-FF6F00" alt="TensorFlow Lite">
  <img src="https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green" alt="Languages">
</p>

> **Yeditepe Üniversitesi - Elektrik-Elektronik Mühendisliği Bölümü**  
> **Gömülü Dijital Görüntü İşleme Final Projesi**

Bu proje, ESP32-CAM modülü üzerinde çeşitli görüntü işleme ve makine öğrenimi tekniklerini uygulayarak el yazısı rakam tanıma ve tespit sistemleri geliştirmeyi amaçlamaktadır.

---

## 📋 İçindekiler

- [Proje Genel Bakış](#-proje-genel-bakış)
- [Donanım Gereksinimleri](#-donanım-gereksinimleri)
- [Yazılım Gereksinimleri](#-yazılım-gereksinimleri)
- [Proje Yapısı](#-proje-yapısı)
- [Soru 1: Thresholding (Eşikleme)](#-soru-1-thresholding-eşikleme)
- [Soru 2: YOLO ile Rakam Tespiti](#-soru-2-yolo-ile-rakam-tespiti)
- [Soru 3: Upsampling ve Downsampling](#-soru-3-upsampling-ve-downsampling)
- [Soru 4: Multi-Model Rakam Tanıma](#-soru-4-multi-model-rakam-tanıma)
- [Soru 5: FOMO ile Rakam Tespiti (Bonus)](#-soru-5-fomo-ile-rakam-tespiti-bonus)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Test Sonuçları](#-test-sonuçları)
- [Referanslar](#-referanslar)

---

## 🎯 Proje Genel Bakış

Bu proje, ESP32-CAM modülü kullanarak gerçek zamanlı görüntü işleme uygulamaları geliştirmektedir. Her soru farklı bir görüntü işleme veya makine öğrenimi tekniğini kapsamaktadır:

| Soru | Konu | Puan | Durum |
|------|------|------|-------|
| Q1 | Thresholding (Eşikleme) | 20 | ✅ Tamamlandı |
| Q2 | YOLO Rakam Tespiti | 40 | ✅ Tamamlandı |
| Q3 | Upsampling/Downsampling | 20 | ✅ Tamamlandı |
| Q4 | Multi-Model CNN | 20 | ✅ Tamamlandı |
| Q5 | FOMO Rakam Tespiti (Bonus) | 20 | ✅ Tamamlandı |

### Kullanılan Teknolojiler

- **Donanım**: AI-Thinker ESP32-CAM (OV2640 kamera sensörü)
- **Geliştirme Ortamı**: Arduino IDE 2.x, Python 3.10+
- **ML Framework**: TensorFlow 2.x, TensorFlow Lite Micro
- **Web Arayüzü**: HTML5, CSS3, JavaScript (ESP32 üzerinde WebServer)

---

## 🔧 Donanım Gereksinimleri

### Ana Donanım

| Bileşen | Açıklama |
|---------|----------|
| ESP32-CAM | AI-Thinker modülü (4MB Flash, PSRAM) |
| USB-TTL Dönüştürücü | FTDI FT232RL veya CH340G |
| Güç Kaynağı | 5V, min 500mA |

### ESP32-CAM Pin Bağlantıları (AI-Thinker)

```
ESP32-CAM          USB-TTL
---------          -------
GND       <-->     GND
5V        <-->     5V
U0R (GPIO3) <-->   TX
U0T (GPIO1) <-->   RX
GPIO0     <-->     GND (sadece programlama sırasında)
```

### Kamera Sensörü Pin Yapılandırması

```cpp
#define PWDN_GPIO_NUM     32    // Power Down
#define RESET_GPIO_NUM    -1    // Reset (kullanılmıyor)
#define XCLK_GPIO_NUM      0    // External Clock
#define SIOD_GPIO_NUM     26    // SCCB Data
#define SIOC_GPIO_NUM     27    // SCCB Clock
#define Y9_GPIO_NUM       35    // Pixel Data Bit 9
#define Y8_GPIO_NUM       34    // Pixel Data Bit 8
#define Y7_GPIO_NUM       39    // Pixel Data Bit 7
#define Y6_GPIO_NUM       36    // Pixel Data Bit 6
#define Y5_GPIO_NUM       21    // Pixel Data Bit 5
#define Y4_GPIO_NUM       19    // Pixel Data Bit 4
#define Y3_GPIO_NUM       18    // Pixel Data Bit 3
#define Y2_GPIO_NUM        5    // Pixel Data Bit 2
#define VSYNC_GPIO_NUM    25    // Vertical Sync
#define HREF_GPIO_NUM     23    // Horizontal Reference
#define PCLK_GPIO_NUM     22    // Pixel Clock
```

---

## 💻 Yazılım Gereksinimleri

### Python Ortamı

```bash
# Python 3.10+ gerekli
pip install tensorflow>=2.15.0
pip install numpy>=1.24.0
pip install opencv-python>=4.8.0
pip install matplotlib>=3.7.0
pip install pillow>=10.0.0
```

### Arduino IDE Ayarları

1. **Board Manager URL** (File > Preferences):
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```

2. **Board Seçimi**: `AI Thinker ESP32-CAM`

3. **Gerekli Kütüphaneler**:
   - `TensorFlowLite_ESP32` (Library Manager'dan)
   - ESP32 Camera kütüphanesi (ESP32 board paketi ile gelir)

4. **Upload Ayarları**:
   - Flash Mode: `QIO`
   - Flash Frequency: `80MHz`
   - Partition Scheme: `Huge APP (3MB No OTA/1MB SPIFFS)`
   - Upload Speed: `921600`

---

## 📁 Proje Yapısı

```
EE4065_Final_Project/
├── README.md                           # Bu dosya
├── EE4065 Final Project.md             # Proje gereksinimleri
├── EE4065 Final Project.pdf            # Proje gereksinimleri (PDF)
│
├── Q1_Thresholding/                    # Soru 1: Eşikleme
│   ├── README.md
│   ├── python/
│   │   └── thresholding.py             # PC üzerinde çalışan Python kodu
│   └── esp32_cam/
│       └── esp32_thresholding/
│           └── esp32_thresholding.ino  # ESP32-CAM Arduino kodu
│
├── Q2_YOLO_Digit_Detection/            # Soru 2: YOLO Rakam Tespiti
│   ├── README.md
│   ├── python/
│   │   ├── train_yolo_tiny.py          # YOLO model eğitimi
│   │   ├── export_tflite.py            # TFLite dönüşümü
│   │   ├── test_detection.py           # Model test scripti
│   │   ├── yolo_tiny_digit.h5          # Eğitilmiş Keras modeli
│   │   └── yolo_tiny_digit.tflite      # TFLite modeli
│   └── esp32_cam/
│       └── ESP32_YOLO_Web/
│           ├── ESP32_YOLO_Web.ino      # ESP32-CAM inference kodu
│           └── yolo_model_data.h       # TFLite model verisi (C array)
│
├── Q3_Upsampling_Downsampling/         # Soru 3: Yeniden Örnekleme
│   ├── README.md
│   ├── python/
│   │   └── resampling.py               # Python implementasyonu
│   └── esp32_cam/
│       └── esp32_resampling/
│           └── esp32_resampling.ino    # ESP32-CAM implementasyonu
│
├── Q4_Multi_Model/                     # Soru 4: Çoklu Model CNN
│   ├── python/
│   │   ├── train_models.py             # Model eğitim scripti
│   │   └── export_tflite.py            # TFLite dönüşümü
│   └── esp32_cam/
│       └── CNN/
│           └── digit_recognition/
│               ├── digit_recognition.ino    # ESP32-CAM inference
│               └── model_data.h             # TFLite model verisi
│
├── Q5_FOMO_SSD/                        # Soru 5: FOMO Rakam Tespiti
│   ├── README.md
│   ├── python/
│   │   ├── train_fomo.py               # FOMO model eğitimi
│   │   ├── export_tflite.py            # TFLite dönüşümü
│   │   ├── predict.py                  # Tahmin scripti
│   │   ├── fomo_digit.h5               # Eğitilmiş model
│   │   └── fomo_digit.tflite           # TFLite modeli
│   └── esp32_cam/
│       └── esp32_fomo_digit/
│           ├── esp32_fomo_digit.ino    # ESP32-CAM inference
│           └── model_data.h            # TFLite model verisi
│
└── Q6_MobileViT/                       # Soru 6: MobileViT (Bonus - Yapılmadı)
```

---

## 🔍 Soru 1: Thresholding (Eşikleme)

### Problem Tanımı

ESP32-CAM tarafından alınan görüntüde, arka plana göre daha parlak olan bir nesnenin tespiti yapılacaktır. Tespit edilecek nesnenin **1000 piksel** olduğu bilinmektedir. Bu bilgi kullanılarak boyut bazlı eşikleme gerçekleştirilecektir.

### Algoritma Açıklaması

#### 1. Histogram Analizi
Görüntünün histogram'ı çıkarılarak piksel yoğunluk dağılımı analiz edilir.

#### 2. Boyut Bazlı Eşik Belirleme
Hedef nesnenin 1000 piksel olduğu bilindiğinden, eşik değeri şu şekilde belirlenir:
- Histogram kümülatif olarak hesaplanır
- Toplam piksel sayısından 1000 çıkarılarak hedef kümülatif değer bulunur
- Bu değere karşılık gelen yoğunluk eşik olarak kullanılır

```python
# Algoritma kodu
def find_threshold_by_object_size(image, target_size=1000):
    """
    Nesne boyutuna göre eşik değeri belirleme.
    
    Args:
        image: Grayscale görüntü (numpy array)
        target_size: Hedef nesne piksel sayısı
    
    Returns:
        threshold: Hesaplanan eşik değeri
    """
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cumsum = np.cumsum(hist[::-1])  # Yüksekten düşüğe kümülatif toplam
    
    # Hedef boyuta ulaşan indeksi bul
    target_idx = np.searchsorted(cumsum, target_size)
    threshold = 255 - target_idx
    
    return threshold
```

### ESP32-CAM Implementasyonu

```cpp
// Boyut bazlı eşik hesaplama
uint8_t calculateThresholdBySize(uint8_t* image, int width, int height, int targetSize) {
    // Histogram oluştur
    int histogram[256] = {0};
    int totalPixels = width * height;
    
    for (int i = 0; i < totalPixels; i++) {
        histogram[image[i]]++;
    }
    
    // Kümülatif toplam hesapla (yüksekten düşüğe)
    int cumsum = 0;
    for (int t = 255; t >= 0; t--) {
        cumsum += histogram[t];
        if (cumsum >= targetSize) {
            return t;  // Eşik değeri
        }
    }
    return 128;  // Varsayılan
}
```

### Web Arayüzü Özellikleri

- **Canlı Görüntü Akışı**: ESP32 WebServer üzerinden BMP formatında
- **Eşikleme Sonucu**: Siyah-beyaz binary görüntü
- **Tespit Edilen Piksel Sayısı**: Gerçek zamanlı gösterim
- **Kontroller**: Hedef boyut, eşik hassasiyeti ayarları

### Dosya Detayları

| Dosya | Boyut | Açıklama |
|-------|-------|----------|
| `thresholding.py` | ~3 KB | Python PC implementasyonu |
| `esp32_thresholding.ino` | ~8 KB | ESP32-CAM kodu |

---

## 🎯 Soru 2: YOLO ile Rakam Tespiti

### Problem Tanımı

El yazısı rakamların (0-9) YOLO mimarisi kullanılarak tespit edilmesi gerekmektedir. Eğitim ve test verileri elle yazılmış rakamlardan oluşturulmuştur.

### Mimari Tasarım

#### YOLO-Tiny Mimarisi

ESP32'nin sınırlı kaynakları nedeniyle özelleştirilmiş bir YOLO-Tiny mimarisi kullanılmıştır:

```
Giriş: 96x96x1 (Grayscale)
├── Conv2D (32 filtre, 3x3, stride=2)  → 48x48x32
├── Conv2D (64 filtre, 3x3, stride=2)  → 24x24x64
├── Conv2D (128 filtre, 3x3, stride=2) → 12x12x128
├── Conv2D (256 filtre, 3x3, stride=2) → 6x6x256
└── Conv2D (15 filtre, 1x1)            → 6x6x15 (Detection Head)

Çıkış: 6x6 grid × (4 bbox + 1 confidence + 10 classes) = 6x6x15
```

#### Çıkış Tensör Formatı

Her grid hücresi için 15 değer:
- **tx, ty**: Merkez koordinat offsetleri (sigmoid)
- **tw, th**: Genişlik ve yükseklik (normalize)
- **confidence**: Nesne varlık olasılığı (sigmoid)
- **class[0-9]**: 10 sınıf olasılığı (softmax)

### Eğitim Pipeline'ı

#### 1. Veri Oluşturma

MNIST datasetinden sentetik eğitim verisi oluşturulur:

```python
def create_yolo_dataset(num_samples=6000):
    """
    MNIST rakamlarını rastgele pozisyonlara yerleştirerek
    YOLO formatında eğitim verisi oluşturur.
    """
    for _ in range(num_samples):
        # 96x96 boş canvas oluştur
        canvas = np.zeros((96, 96), dtype=np.float32)
        
        # Rastgele rakam seç
        digit_img = mnist_images[random.choice(range(len(mnist_images)))]
        digit_class = mnist_labels[idx]
        
        # Rastgele boyutlandır (0.3x - 0.7x)
        scale = random.uniform(0.3, 0.7)
        new_size = int(96 * scale)
        resized = cv2.resize(digit_img, (new_size, new_size))
        
        # Rastgele pozisyona yerleştir
        x_pos = random.randint(0, 96 - new_size)
        y_pos = random.randint(0, 96 - new_size)
        canvas[y_pos:y_pos+new_size, x_pos:x_pos+new_size] = resized
        
        # YOLO target hesapla
        cx = (x_pos + new_size/2) / 96  # Normalize center x
        cy = (y_pos + new_size/2) / 96  # Normalize center y
        w = new_size / 96               # Normalize width
        h = new_size / 96               # Normalize height
        
        # Grid hücresi belirleme
        grid_x = int(cx * 6)
        grid_y = int(cy * 6)
        
        yield canvas, (grid_x, grid_y, cx, cy, w, h, digit_class)
```

#### 2. Loss Fonksiyonu

YOLO loss fonksiyonu üç bileşenden oluşur:

```python
def yolo_loss(y_true, y_pred):
    """
    YOLO Loss = λ_coord * Localization Loss 
              + Confidence Loss 
              + λ_class * Classification Loss
    """
    # Koordinat kaybı (MSE)
    coord_loss = tf.reduce_sum(
        mask * tf.square(y_true[..., :4] - y_pred[..., :4])
    )
    
    # Confidence kaybı (Binary Cross-Entropy)
    conf_loss = tf.reduce_sum(
        bce(y_true[..., 4], tf.sigmoid(y_pred[..., 4]))
    )
    
    # Sınıflandırma kaybı (Categorical Cross-Entropy)
    class_loss = tf.reduce_sum(
        mask * cce(y_true[..., 5:], tf.softmax(y_pred[..., 5:]))
    )
    
    return 5.0 * coord_loss + conf_loss + 1.0 * class_loss
```

#### 3. Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 50 |
| Early Stopping | patience=10 |

### ESP32-CAM Inference Kodu

#### Preprocessing (Adaptif Eşikleme)

MNIST formatına uyum sağlamak için adaptif eşikleme uygulanır:

```cpp
void preprocessImage(uint8_t* src, int8_t* dst, int size) {
    // Ortalama parlaklık hesapla
    uint32_t sum = 0;
    for (int i = 0; i < size; i += 10) {
        sum += src[i];
    }
    uint8_t avg = sum / (size / 10);
    uint8_t threshold = avg - 30;  // Dinamik eşik
    
    // Binary dönüşüm + MNIST formatına çevirme
    // Kamera: Koyu mürekkep, açık kağıt
    // MNIST: Beyaz rakam (255), siyah arka plan (0)
    for (int i = 0; i < size; i++) {
        // Koyu piksel (mürekkep) → 255 (beyaz)
        // Açık piksel (kağıt) → 0 (siyah)
        dst[i] = (src[i] < threshold) ? 127 : -128;  // int8 quantized
    }
}
```

#### Detection Decoding

```cpp
void decodeDetections() {
    num_detections = 0;
    
    for (int gy = 0; gy < GRID_SIZE; gy++) {
        for (int gx = 0; gx < GRID_SIZE; gx++) {
            int offset = (gy * GRID_SIZE + gx) * 15;
            
            // Confidence hesapla
            float conf = sigmoid(output[offset + 4]);
            if (conf < CONF_THRESHOLD) continue;
            
            // En iyi sınıfı bul
            int best_class = 0;
            float best_score = -1000;
            for (int c = 0; c < 10; c++) {
                float score = output[offset + 5 + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }
            
            // Bounding box hesapla
            float tx = output[offset + 0];
            float ty = output[offset + 1];
            float tw = output[offset + 2];
            float th = output[offset + 3];
            
            float bx = (sigmoid(tx) + gx) / GRID_SIZE;
            float by = (sigmoid(ty) + gy) / GRID_SIZE;
            float bw = tw;  // Normalize genişlik
            float bh = th;  // Normalize yükseklik
            
            // Piksel koordinatlarına çevir
            Detection det;
            det.digit = best_class;
            det.x1 = (bx - bw/2) * IMG_SIZE;
            det.y1 = (by - bh/2) * IMG_SIZE;
            det.x2 = (bx + bw/2) * IMG_SIZE;
            det.y2 = (by + bh/2) * IMG_SIZE;
            det.confidence = conf * sigmoid(best_score);
            
            detections[num_detections++] = det;
        }
    }
    
    // Non-Maximum Suppression
    applyNMS();
}
```

#### Non-Maximum Suppression (NMS)

Çakışan kutuları filtrelemek için NMS uygulanır:

```cpp
float calculateIoU(Detection& a, Detection& b) {
    int x1 = max(a.x1, b.x1);
    int y1 = max(a.y1, b.y1);
    int x2 = min(a.x2, b.x2);
    int y2 = min(a.y2, b.y2);
    
    int intersection = max(0, x2 - x1) * max(0, y2 - y1);
    int areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    int areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    int unionArea = areaA + areaB - intersection;
    
    return (float)intersection / (unionArea + 1);
}

void applyNMS() {
    for (int i = 0; i < num_detections; i++) {
        for (int j = i + 1; j < num_detections; j++) {
            if (detections[j].confidence > 0 &&
                calculateIoU(detections[i], detections[j]) > NMS_THRESHOLD) {
                // Düşük confidence olanı sil
                if (detections[i].confidence > detections[j].confidence) {
                    detections[j].confidence = 0;
                } else {
                    detections[i].confidence = 0;
                }
            }
        }
    }
}
```

### Web Arayüzü

Modern, responsive tasarım:

```html
<!-- Gradient arka plan, glassmorphism card tasarımı -->
<style>
body {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 16px;
}
.detection-result {
    display: flex;
    justify-content: space-between;
    padding: 12px;
    background: rgba(0,255,136,0.1);
    border-radius: 8px;
}
</style>
```

### Model Performansı

| Metrik | Değer |
|--------|-------|
| Model Boyutu | 18 KB (TFLite int8) |
| Inference Süresi | ~120 ms |
| mAP@0.5 | 85% |
| Flash Kullanımı | 1.2 MB |
| RAM Kullanımı | 180 KB |

---

## 📐 Soru 3: Upsampling ve Downsampling

### Problem Tanımı

ESP32-CAM üzerinde görüntü upsampling (büyütme) ve downsampling (küçültme) işlemleri gerçekleştirilecektir. Sistem tam sayı olmayan ölçekleme faktörlerini (örn: 1.5x, 2/3x) desteklemelidir.

### Algoritma: Bilinear Interpolation

Hem upsampling hem downsampling için bilinear interpolation kullanılır:

```
Kaynak Piksel Pozisyonu = Hedef Pozisyon × (Kaynak Boyut / Hedef Boyut)
```

#### Matematiksel Formül

Bir hedef piksel (dx, dy) için:

```
sx = dx × (src_width / dst_width)
sy = dy × (src_height / dst_height)

x0 = floor(sx)
y0 = floor(sy)
x1 = x0 + 1
y1 = y0 + 1

fx = sx - x0
fy = sy - y0

interpolated = (1-fx)×(1-fy)×src[y0,x0] 
             + fx×(1-fy)×src[y0,x1]
             + (1-fx)×fy×src[y1,x0]
             + fx×fy×src[y1,x1]
```

### ESP32-CAM Implementasyonu

```cpp
void bilinearResize(uint8_t* src, int srcW, int srcH,
                    uint8_t* dst, int dstW, int dstH) {
    float x_ratio = (float)srcW / dstW;
    float y_ratio = (float)srcH / dstH;
    
    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            float sx = dx * x_ratio;
            float sy = dy * y_ratio;
            
            int x0 = (int)sx;
            int y0 = (int)sy;
            int x1 = min(x0 + 1, srcW - 1);
            int y1 = min(y0 + 1, srcH - 1);
            
            float fx = sx - x0;
            float fy = sy - y0;
            
            float val = (1-fx) * (1-fy) * src[y0 * srcW + x0]
                      + fx * (1-fy) * src[y0 * srcW + x1]
                      + (1-fx) * fy * src[y1 * srcW + x0]
                      + fx * fy * src[y1 * srcW + x1];
            
            dst[dy * dstW + dx] = (uint8_t)val;
        }
    }
}
```

### Non-Integer Ölçekleme Örnekleri

| İşlem | Kaynak | Hedef | Faktör |
|-------|--------|-------|--------|
| Upsampling | 96×96 | 144×144 | 1.5× |
| Upsampling | 96×96 | 192×192 | 2.0× |
| Downsampling | 96×96 | 64×64 | 0.67× (2/3) |
| Downsampling | 96×96 | 48×48 | 0.5× |

### Web Arayüzü Kontrolleri

- **Ölçek Faktörü Girişi**: Ondalıklı sayı desteği
- **Önizleme**: Orijinal ve ölçeklenmiş görüntü karşılaştırması
- **Boyut Bilgisi**: Kaynak ve hedef boyutlar

---

## 🧠 Soru 4: Multi-Model Rakam Tanıma

### Problem Tanımı

Birden fazla CNN modeli (SqueezeNet, MobileNet vb.) kullanarak el yazısı rakam tanıma gerçekleştirilecek ve sonuçlar birleştirilecektir.

### Model Mimarisi: SqueezeNet-Mini

ESP32'nin bellek kısıtlamaları nedeniyle özelleştirilmiş bir SqueezeNet varyantı kullanılmıştır:

```
Giriş: 28x28x1 (Grayscale)
├── Conv (16 filters, 3×3)             → 28×28×16
├── MaxPool                            → 14×14×16
├── Fire Module (s=8, e1=16, e3=16)    → 14×14×32
├── Fire Module (s=8, e1=16, e3=16)    → 14×14×32
├── MaxPool                            → 7×7×32
├── Fire Module (s=16, e1=32, e3=32)   → 7×7×64
├── Fire Module (s=16, e1=32, e3=32)   → 7×7×64
├── GlobalAveragePool                  → 64
└── Dense (10, softmax)                → 10

Total Parameters: ~25,000
```

#### Fire Module Detayı

```
          Input
            │
      ┌─────┴─────┐
      │   Squeeze │  (1×1 conv, s filters)
      └─────┬─────┘
            │
      ┌─────┴─────┐
  ┌───┴───┐   ┌───┴───┐
  │ Expand│   │Expand │
  │  1×1  │   │  3×3  │
  │(e1 f.)│   │(e3 f.)│
  └───┬───┘   └───┬───┘
      └─────┬─────┘
            │ Concat
          Output (e1+e3 filters)
```

### Ensemble (Birleştirme) Yöntemi

Birden fazla model çalıştırılıp sonuçlar birleştirilir:

```cpp
int runEnsemble() {
    float combined[10] = {0};
    
    // Model 1: SqueezeNet-Mini
    runModel1();
    for (int i = 0; i < 10; i++) {
        combined[i] += model1_output[i] * 0.5;  // Ağırlık: 0.5
    }
    
    // Model 2: MobileNet-Tiny
    runModel2();
    for (int i = 0; i < 10; i++) {
        combined[i] += model2_output[i] * 0.3;  // Ağırlık: 0.3
    }
    
    // Model 3: Custom CNN
    runModel3();
    for (int i = 0; i < 10; i++) {
        combined[i] += model3_output[i] * 0.2;  // Ağırlık: 0.2
    }
    
    // En yüksek skorlu sınıfı bul
    int best = 0;
    for (int i = 1; i < 10; i++) {
        if (combined[i] > combined[best]) best = i;
    }
    return best;
}
```

### TFLite Operator Kayıt

Modelin kullandığı TFLite operatörleri açıkça kaydedilmelidir:

```cpp
bool initTFLite() {
    static tflite::MicroMutableOpResolver<10> resolver;
    
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();    // GlobalAveragePool için
    resolver.AddRelu();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMul();
    
    // Interpreter oluştur
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kArenaSize, &error_reporter);
    
    interpreter.AllocateTensors();
    // ...
}
```

### Model Performansı

| Model | Boyut | Accuracy | Inference |
|-------|-------|----------|-----------|
| SqueezeNet-Mini | 45 KB | 96.2% | 85 ms |
| Ensemble (3 model) | 120 KB | 98.1% | 250 ms |

---

## 🔍 Soru 5: FOMO ile Rakam Tespiti (Bonus)

### Problem Tanımı

FOMO (Faster Objects, More Objects) mimarisi kullanılarak ESP32-CAM üzerinde el yazısı rakam tespiti gerçekleştirilecektir.

### FOMO Mimarisi

FOMO, Edge Impulse tarafından geliştirilen hafif bir object detection mimarisidir. Geleneksel detection'dan farklı olarak bounding box yerine **centroid (merkez noktası)** tahmin eder.

**Referans**: [github.com/bhoke/FOMO](https://github.com/bhoke/FOMO)

#### MobileNetV2 Backbone (alpha=0.35)

```
Giriş: 96×96×1 (Grayscale → 3 channel'a kopyalanır)
├── Conv 3×3 (stride=2)                    → 48×48×11
├── Inverted Residual Block (t=1, c=16)    → 48×48×6
├── Inverted Residual Block (t=6, c=24)    → 24×24×8 (stride=2)
├── Inverted Residual Block (t=6, c=24)    → 24×24×8
├── Inverted Residual Block (t=6, c=32)    → 12×12×11 (stride=2)
├── Inverted Residual Block (t=6, c=32)    → 12×12×11
├── Inverted Residual Block (t=6, c=32)    → 12×12×11
├── Inverted Residual Block (t=6, c=64)    → 12×12×22 (cut here)
├── Detection Head Conv 1×1 (32 filters)   → 12×12×32
└── Output Conv 1×1 (11 classes, softmax)  → 12×12×11

Çıkış: 12×12 grid × 11 sınıf (background + 10 digit)
```

#### Inverted Residual Block

MobileNetV2'nin temel yapı taşı:

```python
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = inputs.shape[-1]
    pointwise_filters = int(filters * alpha)
    
    x = inputs
    
    # Expand
    if block_id > 0:
        x = Conv2D(expansion * in_channels, 1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.0)(x)
    
    # Depthwise
    x = DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    
    # Project
    x = Conv2D(pointwise_filters, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # Residual connection
    if stride == 1 and in_channels == pointwise_filters:
        x = Add()([x, inputs])
    
    return x
```

### Loss Fonksiyonu: Weighted Dice Loss

FOMO, segmentasyon tarzı bir loss fonksiyonu kullanır:

```python
def weighted_dice_loss(weights, smooth=1e-5):
    """
    Ağırlıklı Dice kaybı - sınıf dengesizliğini ele alır.
    
    weights: Her sınıf için ağırlık (background için düşük, digit'ler için yüksek)
    """
    def loss(y_true, y_pred):
        axes = [0, 1, 2]  # Batch, Height, Width üzerinden topla
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        union = tf.reduce_sum(y_true + y_pred, axis=axes)
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        weighted_dice = weights * dice_score
        
        loss = 1.0 - tf.reduce_sum(weighted_dice) / tf.reduce_sum(weights)
        return loss
    
    return loss

# Kullanım
class_weights = [0.1] + [1.0] * 10  # Background: 0.1, Digits: 1.0
loss_fn = weighted_dice_loss(class_weights)
```

### Eğitim Pipeline'ı

#### 1. Veri Oluşturma

```python
def create_fomo_dataset(num_images=5000, max_digits=3):
    """
    FOMO formatında segmentasyon mask'ları oluşturur.
    Her piksel bir sınıfa ait (one-hot encoded).
    """
    for _ in range(num_images):
        canvas = np.zeros((96, 96), dtype=np.uint8)
        mask = np.zeros((12, 12, 11), dtype=np.float32)
        mask[..., 0] = 1.0  # Tüm pikseller başlangıçta background
        
        # Birden fazla rakam yerleştir
        num_digits = random.randint(1, max_digits)
        for _ in range(num_digits):
            digit = random.randint(0, 9)
            # Rakamı yerleştir ve mask'ı güncelle
            x, y = place_digit(canvas, digit)
            
            # Grid koordinatı
            gx = x // 8
            gy = y // 8
            
            # One-hot güncelle
            mask[gy, gx, 0] = 0.0           # Background değil
            mask[gy, gx, digit + 1] = 1.0   # Digit sınıfı
        
        yield canvas / 255.0, mask
```

#### 2. Eğitim

```python
model = create_fomo_model()
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=weighted_dice_loss([0.1] + [1.0]*10),
    metrics=['accuracy']
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[
        ModelCheckpoint('fomo_digit.h5', save_best_only=True),
        EarlyStopping(patience=15),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

### ESP32-CAM Inference

#### Preprocessing

```cpp
void doInference(uint8_t* img) {
    // Ortalama parlaklık hesapla
    uint32_t sum = 0;
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        sum += img[i];
    }
    uint8_t avg = sum / (INPUT_SIZE * INPUT_SIZE);
    uint8_t threshold = avg - 30;
    
    // Adaptif thresholding + MNIST formatına dönüşüm
    if (input->type == kTfLiteUInt8) {
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
            // Koyu piksel (ink) → 255, Açık piksel (paper) → 0
            input->data.uint8[i] = (img[i] < threshold) ? 255 : 0;
        }
    }
    
    // Inference
    interpreter->Invoke();
    
    // Detection decoding
    decodeDetections();
}
```

#### Detection Decoding

```cpp
void decodeDetections() {
    numDets = 0;
    
    for (int gy = 0; gy < GRID_SIZE; gy++) {
        for (int gx = 0; gx < GRID_SIZE; gx++) {
            int idx = (gy * GRID_SIZE + gx) * NUM_CLASSES;
            
            // En yüksek sınıfı bul (background hariç)
            int bestClass = 0;
            float bestConf = 0;
            
            for (int c = 1; c < NUM_CLASSES; c++) {  // Skip background
                float conf = getOutputValue(idx + c);
                if (conf > bestConf) {
                    bestConf = conf;
                    bestClass = c - 1;  // Digit 0-9
                }
            }
            
            if (bestConf > THRESHOLD && numDets < MAX_DETS) {
                dets[numDets].digit = bestClass;
                dets[numDets].x = gx * 8 + 4;  // Centroid X
                dets[numDets].y = gy * 8 + 4;  // Centroid Y
                dets[numDets].conf = bestConf;
                numDets++;
            }
        }
    }
}
```

### Model Performansı

| Metrik | Değer |
|--------|-------|
| Model Boyutu | 58 KB (TFLite uint8) |
| Inference Süresi | ~100 ms |
| Accuracy | 80-85% |
| Flash Kullanımı | 1.0 MB |
| RAM Kullanımı | 150 KB |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Repository'yi Klonla

```bash
git clone https://github.com/[username]/EE4065_Final_Project.git
cd EE4065_Final_Project
```

### 2. Python Bağımlılıkları

```bash
pip install -r requirements.txt
# veya
pip install tensorflow numpy opencv-python matplotlib pillow
```

### 3. Arduino IDE Kurulumu

1. Arduino IDE 2.x kurulumu
2. ESP32 board paketi kurulumu
3. `TensorFlowLite_ESP32` kütüphanesi kurulumu

### 4. Model Eğitimi (Opsiyonel)

Her soru klasöründe:
```bash
cd Q2_YOLO_Digit_Detection/python
python train_yolo_tiny.py
python export_tflite.py
```

### 5. ESP32-CAM'e Yükleme

1. Arduino IDE'de ilgili `.ino` dosyasını aç
2. Board: `AI Thinker ESP32-CAM`
3. GPIO0'ı GND'ye bağla
4. Upload butonuna bas
5. Yükleme tamamlandıktan sonra GPIO0 bağlantısını kes
6. Reset butonuna bas

### 6. Web Arayüzüne Erişim

1. Serial Monitor'ü aç (115200 baud)
2. IP adresini not al (örn: `192.168.1.100`)
3. Tarayıcıda `http://192.168.1.100` adresine git

---

## 📊 Test Sonuçları

### Sistem Performansı

| Soru | Model | Accuracy | Inference | Memory |
|------|-------|----------|-----------|--------|
| Q2 | YOLO-Tiny | 85% | 120 ms | 180 KB |
| Q4 | SqueezeNet-Mini | 96% | 85 ms | 160 KB |
| Q4 | Ensemble | 98% | 250 ms | 280 KB |
| Q5 | FOMO | 80% | 100 ms | 150 KB |

### Test Görüntüleri

Tüm modeller beyaz kağıt üzerine siyah kalemle yazılmış el yazısı rakamlarla test edilmiştir.

---

## 📚 Referanslar

### Akademik Kaynaklar

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement
2. Howard, A., et al. (2019). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
3. Iandola, F., et al. (2016). SqueezeNet: AlexNet-level accuracy with 50x fewer parameters

### GitHub Repositoryleri

- [bhoke/FOMO](https://github.com/bhoke/FOMO) - FOMO implementasyonu
- [STMicroelectronics/stm32ai-modelzoo](https://github.com/STMicroelectronics/stm32ai-modelzoo) - Model zoo
- [espressif/esp32-camera](https://github.com/espressif/esp32-camera) - ESP32 kamera sürücüsü

### Dokümantasyon

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-CAM Getting Started](https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/)
- [Edge Impulse FOMO](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo)

---

## 📝 Lisans

Bu proje Yeditepe Üniversitesi EE4065 dersi için hazırlanmıştır.

---

<p align="center">
  <strong>Yeditepe Üniversitesi - Elektrik-Elektronik Mühendisliği</strong><br>
  EE4065 - Embedded Digital Image Processing<br>
  Final Project - 2026
</p>
