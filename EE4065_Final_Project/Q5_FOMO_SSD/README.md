# Question 5a: FOMO Digit Detection on ESP32-CAM

El yazısı rakam tespiti için FOMO (Faster Objects, More Objects) implementasyonu.

## 📖 FOMO Nedir?

FOMO, Edge Impulse tarafından geliştirilen ultra-hafif bir object detection modelidir:
- **Centroid-based detection** - Bounding box yerine merkez nokta tespit eder
- **Heat map output** - Her grid hücresi bir sınıf olasılığı verir
- **Çok hafif** - ~75KB (int8 quantized), 100KB RAM altında çalışır
- **Hızlı** - 30+ FPS ESP32 üzerinde

## 🚀 Kullanım

### 1. Model Eğitimi (Python)

```bash
cd python

# Tam eğitim (~50 epoch, ~30 dakika)
python train_fomo.py

# Hızlı test (5 epoch, ~2 dakika)
python train_fomo.py --test
```

### 2. TFLite Export

```bash
cd python

# Int8 quantized (ESP32 için önerilen)
python export_tflite.py

# veya float16
python export_tflite.py --quantize float16

# Export ve test
python export_tflite.py --test
```

Bu komut:
- `fomo_digit.tflite` oluşturur
- `../esp32_cam/esp32_fomo_digit/model_data.h` dosyasını günceller

### 3. Test ve Görselleştirme

```bash
cd python

# Rastgele test görselleri ile test
python predict.py --test

# Belirli bir görsel ile test
python predict.py --image path/to/image.jpg
```

### 4. ESP32-CAM Deploy

1. Arduino IDE'de `esp32_cam/esp32_fomo_digit/esp32_fomo_digit.ino` dosyasını aç
2. WiFi bilgilerini güncelle:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Board: "AI Thinker ESP32-CAM" seç
4. Upload et
5. Serial Monitor'da IP adresini gör
6. Tarayıcıda IP adresine git

## 📁 Dosya Yapısı

```
Q5_FOMO_SSD/
├── python/
│   ├── train_fomo.py      # Model eğitim scripti
│   ├── export_tflite.py   # TFLite dönüşüm
│   └── predict.py         # Test ve görselleştirme
├── esp32_cam/
│   └── esp32_fomo_digit/
│       ├── esp32_fomo_digit.ino  # ESP32 Arduino kodu
│       └── model_data.h          # TFLite model (C array)
└── README.md
```

## 🔧 Gereksinimler

### Python
- TensorFlow 2.10+
- OpenCV
- NumPy
- Matplotlib

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### Arduino
- ESP32 board package
- TensorFlowLite_ESP32 library

## 📊 Model Detayları

| Özellik | Değer |
|---------|-------|
| Giriş | 96x96x1 (grayscale) |
| Çıkış | 12x12x11 (grid heat map) |
| Backbone | MobileNetV2 (alpha=0.35) |
| Sınıflar | background + 0-9 rakamlar |
| Boyut (int8) | ~75KB |

## 📚 Referanslar

- [bhoke/FOMO GitHub](https://github.com/bhoke/FOMO)
- [Edge Impulse FOMO Documentation](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo)
