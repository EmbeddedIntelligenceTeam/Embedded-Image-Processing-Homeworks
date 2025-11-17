# EE 4065 - Gömülü Dijital Görüntü İşleme: Ödev 2 Raporu

**Hazırlayan:** [Adınız Soyadınız]
**Numara:** [Numaranız]

## 📜 İçindekiler
- [1. Proje Amacı ve Kapsamı](#1-proje-amacı-ve-kapsamı)
- [2. Proje Mimarisi ve Kullanımı](#2-proje-mimarisi-ve-kullanımı)
- [3. Soru 1: Histogram Oluşturma](#3-soru-1-histogram-oluşturma)
  - [3.1. Teori ve Yöntem](#31-teori-ve-yöntem)
  - [3.2. Uygulama Kodu](#32-uygulama-kodu)
  - [3.3. Sonuç ve Doğrulama](#33-sonuç-ve-doğrulama)
- [4. Soru 2: Histogram Eşitleme (HE)](#4-soru-2-histogram-eşitleme-he)
  - [4.1. Teori ve Yöntem (Soru 2a)](#41-teori-ve-yöntem-soru-2a)
  - [4.2. Uygulama Kodu (Soru 2b)](#42-uygulama-kodu-soru-2b)
  - [4.3. Sonuç ve Doğrulama (Soru 2c)](#43-sonuç-ve-doğrulama-soru-2c)
- [5. Soru 3: 2D Konvolüsyon (Filtreleme)](#5-soru-3-2d-konvolüsyon-filtreleme)
  - [5.1. Teori ve Yöntem (Soru 3a)](#51-teori-ve-yöntem-soru-3a)
  - [5.2. Uygulama ve Sonuçlar (Soru 3b: Low Pass)](#52-uygulama-ve-sonuçlar-soru-3b-low-pass)
  - [5.3. Uygulama ve Sonuçlar (Soru 3c: High Pass)](#53-uygulama-ve-sonuçlar-soru-3c-high-pass)
- [6. Soru 4: Medyan Filtre](#6-soru-4-medyan-filtre)
  - [6.1. Teori ve Yöntem (Soru 4a)](#61-teori-ve-yöntem-soru-4a)
  - [6.2. Uygulama Kodu](#62-uygulama-kodu)
  - [6.3. Sonuç ve Doğrulama (Soru 4c)](#63-sonuç-ve-doğrulama-soru-4c)
- [7. Genel Tartışma: Filtre Karşılaştırması](#7-genel-tartışma-filtre-karşılaştırması)

---

## 1. Proje Amacı ve Kapsamı

Bu ödevin temel amacı, bir `STM32F446RE` mikrodenetleyicisi üzerinde, C dilinde temel görüntü işleme algoritmalarını sıfırdan implemente etmektir. Ödev, piksellerin istatistiksel analizini (Histogram), kontrast iyileştirmesini (Histogram Eşitleme) ve uzamsal filtrelemeyi (2D Konvolüsyon ve Medyan Filtre) kapsamaktadır.

Tüm işlemler 128x128 piksel, 8-bit gri tonlamalı (grayscale) görüntüler üzerinde gerçekleştirilmiştir.

## 2. Proje Mimarisi ve Kullanımı

Bu proje, ödev dokümanında belirtilen statik `.h` dosyası yerine, PC ile STM32 arasında dinamik bir görüntü aktarım mimarisi kullanır.

* **Donanım:** `STM32F446RE Nucleo-64` Kartı
* **Haberleşme:** `UART` (2000000 baud)
* **PC Arayüzü:** [Buraya Python/MATLAB/C# vb. ne kullandığınızı yazın]

### 🖥️ Çalışma Akışı
Projenin çalışma mantığı, PC'den gelen komutlara göre görüntü işleme ve geri gönderme üzerine kuruludur:
1.  **Görüntü Yükleme (PC -> STM):** PC tarafındaki arayüz, 128x128 (16384 byte) boyutundaki gri tonlamalı görüntüyü UART üzerinden STM32'ye gönderir.
2.  **STM32'de İşleme:** STM32, gelen veriyi `pImage` adlı global bir diziye kaydeder. `while(1)` döngüsü içinde, bu `pImage` dizisi üzerinde ilgili işleme fonksiyonu (örn: `Homework_Apply_Convolution`) çağrılır.
3.  **Sonuç Gönderme (STM -> PC):** İşlenen görüntü (veya Soru 1'deki gibi histogram verisi), yine UART üzerinden PC'ye geri gönderilir.
4.  **Görselleştirme:** PC arayüzü, gönderdiği "Orijinal Görüntü" ile STM32'den geri aldığı "İşlenmiş Görüntü"yü yan yana göstererek sonuçların anlık olarak karşılaştırılmasını sağlar.

Tüm C implementasyonları `Core/Src/main.c` dosyasındaki `USER CODE` blokları içinde yer almaktadır.

---

## 3. Soru 1: Histogram Oluşturma

### 3.1. Teori ve Yöntem
Histogram, bir görüntüdeki her bir parlaklık seviyesinin (0-255) kaç kez tekrarlandığını sayan bir "nüfus sayımı" işlemidir. Görüntünün karanlık, parlak veya düşük kontrastlı olup olmadığını anlamak için kullanılan temel bir teşhis aracıdır.

Uygulamada, `uint32_t g_histogram_data[256]` adında global bir dizi oluşturulmuştur. `Homework_Calculate_Histogram()` fonksiyonu, 16384 (128x128) pikselin tamamını dolaşır. Her pikselin değerini (`v`) okur ve ilgili sayacı (`g_histogram_data[v]++`) bir artırır. Sayaçların `uint32_t` (32-bit tamsayı) olarak tanımlanmasının sebebi, 16384 pikselin tamamının aynı değerde olması durumunda oluşabilecek taşmayı (overflow) engellemektir.

### 3.2. Uygulama Kodu
Aşağıdaki fonksiyon, `p_gray` (kaynak görüntü) buffer'ını okur ve `p_hist` (hedef histogram dizisi) buffer'ını doldurur.

```c
/**
 * @brief Soru 1a: Verilen 8-bit Grayscale görüntünün histogramını hesaplar.
 * @param p_gray: Kaynak grayscale görüntü buffer'ının pointer'ı (uint8_t*).
 * @param p_hist: Sonuç histogram dizisinin (256 x uint32_t) pointer'ı.
 * @param total_pixels: Toplam piksel sayısı (örn: 16384).
 */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, 
                                  uint32_t total_pixels)
{
    uint32_t i;

    // 1. Adım: Histogram dizisini (sayaçları) sıfırla
    // 256 eleman * 4 byte/eleman = 1024 byte'lık alanı sıfırlar.
    memset(p_hist, 0, 256 * sizeof(uint32_t));

    // 2. Adım: Görüntünün tüm piksellerini dolaş
    for (i = 0; i < total_pixels; i++)
    {
        // 3. Adım: O anki pikselin değerini (0-255) al
        uint8_t pixel_value = p_gray[i]; 
        
        // 4. Adım: O değere karşılık gelen sayacı 1 arttır
        p_hist[pixel_value]++;
    }
}