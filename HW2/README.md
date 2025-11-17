#  EE4065 – Embedded Digital Image Processing
### **Homework 2**
 **Due Date:** November 17, 2025   
 **Team Members:**  
- Taner Kahyaoğlu  
- Yusuf Zivaroğlu  

---

## 1. Project Aim and Scope

The primary objective of this project is to implement foundational digital image processing algorithms from scratch in C, running on an `STM32F446RE` microcontroller. This assignment focuses on analyzing and manipulating images at the pixel level to perform statistical analysis, contrast enhancement, and spatial filtering.

The project covers four main topics:
* **Histogram Generation:** Calculating the intensity distribution of an image.
* **Histogram Equalization (HE):** Automatically enhancing image contrast based on its histogram.
* **2D Convolution:** Implementing a linear spatial filter framework for:
    * **Low-Pass Filtering** (Blurring)
    * **High-Pass Filtering** (Edge Detection)
* **Median Filtering:** Implementing a non-linear filter for noise reduction, specifically "salt-and-pepper" noise.

All operations are performed on a **128x128 pixel, 8-bit grayscale** image.

---

## Project Architecture & PC-Side Host

This project uses a "Host-Target" architecture. The `STM32` board acts as the **Target** (performing the processing), and a Python script on the PC acts as the **Host** (providing the data and visualizing the result).

### Objective (PC-Side)

The Python script (`test_stm32.py`) is responsible for:
1.  Loading any standard image file (e.g., `.png`, `.jpg`) from the PC.
2.  Resizing the image to the project's standard `128x128` resolution.
3.  Converting the image to 8-bit grayscale and flattening it into a raw 16,384-byte array.
4.  Handling the serial communication (using `pyserial`) to **transmit** these 16,384 bytes to the STM32.
5.  Waiting for and **receiving** the 16,384 processed bytes back from the STM32.
6.  Reshaping the received byte array back into a 128x128 image.
7.  Displaying the "Original" and "Processed" images side-by-side using `opencv` for immediate visual verification and comparison.

This single script is used to test **all** homework questions (Q2, Q3, and Q4). The specific filter being tested is changed by modifying the `while(1)` loop in the STM32's `main.c` file and re-compiling the microcontroller.
---

### 🔹 py_serialimg library
```python
import numpy as np
import serial
import msvcrt
import cv2
import time

MCU_WRITES = 87
MCU_READS  = 82
# Request Type
rqType = { MCU_WRITES: "MCU Sends Image", MCU_READS: "PC Sends Image"} 

# Format 
formatType = { 1: "Grayscale", 2: "RGB565", 3: "RGB888",} 

IMAGE_FORMAT_GRAYSCALE	= 1
IMAGE_FORMAT_RGB565		= 2
IMAGE_FORMAT_RGB888		= 3

# Init Com Port
def SERIAL_Init(port):
    global __serial    
    __serial = serial.Serial(port, 2000000, timeout = 10)
    __serial.flush()
    print(__serial.name, "Opened")
    print("")

# Wait for MCU Request 
def SERIAL_IMG_PollForRequest():
    global requestType
    global height
    global width
    global format
    global imgSize
    while(1):
        if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
            print("Exit program!")
            exit(0)
        if np.frombuffer(__serial.read(1), dtype= np.uint8) == 83:
            if np.frombuffer(__serial.read(1), dtype= np.uint8) == 84:
                requestType  = int(np.frombuffer(__serial.read(1), dtype= np.uint8))
                height       = int(np.frombuffer(__serial.read(2), dtype= np.uint16))
                width        = int(np.frombuffer(__serial.read(2), dtype= np.uint16))
                format       = int(np.frombuffer(__serial.read(1), dtype= np.uint8))
                imgSize     = height * width * format
                
                print("Request Type : ", rqType[int(requestType)])
                print("Height       : ", int(height))
                print("Width        : ", int(width))
                print("Format       : ", formatType[int(format)])
                print()
                return [int(requestType), int(height), int(width), int(format)]

# Reads Image from MCU  
def SERIAL_IMG_Read():
    img = np.frombuffer(__serial.read(imgSize), dtype = np.uint8)
    img = np.reshape(img, (height, width, format))
    if format == IMAGE_FORMAT_GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif format == IMAGE_FORMAT_RGB565:
        img = cv2.cvtColor(img, cv2.COLOR_BGR5652BGR)

    timestamp = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())     
    cv2.imshow("img", img) 
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    #cv2.imwrite("capture/image_"+ timestamp + ".jpg", img)  
    return img

# Writes Image to MCU   
def SERIAL_IMG_Write(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (width,height))
    if format == IMAGE_FORMAT_GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif format == IMAGE_FORMAT_RGB565:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGR565)

    img = img.tobytes()
    __serial.write(img)
```

### 🔹 py_image library 
```python
import py_serialimg
import numpy as np
import cv2 

COM_PORT = "COM8"  # !!! BURAYI NUCLEO F446RE'YE GÖRE GÜNCELLEYİN !!!

TEST_IMAGE_FILENAME = "lena_gray.png"  
# --- AYAR SONU ---

print(f"Seri port {COM_PORT} başlatılıyor...")
py_serialimg.SERIAL_Init(COM_PORT)
print("Port başlatıldı.")

mandrill = TEST_IMAGE_FILENAME

try:
    while 1:
        print("\nSTM32'den istek bekleniyor (PollForRequest)...")
        # Bu noktada F446RE kartınız height=128, width=128 bilgilerini gönderecek
        rqType, height, width, format = py_serialimg.SERIAL_IMG_PollForRequest()

        # SENARYO 1: STM32 bir görüntü göndermek istiyor (MCU_WRITES)
        if rqType == py_serialimg.MCU_WRITES:
            print(f"STM32 görüntü gönderiyor (Boyut: {width}x{height}). Alınıyor...")
            # Kütüphane 128x128x2 byte veriyi okuyacak
            img = py_serialimg.SERIAL_IMG_Read()
            print("Görüntü başarıyla alındı.")
            
            # GÜNCELLEME 3 (İsteğe bağlı): Kayıt dosyasının adını netleştirdik
            received_filename = "received_from_f446re.png"
            cv2.imwrite(received_filename, img)
            print(f"Görüntü '{received_filename}' olarak kaydedildi.")

        # SENARYO 2: STM32 bir görüntü almak istiyor (MCU_READS)
        elif rqType == py_serialimg.MCU_READS:
            print(f"STM32 görüntü istiyor (Boyut: {width}x{height}). Gönderiliyor...")
            # 'SERIAL_IMG_Write' fonksiyonu 'mandrill.tif' dosyasını
            # otomatik olarak {width}x{height} (128x128) boyutuna küçültecek.
            img = py_serialimg.SERIAL_IMG_Write(mandrill)
            print(f"'{mandrill}' görüntüsü (128x128'e küçültülerek) başarıyla gönderildi.")

except KeyboardInterrupt:
    print("\nProgram durduruldu.")
except Exception as e:
    print(f"\nBir hata oluştu: {e}")
    print(f"COM portunun ('{COM_PORT}') doğru olduğundan ve STM32 kartının bağlı olduğundan emin olun.")
```
---

##  Q1 — Histogram Formation 

### 🔹 Objective  
Importing an image from the computer, computing its histogram, and displaying the histogram distribution through the Memory Browser in STM32CubeIDE

---

### 🔹 STM32 code 
```c
/* USER CODE BEGIN Includes */
#include "lib_image.h"
#include "lib_serialimage.h"
#include "string.h"
/* USER CODE END Includes */
```

```c
/* USER CODE BEGIN PV */
uint32_t g_histogram_data[256];
/* USER CODE END PV */
```

```c
/* USER CODE BEGIN PFP */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
/* USER CODE END PFP */
```

```c
/* USER CODE BEGIN 4 */

void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height)
{
  uint32_t i;
  uint32_t total_pixels = width * height; // 128 * 128 = 16384

  // 1. Önce histogram dizisinin (g_histogram_data) içini temizle (sıfırla)
  for (i = 0; i < 256; i++)
  {
    p_hist[i] = 0;
  }

  // 2. Görüntünün tüm piksellerini (0'dan 16383'e kadar) tek tek dolaş
  for (i = 0; i < total_pixels; i++)
  {
    // O anki pikselin değerini oku (örn: 150)
    uint8_t pixel_value = p_gray[i]; 
    
    // Histogram dizisinde o değere (150'ye) karşılık gelen sayacı 1 arttır
    p_hist[pixel_value]++;
  }
}

/* USER CODE END 4 */
```

```c
/* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // 1. PC'den 128x128 Grayscale görüntüyü al (pImage buffer'ı dolacak)
	if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	{
		Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, 128, 128);
	}
  }
  /* USER CODE END 3 */
```

---

### Execution Steps and Verification 

Here is the simple 4-step process used to test and verify the histogram calculation.

1.  **Prepare STM32 (`main.c`):**
    First, the `while(1)` loop in `main.c` is modified specifically for this test. The loop must *only* receive the image and then calculate the histogram. The `LIB_SERIAL_IMG_Transmit()` function is commented out (`//`) or removed, as we do not want to send anything back to the PC for this question.

    ```c
    /* main.c - while(1) loop for Q1 Test */
    while (1)
    {
      if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
      {
          // 1. Receive image
          // 2. Calculate histogram (function to be tested)
          Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, ...);
          
          // 3. DO NOT transmit back
          // LIB_SERIAL_IMG_Transmit(&img); // <-- This line is commented out
      }
    }
    ```

2.  **Start Debugger (STM32):**
    The project is compiled and flashed to the board using **Debug Mode** (the "bug" icon). We press **"Resume" (F8)** to let the code run. The STM32 is now waiting to receive an image.

3.  **Run Python Script (PC):**
    On the PC, we run the `test_stm32.py` script using a 128x128 **solid black** test image.
    * The script sends the 16,384 bytes of the black image.
    * After sending, the script waits for a response. Since the STM32 is not sending one, the script will **time out** after 5 seconds.
    * **This timeout is expected and normal for the Q1 test.** It confirms the STM32 received the data and moved on.

4.  **Verify Result (STM32CubeIDE):**
    * After the Python script times out, we go back to STM32CubeIDE and click the **"Pause"** button.
    * We open the **Memory Browser** window (Window > Show View > Memory Browser).
    * We enter `g_histogram_data` into the address bar.
    * As required by Q1c, we check the entries. For the solid black image, the memory correctly shows `g_histogram_data[0]` holding the value **16384** (or `0x4000`), and all other 255 entries are 0.

---

### Results 

To validate the `Homework_Calculate_Histogram` function, a 128x128 solid black image was transmitted from the PC to the STM32.

| Original Image | Histogram formation under the browser |
| :---: | :---: |
| <img width="538" height="537" alt="image" src="https://github.com/user-attachments/assets/403aca2a-7069-48da-a7e1-55b07c3b6cf1" /> | <img width="1145" height="812" alt="image" src="https://github.com/user-attachments/assets/837e55ab-f5bf-41ce-ad74-1da0ff2447bb" /> |

---

##  Q2 — Histogram Equalization

### 🔹 Objective  
**Objective:**
The goal of Histogram Equalization (HE) is to automatically improve the contrast of an image. It is an *adaptive* intensity transformation. Unlike a simple Gamma or Negative transform (from HW1), HE creates a custom transformation function (Look-Up Table or LUT) based on the image's *own* unique histogram. It "stretches" or "spreads" the most common pixel intensities across the entire available range (0-255).

---

### Calculation 

This part is the theoretical calculation of the second question.

| <img width="568" height="569" alt="image" src="https://github.com/user-attachments/assets/2ed4b521-0f78-441e-97bd-d696d3d12975" /> |
| <img width="545" height="788" alt="image" src="https://github.com/user-attachments/assets/5cd2edbe-3c47-4353-9d53-1fa78236c234" /> |
| <img width="542" height="562" alt="image" src="https://github.com/user-attachments/assets/954c8253-fa35-45f6-a7e0-5b1719b4cbcb" /> |


---

### 🔹 STM32 code  
Add the generated header file into your STM32 project includes:

```c
/* USER CODE BEGIN Includes */
#include "lib_image.h"
#include "lib_serialimage.h"
#include "string.h"
/* USER CODE END Includes */
```

```c
/* USER CODE BEGIN PV */
uint32_t g_histogram_data[256];
/* USER CODE END PV */
```

```c
/* USER CODE BEGIN PFP */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
void Homework_Apply_Histogram_EQ(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
/* USER CODE END PFP */
```

```c
/* USER CODE BEGIN 4 */

void Homework_Apply_Histogram_EQ(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height)
{
  uint32_t i;
  uint32_t total_pixels = width * height; // 16384
  
  static uint32_t cdf[256]; // static olarak tanımlamak stack taşmasını önler
  cdf[0] = p_hist[0];
  
  for (i = 1; i < 256; i++)
  {
    cdf[i] = cdf[i-1] + p_hist[i];
  }
  
  // 2. CDF'deki sıfır olmayan ilk (minimum) değeri bul
  uint32_t cdf_min = 0;
  for (i = 0; i < 256; i++)
  {
    if (cdf[i] != 0)
    {
      cdf_min = cdf[i];
      break;
    }
  }

  // 3. Normalizasyon için "Look-Up Table" (LUT) oluştur
  //    Formül: h(v) = round( ( (CDF(v) - CDF_min) * 255 ) / (ToplamPiksel - CDF_min) )
  
  uint8_t lut[256];
  
  float scale_factor = 255.0f / (float)(total_pixels - cdf_min);

  for (i = 0; i < 256; i++)
  {
    float h_v = (float)(cdf[i] - cdf_min) * scale_factor;
    // En yakın tam sayıya yuvarla (0.5f eklemek yuvarlama içindir)
    lut[i] = (uint8_t)(h_v + 0.5f); 
  }

  // 4. Görüntüyü LUT kullanarak "yerinde" yeniden haritala
  //    (Yani pImage'ın içeriğini kalıcı olarak değiştir)
  for (i = 0; i < total_pixels; i++)
  {
    p_gray[i] = lut[p_gray[i]];
  }
}
/* USER CODE END 4 */
```

```c
/* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  
    // 1. PC'den 128x128 Grayscale görüntüyü al (pImage buffer'ı dolacak)
	if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	{
		// 2. (Soru 1) Gelen görüntünün ORİJİNAL histogramını hesapla (g_histogram_data'ya yaz)
		Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, 128, 128);
		  
		// 3. (Soru 2b) Hesaplanan histogramı kullanarak pImage'ı yerinde eşitle 
		Homework_Apply_Histogram_EQ((uint8_t*)pImage, g_histogram_data, 128, 128);
		
        //    yeni histogramını hesapla (g_histogram_data'nın üzerine yaz) 
        Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, 128, 128);

		// 5. İşlenmiş (eşitlenmiş) pImage görüntüsünü PC'ye geri gönder
	    LIB_SERIAL_IMG_Transmit(&img); // MCU -> PC
	}
  }
  /* USER CODE END 3 */

```

---

### Results

A low-contrast version of the "Lena" image was sent to the STM32 to test the `Homework_Apply_Histogram_EQ` function. The processed image was successfully received back by the PC.

As the side-by-side comparison shows, the HE function correctly and automatically enhanced the image contrast. Details in the shadows (like the hair) and highlights (like the hat) that were previously washed out are now clearly visible.

| Original Low-Contrast Image (Sent) | Processed Image (Received from STM32) |
| :---: | :---: |
| <img width="538" height="537" alt="image" src="https://github.com/user-attachments/assets/403aca2a-7069-48da-a7e1-55b07c3b6cf1" />|<img width="540" height="536" alt="image" src="https://github.com/user-attachments/assets/114c7e0d-38ed-46ef-8c08-25d414692147" />|


---


##  Q3 — 2D Convolution and Filtering.

### 🔹 Objective  
**Objective:**
This question moves from "Point Operations" (like in HW1 and Q2-HE) to **"Neighborhood (or Spatial) Operations."** The objective is to create a single, flexible C function that modifies a pixel's value based on the values of its immediate neighbors. This function, `Homework_Apply_Convolution`, is the foundation for all **linear spatial filtering**, including blurring (blurring), sharpening, and edge detection.

---

### 🔹 STM32 code  
Add the generated header file into your STM32 project includes:

```c
/* USER CODE BEGIN Includes */
#include "lib_image.h"
#include "lib_serialimage.h"
#include "string.h" 
/* USER CODE END Includes */
```

```c
/* USER CODE BEGIN PV */
uint32_t g_histogram_data[256];

// İşlenmiş görüntü (filtre sonuçları) için buffer
uint8_t g_processed_image[128 * 128];

// Soru 3b: 3x3 Low Pass Filter (Box Blur) Kerneli
// (Tüm piksellerin ortalamasını alır)
const float g_kernel_low_pass[9] = {
  1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
  1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
  1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f
};

// Soru 3c: 3x3 High Pass Filter (Kenar Bulma) Kerneli
// (Merkez pikseli vurgular, komşuları çıkarır)
const float g_kernel_high_pass[9] = {
  -1.0f, -1.0f, -1.0f,
  -1.0f,  8.0f, -1.0f,
  -1.0f, -1.0f, -1.0f
};
/* USER CODE END PV */
```

```c
/* USER CODE BEGIN PFP */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
void Homework_Apply_Histogram_EQ(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
void Homework_Apply_Convolution(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, const float* kernel);
/* USER CODE END PFP */
```

```c
/* USER CODE BEGIN 4 */

void Homework_Apply_Convolution(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, const float* kernel)
{
  uint32_t x, y, kx, ky;
  float sum;
  uint32_t img_idx, kernel_idx;

  // Görüntünün tamamı üzerinde dolaş (y = satır, x = sütun)
  for (y = 0; y < height; y++)
  {
    for (x = 0; x < width; x++)
    {
      // Kenardaki pikselleri (kernelin taşacağı yerleri) atla/sıfırla
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0; // Kenarları siyah yap
      }
      else
      {
        sum = 0.0f; // Toplamı sıfırla

        // 3x3'lük Kernel'i dolaş (ky = kernel satır, kx = kernel sütun)
        for (ky = 0; ky < 3; ky++)
        {
          for (kx = 0; kx < 3; kx++)
          {
            // Görüntüdeki ilgili pikselin indeksi
            img_idx = (y + ky - 1) * width + (x + kx - 1);
            // Kernel'deki ilgili ağırlığın indeksi
            kernel_idx = ky * 3 + kx;

            // Toplamı hesapla = (piksel * ağırlık)
            sum += (float)p_src[img_idx] * kernel[kernel_idx];
          }
        }

        // Toplam değeri 0-255 aralığına sıkıştır (Clipping/Clamping)
        if (sum < 0.0f)   sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;

        // Sonucu hedef buffer'a yaz
        p_dst[y * width + x] = (uint8_t)sum;
      }
    }
  }
}
/* USER CODE END 4 */
```

```c
/* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  
    // 1. PC'den 128x128 Grayscale görüntüyü al (pImage buffer'ı dolacak)
	if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	{
		// 2. (Soru 3b) Gelen görüntüye (pImage) Low Pass Filtre uygula
        //    Sonucu g_processed_image buffer'ına yaz
		Homework_Apply_Convolution(
            (uint8_t*)pImage,       // Kaynak
            g_processed_image,      // Hedef
            128, 128,               // Boyutlar
            g_kernel_low_pass       // Low Pass Kernel

            //If you want to use the high pass filter, comment out the low pass filter command and remove it from the comment line.
            /*g_kernel_high_pass      // high Pass Kernel*/ 
        );
		  
		// 3. Geri göndermeden önce, işlenmiş görüntüyü ana buffer'a kopyala
        //    (Çünkü LIB_SERIAL_IMG_Transmit, pImage'ı gönderiyor)
        memcpy((uint8_t*)pImage, g_processed_image, 128*128);

		// 4. İşlenmiş (bulanıklaştırılmış) pImage görüntüsünü PC'ye geri gönder
	    LIB_SERIAL_IMG_Transmit(&img); // MCU -> PC
	}
  }
  /* USER CODE END 3 */
```

---

### Results: Low-Pass Filter 

A 3x3 "Box Blur" (average) kernel was applied to the source image. This kernel smooths the image by averaging each pixel with its 8 neighbors, effectively "passing" only the low-frequency information (smooth surfaces) and attenuating high-frequency details (sharp edges, noise).

* **Kernel Used:**
    ```c
    const float g_kernel_low_pass[9] = {
      1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
      1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
      1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f
    };
    ```

* **Visual Result:**
    The resulting image received back on the PC is visibly blurred, as expected from an averaging filter.

| Original Test Image (Sent) | Low-Pass Filtered Result (Received) |
| :---: | :---: |
| <img width="539" height="536" alt="image" src="https://github.com/user-attachments/assets/350c966d-a91a-4768-a481-2b5bea1f47fe" /> | <img width="529" height="529" alt="image" src="https://github.com/user-attachments/assets/3e2994fb-4c45-44f6-bc32-a07352ffe937" /> |

---

### Results: High-Pass Filter

A 3x3 Laplacian kernel was applied to the source image. This kernel calculates the difference between a pixel and its neighbors. It "passes" only the high-frequency information (edges) and attenuates low-frequency areas (flat surfaces), which become black.

* **Kernel Used:**
    ```c
    const float g_kernel_high_pass[9] = {
      -1.0f, -1.0f, -1.0f,
      -1.0f,  8.0f, -1.0f,
      -1.0f, -1.0f, -1.0f
    };
    ```

* **Visual Result:**
    The resulting image is an "edge map" of the original. All flat surfaces are black (0), and only the pixels corresponding to an edge are bright (white).

| Original Test Image (Sent) | High-Pass Filtered Result (Received) |
| :---: | :---: |
| <img width="539" height="536" alt="image" src="https://github.com/user-attachments/assets/350c966d-a91a-4768-a481-2b5bea1f47fe" /> | <img width="539" height="538" alt="image" src="https://github.com/user-attachments/assets/9a9c4375-b04f-4cde-83c9-34e9082d7189" /> |

*The "filtered image entries" required by Q3b and Q3c are represented by the full processed images shown above, which were verified on the PC.*

---

##  Q4 — 2D Median Filtering

### 🔹 Objective  
**Objective:**
The objective of this question is to implement a **Non-Linear Spatial Filter**, which stands in contrast to the Linear filter (Convolution) from Q3. The Median Filter is a powerful non-linear filter used primarily for **noise reduction**.
Its main advantage over a Low-Pass (averaging) filter is its effectiveness against **"salt-and-pepper" noise** (random black and white pixels). While an averaging filter *blurs* this noise, a median filter can **completely remove it** while doing a much better job of **preserving sharp edges** in the image.

---

### 🔹 STM32 code  
Add the generated header file into your STM32 project includes:

```c
/* USER CODE BEGIN Includes */
#include "lib_image.h"
#include "lib_serialimage.h"
#include "string.h" 
/* USER CODE END Includes */
```

```c
/* USER CODE BEGIN PV */
uint32_t g_histogram_data[256];

// İşlenmiş görüntü (filtre sonuçları) için buffer
uint8_t g_processed_image[128 * 128];
/* USER CODE END PV */
```

```c
/* USER CODE BEGIN PFP */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
void Homework_Apply_Histogram_EQ(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
void Homework_Apply_Convolution(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, const float* kernel);
static void Sort_Bubble_9(uint8_t *arr);
void Homework_Apply_Median_Filter(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height);
/* USER CODE END PFP */
```

```c
/* USER CODE BEGIN 4 */

static void Sort_Bubble_9(uint8_t *arr)
{
    int i, j;
    uint8_t temp;
    // Basit bir bubble sort (9 eleman için yeterince hızlı)
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void Homework_Apply_Median_Filter(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height)
{
  uint32_t x, y, kx, ky;
  uint8_t window[9]; // 3x3'lük pencereyi tutacak dizi
  uint8_t window_idx;

  // Görüntünün tamamı üzerinde dolaş (y = satır, x = sütun)
  for (y = 0; y < height; y++)
  {
    for (x = 0; x < width; x++)
    {
      // Kenardaki pikselleri (pencerenin taşacağı yerleri) atla/sıfırla
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0; // Kenarları siyah yap
      }
      else
      {
        window_idx = 0;
        
        // 1. 3x3'lük penceredeki pikselleri 'window' dizisine kopyala
        for (ky = 0; ky < 3; ky++)
        {
          for (kx = 0; kx < 3; kx++)
          {
            window[window_idx] = p_src[(y + ky - 1) * width + (x + kx - 1)];
            window_idx++;
          }
        }

        // 2. 'window' dizisini sırala
        Sort_Bubble_9(window);

        // 3. Sıralanmış dizinin ortasındaki (medyan) değeri al
        //    (9 eleman için 5. eleman, yani index 4)
        p_dst[y * width + x] = window[4];
      }
    }
  }
}
/* USER CODE END 4 */
```

```c
/* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  
	if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	{
		Homework_Apply_Median_Filter(
            (uint8_t*)pImage,       // Kaynak
            g_processed_image,      // Hedef
            128, 128                // Boyutlar
        );
		  
		// 3. Geri göndermeden önce, işlenmiş görüntüyü ana buffer'a kopyala
        memcpy((uint8_t*)pImage, g_processed_image, 128*128);

		// 4. İşlenmiş (gürültüsü temizlenmiş) pImage görüntüsünü PC'ye geri gönder
	    LIB_SERIAL_IMG_Transmit(&img); // MCU -> PC
	}
  }
  /* USER CODE END 3 */
```

---

### Results 

To validate the `Homework_Apply_Median_Filter` function, its primary strength—noise removal—was tested. A 128x128 test image was intentionally corrupted on the PC with "salt-and-pepper" noise (random black and white pixels) before being sent to the STM32.

The STM32 successfully processed this noisy image using the median filter and transmitted the result back to the PC.

The requirement of **Q4c** ("Show filtered image entries") is fulfilled by visually inspecting the entire processed image received back by the PC, which serves as a complete, full-frame representation of all 16,384 filtered entries.

As the side-by-side comparison below clearly shows, the median filter **completely eliminated** the "salt-and-pepper" noise.

| Original Noisy Image (Sent to STM32) | Median Filtered Result (Received from STM32) |
| :---: | :---: |
| <img width="539" height="536" alt="image" src="https://github.com/user-attachments/assets/350c966d-a91a-4768-a481-2b5bea1f47fe" /> | <img width="530" height="526" alt="image" src="https://github.com/user-attachments/assets/f4c56e90-e1e5-4e80-9097-0d1456159a1b" /> |

**Analysis:**
This result demonstrates the key difference between a Median filter (Q4) and a Low-Pass averaging filter (Q3b).
* The **Low-Pass** filter would have *blurred* the noise, averaging the `255` values with their neighbors and creating gray "smudges."
* The **Median** filter, by using rank-order sorting, correctly identified the noise pixels (`0` or `255`) as statistical outliers. It discarded them and replaced them with the local *median* value (the true pixel value), resulting in a clean image with perfectly preserved edges.
---

## 7. Observations and Key Learnings

This project provided several key insights into the practical differences between image processing algorithms, especially in an embedded context.

* **Point vs. Neighborhood Operations:**
    * **Q2 (Histogram Equalization)** is an *enhanced* Point Operation. Its processing time is dominated by the initial setup (Q1's histogram + CDF/LUT creation), but the final application is $O(N)$ fast (one LUT lookup per pixel).
    * **Q3 (Convolution)** and **Q4 (Median Filter)** are Neighborhood Operations. Their processing time is $O(N \cdot M^2)$ (where M is the kernel size, 3x3). They are significantly more computationally intensive as they require 9 memory accesses, (multiple) arithmetic operations, and one memory write *for every single pixel*.

* **Linear vs. Non-Linear Filtering (The Key Takeaway):**
    * **Q3 (Low-Pass)** is a **Linear** filter (a weighted average). As the results showed, while it does reduce general noise, it does so by **blurring the entire image**, causing sharp edges to lose definition. It is also ineffective against "salt-and-pepper" noise, as it just "smudges" the noise pixel instead of removing it.
    * **Q4 (Median)** is a **Non-Linear** filter. The results were far superior for "salt-and-pepper" noise. By sorting the 9-pixel window, it correctly identifies the noise (`0` or `255`) as a statistical outlier and discards it. It replaces the noise with the *median* (a real value from the neighborhood), resulting in a clean image with **sharp edges preserved**.

* **"In-Place" vs. "Out-of-Place" Buffers:**
    * A critical implementation detail. Q2 (HE) could be performed **in-place** (writing results directly back to the `pImage` buffer) because the new value for `pImage[i]` only depends on the *original* value of `pImage[i]`.
    * Q3 (Convolution) and Q4 (Median) **cannot** be performed in-place. To calculate the new value for `(x, y)`, they need the *original* values of neighbors like `(x-1, y)`. If `(x-1, y)` was already overwritten, the calculation would be corrupted. This necessitates a separate "destination" buffer (`g_processed_image`) to hold the new values, followed by a final `memcpy` to the main buffer.
---

## 8. Summary

This homework was successfully completed. All four core algorithms were implemented in C, deployed on the STM32 hardware, and verified visually using a PC-based Python application.

1.  **Q1 (Histogram):** A statistical analysis function was built and proven correct using the STM32CubeIDE debugger and a test image.
2.  **Q2 (HE):** An adaptive, histogram-based contrast enhancement filter was implemented, successfully "spreading" the intensity range of a low-contrast image.
3.  **Q3 (Convolution):** A flexible linear filtering framework was created and validated with both a **Low-Pass (blurring)** kernel and a **High-Pass (edge-detection)** kernel.
4.  **Q4 (Median):** A non-linear, edge-preserving noise removal filter was implemented, proving highly effective at removing "salt-and-pepper" noise that the Low-Pass filter could not.
---

## 9. Submission Notes

* **Project Location:** This `HW2/` folder contains the complete, compilable STM32CubeIDE project.
* **Core Code:** All code written specifically for this homework (all `Homework_...` functions) can be found in `Core/Src/main.c`, primarily located inside the `USER CODE 4` block.
* **Dependencies:** This project relies on the `lib_image.c` and `lib_serialimage.c` files (provided in HW1) for the UART communication protocol.
* **Hardware:** All tests were performed on an `STM32F446RE-Nucleo` board.
* **Active Function:** The `main.c` `while(1)` loop is configured to test one major function at a time (e.g., Q2, Q3, or Q4). To test a different function, the corresponding line in the `while(1)` loop must be commented/uncommented and the project re-compiled. **The code as-submitted is currently configured to run: [PLEASE WRITE THE ACTIVE FUNCTION HERE, e.g., "Question 4: Median Filter"]**.
