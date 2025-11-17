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

##  Q1 — Histogram Formation (40 pts)

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

 <img width="128" height="128" alt="lena_gray" src="https://github.com/user-attachments/assets/5ec46846-03e8-44e6-8ccf-ce7760c08e89" />

 <img width="1145" height="812" alt="image" src="https://github.com/user-attachments/assets/837e55ab-f5bf-41ce-ad74-1da0ff2447bb" />

---

##  Q2 — Histogram Equalization

### 🔹 Objective  
**Objective:**
The goal of Histogram Equalization (HE) is to automatically improve the contrast of an image. It is an *adaptive* intensity transformation. Unlike a simple Gamma or Negative transform (from HW1), HE creates a custom transformation function (Look-Up Table or LUT) based on the image's *own* unique histogram. It "stretches" or "spreads" the most common pixel intensities across the entire available range (0-255).

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

##  Observations  
- Pixel value distributions in **Memory Window** verified that transformations behaved correctly.  
- The system successfully mapped grayscale data to transformed arrays in STM32 memory.  
- All results corresponded with expected theoretical transformations.

---

##  Summary  
- **Python:** Converted an image to grayscale and exported as `.h` array.  
- **STM32:** Applied transformations and verified results via debug memory inspection.  
- **Outcome:** Every transformation (2a–2d) visually confirmed in memory.  

---

##  Submission Notes  
- This `README.md` file contains the complete project report, including all explanations, code, and result images.  
- Repository is private and shared only with the course instructors.
