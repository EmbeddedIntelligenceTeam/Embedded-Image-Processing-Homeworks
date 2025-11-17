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

### Execution Steps

The test for Q2 (and Q3/Q4) uses the full "receive-process-transmit" architecture.

1.  **Prepare STM32 (`main.c`):**
    The `while(1)` loop is configured to run the full processing pipeline. It chains the functions from Q1 and Q2:
    ```c
    /* main.c - Full processing loop */
    while (1)
    {
      if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
      {
          // 1. Calculate original histogram
          Homework_Calculate_Histogram(pImage, g_histogram_data, ...);
          
          // 2. Apply HE function (function being tested)
          Homework_Apply_Histogram_EQ(pImage, g_histogram_data, ...);
          
          // 3. Transmit processed image back to PC
          LIB_SERIAL_IMG_Transmit(&img);
      }
    }
    ```

2.  **Start STM32:**
    The project is compiled and run on the board (either in Debug or normal Run mode).

3.  **Run Python Script (PC):**
    * The `test_stm32.py` script is executed on the PC, this time sending a **low-contrast** (faded or dark) test image.
    * The script sends the original image and then receives the 16,384 bytes of the processed image back from the STM32.

4.  **Verify Result (PC):**
    * The Python script displays the original and processed images side-by-side.
    * We visually confirm that the image received from the STM32 has significantly higher contrast than the original sent.

5.  **Verify Histogram (Q2c - IDE):**
    [cite_start]To verify **Q2c**[cite: 136, 137], the `main.c` loop is temporarily modified to run `Homework_Calculate_Histogram()` a second time, *after* `Homework_Apply_Histogram_EQ()`. The debugger is then paused, and the **Memory Browser** is used to inspect `g_histogram_data`, confirming the histogram is now "spread out".
---

### Results 

To validate the `Homework_Calculate_Histogram` function, a 128x128 solid black image was transmitted from the PC to the STM32.

| Original Test Image (Sent from PC) | Result: STM32CubeIDE Memory Browser |
| :---: | :---: |
| ![q1_test_black](<img width="128" height="128" alt="lena_gray" src="https://github.com/user-attachments/assets/d9728bd7-5d25-43da-aeff-9336a38da44b" />
) | ![q1_memory_result](<img width="1145" height="812" alt="image" src="https://github.com/user-attachments/assets/01069ca4-c3e0-4561-97dd-35abc2acc633" />
) |

---

####  2b — Thresholding  
- **Description:** If pixel intensity > threshold → WHITE, else BLACK.  
 Result:  
`results/output_image under the memory window for tresholding intensity transformation.png`  
![Thresholding Transformation](results/output_image%20under%20the%20memory%20window%20for%20tresholding%20intensity%20transformation.png)

---

####  2c — Gamma Correction  
- **Description:** Adjust image brightness using γ = 3 and γ = 1/3.  

 Gamma = 3:  
`results/output_image under the memory window for Gamma correction with gamma being 3 intensity transformation.png`  
![Gamma 3](results/output_image%20under%20the%20memory%20window%20for%20Gamma%20correction%20with%20gamma%20being%203%20intensity%20transformation.png)

 Gamma = 1/3:  
`results/output_image under the memory window for Gamma correction with gamma being 1 over 3 intensity transformation.png`  
![Gamma 1/3](results/output_image%20under%20the%20memory%20window%20for%20Gamma%20correction%20with%20gamma%20being%201%20over%203%20intensity%20transformation.png)

---

####  2d — Piecewise Linear  
- **Description:** Adjust contrast by defining two linear regions (below and above threshold).  
 Result:  
`results/output_image under the memory window for Piecewise linear intensity transformation.png`  
![Piecewise Linear](results/output_image%20under%20the%20memory%20window%20for%20Piecewise%20linear%20intensity%20transformation.png)

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
