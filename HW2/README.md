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

### 3.4. Results (Q1c)

To validate the `Homework_Calculate_Histogram` function, a 128x128 solid black image was transmitted from the PC to the STM32.

As shown in the table below, the function correctly processed the image. The STM32CubeIDE Memory Browser confirms that the `g_histogram_data` array's first element (`[0]`) holds the value **16384** (which is $128 \times 128$), and all other 255 counters remain at 0. This verifies that every pixel was counted correctly.

| Original Test Image (Sent from PC) | Result: STM32CubeIDE Memory Browser (Q1c) |
| :---: | :---: |
| [**<img width="128" height="128" alt="lena_gray" src="https://github.com/user-attachments/assets/73327c4e-e649-4958-b9a4-8dae367f1e11" />
** <br> *A 128x128 solid black test image.*] | [**<img width="1145" height="812" alt="image" src="https://github.com/user-attachments/assets/89fd5601-d2f2-4d9b-9beb-111ae2314410" />
** <br> *Memory Browser showing `g_histogram_data[0]` = 16384 (`0x4000`)*] |
---

##  Q2 — Intensity Transformations (60 pts)

### 🔹 Objective  
Implement and verify pixel intensity transformations in STM32CubeIDE by observing memory values.

---

### 🔹 STM32 Project Setup  
Add the generated header file into your STM32 project includes:

```c
/* USER CODE BEGIN Includes */
#include "image_data.h"
#include <math.h>
/* USER CODE END Includes */
```

---

### 🔹 STM32 Code (main.c)
```c
/* USER CODE BEGIN 2 */
volatile unsigned char dummy_pixel = my_image_data[0];

#define IMAGE_WIDTH  128
#define IMAGE_HEIGHT 128
#define IMAGE_SIZE   (IMAGE_WIDTH * IMAGE_HEIGHT)

unsigned char output_image[IMAGE_SIZE];

// 2a Negative Image Transformation
/* for (int i = 0; i < IMAGE_SIZE; i++) {
     output_image[i] = 255 - my_image_data[i];
 }*/

// 2b Thresholding Image Transformation
/*int threshold_value = 128;
for (int i = 0; i < IMAGE_SIZE; i++) {
    unsigned char r = my_image_data[i];
    if (r > threshold_value) {
        output_image[i] = 255; // greater than threshold → white
    } else {
        output_image[i] = 0;   // smaller → black
    }
}*/

// 2c Gamma Correction Transformation
/*float gamma_value = 1.0/3.0; // Gamma value
for (int i = 0; i < IMAGE_SIZE; i++) {
    float r_normalized = (float)my_image_data[i] / 255.0;
    float s_normalized = powf(r_normalized, gamma_value);
    output_image[i] = (unsigned char)(s_normalized * 255.0);
}*/

// 2d Piecewise Linear Transformation
/*int r1 = 80;
int s1 = 0;
int r2 = 170;
int s2 = 255;
float slope = (float)(s2 - s1) / (float)(r2 - r1);

for (int i = 0; i < IMAGE_SIZE; i++) {
    unsigned char r = my_image_data[i];
    if (r <= r1) {
        output_image[i] = s1;
    }
    else if (r >= r2) {
        output_image[i] = s2;
    }
    else {
        output_image[i] = (unsigned char)((float)(r - r1) * slope + (float)s1);
    }
}*/

// Prevent optimization
volatile unsigned char dummy_output_pixel = output_image[0];
/* USER CODE END 2 */
```

---

### 🔹 How to Run  
1. Uncomment one transformation block at a time (2a–2d).  
2. Build and run in **Debug Mode**.  
3. Open the **Memory Window** and monitor pixel value changes.  

---

### 🔹 Results  

####  2a — Negative Image  
- **Description:** Inverts all pixel intensities → bright areas become dark and vice versa.  
 Result:  
`results/output_image under the memory window for negative intensity transformation.png`  
![Negative Transformation](results/output_image%20under%20the%20memory%20window%20for%20negative%20intensity%20transformation.png)

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
