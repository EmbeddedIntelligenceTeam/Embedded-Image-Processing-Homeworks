# EE4065 – Embedded Digital Image Processing
### **Homework 3**
**Due Date:** December 19, 2025  
**Team Members:**
- **Taner KAHYAOĞLU**
- **Yusuf ZİVAROĞLU**

---

# 1. Project Aim and Scope

The primary objective of this homework is to implement **advanced image segmentation** and **binary morphological processing** algorithms from scratch using C on an STM32F446RE microcontroller.

While previous assignments focused on linear filtering, this project shifts focus to **non-linear operations** and **statistical image analysis**. The specific goals are:
1.  **Automated Segmentation:** Moving beyond manual thresholding by implementing **Otsu’s Method** to mathematically determine the optimal separation between foreground and background.
2.  **Color Image Processing:** Applying segmentation logic to 16-bit RGB565 images by performing intensity extraction and masking.
3.  **Shape Manipulation:** Implementing morphological operators (Dilation, Erosion, Opening, Closing) to clean noise, fill holes, and refine the binary shapes produced by segmentation.

All processing is performed in real-time on **128×128** pixel images transferred via UART.

---

# 2. Project Architecture

### PC Side (Host)
The Python script acts as the control interface and visualization tool:
- **Preprocessing:** Loads standard image formats and resizes them to 128x128.
- **Serialization:** Converts image data into a raw byte stream (Grayscale or RGB565).
- **Communication:** Transmits data to the MCU via UART at **2 Mbps** baud rate.
- **Visualization:** Receives the processed stream, reconstructs the image, and displays it using OpenCV for verification.

### STM32 Side (Target)
The MCU functions as a dedicated image processing unit:
- **Data Acquisition:** Receives raw pixel data into a memory buffer.
- **Statistical Analysis:** Computes the image histogram for Otsu's method.
- **Algorithm Execution:** Runs the C implementation of thresholding and morphology.
- **Data Transmission:** Sends the modified buffer back to the PC.

---

# 3. Q1 — Otsu Thresholding (Grayscale)

## Objective
The goal is to convert a grayscale image into a binary image (0 or 255) without manually selecting a threshold value. We implement **Otsu’s Method**, a global thresholding algorithm that assumes the image histogram is bimodal (contains two classes: background and foreground). The algorithm iterates through all possible thresholds to find the one that maximizes the **Between-Class Variance** ($\sigma_B^2$).

## STM32 Implementation Details

### Function Prototypes

```c
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
uint8_t Homework_Compute_Otsu_Threshold(uint32_t* p_hist, uint32_t total_pixels);
void Homework_Apply_Threshold(uint8_t* p_gray, uint32_t width, uint32_t height, uint8_t threshold);
```

### Algorithm Logic
1.  **Histogram Calculation:** We traverse the image array to count the frequency of each pixel intensity (0-255).
2.  **Variance Maximization:** We iterate through every possible threshold $t$ from 0 to 255. For each $t$, we calculate:
    * Weight ($w$) of background and foreground.
    * Mean intensity ($\mu$) of background and foreground.
    * Between-Class Variance: $\sigma_B^2 = w_B w_F (\mu_B - \mu_F)^2$.
3.  **Threshold Selection:** The $t$ value that yields the highest $\sigma_B^2$ is selected as the optimal threshold.

```c
/* USER CODE BEGIN 4 */
uint8_t Homework_Compute_Otsu_Threshold(uint32_t* p_hist, uint32_t total_pixels)
{
    uint32_t sum_all = 0, sumB = 0, wB = 0, wF = 0;
    float max_var_between = 0.0f;
    uint8_t threshold = 0;

    // Calculate total moment
    for (uint32_t i = 0; i < 256; i++)
        sum_all += i * p_hist[i];

    for (uint32_t t = 0; t < 256; t++)
    {
        wB += p_hist[t];
        if (wB == 0) continue;

        wF = total_pixels - wB;
        if (wF == 0) break;

        sumB += t * p_hist[t];

        float mB = (float)sumB / wB;
        float mF = (float)(sum_all - sumB) / wF;

        // Calculate Between Class Variance
        float diff = mB - mF;
        float var_between = (float)wB * wF * diff * diff;

        if (var_between > max_var_between)
        {
            max_var_between = var_between;
            threshold = (uint8_t)t;
        }
    }
    return threshold;
}
/* USER CODE END 4 */
```

### Main Loop Execution

```c
#if 1 // Q1 Active
if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
{
    // 1. Compute Histogram from received pImage
    Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, IMG_WIDTH, IMG_HEIGHT);

    // 2. Determine Optimal Otsu Threshold
    uint8_t otsu_T = Homework_Compute_Otsu_Threshold(g_histogram_data, IMG_PIXELS);

    // 3. Apply Threshold (Binarization)
    Homework_Apply_Threshold((uint8_t*)pImage, IMG_WIDTH, IMG_HEIGHT, otsu_T);

    // 4. Send Binary Image back
    LIB_SERIAL_IMG_Transmit(&img);
}
#endif
```

## Results

| Original Grayscale Image | Otsu Binary Output |
| :---: | :---: |
| <img width="256" height="256" alt="cameraman" src="https://github.com/user-attachments/assets/f1c2cfc3-08cf-4200-a330-0a0b971bb1ba" /> | <img width="256" height="256" alt="received_from_f446re(otsu_gray)" src="https://github.com/user-attachments/assets/0860222f-f892-459f-9493-a631c625cf3e" />|

---

# 4. Q2 — Otsu Thresholding on Color Images (RGB565)

## Objective
The goal is to extend Otsu's method to color images using a **Multi-Channel Approach**. Instead of converting the image to grayscale and finding a single threshold, we analyze the **Red, Green, and Blue** channels independently. We calculate three separate thresholds ($T_R, T_G, T_B$) and binarize each channel individually before recombining them.

## Implementation Setup
To enable color processing, the logic in `main.c` adapts dynamically based on the active question block:

1.  **Global Buffer Size:** The global image buffer was defined to accommodate the larger size of RGB565 images (2 bytes per pixel).
    ```c
    volatile uint8_t pImage[IMG_PIXELS * 2]; // 128*128*2 bytes to support both modes
    ```

2.  **Context-Specific Initialization:** Inside the Q2 active block (`#if 1`), the image structure is explicitly initialized for the **RGB565** format, whereas for Q1 and Q3, it defaults to Grayscale.
    ```c
    // Inside the Q2 logic block:
    img.format = IMAGE_FORMAT_RGB565;
    LIB_IMAGE_InitStruct(&img, (uint8_t*)pImage, IMG_HEIGHT, IMG_WIDTH, IMAGE_FORMAT_RGB565);
    ```

3.  **PC Configuration:** The input file in the Python script (`py_image.py`) is set to the color test image:
    ```python
    TEST_IMAGE_FILENAME = "lena_color.png"
    ```

## STM32 Implementation Details

### Multi-Channel Histogram & Segmentation Logic
The code performs the following steps:
1.  **Extraction:** Iterates through the RGB565 buffer and extracts 8-bit R, G, and B components.
2.  **Histogram Analysis:** Builds three separate histograms (`histR`, `histG`, `histB`).
3.  **Thresholding:** Computes optimal Otsu thresholds for each channel independently.
4.  **Reconstruction:** Rebuilds the pixel by setting each channel to its maximum value if it exceeds the threshold, or 0 otherwise.

```c
#if 1 // Q2 Active: Multi-Channel Otsu
if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK) // RGB565 Received
{
    uint16_t* p_src_rgb = (uint16_t*)pImage;

    // 3 Separate Histograms (Static to prevent stack overflow)
    static uint32_t histR[256];
    static uint32_t histG[256];
    static uint32_t histB[256];

    // 1. Clear Histograms
    memset(histR, 0, 256 * sizeof(uint32_t));
    memset(histG, 0, 256 * sizeof(uint32_t));
    memset(histB, 0, 256 * sizeof(uint32_t));

    // 2. Build Histograms
    for (uint32_t i = 0; i < IMG_PIXELS; i++)
    {
        uint16_t pixel = p_src_rgb[i];
        // Extract 8-bit values
        uint8_t r = ((pixel >> 11) & 0x1F) << 3;
        uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
        uint8_t b = ((pixel)       & 0x1F) << 3;

        histR[r]++; histG[g]++; histB[b]++;
    }

    // 3. Compute 3 Separate Thresholds
    uint8_t T_red   = Homework_Compute_Otsu_Threshold(histR, IMG_PIXELS);
    uint8_t T_green = Homework_Compute_Otsu_Threshold(histG, IMG_PIXELS);
    uint8_t T_blue  = Homework_Compute_Otsu_Threshold(histB, IMG_PIXELS);

    // 4. Binarize Each Channel & Reconstruct
    for (uint32_t i = 0; i < IMG_PIXELS; i++)
    {
        uint16_t pixel = p_src_rgb[i];
        
        uint8_t r = ((pixel >> 11) & 0x1F) << 3;
        uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
        uint8_t b = ((pixel)       & 0x1F) << 3;

        // If channel > Threshold -> Set to MAX, else 0
        // RGB565 Max Values: R=31 (0x1F), G=63 (0x3F), B=31 (0x1F)
        uint16_t new_r = (r > T_red)   ? 0x1F : 0;
        uint16_t new_g = (g > T_green) ? 0x3F : 0;
        uint16_t new_b = (b > T_blue)  ? 0x1F : 0;

        // Pack back into RGB565
        p_src_rgb[i] = (new_r << 11) | (new_g << 5) | (new_b);
    }

    img.format = IMAGE_FORMAT_RGB565;
    LIB_SERIAL_IMG_Transmit(&img);
}
#endif
```

## Results

| Original Color Image | Multi-Channel Otsu Output |
| :---: | :---: |
|<img width="256" height="256" alt="lena_color" src="https://github.com/user-attachments/assets/c01eee8c-59b9-419a-855a-ca7809552927" />| <img width="256" height="256" alt="received_from_f446re(otsu_color3)" src="https://github.com/user-attachments/assets/d2e67bd2-0c5d-4371-8d2f-a9b41e039aff" />|

---

# 5. Q3 — Morphological Operations

## Objective
The goal is to implement fundamental morphological operations to process binary images. These operations are essential for cleaning noise, separating objects, or filling gaps.

**Process Flow:**
1.  **Input:** The PC sends the original **Grayscale** image (`cameraman.png`).
2.  **Internal Binarization:** The STM32 automatically calculates the Otsu threshold (reusing the logic from Q1) and converts the image to **Binary** in memory.
3.  **Morphology:** The selected operation (Dilation, Erosion, etc.) is applied to this binary data.
4.  **Output:** The processed binary image is sent back to the PC.

## STM32 Implementation Details

**Mechanism:** Since we cannot run all filters simultaneously, we use preprocessor directives (`#if 1` / `#if 0`) in `main.c` to select which operation to compile and execute.

**Buffering:** Morphological operations rely on neighboring pixels. We read from the source buffer (`pImage`) and write the result to a destination buffer (`g_processed_image`) to avoid data corruption during processing.

### Main Loop Execution (Selectable Operations)
The code block below demonstrates how the system receives the image, **first binarizes it using Otsu**, and then applies the active morphological filter.

```c
/* USER CODE BEGIN WHILE */
while (1)
{
    // 1. Receive Image
    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
         // 2. PRE-PROCESSING: Apply Otsu Thresholding
         // We must convert the grayscale image to binary before morphology.
         Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, IMG_WIDTH, IMG_HEIGHT);
         uint8_t otsu_T = Homework_Compute_Otsu_Threshold(g_histogram_data, IMG_PIXELS);
         Homework_Apply_Threshold((uint8_t*)pImage, IMG_WIDTH, IMG_HEIGHT, otsu_T);
         
         // At this point, 'pImage' holds the Binary Image (0 or 255).
         
         // 3. APPLY ACTIVE MORPHOLOGICAL FILTER
         // Only one block should be set to #if 1 at a time.

#if 1   // TEST: Dilation (Active)
         Homework_Dilation_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT);
         memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS); 
#endif

#if 0   // TEST: Erosion (Inactive)
         Homework_Erosion_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT);
         memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
#endif

#if 0   // TEST: Opening (Inactive)
         Homework_Opening_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT, g_tmp_image);
         memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
#endif

#if 0   // TEST: Closing (Inactive)
         Homework_Closing_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT, g_tmp_image);
         memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
#endif

         // 4. Send Final Result
         LIB_SERIAL_IMG_Transmit(&img);
    }
}
```

### Morphological Functions

#### 1. Dilation
**Logic:** Sets pixel to 255 if **any** neighbor is 255 (Max Filter).
**Effect:** Expands white regions, fills black holes.

```c
void Homework_Dilation_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height)
{
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
       uint8_t max_val = 0;
       for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
             uint8_t v = p_src[(y + ky) * width + (x + kx)];
             if (v > max_val) max_val = v;
          }
       }
       p_dst[y * width + x] = max_val;
    }
  }
}
```

#### 2. Erosion
**Logic:** Sets pixel to 255 only if **all** neighbors are 255 (Min Filter).
**Effect:** Shrinks white regions, removes white noise spots.

```c
void Homework_Erosion_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height)
{
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
       uint8_t min_val = 255; 
       for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
             uint8_t v = p_src[(y + ky) * width + (x + kx)];
             if (v < min_val) min_val = v;
          }
       }
       p_dst[y * width + x] = min_val;
    }
  }
}
```

#### 3. Opening & Closing
* **Opening:** Erosion -> Dilation. (Removes Salt Noise).
* **Closing:** Dilation -> Erosion. (Fills Pepper Noise).

```c
void Homework_Opening_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, uint8_t* p_tmp)
{
  Homework_Erosion_3x3(p_src, p_tmp, width, height); 
  Homework_Dilation_3x3(p_tmp, p_dst, width, height); 
}

void Homework_Closing_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, uint8_t* p_tmp)
{
  Homework_Dilation_3x3(p_src, p_tmp, width, height); 
  Homework_Erosion_3x3(p_tmp, p_dst, width, height); 
}
```

## Results

To clearly demonstrate the effects, the operations were applied to the **Cameraman** image (which was internally binarized by the STM32).

| Baseline (Otsu Only) | Dilation | 
| :---: | :---: | 
| <img width="256" height="256" alt="received_from_f446re(otsu_gray)" src="https://github.com/user-attachments/assets/ccd7f26b-72b6-46e6-ac12-0142c4a85404" /> | <img width="256" height="256" alt="received_from_f446re(dilation)" src="https://github.com/user-attachments/assets/bd3b5e1d-5391-4846-9c4f-00a0dc269ecc" /> |

| Baseline (Otsu Only) | Erosion |
| :---: | :---: |
| <img width="256" height="256" alt="received_from_f446re(otsu_gray)" src="https://github.com/user-attachments/assets/ccd7f26b-72b6-46e6-ac12-0142c4a85404" /> | <img width="256" height="256" alt="received_from_f446re(erosion)" src="https://github.com/user-attachments/assets/30d37bf6-8db3-4f89-a3db-7d7ed186bc85" /> |

| Baseline (Otsu Only) | Opening |
| :---: | :---: |
| <img width="256" height="256" alt="received_from_f446re(otsu_gray)" src="https://github.com/user-attachments/assets/ccd7f26b-72b6-46e6-ac12-0142c4a85404" /> | <img width="256" height="256" alt="received_from_f446re(opening)" src="https://github.com/user-attachments/assets/9888a595-d84b-4c4a-9d08-ac63e0575717" /> |

| Baseline (Otsu Only) | Closing |
| :---: | :---: |
| <img width="256" height="256" alt="received_from_f446re(otsu_gray)" src="https://github.com/user-attachments/assets/ccd7f26b-72b6-46e6-ac12-0142c4a85404" /> | <img width="256" height="256" alt="received_from_f446re(closing)" src="https://github.com/user-attachments/assets/b3d8279a-af53-4169-869a-db8552ebeaa2" /> |

---

# 6. Observations and Key Learnings

* **Adaptive Thresholding (Otsu):** Unlike manual thresholding which fails under varying lighting conditions, Otsu's method demonstrated robustness by automatically maximizing the inter-class variance. This proves highly effective for automating segmentation tasks in embedded vision applications.
* **Non-Linear vs. Linear Processing:** In HW2, linear filters (Convolution) preserved the overall energy of the image. In contrast, Morphological operations (Q3) are non-linear and destructive; they intentionally add or remove pixels based on geometric shapes. This distinction is critical: filtering is for *enhancing* images, while morphology is for *analyzing and cleaning* shapes.
* **Memory Constraints & Buffering:** A critical embedded system lesson was the necessity of **Double Buffering**. Since morphological operations require the original values of neighboring pixels to compute the current pixel, in-place modification would propagate errors. We utilized `g_processed_image` (and `g_tmp_image` for compound operations) to handle this, highlighting the trade-off between algorithm complexity and RAM usage ($O(N)$ auxiliary space).
* **Color Space Manipulation:** Masking in Q2 reinforced the concept that processing color images often involves dimensionality reduction (RGB -> Grayscale) for analysis (thresholding), followed by re-application to the original high-dimensional data (Masking), a common pattern in computer vision pipelines.
