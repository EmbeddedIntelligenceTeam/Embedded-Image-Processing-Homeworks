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
Otsu's method relies on intensity distribution, which is not directly available in a color image. The objective here is to adapt the algorithm for **RGB565** format. We convert the color image to a temporary grayscale representation to calculate the threshold, and then use that threshold to create a **Mask**.
* **Background ($Intensity \le T$):** Pixels are set to Black (`0x0000`).
* **Foreground ($Intensity > T$):** Pixels retain their original color.
## Implementation Setup
To enable color processing, the following configurations were updated in the code:

1.  **Buffer Expansion:** Since RGB565 uses 2 bytes per pixel, the buffer size was doubled in `main.c`:
    ```c
    volatile uint8_t pImage[IMG_PIXELS * 2]; // 128*128*2 bytes
    ```
2.  **Initialization:** The image structure was re-initialized for color format:
    ```c
    LIB_IMAGE_InitStruct(&img, (uint8_t*)pImage, IMG_HEIGHT, IMG_WIDTH, IMAGE_FORMAT_RGB565);
    ```
3.  **Active Logic:** The preprocessor directive for the Q2 block was set to `#if 1`.
4.  **PC Configuration:** The input file in `py_image.py` was set to `"sss_image.png"`.
## STM32 Implementation Details

### Bitwise Extraction & Masking
Since RGB565 packs pixel data into 16 bits (5 bits Red, 6 bits Green, 5 bits Blue), we use bit-shifting to extract components and calculate luminance.

```c
#if 1 // Q2 Active
if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK) // RGB565 Received
{
    uint16_t* p_src_rgb = (uint16_t*)pImage;

    // 1. Dimensionality Reduction: RGB565 -> Grayscale
    // We use a temporary buffer (g_tmp_image) to hold the intensity values.
    for (uint32_t i = 0; i < IMG_PIXELS; i++)
    {
        uint16_t pixel = p_src_rgb[i];
        
        // Extract components using bitmasks and shifts
        uint8_t r = ((pixel >> 11) & 0x1F) << 3; // Expand to 8-bit range
        uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
        uint8_t b = ((pixel)       & 0x1F) << 3;
        
        // Simple Average Method for Grayscale
        g_tmp_image[i] = (r + g + b) / 3;
    }

    // 2. Calculate Histogram & Threshold on the Grayscale Map
    Homework_Calculate_Histogram(g_tmp_image, g_histogram_data, IMG_WIDTH, IMG_HEIGHT);
    uint8_t otsu_T = Homework_Compute_Otsu_Threshold(g_histogram_data, IMG_PIXELS);

    // 3. Apply Mask to the Original RGB Image
    for (uint32_t i = 0; i < IMG_PIXELS; i++)
    {
        // Check intensity against threshold
        if (g_tmp_image[i] <= otsu_T)
        {
            p_src_rgb[i] = 0x0000; // Mask Background (Black)
        }
        // Else: Leave p_src_rgb[i] untouched (Foreground Color Preserved)
    }

    img.format = IMAGE_FORMAT_RGB565;
    LIB_SERIAL_IMG_Transmit(&img);
}
#endif
```

## Results

| Original Color Image | Otsu Masked Output |
| :---: | :---: |
| <img width="256" height="256" alt="sss_image" src="https://github.com/user-attachments/assets/d2bbd9b9-2686-4a05-944d-7baf2c4b3c70" />| <img width="256" height="256" alt="received_from_f446re" src="https://github.com/user-attachments/assets/d6bae0a7-2874-4ef8-a000-5209ef8f37e0" />|

---

# 5. Q3 — Morphological Operations

## Objective
Binary images often contain noise (small white specks) or imperfections (small black holes). Morphological operations process images based on shapes using a **Structuring Element (Kernel)**. We implemented a 3x3 square kernel to perform:

1.  **Dilation:** Sets the pixel value to the **maximum** of its neighbors. This expands white regions and fills holes ("pepper" noise).
2.  **Erosion:** Sets the pixel value to the **minimum** of its neighbors. This shrinks white regions and eliminates small white specks ("salt" noise).
3.  **Opening:** Erosion followed by Dilation. Removes noise from the background while maintaining the object's approximate size.
4.  **Closing:** Dilation followed by Erosion. Fills holes inside the object while maintaining the object's approximate size.

## STM32 Implementation Details

**Critical Note on Buffering:** Unlike point operations (like brightness adjustment), morphological operations depend on the values of neighboring pixels. Therefore, they cannot be performed "in-place". We must read from a source buffer and write the result to a separate destination buffer to avoid data corruption.

### Morphological Functions

```c
/* USER CODE BEGIN 4 */

// Dilation: Max Filter
void Homework_Dilation_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height)
{
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
       
       // Boundary checks omitted for brevity (handled in full code)
       
       uint8_t max_val = 0;
       // Iterate 3x3 Kernel
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

// Opening: Erosion -> Dilation
void Homework_Opening_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, uint8_t* p_tmp)
{
  // Step 1: Erode source into temp buffer
  Homework_Erosion_3x3(p_src, p_tmp, width, height); 
  // Step 2: Dilate temp buffer into destination buffer
  Homework_Dilation_3x3(p_tmp, p_dst, width, height); 
}
/* USER CODE END 4 */
```

## Results

| Original Noisy Binary | Dilation | Erosion | Opening |
|-----------------------|----------|---------|---------|
| ![Original](noisy_original.png) | ![Dilation](dilation.png) | ![Erosion](erosion.png) | ![Opening](opening.png) |

---

# 6. Observations and Key Learnings

* **Algorithmic Efficiency:** Otsu's method is computationally efficient ($O(N)$) for embedded systems as it relies on a single pass to build the histogram and a loop of 256 iterations to find the threshold, making it suitable for real-time applications on the STM32F446RE.
* **Memory Management:** Morphological operations require $O(N)$ auxiliary memory. On a microcontroller with limited RAM, managing these buffers (e.g., `g_tmp_image`) is critical. We reused the same buffer for different stages of the "Opening" operation to optimize usage.
* **Effect of Kernel Size:** While we used a 3x3 kernel, larger kernels (e.g., 5x5) would remove larger noise particles but would also cause more aggressive distortion (shrinking/expanding) of the main object features.
* **RGB565 Complexity:** Processing color images requires explicit handling of bit-fields. Masking demonstrates how we can use the result of a grayscale analysis (the threshold) to manipulate the original color data selectively.
