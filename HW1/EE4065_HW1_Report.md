# 🧠 EE4065 – Embedded Digital Image Processing
### **Homework 1**
📅 **Due Date:** November 7, 2025 — 23:59  
👥 **Team Members:**  
- Taner Kahyaoğlu  
- Yusuf Zivaroğlu  

---

## 📘 Description  
This project demonstrates the process of converting an image into a grayscale `.h` header file using Python,  
then applying various pixel intensity transformations (Negative, Thresholding, Gamma, Piecewise Linear)  
on STM32 via STM32CubeIDE, and observing the results through the **Memory Window**.

---

## 🧩 Q1 — Grayscale Image Formation (40 pts)

### 🔹 Objective  
Convert a selected image into grayscale and store it as a `.h` header file to visualize in STM32 memory.

---

### 🔹 Python Code — *convert.py*
```python
from PIL import Image
import os

IMAGE_FILE = "lena_gray.png"  # Name of the image you will convert
OUTPUT_FILE = "image_data.h"  # Name of the output .h file
ARRAY_NAME = "my_image_data"  # Name of the C array
WIDTH = 128
HEIGHT = 128
# ----------------------

# 1. Open the image, convert to grayscale ('L' mode), and resize
img = Image.open(IMAGE_FILE).convert('L').resize((WIDTH, HEIGHT))

# 2. Get the pixels as a list
pixels = list(img.getdata())

# 3. Create the .h file and write into it
with open(OUTPUT_FILE, 'w') as f:
    f.write(f"// Image: {IMAGE_FILE}, Size: {WIDTH}x{HEIGHT}\n")
    f.write(f"unsigned char {ARRAY_NAME}[{len(pixels)}] = {{\n ")

    # Write pixels line by line for readability
    for i, pixel in enumerate(pixels):
        f.write(f"{pixel}, ")
        if (i + 1) % 16 == 0:  # 16 pixels per line
            f.write("\n ")

    f.write("\n};\n")

print(f"'{OUTPUT_FILE}' file was created successfully!")
```

---

### 🔹 Execution Steps  
1. Open **Command Prompt (CMD)**  
2. Install Pillow library:  
   ```bash
   pip install Pillow
   ```
3. Place your image (`lena_gray.png`) in the same directory as `convert.py`  
4. Run the script:  
   ```bash
   python convert.py
   ```
5. The file `image_data.h` will be generated automatically and placed under:  
   ```
   python image converter/
   ```
6. Move the generated `.h` file into STM32 project path:  
   ```
   HW1/STM32CubeIDE/Core/Inc/
   ```

---

### 🔹 Results  

🖼️ **Original Grayscale Image:**  
`python image converter/lena_gray.png`  
![Lena Gray](python%20image%20converter/lena_gray.png)

💾 **Memory Observation:**  
`results/my_image_data under the memory window.png`  
![Memory Window](results/my_image_data%20under%20the%20memory%20window.png)

---

## 🧩 Q2 — Intensity Transformations (60 pts)

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

#### 🧪 2a — Negative Image  
- **Description:** Inverts all pixel intensities → bright areas become dark and vice versa.  
📸 Result:  
`results/output_image under the memory window for negative intesity transformation.png`  
![Negative Transformation](results/output_image%20under%20the%20memory%20window%20for%20negative%20intesity%20transformation.png)

---

#### 🧪 2b — Thresholding  
- **Description:** If pixel intensity > threshold → WHITE, else BLACK.  
📸 Result:  
`results/output_image under the memory window for tresholding intesity transformation.png`  
![Thresholding Transformation](results/output_image%20under%20the%20memory%20window%20for%20tresholding%20intesity%20transformation.png)

---

#### 🧪 2c — Gamma Correction  
- **Description:** Adjust image brightness using γ = 3 and γ = 1/3.  

📸 Gamma = 3:  
`results/output_image under the memory window for Gamma correction with gamma being 3 intesity transformation.png`  
![Gamma 3](results/output_image%20under%20the%20memory%20window%20for%20Gamma%20correction%20with%20gamma%20being%203%20intesity%20transformation.png)

📸 Gamma = 1/3:  
`results/output_image under the memory window for Gamma correction with gamma being 1 over 3 intesity transformation.png`  
![Gamma 1/3](results/output_image%20under%20the%20memory%20window%20for%20Gamma%20correction%20with%20gamma%20being%201%20over%203%20intesity%20transformation.png)

---

#### 🧪 2d — Piecewise Linear  
- **Description:** Adjust contrast by defining two linear regions (below and above threshold).  
📸 Result:  
`results/output_image under the memory window for Piecewise linear intesity transformation.png`  
![Piecewise Linear](results/output_image%20under%20the%20memory%20window%20for%20Piecewise%20linear%20intesity%20transformation.png)

---

## 🧮 Observations  
- Pixel value distributions in **Memory Window** verified that transformations behaved correctly.  
- The system successfully mapped grayscale data to transformed arrays in STM32 memory.  
- All results corresponded with expected theoretical transformations.

---

## ✅ Summary  
- **Python:** Converted an image to grayscale and exported as `.h` array.  
- **STM32:** Applied transformations and verified results via debug memory inspection.  
- **Outcome:** Every transformation (2a–2d) visually confirmed in memory.  

---

## 📬 Submission Notes  
- This `README.md` file contains the complete project report, including all explanations, code, and result images.  
- Repository is private and shared only with the course instructors.
