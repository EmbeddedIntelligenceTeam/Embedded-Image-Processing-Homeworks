#  EE4065 – Embedded Image Processing
### **Homework 1**
 **Due Date:** November 7, 2025 — 23:59  
 **Team Members:**  
- Taner Kahyaoğlu  
- Yusuf Zivaroğlu

---

##  Description
This repository contains the implementation, source codes, and documentation for **Homework 1** of the *EE4065 Embedded Image Processing* course.  
The project combines **Python-based image preprocessing** and **STM32 embedded execution** to perform grayscale image display and intensity transformation analysis.

---

##  Homework Tasks

### **Q1 – Grayscale Image Formation (40 points)**  
- Create a **grayscale image** on your PC with a suitable size.  
- Convert it to a **header file (`.h`)** containing the pixel array.  
- Add this header file to your STM32 project and observe the image data in memory.

---

### **Q2 – Intensity Transformations (60 points)**  
Implement and verify the following transformations:

| Part | Transformation Type | Description |
|------|---------------------|--------------|
| 2a | Negative Image | Invert all pixel intensities |
| 2b | Thresholding | Apply thresholding to create binary output |
| 2c | Gamma Correction | Perform gamma adjustment with γ = 3 and γ = 1/3 |
| 2d | Piecewise Linear | Implement piecewise linear mapping using threshold value |

 Results are observed using **Memory Window** under **STM32CubeIDE**.

---


##  Project Execution Workflow

This section explains the complete development and execution process step-by-step.

###  Step 1 — Preparing the Environment
1. Open **Command Prompt (CMD)**.  
2. Install the Python image library:
   ```bash
   pip install Pillow
   ```
3. Place the image you want to convert (e.g., `input_image.jpg`) in the same folder as `convert.py`.

---

###  Step 2 — Generating the Header File
1. Run the conversion script:
   ```bash
   python convert.py
   ```
2. The script creates:
   ```
   image_data.h
   ```
   — a header file containing the image’s grayscale pixel array.

---

###  Step 3 — STM32 Project Setup
1. Create a new STM32 project in **STM32CubeIDE**.  
2. Copy the generated `image_data.h` file into:
   ```
   Core/Inc/
   ```
3. Include the header in `main.c` and write the code to process or display pixel data.

---

###  Step 4 — Debugging and Memory Observation
1. Build and run the project in **Debug Mode**.  
2. Open the **Memory Window** in STM32CubeIDE.  
3. Observe the pixel intensity data directly from the MCU memory.

---

###  Step 5 — Question 2 (Intensity Transformations)
All transformation codes are included in the main source but **commented out** for clarity.

Within your `main.c`, find:
```c
/* USER CODE BEGIN 2 */
// 2a - Negative Image
// 2b - Thresholding
// 2c - Gamma Correction (γ = 3, γ = 1/3)
// 2d - Piecewise Linear Transformation
/* USER CODE END 2 */
```

To test a specific part:
1. Uncomment the relevant section (`2a`, `2b`, `2c`, or `2d`).  
2. Rebuild and run in **Debug Mode**.  
3. Observe corresponding transformations in the **Memory Window**.

---

##  Summary
- `convert.py` converts images into C-compatible header files.  
- STM32CubeIDE project loads and manipulates this image data.  
- Each question step can be verified visually through the memory view.  
- Commented code sections make it easy to switch between transformation types.

---

##  Submission Notes
- This GitHub repository is **private** and shared only with the course instructors.  
- Commit history shows incremental progress on both Python preprocessing and STM32 embedded tasks.
