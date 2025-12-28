# EE4065 – Embedded Digital Image Processing
### **Homework 5**
**Due Date:** December 26, 2025  
**Team Members:**
- **Taner KAHYAOĞLU**
- **Yusuf ZİVAROĞLU**

**High Important** you should use python version 3.8 for this homework codes
---

# HW5 - EOC2: Keyword Spotting (KWS) on STM32 using TFLite Micro

This project demonstrates the end-to-end development of a Keyword Spotting (KWS) system to recognize spoken digits (0-9). The workflow includes data preprocessing, training a Multi-Layer Perceptron (MLP) model, and deploying it onto an STM32F446RE microcontroller using the **TensorFlow Lite Micro** framework. This implementation follows the methodology described in Chapter 12 of the reference textbook.

---

## 1. Feature Extraction (MFCC)

Raw audio data is high-dimensional and computationally expensive for microcontrollers. Therefore, **Mel-Frequency Cepstral Coefficients (MFCC)** are used to extract compact features that represent the human auditory system's perception of sound. To ensure mathematical consistency between the PC training environment and the MCU inference, the `cmsisdsp` library was utilized.

### Technical Specifications:
* **Dataset:** Free Spoken Digit Dataset (FSDD).
* **Sampling Rate:** 8000 Hz.
* **FFT Size:** 1024.
* **Mel Filters:** 20.
* **DCT Outputs:** 13.
* **Windowing:** Hamming window ($N=1024$).
* **Feature Vector:** Each audio recording is fixed to 2048 samples ($2 \times FFTSize$) and split into two halves. 13 MFCC coefficients are extracted from each half, resulting in a total input vector of **26 features**.



---

## 2. Model Architecture and Training

A Multi-Layer Perceptron (MLP) was designed using the Keras API to classify the extracted MFCC features.

### Model Summary:
* **Input Layer:** 26 Neurons.
* **Hidden Layer 1:** 100 Neurons (ReLU activation).
* **Hidden Layer 2:** 100 Neurons (ReLU activation).
* **Output Layer:** 10 Neurons (Softmax activation for digits 0-9).

### Training Hyperparameters:
* **Optimizer:** Adam ($1 \times 10^{-3}$ learning rate).
* **Loss Function:** Categorical Crossentropy.
* **Epochs:** 100.
* **Validation:** The dataset was split by speaker, using "yweweler" as the test set to evaluate cross-speaker generalization.



---

## 3. Model Conversion and Deployment

To deploy the trained model onto the STM32, the `.h5` Keras model must be converted into a format compatible with embedded memory.

### TFLite Conversion:
The model was first converted to a flatbuffer format using the `TFLiteConverter`.

```python
import tensorflow as tf
from keras.models import load_model

model = load_model("kws_mlp.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

---

## 4. Hardware Implementation (STM32 CubeIDE)

After converting the model into C arrays, the firmware was developed in STM32 CubeIDE to handle real-time audio capture, signal processing, and neural network inference.

### 4.1 Prerequisites & Libraries
The implementation relies on several custom libraries to abstract hardware complexity:

`lib_model.h`: Manages the TensorFlow Lite Micro interpreter, tensor arena, and inference execution.

`lib_audio.h`: Interface for the onboard microphone to capture raw PCM data.

`ks_feature_extraction.h`: Implements the MFCC algorithm (utilizing CMSIS-DSP) to match the training pipeline.

`mlp_fsdd_model.h`: Contains the exported C array of the trained TFLite model.

`lib_serial.h`: Handles high-speed data transmission to the PC via UART.


### 4.2 System Configuration

*Microcontroller*: STM32 (High-performance series).

*UART Baudrate*: 2,000,000 bps (configured for high-speed transmission of audio buffers and inference results).

*Tensor Arena Size*: 136 KB (allocated for TFLite Micro intermediate calculations).

*Audio Setup*: Raw recording downsampled to 8 kHz Mono to match the FSDD dataset specifications.
