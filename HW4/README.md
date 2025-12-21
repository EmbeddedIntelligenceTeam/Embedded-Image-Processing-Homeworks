# EE4065 – Embedded Digital Image Processing
### **Homework 3**
**Due Date:** December 19, 2025  
**Team Members:**
- **Taner KAHYAOĞLU**
- **Yusuf ZİVAROĞLU**

---

---

# 7. Application 1 — Single Neuron Classifier with Hu Moments

## 7.1 Objective
In this application, we shift focus from classical image processing to **TinyML (Embedded Machine Learning)**. The goal is to build a lightweight classifier capable of recognizing handwritten digits (specifically distinguishing the digit **'0'** from all other digits) using the **MNIST** dataset.

Directly processing raw images (28x28 = 784 pixels) is computationally expensive for resource-constrained microcontrollers. To address this, we implement a **Feature Extraction** pipeline using **Hu Moments**. This allows us to reduce the dimensionality of the input data from **784 raw pixels** down to just **7 floating-point values**, enabling the use of a **Single Neuron (Perceptron)** model for classification.

## 7.2 Theory: Hu Moments & Dimensionality Reduction
**Hu Moments** are a set of 7 statistical moments calculated from the image intensity function. They are critical for this application because they are **invariant** to:
1.  **Translation** (Position of the digit doesn't matter).
2.  **Scale** (Size of the digit doesn't matter).
3.  **Rotation** (Angle of the digit doesn't matter).

By feeding these 7 invariant features into the neural network instead of raw pixels, we drastically reduce the model size and processing time, making it suitable for embedded implementation.

## 7.3 Implementation Details

### Data Preparation & Feature Extraction
We utilize the **OpenCV** library to calculate the Hu Moments for every image in the MNIST training and test sets.
* **Input:** 28x28 Grayscale Image.
* **Process:** `cv2.moments()` $\rightarrow$ `cv2.HuMoments()`.
* **Output:** Vector of size [7].

### Normalization
Since Hu Moments can have vastly different ranges (some are very small, others large), we apply **Z-score Normalization** (Standard Standardization) to ensure stable convergence during training.
$$X_{norm} = \frac{X - \mu}{\sigma}$$
The mean ($\mu$) and standard deviation ($\sigma$) are calculated from the training set and applied to the test set.

### Model Architecture (Single Neuron)
We design the simplest possible neural network: a **Single Neuron** with a **Sigmoid** activation function. This effectively acts as a Logistic Regression classifier.
* **Input Layer:** 7 Nodes (Hu Moments).
* **Dense Layer:** 1 Node (Sigmoid Activation).
* **Loss Function:** Binary Crossentropy.
* **Optimizer:** Adam ($lr=0.001$).

### Handling Class Imbalance
The dataset is modified to a **Binary Classification** problem:
* **Class 0:** Represents the digit '0'.
* **Class 1:** Represents all other digits ('1' through '9').

Since Class 1 is roughly 9 times more frequent than Class 0, we introduce **Class Weights** (`{0:8, 1:1}`) during training to penalize misclassifying the minority class (digit '0') more heavily.

## 7.4 Python Implementation Code

```python
import os 
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras
from matplotlib import pyplot as plt

# 1. Load MNIST Data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 2. Feature Extraction (Hu Moments)
print("Extracting Hu Moments...")
train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True) 
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True) 
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# 3. Data Normalization (Manual Standardization)
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# 4. Define Single Neuron Model
model = keras.models.Sequential([
  keras.layers.Dense(1, input_shape=[7], activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.BinaryAccuracy()])

# 5. Prepare Binary Labels (0 vs Rest) & Train
train_labels[train_labels != 0] = 1 # 0 stays 0, others become 1
test_labels[test_labels != 0] = 1

print("Training Single Neuron Perceptron...")
model.fit(train_huMoments,
          train_labels, 
          batch_size=128, 
          epochs=50, 
          class_weight={0:8, 1:1}, # Handle Imbalance
          verbose=1)

# 6. Evaluation
perceptron_preds = model.predict(test_huMoments)

# 7. Visualization
conf_matrix = confusion_matrix(test_labels, perceptron_preds > 0.5)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot()
cm_display.ax_.set_title("Single Neuron Classifier Confusion Matrix")
plt.show()

# Save model for potential MCU deployment
model.save("hdr_perceptron.h5")
