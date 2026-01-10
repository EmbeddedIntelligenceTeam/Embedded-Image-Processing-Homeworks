"""
EE4065 Homework 6 - Automated Screenshot Capture
Saves test results as image files.
"""

import os
import sys

os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'

import serial
import serial.tools.list_ports
import numpy as np
import cv2
import time
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

# ============================================================================
# Configuration
# ============================================================================
PORT = 'COM8'
BAUD_RATE = 115200
TIMEOUT = 10

SYNC_BYTE = 0xAA
ACK_BYTE = 0x55

IMAGE_SIZE = 32
NUM_CLASSES = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_DIR = os.path.join(SCRIPT_DIR, "..", "tflite_models")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "Images")

# Global variables
mnist_images = None
mnist_labels = None
tflite_interpreters = {}

MODEL_INFO = {
    'SqueezeNetMini': {'size_kb': 54.2, 'fits_f446': True, 'recommended': True, 'status': 'WORKING'},
    'MobileNetV2Mini': {'size_kb': 106.5, 'fits_f446': True, 'recommended': True, 'status': 'WORKING'},
    'ResNet8': {'size_kb': 96.3, 'fits_f446': True, 'recommended': False, 'status': 'Op Error'},
    'EfficientNetMini': {'size_kb': 108.5, 'fits_f446': True, 'recommended': False, 'status': 'Op Error'},
    'ResNet14': {'size_kb': 201.7, 'fits_f446': True, 'recommended': False, 'status': 'Op Error'},
}

# ============================================================================
# Initialization
# ============================================================================

def init_models():
    global mnist_images, mnist_labels, tflite_interpreters
    
    print("Loading MNIST dataset...")
    (_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    mnist_images = []
    for img in test_images:
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mnist_images.append(img_resized)
    mnist_images = np.array(mnist_images)
    mnist_labels = test_labels
    print(f"✓ {len(mnist_images)} test images loaded")
    
    print("\nLoading TFLite models...")
    for model_name, info in MODEL_INFO.items():
        if info['fits_f446']:
            tflite_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")
            if os.path.exists(tflite_path):
                try:
                    interpreter = tf.lite.Interpreter(
                        model_path=tflite_path,
                        num_threads=1,
                        experimental_delegates=[]
                    )
                    interpreter.allocate_tensors()
                    tflite_interpreters[model_name] = interpreter
                    print(f"  ✓ {model_name} ({info['size_kb']:.1f} KB)")
                except Exception as e:
                    print(f"  ✗ {model_name}: {e}")

# ============================================================================
# PC Inference
# ============================================================================

def pc_inference(image, model_name='SqueezeNetMini'):
    if model_name not in tflite_interpreters:
        return None, None
    
    interpreter = tflite_interpreters[model_name]
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    img = image.astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    img = np.expand_dims(img, axis=0)
    
    if input_details['dtype'] == np.uint8:
        scale, zero_point = input_details['quantization']
        img = (img / scale + zero_point).astype(np.uint8)
    
    interpreter.set_tensor(input_details['index'], img)
    
    try:
        start = time.perf_counter()
        interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000
    except RuntimeError:
        return None, None
        
    output = interpreter.get_tensor(output_details['index'])
    
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    
    return output[0], elapsed

# ============================================================================
# STM32 Communication
# ============================================================================

def send_image_get_prediction(ser, image):
    try:
        ser.reset_input_buffer()
        
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        image_flat = image.flatten()
        ser.write(bytes([SYNC_BYTE]))
        
        start = time.time()
        while time.time() - start < 2:
            if ser.in_waiting > 0:
                ack = ser.read(1)
                if len(ack) > 0 and ack[0] == ACK_BYTE:
                    break
        else:
            return None, None, "ACK timeout"
        
        ser.write(image_flat.tobytes())
        
        start = time.time()
        response = b''
        expected_size = NUM_CLASSES
        
        while len(response) < expected_size and time.time() - start < 5:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting)
            time.sleep(0.01)
        
        if len(response) < expected_size:
            return None, None, f"Incomplete response ({len(response)} bytes)"
        
        predictions = np.frombuffer(response[:expected_size], dtype=np.uint8)
        
        time_data = b''
        start = time.time()
        while len(time_data) < 4 and time.time() - start < 1:
            if ser.in_waiting > 0:
                time_data += ser.read(min(4 - len(time_data), ser.in_waiting))
            time.sleep(0.01)
        
        if len(time_data) >= 4:
            inference_time = struct.unpack('<I', time_data[:4])[0]
        else:
            inference_time = 0
        
        return predictions, inference_time, None
        
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# Image Saving Functions
# ============================================================================

def save_single_test_image(image, true_label, pc_pred, pc_time, mcu_preds, mcu_time, model_name, filename):
    """Save single image test result."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'EE4065 HW6 - Single Image Test ({model_name})\nYusuf Zivaroğlu & Taner Kahyaoğlu', 
                 fontsize=14, fontweight='bold')
    
    # Left: Image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'True Label: {true_label}', fontsize=12)
    axes[0].axis('off')
    
    # Middle: Prediction bars
    ax1 = axes[1]
    mcu_pred = np.argmax(mcu_preds)
    colors = ['#2ecc71' if i == mcu_pred else '#3498db' for i in range(10)]
    if mcu_pred != true_label:
        colors[true_label] = '#e74c3c'
    
    bars = ax1.barh(range(10), mcu_preds, color=colors)
    ax1.set_yticks(range(10))
    ax1.set_yticklabels([str(i) for i in range(10)])
    ax1.set_xlabel('Confidence Score')
    ax1.set_title('MCU (STM32) Predictions', fontsize=12)
    ax1.invert_yaxis()
    
    # Right: Result summary
    ax2 = axes[2]
    ax2.axis('off')
    
    mcu_correct = "✓ CORRECT" if mcu_pred == true_label else "✗ WRONG"
    pc_correct = "✓ CORRECT" if pc_pred == true_label else "✗ WRONG"
    
    result_text = f"""
    ╔══════════════════════════════════════╗
    ║         RESULT COMPARISON            ║
    ╠══════════════════════════════════════╣
    ║  True Value:       {true_label}                   ║
    ╟──────────────────────────────────────╢
    ║  PC Prediction:    {pc_pred} {pc_correct:12s}  ║
    ║  PC Time:          {pc_time:.2f} ms            ║
    ╟──────────────────────────────────────╢
    ║  MCU Prediction:   {mcu_pred} {mcu_correct:12s}  ║
    ║  MCU Time:         {mcu_time} ms               ║
    ╟──────────────────────────────────────╢
    ║  PC-MCU Match:     {'YES ✓' if pc_pred == mcu_pred else 'NO ✗':12s}      ║
    ╚══════════════════════════════════════╝
    """
    ax2.text(0.1, 0.5, result_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")
    return filepath

def save_batch_test_image(results, model_name, filename):
    """Save batch test result."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'EE4065 HW6 - Batch Test Results ({model_name})\nYusuf Zivaroğlu & Taner Kahyaoğlu', 
                 fontsize=14, fontweight='bold')
    
    correct_mcu = sum(1 for r in results if r['mcu_pred'] == r['true_label'])
    correct_pc = sum(1 for r in results if r['pc_pred'] == r['true_label'])
    total = len(results)
    
    # Top left: Accuracy chart
    ax1 = axes[0, 0]
    categories = ['PC', 'MCU (STM32)']
    accuracies = [100*correct_pc/total, 100*correct_mcu/total]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(categories, accuracies, color=colors)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%', ha='center', fontweight='bold')
    
    # Top right: Inference time chart
    ax2 = axes[0, 1]
    avg_pc_time = np.mean([r['pc_time'] for r in results if r['pc_time'] is not None])
    avg_mcu_time = np.mean([r['mcu_time'] for r in results if r['mcu_time'] is not None])
    times = [avg_pc_time, avg_mcu_time]
    bars = ax2.bar(categories, times, color=colors)
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Average Inference Time')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{t:.1f}ms', ha='center', fontweight='bold')
    
    # Bottom left: Result table
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    table_data = []
    for i, r in enumerate(results[:10]):  # First 10 results
        mcu_ok = "✓" if r['mcu_pred'] == r['true_label'] else "✗"
        pc_ok = "✓" if r['pc_pred'] == r['true_label'] else "✗"
        table_data.append([i+1, r['true_label'], f"{r['pc_pred']} {pc_ok}", 
                          f"{r['mcu_pred']} {mcu_ok}", f"{r['mcu_time']}"])
    
    table = ax3.table(cellText=table_data,
                     colLabels=['#', 'True', 'PC', 'MCU', 'Time(ms)'],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax3.set_title('Test Results', fontsize=12, pad=20)
    
    # Bottom right: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    match_count = sum(1 for r in results if r['pc_pred'] == r['mcu_pred'])
    
    summary_text = f"""
    ╔══════════════════════════════════════════╗
    ║            SUMMARY RESULTS               ║
    ╠══════════════════════════════════════════╣
    ║  Test Count:         {total:4d}                 ║
    ╟──────────────────────────────────────────╢
    ║  PC Accuracy:        {correct_pc}/{total} = {100*correct_pc/total:5.1f}%       ║
    ║  MCU Accuracy:       {correct_mcu}/{total} = {100*correct_mcu/total:5.1f}%       ║
    ╟──────────────────────────────────────────╢
    ║  PC-MCU Match:       {match_count}/{total} = {100*match_count/total:5.1f}%       ║
    ╟──────────────────────────────────────────╢
    ║  Avg PC Time:        {avg_pc_time:6.2f} ms          ║
    ║  Avg MCU Time:       {avg_mcu_time:6.1f} ms           ║
    ╚══════════════════════════════════════════╝
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")
    return filepath

def save_digit_test_image(results, digit, model_name, filename):
    """Save specific digit test result."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'EE4065 HW6 - Digit {digit} Test ({model_name})\nYusuf Zivaroğlu & Taner Kahyaoğlu', 
                 fontsize=14, fontweight='bold')
    
    correct = sum(1 for r in results if r['mcu_pred'] == r['true_label'])
    total = len(results)
    
    # Left: Image grid
    ax1 = axes[0]
    n_cols = min(5, total)
    n_rows = (total + n_cols - 1) // n_cols
    
    for i, r in enumerate(results):
        row = i // n_cols
        col = i % n_cols
        img = r['image']
        ax1.imshow(img, cmap='gray', extent=[col*35, col*35+32, (n_rows-row)*35-32, (n_rows-row)*35])
        
        color = 'green' if r['mcu_pred'] == r['true_label'] else 'red'
        ax1.text(col*35+16, (n_rows-row)*35+5, f"P:{r['mcu_pred']}", 
                ha='center', fontsize=8, color=color, fontweight='bold')
    
    ax1.set_xlim(-5, n_cols*35+5)
    ax1.set_ylim(-5, n_rows*35+10)
    ax1.axis('off')
    ax1.set_title(f'Test Images (Digit {digit})')
    
    # Right: Result
    ax2 = axes[1]
    ax2.axis('off')
    
    result_text = f"""
    ╔══════════════════════════════════════╗
    ║     DIGIT {digit} TEST RESULTS          ║
    ╠══════════════════════════════════════╣
    ║  Test Count:      {total:4d}              ║
    ║  Correct:         {correct:4d}              ║
    ║  Accuracy:        {100*correct/total:5.1f}%           ║
    ╟──────────────────────────────────────╢
    """
    
    for i, r in enumerate(results):
        mcu_ok = "✓" if r['mcu_pred'] == r['true_label'] else "✗"
        result_text += f"    ║  [{i+1}] Prediction: {r['mcu_pred']} {mcu_ok}  ({r['mcu_time']}ms)  ║\n"
    
    result_text += "    ╚══════════════════════════════════════╝"
    
    ax2.text(0.1, 0.5, result_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")
    return filepath

def save_menu_image():
    """Save menu image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    menu_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EE4065 - Handwritten Digit Recognition                    ║
║               STM32 Nucleo-F446RE + TensorFlow Lite Micro                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Team: Yusuf Zivaroğlu & Taner Kahyaoğlu                                     ║
║  Date: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌────────────────────────────────────────────────────────┐
  │  Active Model (PC comparison): SqueezeNetMini         │
  └────────────────────────────────────────────────────────┘

  ╔══════════════════════════════════════════════════════╗
  ║                    MAIN MENU                         ║
  ╠══════════════════════════════════════════════════════╣
  ║  [1] Single image test (random)                      ║
  ║  [2] Batch test (10 images)                          ║
  ║  [3] Batch test (custom count)                       ║
  ║  [4] Specific digit test (0-9)                       ║
  ╟──────────────────────────────────────────────────────╢
  ║  [5] Show model compatibility                        ║
  ║  [6] Change PC comparison model                      ║
  ╟──────────────────────────────────────────────────────╢
  ║  [i] STM32 info                                      ║
  ║  [q] Quit                                            ║
  ╚══════════════════════════════════════════════════════╝

  📡 Connected: COM8 (STMicroelectronics STLink Virtual COM Port)
  🧠 Model: SqueezeNetMini (54.2 KB)
"""
    
    ax.text(0.05, 0.95, menu_text, fontsize=10, fontfamily='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.9),
            color='#00ff00')
    
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    filepath = os.path.join(IMAGES_DIR, 'Menu.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  ✓ Saved: Menu.png")
    return filepath

def save_model_compatibility_image():
    """Save model compatibility image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = [
        ('SqueezeNetMini', 54.2, True, 'WORKING', True),
        ('ResNet8', 96.3, True, 'Op Error', False),
        ('MobileNetV2Mini', 106.5, True, 'WORKING', True),
        ('EfficientNetMini', 108.5, True, 'Op Error', False),
        ('ResNet14', 201.7, True, 'Op Error', False),
        ('ShuffleNetMini', 253.2, False, 'Too Large', False),
        ('ResNet20', 307.2, False, 'Too Large', False),
        ('MobileNetV2', 492.1, False, 'Too Large', False),
        ('EfficientNet', 767.1, False, 'Too Large', False),
        ('SqueezeNet', 805.8, False, 'Too Large', False),
        ('ShuffleNet', 1159.0, False, 'Too Large', False),
    ]
    
    # Horizontal bar chart
    names = [m[0] for m in models]
    sizes = [m[1] for m in models]
    colors = ['#2ecc71' if m[3] == 'WORKING' else '#e74c3c' if not m[2] else '#f39c12' for m in models]
    
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, sizes, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('TFLite Model Size (KB)')
    ax.set_title('STM32 Nucleo-F446RE Model Compatibility (512KB Flash, 128KB RAM)\nYusuf Zivaroğlu & Taner Kahyaoğlu', 
                fontsize=12, fontweight='bold')
    
    # Flash limit line
    ax.axvline(x=250, color='red', linestyle='--', linewidth=2, label='Practical Flash Limit (~250KB)')
    
    # Status labels
    for i, (bar, model) in enumerate(zip(bars, models)):
        width = bar.get_width()
        ax.text(width + 10, bar.get_y() + bar.get_height()/2,
               f'{model[3]}', va='center', fontsize=9,
               color='green' if model[3] == 'WORKING' else 'red')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Working'),
        mpatches.Patch(color='#f39c12', label='Op Error'),
        mpatches.Patch(color='#e74c3c', label='Too Large'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    filepath = os.path.join(IMAGES_DIR, 'ModelCompatibly.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: ModelCompatibly.png")
    return filepath

# ============================================================================
# Main Test Functions
# ============================================================================

def run_all_tests(ser, model_name='SqueezeNetMini'):
    """Run all tests and save images."""
    
    print("\n" + "="*60)
    print("  AUTOMATED TEST AND SCREENSHOT CAPTURE")
    print("="*60)
    
    # Menu image
    print("\n[1/6] Creating menu image...")
    save_menu_image()
    
    # Model compatibility image
    print("\n[2/6] Creating model compatibility image...")
    save_model_compatibility_image()
    
    # Single image test
    print(f"\n[3/6] Single image test ({model_name})...")
    idx = np.random.randint(0, len(mnist_images))
    image = mnist_images[idx]
    true_label = mnist_labels[idx]
    
    pc_preds, pc_time = pc_inference(image, model_name)
    pc_pred = np.argmax(pc_preds) if pc_preds is not None else -1
    
    mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
    if mcu_preds is not None:
        mcu_pred = np.argmax(mcu_preds)
        print(f"    True: {true_label}, PC: {pc_pred}, MCU: {mcu_pred}")
        prefix = "SqueezeNet" if model_name == "SqueezeNetMini" else "MobileNetV2Mini"
        save_single_test_image(image, true_label, pc_pred, pc_time, mcu_preds, mcu_time, 
                               model_name, f'{prefix}_option1.png')
    else:
        print(f"    Error: {error}")
    
    # Batch test (10 samples)
    print(f"\n[4/6] Batch test - 10 samples ({model_name})...")
    results = []
    indices = np.random.choice(len(mnist_images), 10, replace=False)
    
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        pc_preds, pc_time = pc_inference(image, model_name)
        pc_pred = np.argmax(pc_preds) if pc_preds is not None else -1
        
        mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
        if mcu_preds is not None:
            mcu_pred = np.argmax(mcu_preds)
            results.append({
                'image': image,
                'true_label': true_label,
                'pc_pred': pc_pred,
                'pc_time': pc_time,
                'mcu_pred': mcu_pred,
                'mcu_time': mcu_time
            })
            mcu_ok = "✓" if mcu_pred == true_label else "✗"
            print(f"    [{i+1:2d}/10] True={true_label} MCU={mcu_pred} {mcu_ok}")
    
    if results:
        prefix = "SqueezeNet" if model_name == "SqueezeNetMini" else "MobileNetV2Mini"
        save_batch_test_image(results, model_name, f'{prefix}_option2.png')
    
    # Batch test (custom count - 25 samples)
    print(f"\n[5/6] Batch test - 25 samples ({model_name})...")
    results = []
    indices = np.random.choice(len(mnist_images), 25, replace=False)
    
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        pc_preds, pc_time = pc_inference(image, model_name)
        pc_pred = np.argmax(pc_preds) if pc_preds is not None else -1
        
        mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
        if mcu_preds is not None:
            mcu_pred = np.argmax(mcu_preds)
            results.append({
                'image': image,
                'true_label': true_label,
                'pc_pred': pc_pred,
                'pc_time': pc_time,
                'mcu_pred': mcu_pred,
                'mcu_time': mcu_time
            })
    
    if results:
        correct_mcu = sum(1 for r in results if r['mcu_pred'] == r['true_label'])
        print(f"    Accuracy: {correct_mcu}/25 = {100*correct_mcu/25:.1f}%")
        prefix = "SqueezeNet" if model_name == "SqueezeNetMini" else "MobileNetV2Mini"
        save_batch_test_image(results, model_name, f'{prefix}_option3.png')
    
    # Specific digit test
    print(f"\n[6/6] Specific digit test - Digit 5 ({model_name})...")
    digit = 5
    digit_indices = np.where(mnist_labels == digit)[0]
    indices = np.random.choice(digit_indices, min(5, len(digit_indices)), replace=False)
    
    results = []
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
        if mcu_preds is not None:
            mcu_pred = np.argmax(mcu_preds)
            results.append({
                'image': image,
                'true_label': true_label,
                'mcu_pred': mcu_pred,
                'mcu_time': mcu_time
            })
            mcu_ok = "✓" if mcu_pred == true_label else "✗"
            print(f"    [{i+1}] Prediction={mcu_pred} {mcu_ok}")
    
    if results:
        prefix = "SqueezeNet" if model_name == "SqueezeNetMini" else "MobileNetV2Mini"
        save_digit_test_image(results, digit, model_name, f'{prefix}_option4.png')
    
    print("\n" + "="*60)
    print("  ALL TESTS COMPLETED!")
    print("="*60)

# ============================================================================
# Main Function
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  EE4065 HW6 - Automated Test and Screenshot Capture")
    print("  Yusuf Zivaroğlu & Taner Kahyaoğlu")
    print("="*60)
    
    # Create Images directory
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Initialize models
    init_models()
    
    # List ports
    ports = serial.tools.list_ports.comports()
    print("\nAvailable COM ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device}: {port.description}")
    
    try:
        print(f"\nConnecting to {PORT}...")
        ser = serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"✓ Connected: {PORT}")
        
        # SqueezeNetMini tests
        print("\n" + "="*60)
        print("  SqueezeNetMini TESTS")
        print("="*60)
        run_all_tests(ser, 'SqueezeNetMini')
        
        # MobileNetV2Mini tests (optional)
        # Note: Different firmware must be loaded on STM32 for different model
        # print("\n" + "="*60)
        # print("  MobileNetV2Mini TESTS")
        # print("="*60)
        # run_all_tests(ser, 'MobileNetV2Mini')
        
        ser.close()
        print("\n✓ Port closed.")
        
    except serial.SerialException as e:
        print(f"\n✗ Serial port error: {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
