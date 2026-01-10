"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       EE4065 Homework 6 - CNN Handwritten Digit Recognition Application     ║
║                   STM32 Nucleo-F446RE + TensorFlow Lite Micro                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Team Members:                                                               ║
║    • Yusuf Zivaroğlu                                                         ║
║    • Taner Kahyaoğlu                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Compatible Models (STM32 F446RE - 512KB Flash, 128KB RAM):                 ║
║    • SqueezeNetMini:    54.2 KB  (SMALLEST - Recommended)                   ║
║    • ResNet8:           96.3 KB                                              ║
║    • MobileNetV2Mini:  106.5 KB                                              ║
║    • EfficientNetMini: 108.5 KB                                              ║
║    • ResNet14:         201.7 KB  (RAM limit)                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Note: Due to Flash size constraints, only one model can be tested at a time.
      To test different models, change the model in STM32 project and reflash.
"""

import os
import sys
from datetime import datetime

# Disable XNNPACK delegate (must be set BEFORE importing TensorFlow)
os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'

import serial
import serial.tools.list_ports
import numpy as np
import cv2
import time
import struct

import tensorflow as tf
from tensorflow import keras

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIGURATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
PORT = 'COM8'          # STM32 COM port (update if needed)
BAUD_RATE = 115200
TIMEOUT = 10

# Protocol bytes
SYNC_BYTE = 0xAA
ACK_BYTE = 0x55
CMD_INFERENCE = 0x01
CMD_INFO = 0x02
CMD_READY = 0x03

# Image parameters (must match STM32 model input)
IMAGE_SIZE = 32
NUM_CLASSES = 10

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_DIR = os.path.join(SCRIPT_DIR, "..", "tflite_models")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              GLOBAL VARIABLES                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
mnist_images = None
mnist_labels = None
tflite_interpreters = {}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MODEL INFORMATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
MODEL_INFO = {
    'SqueezeNetMini': {'size_kb': 54.2, 'fits_f446': True, 'recommended': True, 'status': 'WORKING'},
    'ResNet8': {'size_kb': 96.3, 'fits_f446': True, 'recommended': False, 'status': 'Op Error'},
    'MobileNetV2Mini': {'size_kb': 106.5, 'fits_f446': True, 'recommended': True, 'status': 'WORKING'},
    'EfficientNetMini': {'size_kb': 108.5, 'fits_f446': True, 'recommended': False, 'status': 'Op Error'},
    'ResNet14': {'size_kb': 201.7, 'fits_f446': True, 'recommended': False, 'status': 'Op Error'},
    'ResNet20': {'size_kb': 307.2, 'fits_f446': False, 'recommended': False, 'status': 'Too Large'},
    'MobileNetV2': {'size_kb': 492.1, 'fits_f446': False, 'recommended': False, 'status': 'Too Large'},
    'SqueezeNet': {'size_kb': 805.8, 'fits_f446': False, 'recommended': False, 'status': 'Too Large'},
    'EfficientNet': {'size_kb': 767.1, 'fits_f446': False, 'recommended': False, 'status': 'Too Large'},
    'ShuffleNetMini': {'size_kb': 253.2, 'fits_f446': False, 'recommended': False, 'status': 'Too Large'},
    'ShuffleNet': {'size_kb': 1159.0, 'fits_f446': False, 'recommended': False, 'status': 'Too Large'},
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              HELPER FUNCTIONS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def clear_screen():
    """Clear the terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print main header."""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "EE4065 - Handwritten Digit Recognition" + " "*19 + "║")
    print("║" + " "*15 + "STM32 Nucleo-F446RE + TensorFlow Lite Micro" + " "*18 + "║")
    print("╠" + "═"*78 + "╣")
    print("║  Team: Yusuf Zivaroğlu & Taner Kahyaoğlu" + " "*36 + "║")
    print("║  Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " "*49 + "║")
    print("╚" + "═"*78 + "╝")

def print_success(msg):
    """Print success message."""
    print(f"  ✅ {msg}")

def print_error(msg):
    """Print error message."""
    print(f"  ❌ {msg}")

def print_warning(msg):
    """Print warning message."""
    print(f"  ⚠️  {msg}")

def print_info(msg):
    """Print info message."""
    print(f"  ℹ️  {msg}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              INITIALIZATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def init_models():
    """Load MNIST dataset and TFLite interpreters for PC comparison."""
    global mnist_images, mnist_labels, tflite_interpreters
    
    print("\n┌" + "─"*58 + "┐")
    print("│" + " "*20 + "Model Initialization" + " "*18 + "│")
    print("└" + "─"*58 + "┘")
    
    print("\n📊 Loading MNIST dataset...")
    (_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    # Preprocessing: resize to 32x32
    mnist_images = []
    for img in test_images:
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mnist_images.append(img_resized)
    mnist_images = np.array(mnist_images)
    mnist_labels = test_labels
    print_success(f"{len(mnist_images)} test images loaded")
    
    # Load TFLite interpreters for compatible models
    print("\n🧠 Loading TFLite models for PC comparison...")
    loaded_count = 0
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
                    status_icon = "✓" if info['status'] == 'WORKING' else "○"
                    print(f"  {status_icon} {model_name:20s} ({info['size_kb']:6.1f} KB)")
                    loaded_count += 1
                except Exception as e:
                    print_warning(f"{model_name}: {e}")
            else:
                print(f"  ○ {model_name:20s} (not found)")
    
    print_success(f"Total {loaded_count} models loaded")

def list_available_ports():
    """List available COM ports."""
    ports = serial.tools.list_ports.comports()
    print("\n📡 Available COM Ports:")
    print("  ┌" + "─"*50 + "┐")
    for i, port in enumerate(ports):
        print(f"  │ [{i}] {port.device:8s} - {port.description[:35]:35s} │")
    print("  └" + "─"*50 + "┘")
    return ports

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              PC INFERENCE (TFLite)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def pc_inference(image, model_name='SqueezeNetMini'):
    """Perform inference on PC using TFLite interpreter."""
    if model_name not in tflite_interpreters:
        return None, None
    
    interpreter = tflite_interpreters[model_name]
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Prepare image: expand to 3 channels, normalize
    img = image.astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    img = np.expand_dims(img, axis=0)
    
    # Quantize for INT8 model
    if input_details['dtype'] == np.uint8:
        scale, zero_point = input_details['quantization']
        img = (img / scale + zero_point).astype(np.uint8)
    
    interpreter.set_tensor(input_details['index'], img)
    
    try:
        start = time.perf_counter()
        interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000
    except RuntimeError as e:
        return None, None
        
    output = interpreter.get_tensor(output_details['index'])
    
    # Dequantize if INT8
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    
    return output[0], elapsed

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              STM32 COMMUNICATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def send_image_get_prediction(ser, image):
    """Send image to STM32 and receive prediction."""
    try:
        ser.reset_input_buffer()
        
        # Prepare image: grayscale uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        image_flat = image.flatten()
        
        # Send sync byte
        ser.write(bytes([SYNC_BYTE]))
        
        # Wait for ACK
        start = time.time()
        while time.time() - start < 2:
            if ser.in_waiting > 0:
                ack = ser.read(1)
                if len(ack) > 0 and ack[0] == ACK_BYTE:
                    break
        else:
            return None, None, "ACK timeout"
        
        # Send image data (32*32 = 1024 bytes)
        ser.write(image_flat.tobytes())
        
        # Wait for predictions
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
        
        # Read inference time (4 bytes, uint32)
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

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              DISPLAY FUNCTIONS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def display_image_ascii(image):
    """Display image as ASCII art."""
    print("\n  📷 Image Preview:")
    print("  ┌" + "─"*64 + "┐")
    for row in range(0, IMAGE_SIZE, 2):
        line = "  │ "
        for col in range(0, IMAGE_SIZE, 1):
            pixel = image[row, col] if row < IMAGE_SIZE else 0
            if pixel > 200:
                line += "██"
            elif pixel > 150:
                line += "▓▓"
            elif pixel > 100:
                line += "▒▒"
            elif pixel > 50:
                line += "░░"
            else:
                line += "  "
        line += " │"
        print(line)
    print("  └" + "─"*64 + "┘")

def display_predictions(predictions, label=None, show_graph=True):
    """Display prediction bar graph."""
    predicted = np.argmax(predictions)
    max_val = max(predictions) if max(predictions) > 0 else 1
    
    if show_graph:
        print("\n  📊 Prediction Results:")
        print("  ┌" + "─"*50 + "┐")
        for i in range(NUM_CLASSES):
            val = predictions[i]
            bar_len = int(val / max_val * 25) if max_val > 0 else 0
            bar = '█' * bar_len + '░' * (25 - bar_len)
            marker = " ◄── PREDICTION" if i == predicted else ""
            if label is not None and i == label:
                marker += " (TRUE)" if i != predicted else ""
            confidence = val / 255 * 100 if max_val > 0 else 0
            print(f"  │ {i}: [{bar}] {confidence:5.1f}%{marker:20s} │")
        print("  └" + "─"*50 + "┘")
    
    return predicted

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              TEST FUNCTIONS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_single_test(ser, model_name='SqueezeNetMini'):
    """Perform single test with a random image."""
    idx = np.random.randint(0, len(mnist_images))
    image = mnist_images[idx]
    true_label = mnist_labels[idx]
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*20 + "SINGLE IMAGE TEST" + " "*21 + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Image Index: {idx:5d}" + " "*38 + "║")
    print(f"║  True Label:  {true_label}" + " "*42 + "║")
    print("╚" + "═"*58 + "╝")
    
    display_image_ascii(image)
    
    # PC inference
    print("\n  🖥️  PC Inference:")
    pc_preds, pc_time = pc_inference(image, model_name)
    if pc_preds is not None:
        pc_preds_scaled = (pc_preds * 255).astype(np.uint8)
        pc_pred = np.argmax(pc_preds_scaled)
        pc_correct = "✓" if pc_pred == true_label else "✗"
        print(f"     Model: {model_name}")
        print(f"     Prediction: {pc_pred} {pc_correct}  (Time: {pc_time:.2f}ms)")
    else:
        pc_pred = -1
        print_warning(f"{model_name}: Inference failed")
    
    # MCU inference
    print("\n  🔌 STM32 (MCU) Inference:")
    mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
    
    if mcu_preds is not None:
        mcu_pred = display_predictions(mcu_preds, true_label)
        mcu_correct = "✓" if mcu_pred == true_label else "✗"
        print(f"\n     Prediction: {mcu_pred} {mcu_correct}  (Inference Time: {mcu_time}ms)")
        
        # Comparison
        print("\n  ═══════════════════════════════════════════════════════")
        print("  📋 RESULT COMPARISON:")
        print("  ─────────────────────────────────────────────────────────")
        print(f"     True Value:    {true_label}")
        if pc_pred >= 0:
            print(f"     PC Prediction: {pc_pred} {pc_correct}")
        print(f"     MCU Prediction:{mcu_pred} {mcu_correct}")
        match_status = "MATCH ✓" if pc_pred == mcu_pred else "DIFFERENT ✗"
        if pc_pred >= 0:
            print(f"     PC-MCU Status: {match_status}")
        print("  ═══════════════════════════════════════════════════════")
    else:
        print_error(f"MCU Error: {error}")
        mcu_pred = -1
    
    return true_label, pc_pred, mcu_pred

def run_batch_test(ser, num_samples=10, model_name='SqueezeNetMini'):
    """Perform batch test."""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*22 + "BATCH TEST" + " "*26 + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Sample Count: {num_samples:4d}" + " "*37 + "║")
    print(f"║  Model:        {model_name:20s}" + " "*17 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    indices = np.random.choice(len(mnist_images), num_samples, replace=False)
    
    correct_pc = 0
    correct_mcu = 0
    total = 0
    total_pc_time = 0
    total_mcu_time = 0
    pc_mcu_match = 0
    
    print("  ┌───────┬───────┬────────────┬────────────┬──────────┐")
    print("  │ No.   │ True  │ PC Pred    │ MCU Pred   │ Time(ms) │")
    print("  ├───────┼───────┼────────────┼────────────┼──────────┤")
    
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        # PC inference
        pc_preds, pc_time = pc_inference(image, model_name)
        if pc_preds is not None:
            pc_pred = np.argmax(pc_preds)
            total_pc_time += pc_time
            if pc_pred == true_label:
                correct_pc += 1
        else:
            pc_pred = -1
        
        # MCU inference
        mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
        
        if mcu_preds is not None:
            mcu_pred = np.argmax(mcu_preds)
            total_mcu_time += mcu_time
            if mcu_pred == true_label:
                correct_mcu += 1
            total += 1
            
            if pc_pred == mcu_pred:
                pc_mcu_match += 1
            
            mcu_ok = "✓" if mcu_pred == true_label else "✗"
            pc_str = f"{pc_pred} {'✓' if pc_pred == true_label else '✗'}" if pc_pred >= 0 else "N/A"
            
            print(f"  │ {i+1:5d} │   {true_label}   │    {pc_str:6s}   │    {mcu_pred} {mcu_ok}     │  {mcu_time:6d}  │")
        else:
            print(f"  │ {i+1:5d} │   {true_label}   │   ERROR    │   ERROR    │    -     │")
    
    print("  └───────┴───────┴────────────┴────────────┴──────────┘")
    
    # Summary
    if total > 0:
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " "*24 + "RESULTS" + " "*27 + "║")
        print("╠" + "═"*58 + "╣")
        if total_pc_time > 0:
            pc_acc = 100*correct_pc/total
            print(f"║  PC Accuracy:     {correct_pc}/{total} = {pc_acc:5.1f}%" + " "*26 + "║")
            print(f"║  Avg PC Time:     {total_pc_time/total:.2f}ms" + " "*35 + "║")
        else:
            print("║  PC: Skipped (delegate error)" + " "*26 + "║")
        mcu_acc = 100*correct_mcu/total
        print(f"║  MCU Accuracy:    {correct_mcu}/{total} = {mcu_acc:5.1f}%" + " "*25 + "║")
        print(f"║  Avg MCU Time:    {total_mcu_time/total:.1f}ms" + " "*35 + "║")
        print("╠" + "═"*58 + "╣")
        match_rate = 100*pc_mcu_match/total
        print(f"║  PC-MCU Match:    {pc_mcu_match}/{total} = {match_rate:5.1f}%" + " "*24 + "║")
        print("╚" + "═"*58 + "╝")

def run_digit_test(ser, digit, model_name='SqueezeNetMini'):
    """Test a specific digit."""
    print("\n" + "╔" + "═"*58 + "╗")
    print(f"║" + f"   SPECIFIC DIGIT TEST: {digit}" + " "*32 + "║")
    print("╚" + "═"*58 + "╝")
    
    # Find samples of this digit
    digit_indices = np.where(mnist_labels == digit)[0]
    indices = np.random.choice(digit_indices, min(5, len(digit_indices)), replace=False)
    
    correct_mcu = 0
    total = 0
    
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
        
        if mcu_preds is not None:
            mcu_pred = np.argmax(mcu_preds)
            if mcu_pred == true_label:
                correct_mcu += 1
            total += 1
            
            mcu_ok = "✓" if mcu_pred == true_label else "✗"
            print(f"  [{i+1}] True={true_label}  MCU={mcu_pred} {mcu_ok}  ({mcu_time}ms)")
        else:
            print_error(f"[{i+1}] ERROR: {error}")
    
    if total > 0:
        print(f"\n  📊 Accuracy: {correct_mcu}/{total} = {100*correct_mcu/total:.1f}%")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MENU FUNCTIONS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def show_model_compatibility():
    """Show models compatible with F446RE."""
    print("\n" + "╔" + "═"*70 + "╗")
    print("║" + " "*15 + "STM32 Nucleo-F446RE Model Compatibility" + " "*14 + "║")
    print("║" + " "*20 + "(512KB Flash, 128KB RAM)" + " "*26 + "║")
    print("╠" + "═"*70 + "╣")
    print("║  Model               │ Size       │ Fits    │ Status             ║")
    print("╟──────────────────────┼────────────┼─────────┼────────────────────╢")
    
    for name, info in sorted(MODEL_INFO.items(), key=lambda x: x[1]['size_kb']):
        size_str = f"{info['size_kb']:.1f} KB"
        compat = "✓ YES" if info['fits_f446'] else "✗ NO"
        status = info['status']
        rec = " ★" if info['recommended'] else ""
        print(f"║  {name:18s}{rec:2s}│ {size_str:10s} │ {compat:7s} │ {status:18s} ║")
    
    print("╚" + "═"*70 + "╝")
    print("\n  ★ = Recommended model (tested and working on STM32)")
    print("  Note: Only ONE model can be loaded at a time due to Flash size.")

def interactive_menu(ser):
    """Interactive menu loop."""
    current_model = 'SqueezeNetMini'  # Default model
    
    while True:
        print_header()
        print("\n  ┌" + "─"*56 + "┐")
        print(f"  │  Active Model (PC comparison): {current_model:18s} │")
        print("  └" + "─"*56 + "┘")
        
        print("\n  ╔══════════════════════════════════════════════════════╗")
        print("  ║                    MAIN MENU                         ║")
        print("  ╠══════════════════════════════════════════════════════╣")
        print("  ║  [1] Single image test (random)                      ║")
        print("  ║  [2] Batch test (10 images)                          ║")
        print("  ║  [3] Batch test (custom count)                       ║")
        print("  ║  [4] Specific digit test (0-9)                       ║")
        print("  ╟──────────────────────────────────────────────────────╢")
        print("  ║  [5] Show model compatibility                        ║")
        print("  ║  [6] Change PC comparison model                      ║")
        print("  ╟──────────────────────────────────────────────────────╢")
        print("  ║  [i] STM32 info                                      ║")
        print("  ║  [q] Quit                                            ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        
        choice = input("\n  👉 Your choice: ").strip().lower()
        
        if choice == '1':
            run_single_test(ser, current_model)
            input("\n  Press Enter to continue...")
        
        elif choice == '2':
            run_batch_test(ser, 10, current_model)
            input("\n  Press Enter to continue...")
        
        elif choice == '3':
            try:
                n = int(input("  How many samples? (1-100): ").strip())
                n = max(1, min(100, n))
                run_batch_test(ser, n, current_model)
            except:
                print_error("Invalid number!")
            input("\n  Press Enter to continue...")
        
        elif choice == '4':
            digit = input("  Enter digit (0-9): ").strip()
            if digit in '0123456789':
                run_digit_test(ser, int(digit), current_model)
            else:
                print_error("Invalid digit!")
            input("\n  Press Enter to continue...")
        
        elif choice == '5':
            show_model_compatibility()
            input("\n  Press Enter to continue...")
        
        elif choice == '6':
            print("\n  Available models (for PC comparison):")
            models = list(tflite_interpreters.keys())
            for i, name in enumerate(models):
                marker = " (active)" if name == current_model else ""
                print(f"    [{i+1}] {name}{marker}")
            try:
                idx = int(input("\n  Select model: ").strip()) - 1
                if 0 <= idx < len(models):
                    current_model = models[idx]
                    print_success(f"Model changed: {current_model}")
            except:
                print_error("Invalid selection!")
        
        elif choice == 'i':
            ser.reset_input_buffer()
            ser.write(b'i')
            time.sleep(0.5)
            response = ""
            while ser.in_waiting:
                response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                time.sleep(0.1)
            print(response if response else "No response from STM32")
            input("\n  Press Enter to continue...")
        
        elif choice == 'q':
            print_info("Exiting program...")
            break
        
        else:
            print_error("Invalid selection. Try again.")
        
        ser.reset_input_buffer()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN FUNCTION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    clear_screen()
    print_header()
    
    ser = None
    
    try:
        # Initialize models
        init_models()
        
        # Show model compatibility
        show_model_compatibility()
        
        # List available ports
        ports = list_available_ports()
        
        if not ports:
            print_error("No COM port found!")
            print_info("Make sure STM32 Nucleo board is connected via USB.")
            return
        
        # Select port
        print(f"\n  Default port: {PORT}")
        print("  Enter COM port (or Enter for default, 'q' to quit):")
        port_input = input("  > ").strip()
        
        if port_input.lower() == 'q':
            return
        
        if port_input:
            try:
                idx = int(port_input)
                if 0 <= idx < len(ports):
                    port = ports[idx].device
                else:
                    port = port_input
            except ValueError:
                port = port_input
        else:
            port = PORT
        
        # Connect
        print(f"\n  🔗 Connecting to {port} at {BAUD_RATE} baud...")
        ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # Wait for STM32 reset
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print_success(f"Connected to {port}!")
        
        # Read startup message
        time.sleep(0.5)
        while ser.in_waiting:
            print(ser.read(ser.in_waiting).decode('utf-8', errors='ignore'), end='')
        
        # Interactive menu
        interactive_menu(ser)
    
    except serial.SerialException as e:
        print_error(f"Serial Port Error: {e}")
        print_info("Check:")
        print("       1. Is STM32 connected?")
        print("       2. Is another program using the port?")
        print("       3. Is the correct firmware loaded?")
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ser and ser.is_open:
            ser.close()
            print_info("Port closed.")

if __name__ == "__main__":
    main()
