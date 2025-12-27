import cv2
import py_moments
import py_serial
import py_serialimg
import calculate_moments as cm
import tensorflow as tf
import numpy as np

py_serial.SERIAL_Init("COM5")
py_serialimg.__serial = py_serial.__serial
interpreter = tf.lite.Interpreter(model_path = "hdr_mlp.tflite")
my_signature = interpreter.get_signature_runner()
while 1:
    rqType, height, width, format  = py_serialimg.SERIAL_IMG_PollForRequest()
    if rqType == py_serial.MCU_WRITES:
        img = py_serialimg.SERIAL_IMG_Read()
        moments, huMoments = py_moments.MOMENTS_Run(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        output = my_signature(dense_input = huMoments.astype(np.float32))['dense_2']
        rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
        if rqType == py_serial.MCU_WRITES:
            mcuOutputs = py_serial.SERIAL_Read()
            print()
            print("PC OUTPUT:")
            print(output)
            print("MCU OUTPUT:")
            print(mcuOutputs)
            print()

