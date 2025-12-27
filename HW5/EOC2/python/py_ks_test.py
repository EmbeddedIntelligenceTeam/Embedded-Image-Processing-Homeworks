import numpy as np
import scipy.signal as sig
import py_serial
import py_mfcc
import tensorflow as tf
from scipy.io.wavfile import write

py_serial.SERIAL_Init("COM5")
sampleRate = 8000
fftSize = 1024
numOfMelFilters = 20
numOfDctOutputs = 13
window = sig.get_window("hamming", fftSize)
py_mfcc.MFCC_Init(fftSize, sampleRate, numOfMelFilters, numOfDctOutputs, window)
interpreter = tf.lite.Interpreter(model_path="mlp_fsdd_model.tflite")
my_signature = interpreter.get_signature_runner()
while 1:
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == py_serial.MCU_WRITES:
        data = py_serial.SERIAL_Read()
        #write('test.wav', sampleRate, data.astype(np.float32))
        mfccFeatures = py_mfcc.MFCC_Run(data)
        output = list(my_signature(dense_input=mfccFeatures.reshape((1,26)))['dense_2'][0])
        rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
        if rqType == py_serial.MCU_WRITES:
            mcuOutputs = py_serial.SERIAL_Read()
            print()
            print("PC OUTPUT:")
            print(output)
            print("MCU OUTPUT:")
            print(mcuOutputs)
            print()




    
        



