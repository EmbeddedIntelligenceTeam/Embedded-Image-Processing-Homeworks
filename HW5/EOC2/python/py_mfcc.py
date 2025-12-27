import os
import numpy as np
from scipy.io import wavfile
import cmsisdsp as dsp
import cmsisdsp.mfcc as mfcc
from cmsisdsp.datatype import F32

def MFCC_Init(__fftSize, __sampleRate, __numOfMelFilters, __numOfDctOutputs, __window):
    global fftSize
    global sampleRate
    global numOfMelFilters
    global numOfDctOutputs
    global window
    global mfccf32

    fftSize = __fftSize
    sampleRate = __sampleRate
    numOfMelFilters = __numOfMelFilters 
    numOfDctOutputs = __numOfDctOutputs
    window = __window
    
    freq_min = 20
    freq_high = sampleRate / 2
    filtLen, filtPos, packedFilters = mfcc.melFilterMatrix(
        F32, freq_min, freq_high, numOfMelFilters, sampleRate, fftSize
    )

    dctMatrixFilters = mfcc.dctMatrix(F32, numOfDctOutputs, numOfMelFilters)
    mfccf32 = dsp.arm_mfcc_instance_f32()
    status = dsp.arm_mfcc_init_f32(
        mfccf32,
        fftSize,
        numOfMelFilters,
        numOfDctOutputs,
        dctMatrixFilters,
        filtPos,
        filtLen,
        packedFilters,
        window,
    )


def MFCC_Run(sample):
    sample = sample.astype(np.float32)
    sample = sample / max(abs(sample))
    first_half = sample[:fftSize]
    second_half = sample[fftSize:2*fftSize]
    tmp = np.zeros(fftSize + 2)
    mfcc_features = dsp.arm_mfcc_f32(mfccf32, first_half, tmp)
    mfcc_features_2 = dsp.arm_mfcc_f32(mfccf32, second_half, tmp)
    mfcc_feature = np.concatenate((mfcc_features, mfcc_features_2))
        
    return mfcc_feature