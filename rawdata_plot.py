import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram, convolve2d
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft
from scipy.signal import stft
import scipy.signal as signal
import os
import re
import time
from collections import Counter
import pandas as pd

fs = 48000


def main():
    getData()
    return

def medfilter (data):
    pdata=data.copy()
    for j in range(3, len(data)-2):
        pdata[j] = 0.15 * data[j - 2] + 0.2 * data[j - 1] + 0.3 * data[j] + 0.2 * data[j + 1] + 0.15 * data[j + 2]
    return pdata

def getData():
    filepath = 'D:/zq/OneDrive/experiments/2020/soundsign/'
    files=os.listdir(filepath)
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern,file)
        if len(res) == 1 and int(res[0]) == 52:
            filename = filepath+file
            starttime = time.time()
            rawdata = np.memmap(filename, dtype=np.float32, mode='r')
            # plt.figure()
            # plt.plot(rawdata)
            # plt.xlim((0, 20000))
            data = butter_bandpass_filter(rawdata, 16500, 22500, 48000)
            data = data[24000:]
            startF = 17000
            B = 5000
            duration = 0.04
            I1 = getI(data, startF, B, duration)
            fs = 48000
            I = butter_lowpass_filter(I1, B/2, fs, order=5)
            # plt.figure()
            # plt.plot(I)
            Q1 = getQ(data, startF, B, duration)
            Q = butter_lowpass_filter(Q1, B / 2, fs, order=5)
            signaldata = []
            for i in range(0, I.shape[0]):
                signalsample = complex(Q[i], I[i])
                signaldata.append(signalsample)
            signal_ph = np.angle(signaldata)
            # signal_ph = np.unwrap(signal_ph)
            # plt.figure()
            # plt.plot(signal_ph)
            freq, t, zxx = signal.stft(I, fs, nperseg=fs * 0.05, noverlap=fs * 0.045, nfft=8192)
            zxx = np.abs(zxx)
            zxx = 20*np.log10(zxx)
            maxindex = zxx.argmax(axis=0)
            tmp = Counter(maxindex).most_common(1)
            tmp = tmp[0]
            tmp = tmp[0]
            # zxx = np.diff(zxx)
            print(tmp, int(res[0]))
            # plt.figure()
            # plt.pcolormesh(t[:], freq, zxx)
            # plt.figure()
            signal_multi = []
            signal_mean = []
            for i in range(max([0, tmp-30]), tmp+30):
                signal_path = zxx[i,:]
                signal_path = butter_lowpass_filter(signal_path, 8, 200, order=5)
                signal_path = signal_path[200:]
                # if np.mean(signal_path[:100]) >= -80:
                #     chID = '%d' % i
                #     cc = np.mean(signal_path[:100])
                #     signal_path = signal_path - cc
                #     plt.plot(signal_path, label=chID)
                #     # plt.plot(np.diff(signal_path), label=chID)
                #     plt.legend()
                #     # cc.append(np.diff(signal_path))
                # cc = np.mean(signal_path[:100])
                # signal_path = signal_path - cc
                signal_multi.append(signal_path)
                signal_mean.append(np.mean(signal_path[:100]))
            tmp = pd.Series(signal_mean).sort_values().index[-4:]
            tmp = list(tmp)
            tmp.sort()
            datapro = [signal_multi[i] for i in tmp]
            datapro = np.array(datapro)
            # datapro = np.diff(datapro, axis=1)
            for i in range(len(datapro)):
                plt.plot(datapro[i])
            # plt.ylim((0, 300))
            # plt.figure()
            # plt.pcolormesh(cc)
    plt.show()


def move_average(data):
    win_size = 200
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    data = data.reshape((new_len, win_size))
    result = np.zeros(new_len)
    for i in range(new_len):
        result[i] = np.mean(data[i, :])
    return result


def move_average_overlap(data):
    win_size = 200
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    new_len = new_len*2
    result = np.zeros(new_len)
    for index in range(0, new_len):
        start =  (index/2)*win_size
        end = (index/2+1)*win_size
        result[index] = np.mean(data[int(start):int(end)])
    return result

def getI(data, startF, B, duration):
    times = np.linspace(0, duration, int(fs * duration), False)
    rep = int(len(data)/len(times))+1
    cosdata = []
    for i in range(rep):
        tmp = np.cos(2 * np.pi * (startF * times + B * times * times / (2 * duration)))
        cosdata.extend(tmp.tolist())
    cosdata = np.array(cosdata)
    mulCos = cosdata[:len(data)] * data
    return mulCos



def getQ(data, startF, B, duration):
    times = np.linspace(0, duration, int(fs * duration), False)
    rep = int(len(data)/len(times))+1
    sindata = []
    for i in range(rep):
        tmp = np.sin(2 * np.pi * (startF * times + B * times * times / (2 * duration)))
        sindata.extend(tmp.tolist())
    sindata = np.array(sindata)
    mulSin = sindata[:len(data)] * data
    return mulSin


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    main()
