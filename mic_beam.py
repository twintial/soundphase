import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks_cwt, spectrogram, convolve2d
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.signal import stft
import os
import re
import time
import wave

fs = 48000


def get_dtype_from_width(width, unsigned=True):
    if width == 1:
        if unsigned:
            return np.uint8
        else:
            return np.int8
    elif width == 2:
        return np.int16
    elif width == 3:
        raise ValueError("unsupported type: int24")
    elif width == 4:
        return np.float32
    else:
        raise ValueError("Invalid width: %d" % width)
def load_audio_data(filename, type='wav'):
    if type == 'pcm':
        rawdata = np.memmap(filename, dtype=np.float32, mode='r')
        return rawdata, 48e3
    elif type == 'wav':
        wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
        num_frame = wav.getnframes()  # 获取帧数
        num_channel = wav.getnchannels()  # 获取声道数
        framerate = wav.getframerate()  # 获取帧速率
        num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
        str_data = wav.readframes(num_frame)  # 读取全部的帧
        wav.close()  # 关闭流
        wave_data = np.frombuffer(str_data, dtype=get_dtype_from_width(num_sample_width))  # 将声音文件数据转换为数组矩阵形式
        wave_data = wave_data.reshape((-1, num_channel))  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
        return wave_data, framerate

def main():
    getData()
    return

def medfilter (data):
    pdata=data.copy()
    for j in range(3, len(data)-2):
        pdata[j] = 0.15 * data[j - 2] + 0.2 * data[j - 1] + 0.3 * data[j] + 0.2 * data[j + 1] + 0.15 * data[j + 2]
    return pdata

def draw_circle(I, Q):
    plt.figure()
    for i in range(0, len(I)):
        plt.clf()
        plt.plot(I[:i], Q[:i])
        plt.pause(0.01)
def getData():
    filepath = r'D:\实验数据\2021\毕设\micarrayspeaker\raw\sjj\gesture2/'
    files=os.listdir(filepath)
    for file in files:
        if file == '60.wav':
            filename = filepath+file
            print(filename)
            rawdata, _ = load_audio_data(filename)
            rawdata = rawdata[6000:,:]
            for channelID in [0]:
                fc = 17350+700*channelID
                dataphase = []
                dataam = []
                for micID in range(7):
                    data = butter_bandpass_filter(rawdata[:, micID], fc-100, fc+100, 48000)
                    data = data[48000:]
                    f = fc
                    I1 = getI(data, f)
                    I = move_average_overlap(I1)
                    Q1 = getQ(data, f)
                    Q = move_average_overlap(Q1)
                    decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
                    trendQ = decompositionQ.trend
                    trendQ = np.hstack((np.array([trendQ[10]] * 10), trendQ[10:]))

                    decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
                    trendI = decompositionI.trend
                    trendI = np.hstack((np.array([trendI[10]] * 10), trendI[10:]))
                    signaldata = []

                    for i in range(0, trendI.shape[0]):
                        signalsample = complex(trendQ[i], trendI[i])
                        signaldata.append(signalsample)

                    signal_ph = np.angle(signaldata)
                    signal_ph = np.unwrap(signal_ph)
                    signal_am = np.abs(signaldata)

                    # x = signal_am*np.exp(1j*signal_ph)
                    # draw_circle(np.real(x), np.imag(x))
                    dataphase.append(signal_ph)
                    dataam.append(signal_am)
                dataphase = np.array(dataphase)
                dataam = np.array(dataam)
                plt.figure()
                plt.title('phase')
                plt.plot((dataphase[2,:]))
            pos_aoa = np.linspace(0, 360, 3600)
            pos_aoa = pos_aoa*2*np.pi/360
            steering_vector = [np.array([1]*len(pos_aoa))]
            r = 0.043
            wavelength = 343/17000
            for micID in range(1, 7):
                theta = np.pi * (micID - 1) / 3
                ang = pos_aoa - theta
                deltaph = 2 * np.pi * r * np.cos(ang) / wavelength
                tmp = np.exp(1j*deltaph)
                steering_vector.append(tmp)
            steering_vector = np.array(steering_vector)
            signal_vector = dataam*np.exp(1j*dataphase)
            signal_vector = np.transpose(signal_vector)
            signal_vector = np.dot(signal_vector, steering_vector)
            signal_sum = np.angle(signal_vector)
            signal_sum = np.unwrap(signal_sum, axis=0)
            signal_sum = np.diff(signal_sum, axis=0)
            signal_sum = np.transpose(signal_sum)
            sumsum = np.sum(np.abs(signal_sum), axis=1)
            # plt.figure()
            # plt.plot(sumsum)
            signal_phase = np.angle(signal_vector)
            sumsum = list(sumsum)
            ii = sumsum.index(max(sumsum))
            print(ii)
            maxrange = []
            for i in range(len(sumsum)):
                cc = np.unwrap(signal_phase[:, i])
                maxrange.append(np.max(cc) - np.min(cc))
            # plt.figure()
            # plt.plot(maxrange, '.')
            iii = maxrange.index(max(maxrange))
            print(iii)
            # plt.figure()
            # plt.pcolormesh(signal_sum)

            # draw_circle(np.real(signal_vector[:, iii]), np.imag(signal_vector[:, iii]))

            plt.figure()
            plt.plot((np.unwrap(signal_phase[:, iii])), '.-')
            plt.show()
    plt.show()


def get_distance(dataphase, dataam):
    channel_num = dataam.shape[0]
    delta = [4*np.pi*i*700/340 for i in range(channel_num)]
    delta = delta * dataam.shape[1]
    delta = np.array(delta)
    delta = np.reshape(delta, [dataam.shape[1], dataam.shape[0]])
    delta = np.transpose(delta)
    distance = []
    for i in range(100):
        d = i/100
        deltaphase = dataphase - delta*d
        deltaphase = np.exp(1j*deltaphase)
        tmp = deltaphase
        dp = np.sum(tmp,axis=0)
        dp = np.abs(dp)
        distance.append(dp)
    distance = np.array(distance)
    return distance



def getPhase1(I, Q):
    derivativeQ = getDerivative(Q)
    derivativeI = getDerivative(I)
    # phase=np.unwrap(2*())+np.pi/2))/2
    # distance=distanceLine(phase,20000)
    # plt.plot(distance)
    # plt.show()
    derivativeQ = np.asarray(derivativeQ)
    derivativeQ[np.where(derivativeQ==0)]=0.000001
    arcValue = np.arctan(-np.asarray(derivativeI) / (derivativeQ))
    newData = unwrap(arcValue)
    plt.plot(newData)
    plt.show()


def unwrap(data):
    resultData = []
    diffs = np.roll(data, -1) - data
    diffs = diffs[:len(data) - 1]
    first_value = data[0]
    resultData.append(first_value)
    previous_value = first_value
    current_value=None
    for diff in diffs:
        if diff > np.pi / 2:
            current_value = previous_value + diff - np.pi
            resultData.append(current_value)
        elif diff < -np.pi / 2:
            current_value = previous_value + diff + np.pi
            resultData.append(current_value)
        else:
            current_value=previous_value+diff
            resultData.append(current_value)
        previous_value = current_value
    return np.asarray(resultData)

def getDerivative(data):
    derivativeQ = []
    for i in range(len(data) - 1):
        derivativeQ.append((data[i + 1] - data[i]))
    return derivativeQ


def removeDC(data):
    return data - np.mean(data)


def distanceLine(phase, freq):
    distances = np.zeros(len(phase) - 1)
    for i in np.arange(1, len(phase)):
        phaseDiff = phase[0] - phase[i]
        distanceDiff = 343 / (2 * np.pi * freq) * phaseDiff
        distances[i - 1] = distanceDiff
    distances = distances / 2
    return distances


def getPhase(Q, I):
    if I == 0 and Q > 0:
        return np.pi / 2
    elif I == 0 and Q < 0:
        return 3 / 2 * np.pi
    elif Q == 0 and I > 0:
        return 0
    elif Q == 0 and I < 0:
        return np.pi
    tanValue = Q / I
    tanPhase = np.arctan(tanValue)
    resultPhase = 0
    if I > 0 and Q > 0:
        resultPhase = tanPhase
    elif I < 0 and Q > 0:
        resultPhase = np.pi + tanPhase
    elif I < 0 and Q < 0:
        resultPhase = np.pi + tanPhase
    elif I > 0 and Q < 0:
        resultPhase = 2 * np.pi + tanPhase
    return resultPhase


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
    win_size = 400
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    new_len = new_len*2
    result = np.zeros(new_len)
    for index in range(0, new_len):
        start =  (index/2)*win_size
        end = (index/2+1)*win_size
        result[index] = np.mean(data[int(start):int(end)])
    return result

def getI(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulCos = np.cos(2 * np.pi * f * times) * data
    return mulCos



def getQ(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulSin = -np.sin(2 * np.pi * f * times) * data
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
