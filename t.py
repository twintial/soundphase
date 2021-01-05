import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''


    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取帧速率
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    wave_data = np.frombuffer(str_data, dtype='int16')  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    wave_data = wave_data
    return wave_data, framerate


def wav_show(wave_data, fs):  # 显示出来声音波形
    time = np.arange(0, len(wave_data)) * (1.0 / fs)  # 计算声音的播放时间，单位为秒
    # 画声音波形
    plt.plot(time, wave_data)
    plt.show()

if (__name__ == '__main__'):
    # import matplotlib.pyplot as plt
    # from scipy.misc import electrocardiogram
    # from scipy.signal import find_peaks
    # x = electrocardiogram()[2000:4000]
    # peaks, _ = find_peaks(x, height=0)
    # plt.plot(x)
    # plt.plot(peaks, x[peaks], "x")
    # # plt.plot(np.zeros_like(x), "--", color="gray")
    # plt.show()
    # from sklearn.preprocessing import normalize
    # a = np.arange(10).reshape(1,-1)
    # print(a)
    # print(normalize(a,norm='max'))
    a = np.array([[0,1,2], [1,3,0]])
    b = np.hstack((a,a))
    x = np.argmax(a, axis=1)
    print(x)