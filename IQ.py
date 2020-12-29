import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter, find_peaks_cwt, find_peaks, normalize
from scipy import signal
from unwrap import *
import wave
from statsmodels.tsa.seasonal import seasonal_decompose
import sklearn
import time

index = 400
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

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
def load_audio_data(filename, type='pcm'):
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


# butter band滤波
def butter_bandpass_filter(data, lowcut, highcut, fs=48e3, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=-1)


# butter lowpass
def butter_lowpass_filter(data, cutoff, fs=48e3, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def power2db(power):
    if np.any(power < 0):
        raise ValueError("power less than 0")
    db = (10 * np.log10(power) + 300) - 300
    return db
def normalized_signal_fft(data, fs=48e3, figure=False):
    N = len(data)
    y = np.abs(fft(data)) / N
    # 这里要不要乘2？
    y_signle = y[:int(N / 2)] * 2
    x = fftfreq(N) * fs
    x = x[x >= 0]
    db = power2db(y_signle)
    if figure:
        plt.figure()
        plt.plot(x, y_signle)
        plt.xlim((-1000, 5000))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('power')
        plt.title('单边边振幅谱（归一化）')
    return x, y_signle
def draw_spec(x, fs):
    f, t, Sxx = signal.spectrogram(x, fs)
    plt.figure()
    plt.pcolormesh(t, f, Sxx, shading='auto')
    plt.ylabel('Frequency [Hz]')
    # plt.ylim((15e3, 21e3))
    plt.xlabel('Time [sec]')
    plt.show()



def get_IQ(data, f, fs=48e3, figure=False):
    # tx为频率为f的cos signal
    # 不支持多声道
    times = np.arange(0, len(data)) * 1 / fs
    I_raw = np.cos(2 * np.pi * f * times) * data
    Q_raw = -np.sin(2 * np.pi * f * times) * data

    I = butter_lowpass_filter(I_raw, 200)
    Q = butter_lowpass_filter(Q_raw, 200)
    if figure:
        plt.figure()
        plt.plot(times, I, label='I')
        plt.plot(times, Q, label='Q')
        plt.xlabel('Time (s)')
        plt.ylabel('I/Q')
        plt.title('I/Q')
        plt.legend()
    return I, Q

def find_period(data):
    peaks, _ = find_peaks(data)
    internal = np.mean(np.diff(peaks))
    return int(internal)
    # plt.figure()
    # plt.plot(data)
    # plt.plot(peaks, data[peaks], "x")
    # plt.plot(np.zeros_like(x), "--", color="gray")

def denoised_IQ(I, Q, period, fs=48e3, figure=False):
    # 用seasonal_decompose
    I_d = seasonal_decompose(I, period=period, two_sided=False).trend
    I_d = np.hstack((np.array([I_d[period]] * period), I_d[period:]))
    Q_d = seasonal_decompose(Q, period=period, two_sided=False).trend
    Q_d = np.hstack((np.array([Q_d[period]] * period), Q_d[period:]))
    if figure:
        plt.figure()
        plt.plot(np.arange(0, len(I_d)) * 1 / fs, I_d, label='I_d')
        plt.plot(np.arange(0, len(Q_d)) * 1 / fs, Q_d, label='Q_d')
        plt.xlabel('Time (s)')
        plt.ylabel('I/Q')
        plt.title('I/Q')
        plt.legend()
    return I_d, Q_d

def get_phase(I, Q, fs=48e3, figure=False):
    # s_u_a = get_phase_new(I, Q)
    signal = I + 1j*Q
    angle = np.angle(signal)
    u_a = np.unwrap(angle)
    # d = np.diff(u_a)
    if figure:
        plt.figure()
        plt.plot(np.arange(0, len(u_a)) * 1 / fs, u_a)
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (rad)')
        plt.title('Phase')
    return u_a

# 画圈圈，使用时要保证不能有其他图画，不然会有问题
def draw_circle(I, Q):
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.5])
    ax.set_xlim([0, 1.5])
    circle, = ax.plot(0, 0, label='I/Q')
    timer = fig.canvas.new_timer(interval=100)
    def OnTimer(ax):
        global index
        n = 300
        circle.set_ydata(Q[:index*n])
        circle.set_xdata(I[:index*n])
        index = index + 1
        ax.draw_artist(circle)
        ax.figure.canvas.draw()
        if index*n > len(Q):
            print("end")
        else:
            print(f"time:{n / 48000 * index}")
    timer.add_callback(OnTimer, ax)
    timer.start()
    plt.show()


def normalize_max_min(x):
    max = np.max(x)
    min = np.min(x)
    return (x-min)/(max-min)


if __name__ == '__main__':
    from staticremove import *
    data, fs = load_audio_data(r'0.pcm', 'pcm')
    # data = data[:, 0].T
    data = data[48000:]
    fc = 17350 + 700*0
    data = butter_bandpass_filter(data, fc-250, fc+250)
    # normalized_signal_fft(data, figure=True)
    I, Q = get_IQ(data, fc, figure=False)
    p = find_period(I[48000:3 * 48000])
    print(p)
    I_denoised, Q_denoised = denoised_IQ(I, Q, p, figure=False) # p=500还不错
    static_I = LEVD(I_denoised, Thr=0.0015)
    static_Q = LEVD(Q_denoised, Thr=0.0015)
    # print(np.max(static_Q))
    # plt.figure()
    # plt.plot(static_Q)
    # plt.figure()
    # plt.plot(static_I)

    # n_I = normalize_max_min(I_denoised - static_I)
    # n_Q = normalize_max_min(Q_denoised - static_Q)
    # draw_circle(n_I, n_Q)

    #p = find_period(I[48000:4*48000])
    # print(p)
    #I_denoised, Q_denoised = denoised_IQ(I, Q, p, figure=True)
    #
    phase = get_phase(I_denoised - static_I, Q_denoised - static_Q, figure=True)
    print(phase[48000*6])
    print(phase[48000*1])
    t1 = 6
    t0 = 1
    d = -((phase[48000*t1]-phase[48000*t0])/(2*np.pi))*343/fc
    print(d)
    # get_phase(normalize(I), normalize(Q), figure=True)


    # normalized_signal_fft(I, figure=True)
    plt.show()
