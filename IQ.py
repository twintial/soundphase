import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter, find_peaks_cwt, find_peaks, normalize, filtfilt
from scipy import signal
from unwrap import *
import wave
from statsmodels.tsa.seasonal import seasonal_decompose
from staticremove import *
import sklearn
import time

index = 0
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
    return filtfilt(b, a, data, axis=-1)


# butter lowpass
def butter_lowpass_filter(data, cutoff, fs=48e3, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def power2db(power):
    if np.any(power < 0):
        raise ValueError("power less than 0")
    db = (10 * np.log10(power) + 300) - 300
    return db
def normalized_signal_fft(data, fs=48e3, figure=False, xlim = (0,25e3)):
    N = len(data)
    y = np.abs(fft(data)) / N
    # 这里要不要乘2？
    y_signle: np.ndarray = y[:int(np.round(N / 2))] * 2
    x = fftfreq(N) * fs
    x = x[x >= 0]
    db = power2db(y_signle)

    # 用于调制振幅
    peaks, _ = find_peaks(y_signle, height=20)
    plt.plot(y_signle)
    plt.plot(peaks, y_signle[peaks], "x")
    plt.plot(np.zeros_like(y_signle), "--", color="gray")
    plt.show()
    print(x[peaks])
    print(y_signle[peaks])

    if figure:
        plt.figure()
        plt.plot(x, y_signle)
        plt.xlim(xlim)
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
    times = times + 2347
    I_raw = np.cos(2 * np.pi * f * times) * data
    Q_raw = -np.sin(2 * np.pi * f * times) * data
    #
    # I = butter_lowpass_filter(I_raw, 200)
    # Q = butter_lowpass_filter(Q_raw, 200)
    I = my_move_average_overlap(I_raw, 200)
    Q = my_move_average_overlap(Q_raw, 200)

    if figure:
        plt.figure()
        plt.plot(I.reshape((-1)), label='I')
        plt.plot(Q.reshape((-1)), label='Q')
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
    # angle = angle[500:]
    # angle[0:400] = angle[400:800]  # 暂时的方法
    u_a = angle
    # u_a = np.unwrap(angle)
    # u_a = angle
    # u_a = np.diff(u_a)
    if figure:
        plt.figure()
        plt.plot(np.arange(0, len(u_a.reshape(-1))) * 1 / fs, u_a.reshape(-1))
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (rad)')
        plt.title('Phase')
    return u_a

def get_magnitude(I: np.ndarray, Q: np.ndarray, figure=False) -> np.ndarray:
    signal = I + 1j * Q
    magn = np.abs(signal)
    magn = 10*np.log10(magn)
    # 用麦克风阵列最后几个值会有一些问题，去掉比较好
    magn = magn[:-10]
    if figure:
        plt.figure()
        plt.plot(np.diff(magn.reshape(-1)))
        plt.ylabel('Magnitude (power)')
        plt.title('Magnitude')
    return magn

def move_average_overlap(data):
    win_size = 200
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    new_len = new_len*2
    result = np.zeros(new_len)
    for index in range(0, new_len):
        start = (index/2)*win_size
        end = (index/2+1)*win_size
        result[index] = np.mean(data[int(start):int(end)])
    return result


def my_move_average_overlap(data, n):
    if len(data.shape) == 1:
        data = data.reshape((1, -1))
    ret = np.cumsum(data, axis=-1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    b = ret[:, n - 1:] / n
    index = np.arange(0, b.shape[1], 100)
    return b[:, index]

# 画圈圈，使用时要保证不能有其他图画，不然会有问题
def draw_circle(I, Q, fs=48e3):
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.5])
    ax.set_xlim([0, 1.5])
    circle, = ax.plot(0, 0, label='I/Q')
    timer = fig.canvas.new_timer(interval=100)
    def OnTimer(ax):
        global index
        speed = 300
        circle.set_ydata(Q[:index*speed])
        circle.set_xdata(I[:index*speed])
        index = index + 1
        ax.draw_artist(circle)
        ax.figure.canvas.draw()
        if index*speed > len(Q):
            print("end")
        else:
            print(f"time:{(index*speed)/fs}")
    timer.add_callback(OnTimer, ax)
    timer.start()
    plt.show()


def normalize_max_min(x):
    max = np.max(x)
    min = np.min(x)
    return (x-min)/(max-min)


# free device的实现
def path_length_change_estimation(data):
    for fID in range(0, 7):
        fc = 17350 + 700 * fID
        data_filter = butter_bandpass_filter(data, fc - 150, fc + 150)
        normalized_signal_fft(data_filter, figure=False)
        plt.show()
        I, Q = get_IQ(data_filter, fc, figure=False)
        I_denoised, Q_denoised = denoised_IQ(I, Q, 300, figure=False)
        static_I = LEVD(I_denoised, Thr=0.0015)
        static_Q = LEVD(Q_denoised, Thr=0.0015)
        phase = get_phase(I_denoised-static_I, Q_denoised-static_Q, figure=False)
        # plt.show()
        # if fID == 5:
        #     n_I = normalize_max_min(I_denoised)
        #     n_Q = normalize_max_min(Q_denoised)
        #     draw_circle(n_I, n_Q)
        d = -((phase[-1] - phase[0]) / (2 * np.pi)) * 343 / fc
        print(f"fc={fc}, d={d}")


def demo():
    # data, fs = load_audio_data(r'D:\experiment\data\2021\毕业设计\soundgestreco\0\0\1.wav', 'wav')
    data, fs = load_audio_data(r'D:\实验数据\2021\毕设\micarray\sjj\gesture1\0.wav', 'wav')
    data = data[48000 * 2:, 3].T
    # data, fs = load_audio_data(r'D:\projects\pyprojects\andriodfaceidproject\temp\word1\shenjunjie\0.pcm', 'pcm')
    # data, fs = load_audio_data(r'gest\sjj\gesture1\1.pcm', 'pcm')
    # data = data[48000*1:]
    data_filter = butter_bandpass_filter(data, 15e3, 23e3)
    # data = data[48000:, 0]
    normalized_signal_fft(data_filter, figure=True, xlim=(15e3, 23e3))
    plt.show()
    # fc = 17350 + 700 * 0
    step = 350
    fc = 17350
    for i in range(8):
        fc = 17350 + step * i
        data_filter = butter_bandpass_filter(data, fc - 150, fc + 150)
        # data = data[48000:, 0]
        normalized_signal_fft(data_filter, figure=True, xlim=(15e3, 23e3))
        plt.show()

        I, Q = get_IQ(data_filter, fc, figure=True)
        # denoise
        decompositionQ = seasonal_decompose(Q.T, period=10, two_sided=False)
        trendQ = decompositionQ.trend[10:]
        decompositionI = seasonal_decompose(I.T, period=10, two_sided=False)
        trendI = decompositionI.trend[10:]

        # static_I = LEVD(I_denoised, Thr=0.0015)
        # static_Q = LEVD(Q_denoised, Thr=0.0015)

        trendI = trendI[10:]
        trendQ = trendQ[10:]
        # trendI[:10] = trendI[10:]
        plt.plot(trendI)
        plt.plot(trendQ)
        plt.show()


        # phase = get_phase(trendI.reshape(1,-1), trendQ.reshape(1,-1), figure=True)
        # print(f"标准差:{np.std(phase)}")
        # plt.show()
        magn = get_magnitude(trendI, trendQ, figure=True)
        plt.show()
    fc = 17350
    for win in range(0, len(data), 2048):
        data_win = data[win:win+2048]
        print(f"time:{1 + win / fs}")
        for i in range(1):
            fc = 17350 + 700 * i
            data_filter = butter_bandpass_filter(data_win, fc - 250, fc + 250)
            # data = data[48000:, 0]
            # normalized_signal_fft(data_filter, figure=True, xlim=(15e3, 23e3))
            # plt.show()

            I, Q = get_IQ(data_filter, fc, figure=False)
            phase = get_phase(I, Q, figure=True)
            print(f"标准差:{np.std(phase)}")
            plt.show()


    p = find_period(I[48000:3 * 48000])
    print(p)
    I_denoised, Q_denoised = denoised_IQ(I, Q, p, figure=False)  # p=500还不错
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

    # p = find_period(I[48000:4*48000])
    # print(p)
    # I_denoised, Q_denoised = denoised_IQ(I, Q, p, figure=True)
    #
    phase = get_phase(I_denoised - static_I, Q_denoised - static_Q, figure=True)
    print(phase[48000 * 6])
    print(phase[48000 * 1])
    t1 = 6
    t0 = 1
    d = -((phase[48000 * t1] - phase[48000 * t0]) / (2 * np.pi)) * 343 / fc
    print(d)
    # get_phase(normalize(I), normalize(Q), figure=True)

    # normalized_signal_fft(I, figure=True)
    plt.show()


def phasediff_between_mic():
    data, fs = load_audio_data(r'D:\实验数据\2021\毕设\micarray\sjj\gesture1\14.wav', 'wav')
    # plt.plot(data[:,0])
    # plt.show()
    data = data[48000 * 2:, :].T
    step = 350

    all_mic_phases = []
    for mic_num in range(7):
        cur_data = data[mic_num]
        phases = []
        for i in range(8):
            fc = 17350 + step * i
            data_filter = butter_bandpass_filter(cur_data, fc - 150, fc + 150)

            I, Q = get_IQ(data_filter, fc, figure=False)
            # denoise
            decompositionQ = seasonal_decompose(Q.T, period=10, two_sided=False)
            trendQ = decompositionQ.trend[10:]
            decompositionI = seasonal_decompose(I.T, period=10, two_sided=False)
            trendI = decompositionI.trend[10:]
            phase = get_magnitude(trendI.reshape(1, -1), trendQ.reshape(1, -1), figure=False)
            phases.append(phase.reshape(-1))
        all_mic_phases.append(phases)
    all_mic_phases = np.array(all_mic_phases)
    # for i in range(7):
    #     temp = all_mic_phases[:, i, :]
    #     plt.plot(temp.T)
    #     plt.show()
    # axis=0不同麦克分之间差分，axis=-1时间维度做差分
    print(all_mic_phases.shape)
    phase_diff = np.diff(all_mic_phases, axis=-1)
    print(phase_diff.shape)
    # u_p_ds = np.unwrap(phase_diff, axis=-1)
    u_p_ds = phase_diff
    for i in range(6):
        u_p_d = u_p_ds[i]
        print(i)
        for x in u_p_d:
            plt.plot(x)
        plt.show()



if __name__ == '__main__':
    # data, fs = load_audio_data(r'0.pcm', 'pcm')
    # data = data[48000:]
    # for i in range(0, len(data), 512):
    #     path_length_change_estimation(data[i:i+512])
    demo()
    # phasediff_between_mic()