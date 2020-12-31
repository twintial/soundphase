import scipy.signal as signal
import numpy as np
from IQ import *


def get_xm(data, fs=48e3):
    # 不支持多声道
    assert len(data.shape) == 1
    times = np.arange(0, len(data)) * 1 / fs
    # s = signal.chirp(times, 18e3, 0.04, 20e3, method='linear') * data
    # fmcw_t = FMCW_wave_d(48e3, times, 0.04, 18e3, 20e3, 0)
    # s = fmcw_t[48000:] * data[48000:]
    s = FMCW_wave_d(48e3, times, 0.02, 18e3, 22e3, 0) * data
    # 1000应该可以支持1.5m以上的距离变动
    return butter_lowpass_filter(s, 5000)


def FMCW_wave(fs, times, T, f0, f1):
    ts = np.arange(0, T, 1.0 / fs)
    sweep = signal.chirp(ts, f0, T, f1, method='linear').astype(np.float32)
    c = int(times[-1] / T) + 1  # 这里取整
    rest = len(times) - c * len(ts)
    y = sweep
    for i in range(c - 1):
        y = np.hstack((y, sweep))
    y = np.hstack((y, sweep[:rest]))
    return y


def FMCW_wave_d(fs, times, T, f0, f1, tw: float = 0):
    # 有问题,tw需要>0
    ts = np.arange(0, T, 1.0 / fs)
    # -T<tw<T
    if tw >= T or tw <= -T:
        tw = tw % T
    # ts = ts + tw
    B = f1 - f0
    k = B / T
    sweep = np.exp(1j * 2 * np.pi * (f0 * ts + (1 / 2) * k * np.power(ts, 2)))
    # c = int((times[-1]+1/fs) / T)  # 这里取整
    c = int(len(times)/len(ts))
    rest = len(times) - c * len(ts)
    y = []
    for i in range(c):
        y = np.hstack((y, sweep.real))
    y = np.hstack((y, sweep.real[:rest]))
    # 平移
    if tw != 0:
        l = int(tw * fs)
        temp = np.hstack((sweep.real, sweep.real))
        # 不知道对不对
        y = np.hstack((y[l:], temp[rest:rest + l]))
    return y


def FMCW_wave_d_with_sep(fs, times, T, f0, f1, tw: float = 0):
    # 还有点问题不能使用
    ts = np.arange(0, T, 1.0 / fs)
    # -T<tw<T
    if tw >= T or tw <= -T:
        tw = tw % T
    # ts = ts + tw
    B = f1 - f0
    k = B / T
    sweep = np.exp(1j * 2 * np.pi * (f0 * ts + (1 / 2) * k * np.power(ts, 2)))
    sweep = np.hstack((sweep, np.zeros_like(sweep)))
    c = int((times[-1]+1/fs) / (T))  # 这里取整
    rest = len(times) - c * len(ts)
    y = []
    for i in range(c):
        y = np.hstack((y, sweep.real)) # 增加了间隔
    y = np.hstack((y, sweep.real[:rest]))
    # 平移
    if tw != 0:
        l = int(tw * fs)
        y = np.hstack((y[l:], sweep.real[rest:rest + l]))
    return y
def get_xm_sep(data, fs=48e3):
    # 还有点问题不能使用
    # 不支持多声道
    assert len(data.shape) == 1
    times = np.arange(0, len(data)) * 1 / fs
    # s = signal.chirp(times, 18e3, 0.04, 20e3, method='linear') * data
    # fmcw_t = FMCW_wave_d(48e3, times, 0.04, 18e3, 20e3, 0)
    # s = fmcw_t[48000:] * data[48000:]
    s = FMCW_wave_d_with_sep(48e3, np.arange(0, 0.08, 1 / 48e3), 0.04, 18e3, 20e3, 0) * data
    # 1000应该可以支持1.5m以上的距离变动
    return butter_lowpass_filter(s, 5000)


def get_virtual_xm(data, fs, T, f0, f1):
    virtual_xm = 0
    times = np.arange(0, len(data)) * 1 / fs
    B = f1 - f0
    # k = B / T
    xm = get_xm(data, fs)
    f, y = normalized_signal_fft(xm, figure=True)
    plt.show()
    fd_index = np.argmax(y)
    fd = f[fd_index]
    tw = fd * T / (B)  # 为什么是2B,好像应该是B
    while fd != 0:
        tw = T - tw
        virtual_xt = FMCW_wave_d(fs, times, T, f0, f1, tw=tw)
        s = virtual_xt * data
        virtual_xm = butter_lowpass_filter(s, 5000)
        f, y = normalized_signal_fft(virtual_xm, figure=True)
        plt.show()
        fd_index = np.argmax(y)
        fd = f[fd_index]
    return virtual_xm
def get_virtual_xm_with_xm(xm, fs, T, f0, f1):
    virtual_xm = 0
    times = np.arange(0, len(data)) * 1 / fs
    B = f1 - f0
    # k = B / T
    f, y = normalized_signal_fft(xm, figure=False)
    fd_index = np.argmax(y)
    fd = f[fd_index]
    tw = fd * T / (B)  # 为什么是2B,好像应该是B
    while fd - 0 > 50:
        tw = T - tw
        virtual_xt = FMCW_wave_d(fs, times, T, f0, f1, tw=tw)
        s = virtual_xt * data
        virtual_xm = butter_lowpass_filter(s, 5000)
        f, y = normalized_signal_fft(virtual_xm, figure=True)
        plt.show()
        fd_index = np.argmax(y)
        fd = f[fd_index]
    return virtual_xm


def windowed_analyze(data, window_size=2048):
    xm = get_xm(data)
    for i in range(0, len(data), window_size):
        # sample_window = data[i:i+window_size]
        # # draw_spec(sample_window, fs)
        # xm = get_xm(sample_window)
        # draw_spec(xm, fs)
        x, y = normalized_signal_fft(xm[i:i+window_size], figure=True)
        fd_index = np.argmax(y)
        fd = x[fd_index]
        print(fd)
        print(f"time:{1+i/fs}")
        plt.show()
        v_xm = get_virtual_xm_with_xm(xm[i:i+window_size], fs, 0.04, 18e3, 20e3)

if __name__ == '__main__':
    # 读取数据
    # 2,3 很正常,0,1有问题
    data, fs = load_audio_data('chirp/20ms18-22/3.wav', type='wav')
    # data = np.diff(data[:, 0])
    data = data[48000:, 0]  # 只用一个声道
    data = butter_bandpass_filter(data, 15e3, 23e3)
    # windowed_analyze(data)
    # 模拟
    # t = 10
    # fs = 48e3
    # data1 = FMCW_wave_d(48e3, np.arange(0, t, 1 / 48e3), 0.02, 18e3, 22e3, tw=0.005)
    # data2 = FMCW_wave_d(48e3, np.arange(0, t, 1 / 48e3), 0.02, 18e3, 22e3, tw=0.0025)
    # data = data1+data2
    # data = data[48000:]

    draw_spec(data, fs)
    xm = get_xm(data)
    draw_spec(xm, fs)
    x, y = normalized_signal_fft(xm, figure=True, xlim=(-10, 6000))
    #
    fd_index = np.argmax(y)
    fd = x[fd_index]
    print(fd)
    #
    # v_xm = get_virtual_xm(data, fs, 0.02, 16e3, 21e3)

    # v_xm_complex = signal.hilbert(v_xm)
    # get_phase(v_xm_complex.real, v_xm_complex.imag, figure=True)
    # draw_circle(normalize_max_min(v_xm_complex.real), normalize_max_min(v_xm_complex.imag))
    plt.show()
