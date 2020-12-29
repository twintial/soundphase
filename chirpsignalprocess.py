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
    s = FMCW_wave_d(48e3, times, 0.04, 18e3, 20e3, 0) * data
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
    # 存在问题
    ts = np.arange(0, T, 1.0 / fs)
    # -T<tw<T
    if tw >= T or tw <= -T:
        tw = tw % T
    # ts = ts + tw
    B = f1 - f0
    k = B / T
    sweep = np.exp(1j * 2 * np.pi * (f0 * ts + (1 / 2) * k * np.power(ts, 2)))
    c = int(times[-1] / T)  # 这里取整
    rest = len(times) - c * len(ts)
    y = []
    for i in range(c):
        y = np.hstack((y, sweep.real))
    y = np.hstack((y, sweep.real[:rest]))
    l = int(tw * fs)
    y = np.hstack((y[l:], y[:l]))
    return y


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


if __name__ == '__main__':
    # 读取数据
    # 2,3 很正常,0,1有问题
    data, fs = load_audio_data('chirp/0/4.wav', type='wav')
    data = data[48000:, 0]  # 只用一个声道
    data = butter_bandpass_filter(data, 15e3, 22e3)

    # 模拟
    # t = 10
    # fs = 48e3
    # data = FMCW_wave_d(48e3, np.arange(0, t, 1 / 48e3), 0.04, 18e3, 20e3, tw=0.01)
    # data = data[48000:]

    draw_spec(data, fs)
    xm = get_xm(data)
    draw_spec(xm, fs)
    x, y = normalized_signal_fft(xm, figure=True)

    fd_index = np.argmax(y)
    fd = x[fd_index]
    print(fd)

    v_xm = get_virtual_xm(data, fs, 0.04, 18e3, 20e3)
    # v_xm_complex = signal.hilbert(v_xm)
    # get_phase(v_xm_complex.real, v_xm_complex.imag, figure=True)
    # draw_circle(normalize_max_min(v_xm_complex.real), normalize_max_min(v_xm_complex.imag))
    plt.show()
