import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
def cos_wave(A, f, fs, t, phi=0):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    # 默认类型为float32
    y = A * np.cos(2 * np.pi * f * n * Ts + phi * (np.pi / 180)).astype(np.float32)
    return y

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

np.random.seed(1123)
f = 1000
fs = 16000
t = 10
signal = cos_wave(1, f, fs, t, phi=0)
noise = wgn(signal, 10)
x = signal + noise

cal_snr = 10*np.log10(np.mean(signal**2) / np.mean(noise ** 2))
print('calculate snr in time domain: ', cal_snr)

NFFT = 2048
# 要除NFFT
X = fftshift(fft(x, NFFT)) / NFFT
f = fftshift(fftfreq(NFFT, d=1/fs))
f_single = f[f>=0]
X_signal = X[f>=0]
X_signal[1:] = X_signal[1:] * 2

signal_power = np.abs(X_signal[np.where(f_single == 1000)[0]])[0]
print('signal power: ', signal_power)
SNR = signal_power / ((np.sum(np.abs(X))-signal_power)/(NFFT-1))
print('calculate snr in freq domain: ', 10 * np.log10(SNR))

db = 10 * np.log10(np.abs(X_signal))

plt.plot(f_single, db)
plt.show()
