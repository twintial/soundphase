import numpy as np
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

f = 1000
fs = 16000
t = 1
signal = cos_wave(1, f, fs, t, phi=0)

SNR = 10
noise = np.random.randn(len(signal))
noise = noise / np.linalg.norm(noise) * np.linalg.norm(signal) / (10**(SNR/20))

x = signal + noise


actualSNR = 20*np.log10(np.linalg.norm(signal)/np.linalg.norm(x - signal))

print(actualSNR)

NFFT = 2048
X = fftshift(fft(x, NFFT))
f = fftshift(fftfreq(NFFT, d=1/fs))
f_single = f[f>=0]
X = X[f>=0]
X[1:] = X[1:] * 2

signal_power = np.abs(X[np.where(f_single == 1000)[0]])[0]
SNR = signal_power / ((np.sum(np.abs(X))-signal_power)/(NFFT-1))
print(20 * np.log10(SNR))

import matplotlib.pyplot as plt
# plt.plot(f_single, np.abs(X))
# plt.show()
plt.psd(x, NFFT, fs)
plt.show()
