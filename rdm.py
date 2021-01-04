# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

# parameters setting

# 无法在声波中使用，有时间可以学习一下

B = 135e6  # Sweep Bandwidth
T = 36.5e-6  # Sweep Time
c = 3e8  # Speed of Light
f0 = 76.5e9  # Start Frequency
# B = 4e3
# T = 0.04
# c = 343
# f0 = 18e3

N = 512  # Sample Length
L = 128  # Chirp Total
NumRangeFFT = 512  # Range FFT Length
NumDopplerFFT = 128  # Doppler FFT Length
rangeRes = c / 2 / B  # Range Resolution
velRes = c / 2 / f0 / T / NumDopplerFFT  # Velocity Resolution
maxRange = rangeRes * NumRangeFFT  # Max Range
maxVel = velRes * NumDopplerFFT / 2  # Max Velocity
tarR = [50, 90]  # Target Range
tarV = [3, 20]  # Target Velocity

# generate receive signal

S1 = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    for n in range(0, N):
        S1[l][n] = np.exp(np.complex(0, 1) * 2 * np.pi * (
                    ((2 * B * (tarR[0] + tarV[0] * T * l)) / (c * T) + (2 * f0 * tarV[0]) / c) * (T / N) * n + (
                        2 * f0 * (tarR[0] + tarV[0] * T * l)) / c))

S2 = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    for n in range(0, N):
        S2[l][n] = np.exp(np.complex(0, 1) * 2 * np.pi * (
                    ((2 * B * (tarR[1] + tarV[1] * T * l)) / (c * T) + (2 * f0 * tarV[1]) / c) * (T / N) * n + (
                        2 * f0 * (tarR[1] + tarV[1] * T * l)) / c))

sigReceive = S1 + S2

# range win processing

sigRangeWin = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeWin[l] = np.multiply(sigReceive[l], np.hamming(N).T)

# range fft processing

sigRangeFFT = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeFFT[l] = np.fft.fft(sigRangeWin[l], NumRangeFFT)

# doppler win processing

sigDopplerWin = np.zeros((L, N), dtype=complex)
for n in range(0, N):
    sigDopplerWin[:, n] = np.multiply(sigRangeFFT[:, n], np.hamming(L).T)

# doppler fft processing

sigDopplerFFT = np.zeros((L, N), dtype=complex)
for n in range(0, N):
    # 区别在这里
    sigDopplerFFT[:, n] = np.fft.fftshift(np.fft.fft(sigDopplerWin[:, n], NumDopplerFFT))

fig = plt.figure()
ax = Axes3D(fig)

x = np.arange(0, NumRangeFFT * rangeRes, rangeRes)
y = np.arange((-NumDopplerFFT / 2) * velRes, (NumDopplerFFT / 2) * velRes, velRes)
# x = np.arange(NumRangeFFT)
# y = np.arange(NumDopplerFFT)
# print(len(x))
# print(len(y))
X, Y = np.meshgrid(x, y)
Z = np.abs(sigDopplerFFT)
r, c = divmod(np.argmax(Z), Z.shape[1])
print(X[r][c], Y[r][c])
ax.plot_surface(X, Y, Z,
                rstride=1,  # rstride（row）指定行的跨度
                cstride=1,  # cstride(column)指定列的跨度
                cmap=plt.get_cmap('rainbow'))  # 设置颜色映射

ax.invert_xaxis()  # x轴反向

plt.show()
