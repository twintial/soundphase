'''
程序中用hist()检查噪声是否是高斯分布，psd()检查功率谱密度是否为常数。
'''
import numpy as np
import pylab as plt

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

plt.figure()
t = np.arange(0, 10000000) * 0.1
x = np.sin(t)
n = wgn(x, 6)
xn = x+n # 增加了6dBz信噪比噪声的信号
plt.subplot(221)
plt.plot(t[:1024],x[:1024])
plt.title('The original signal-x')

plt.subplot(222)
plt.plot(t[:1024],xn[:1024])
plt.title('The original sinal with Gauss White Noise')

plt.subplot(223)
plt.hist(n, bins=100)
plt.title('Gauss Noise Distribution')

plt.subplot(224)
plt.psd(n)
plt.title('PSD')
plt.tight_layout()

plt.savefig('show the result.png',dpi=600)