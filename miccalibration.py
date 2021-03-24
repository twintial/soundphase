'''
测试用于麦克风阵列的时延迟，用于校准
'''
from IQ import load_audio_data
import numpy as np
import matplotlib.pyplot as plt
from arlpy import bf

def cons_uca(r):
    theta = np.pi / 3
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    return np.array(pos)

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

def beamform_real(data, sd, fs=48000.0):
    """
    这种方法分辨率只有15度，越靠近15的倍数度数效果越好
    :param data:
    :param sd:
    :param fs:
    :return:
    """
    tailor_frames_nums = np.round(sd[0] * fs) + 6
    tailor_frames_nums = tailor_frames_nums.astype(np.int16)
    print(tailor_frames_nums)
    # 防止 max_tailor_frames - tailor_frames_num == 0
    max_tailor_frames = np.max(tailor_frames_nums) + 1
    beamformed_data = []
    for i, tailor_frames_num in enumerate(tailor_frames_nums):
        beamformed_data.append(data[i, tailor_frames_num:tailor_frames_num - max_tailor_frames])
    return np.array(beamformed_data)

def pearsonCroCor(data, sample, mode=0):
        if mode == 2:
            corresult = np.correlate(data, sample)
        else:
            corresult = np.zeros(data.size - sample.size + 1, dtype=np.float)
            # slide the data, every time pick sample.size elements to do pearson with sample
            i = 0
            sample_size = sample.size
            test_rel = np.correlate(sample, sample)
            while i + sample_size <= data.size:
                if mode == 1:
                    corresult[i] = np.correlate(data[i:i + sample_size], sample)[0]
                else:
                    # corresult[i]=pearsonr(data[i:i+sample_size],sample)[0]
                    corresult[i] = np.corrcoef(data[i:i + sample_size], sample)[0][1]

                i += 1
        return corresult
def crosscor():
    # data, fs = load_audio_data(r'D:\projects\pyprojects\soundphase\calib\0\0.wav', 'wav')
    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\1khz\0.wav', 'wav')
    t1 = 2
    t2 = 3
    data1 = data[48000 * t1:48000 * t2 + 500, 0].T
    data2 = data[48000 * t1:48000 * t2, 1].T
    print(len(data2)/48000)
    plt.plot((np.arange(48000 * t1, len(data2) + 48000 * t1))/48000, data2)
    plt.show()

    '''
    测试了mic之间的时间延迟，效果和理论计算一致
    '''
    corr = pearsonCroCor(data1, data2)
    # npsignalplot(corr)
    plt.plot(corr)
    plt.show()

    index = np.argmax(corr)
    print(index)

def get_sinusoid(fs, t):
    A = [1, 1, 1, 1, 1, 1, 1, 1]
    alpha = 1 / sum(A)
    y = A[0] * cos_wave(1, 18000, fs, t)
    for i in range(1, 8):
        y = y + A[i] * cos_wave(1, 17000 + i * 400, fs, t)
    signal = alpha * y
    return signal


def simu():
    fs = 48e3
    pos = cons_uca(0.043)
    # data_1 = cos_wave(1, 20e3, fs, 10)
    data_1 = get_sinusoid(fs, 10)
    data_7 = np.tile(data_1, (7, 1))
    theta = [[0, 0]]
    sd = -bf.steering_plane_wave(pos, 343, theta)
    data = beamform_real(data_7, sd, fs)
    data = data.T

    t1 = 2
    t2 = 3
    data1 = data[int(fs * t1):int(fs * t2 + 500), 4].T
    data2 = data[int(fs * t1):int(fs * t2), 1].T
    # print(len(data2)/48000)
    # plt.plot((np.arange(48000 * t1, len(data2) + 48000 * t1))/48000, data2)
    # plt.show()

    '''
    测试了mic之间的时间延迟，效果和理论计算一致
    '''
    corr = pearsonCroCor(data1, data2, mode=1)
    # npsignalplot(corr)
    # plt.scatter(np.arange(len(corr)), corr)
    plt.plot(corr)
    plt.show()

    index = np.argmax(corr)
    print(index)




if __name__ == '__main__':
    simu()
    # crosscor()