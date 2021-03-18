'''
测试用于麦克风阵列的时延迟，用于校准
'''
from IQ import load_audio_data
import numpy as np
import matplotlib.pyplot as plt


def crosscor():
    def pearsonCroCor(data, sample, mode=0):
        if mode == 2:
            corresult = np.correlate(data, sample)
        else:
            corresult = np.zeros(data.size - sample.size + 1, dtype=np.complex128)
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
        return np.abs(corresult)
    # data, fs = load_audio_data(r'D:\projects\pyprojects\soundphase\calib\0\0.wav', 'wav')
    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\sound\0.wav', 'wav')
    t1 = 1
    t2 = 15
    data1 = data[48000 * t1:48000 * t2 + 500, 0].T
    data2 = data[48000 * t1:48000 * t2, 1].T
    print(len(data2)/48000)
    plt.plot((np.arange(48000, len(data2) + 48000))/48000, data2)
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


if __name__ == '__main__':
    crosscor()