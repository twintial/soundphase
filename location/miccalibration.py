'''
测试用于麦克风阵列的时延迟，用于校准
'''
import time
from arlpy import bf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft, fftshift
import wave

from scipy.signal import butter, filtfilt


def steering_plane_wave(pos, c, theta):
    """Compute steering delays assuming incoming signal has a plane wavefront.

    For linear arrays, pos is 1D array. For planar and 3D arrays, pos is a 2D array with a
    sensor position vector in each row.

    For linear arrays, theta is a 1D array of angles (in radians) with 0 being broadside. For
    planar and 3D arrays, theta is a 2D array with an (azimuth, elevation) pair in each row.
    Such arrays can be easily generated using the :func:`arlpy.utils.linspace2d` function.

    The broadside direction is along the x-axis of a right-handed coordinate system with z-axis pointing
    upwards, and has azimuth and elevation as 0. In case of linear arrays, the y-coordinate is the
    sensor position. In case of planar arrays, if only 2 coordinates are provided, these coordinates
    are assumed to be y and z.

    :param pos: sensor positions (m)
    :param c: signal propagation speed (m/s)
    :param theta: steering directions (radians)
    :returns: steering delays with a row for each direction (s)

    >>> import numpy as np
    >>> from arlpy import bf, utils
    >>> pos1 = [0.0, 0.5, 1.0, 1.5, 2.0]
    >>> a1 = bf.steering_plane_wave(pos1, 1500, np.deg2rad(np.linspace(-90, 90, 181)))
    >>> pos2 = [[0.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.0],
                [0.5, 0.5]]
    >>> a2 = bf.steering_plane_wave(pos2, 1500, np.deg2rad(utils.linspace2d(-20, 20, 41, -10, 10, 21)))
    """
    pos = np.array(pos, dtype=np.float)
    theta = np.asarray(theta, dtype=np.float)
    if pos.ndim == 1:
        # pos -= np.mean(pos)
        dist = pos[:, np.newaxis] * np.sin(theta)
    else:
        if pos.shape[1] != 2 and pos.shape[1] != 3:
            raise ValueError('Sensor positions must be either 2d or 3d vectors')
        # pos -= np.mean(pos, axis=0)
        if pos.shape[1] == 2:
            pos = np.c_[np.zeros(pos.shape[0]), pos]
        azim = theta[:, 0]
        elev = theta[:, 1]
        dvec = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        dist = np.dot(pos, dvec)
    return -dist.T / c

def butter_bandpass_filter(data, lowcut, highcut, fs=48e3, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


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


def cons_uca(r):
    theta = np.pi / 3
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    return np.array(pos)

def get_sinusoid(f0, step, n, fs, t):
    A = [1] * n
    alpha = 1 / sum(A)
    y = A[0] * cos_wave(1, f0, fs, t)
    for i in range(1, n):
        y = y + A[i] * cos_wave(1, f0 + i * step, fs, t)
    signal = alpha * y
    return signal
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
    tailor_frames_nums = sd[0] * fs
    print(tailor_frames_nums)
    tailor_frames_nums = tailor_frames_nums.astype(np.int16)
    print(tailor_frames_nums)
    beamformed_data = []
    for i, tailor_frames_num in enumerate(tailor_frames_nums):
        if tailor_frames_num < 0:
            temp_data = np.hstack((data[i, -tailor_frames_num:], np.zeros(-tailor_frames_num)))
        elif tailor_frames_num > 0:
            temp_data = np.hstack((np.zeros(tailor_frames_num), data[i, :-tailor_frames_num]))
        else:
            temp_data = data[i]
        beamformed_data.append(temp_data)
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


def normalized_signal_fft_with_fft(fft, title, fs=48e3, figure=True, xlim=(19e3, 25e3)):
    N = len(fft)
    y_signle: np.ndarray = fft[:int(np.round(N / 2))] * 2
    x = fftfreq(N) * fs
    x = x[x >= 0]
    if figure:
        plt.figure()
        plt.plot(x, y_signle)
        plt.xlim(xlim)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('power')
        plt.title(title)
    return x, y_signle
def gcc_phat(a, b):
    w = np.hanning(len(a))
    # w = np.ones(len(a))
    f_s1 = fft(a * w)
    normalized_signal_fft_with_fft(np.abs(f_s1), 'f_s1',xlim=(0,20e3))
    f_s2 = fft(b * w)
    # plt.plot(np.abs(f_s2))
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = np.abs(f_s) + np.finfo(np.float32).eps
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
    return np.abs(ifft(f_s))


# srp phat
def get_steering_vector(pos_i, pos_j, c, dvec):
    """
    :param pos_i:
    :param pos_j:
    :param c:
    :param dvec: np array, shape=(10 × 4^L + 2, 3)
    :return:
    """
    pos_i = np.array(pos_i, dtype=np.float)
    pos_j = np.array(pos_j, dtype=np.float)
    dist = np.dot((pos_i - pos_j).reshape(1, 3), dvec.T)
    return (-dist.T / c).reshape(-1)  # 这个负号要不要?
def vec2theta(vec):
    vec = np.array(vec)
    r = np.linalg.norm(vec[0])
    theta = np.zeros((vec.shape[0], 2))
    theta[:, 0] = np.arctan2(vec[:, 1], vec[:, 0])
    theta[:, 1] = np.arcsin(vec[:, 2]/r)
    return theta
def theta2vec(theta, r=1):
    # vectors = np.zeros((theta.shape[0], 3))
    # vectors[:, 0] = r*np.cos(theta[:, 1])*np.cos(theta[:, 0])
    # vectors[:, 1] = r*np.cos(theta[:, 1])*np.sin(theta[:, 0])
    # vectors[:, 2] = r*np.sin(theta[:, 1])
    vectors = []
    for t in theta:
        vectors.append(
            [r * np.cos(t[1]) * np.cos(t[0]),
             r * np.cos(t[1]) * np.sin(t[0]),
             r * np.sin(t[1])])
    return np.array(vectors)
def generate_big_grid():
    aziang = np.arange(-180, 180)
    eleang = np.arange(0, 90)
    scan_az, scan_el = np.meshgrid(aziang, eleang)
    scan_angles = np.vstack((scan_az.reshape(-1, order='F'), scan_el.reshape(-1, order='F')))
    scan_angles = np.deg2rad(scan_angles.T)
    return theta2vec(scan_angles)
def gcc_phat_search(x_i, x_j, fs, tau):
    """
    :param x_i: real signal of mic i
    :param x_j: real signal of mic j
    :param fs: sample rate
    :param search_grid: grid for search, each point in grid is a 3-D vector
    :return: np array, shape = (n_frames, num_of_search_grid)
    """
    w = np.hanning(len(x_j))
    # w = np.ones(len(x_j))
    P = fft(x_i * w) * fft(x_j * w).conj()
    A = P / (np.abs(P)+np.finfo(np.float32).eps)

    # 为之后使用窗口做准备
    A = A.reshape(1, -1)

    num_bins = A.shape[1]
    k = np.arange(num_bins)
    # t1 = time.time()
    exp_part = np.outer(k, 2j * np.pi * tau * fs/num_bins)
    # t2 = time.time()
    # print('ifft1 time consuption: ', t2 - t1)
    # t1 = time.time()
    R = np.dot(A, np.exp(exp_part)) / num_bins
    # t2 = time.time()
    # print('ifft2 time consuption: ', t2 - t1)
    return np.abs(R)
def srp_phat(raw_signal, mic_array_pos, search_grid, c, fs):
    assert raw_signal.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    # grid, _ = create_spherical_grids(level)
    # print(grid.shape)
    E_d = np.zeros((1, search_grid.shape[0]))  # (num_frames, num_points), 之后要改
    for i in range(mic_num):
        for j in range(i + 1, mic_num):
            # tau is given in second, 这个也可以提前计算
            tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, search_grid)
            # t1 = time.time()
            R_ij = gcc_phat_search(raw_signal[i], raw_signal[j], fs, tau)
            # t2 = time.time()
            E_d += R_ij
            # print('each pair time consuption: ', t2 - t1)
    return E_d

def one_frame(data=None):
    if data is None:
        data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\1khz\0.wav', 'wav')
        data = data.T
        t1 = 2.5
        t2 = 3
        data_seg = data[:-1, int(48000 * t1):int(48000 * t2)]

        data2 = data_seg[0]
        plt.figure()
        plt.plot((np.arange(48000 * t1, len(data2) + 48000 * t1)) / 48000, data2)
        plt.show()

        frame_len = 2048
        data_seg = data_seg[:, :frame_len]
    else:
        data_seg = data
    fs = 48000
    c = 343
    level = 4

    grid: np.ndarray = np.load(rf'grid/{level}_north.npz')['grid']
    # grid = generate_big_grid()
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(grid[:,0],grid[:,1],grid[:,2])
    # plt.show()
    # grid = cons_uca(1)[1:]
    # mic mem pos
    pos = cons_uca(0.043)
    t1 = time.time()
    E = srp_phat(data_seg, pos, grid, c, fs)
    i_sort = np.argsort(E.reshape(-1))
    g_sort = grid[i_sort][::-1]
    for i in range(20):
        print(rf'top {i} is ', np.rad2deg(vec2theta([g_sort[i]])))
    # E = srp_phat_muti_thread(data, pos, grid, c, fs)
    # E = srp_phat_previous_tau(data_seg, pos, grid, pairs_tau, fs)
    t2 = time.time()
    print('srp_phat time consumption: ', t2 - t1)
    sdevc = grid[np.argmax(E, axis=1)]  # source direction vector
    # print(sdevc)
    print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
    print('=' * 50)
def split_frame():
    c = 343
    frame_count = 1024
    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\1khz\0.wav', 'wav')
    skip_time = int(fs * 3)
    data = data[skip_time:, :-1].T
    # search unit circle
    level = 3
    grid: np.ndarray = np.load(rf'grid/{level}_north.npz')['grid']
    # mic mem pos
    pos = cons_uca(0.043)
    # calculate tau previously
    # pairs_tau = calculate_pairs_tau(pos, grid, c)
    for i in range(0, data.shape[1], frame_count):
        data_seg = data[:, i:i+frame_count]
        # 噪声不做,随便写的
        if np.max(abs(fft(data_seg[0] / len(data_seg[0])))) < 10:
            continue
        print('time: ', (skip_time + i)/fs)
        t1 = time.time()
        E = srp_phat(data_seg, pos, grid, c, fs)
        # E = srp_phat_muti_thread(data, pos, grid, c, fs)
        # E = srp_phat_previous_tau(data_seg, pos, grid, pairs_tau, fs)
        t2 = time.time()
        print('srp_phat time consumption: ', t2-t1)
        sdevc = grid[np.argmax(E, axis=1)]  # source direction vector
        # print(sdevc)
        print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
        print('='*50)
        # plot_angspect(E_d[0], grid)


def crosscor():
    # data, fs = load_audio_data(r'D:\projects\pyprojects\soundphase\calib\0\0.wav', 'wav')
    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\sinusoid2\0.wav', 'wav')
    data = butter_bandpass_filter(data.T, 15e3, 23e3)
    # data = data.T
    t1 = 2
    t2 = 3
    data1 = data[0, int(48000 * t1):int(48000 * t2 + 500)]
    data2 = data[1, int(48000 * t1):int(48000 * t2)]
    # print(len(data2)/48000)
    plt.figure()
    plt.plot((np.arange(48000 * t1, len(data2) + 48000 * t1))/48000, data2)
    plt.show()

    '''
    测试了mic之间的时间延迟，效果和理论计算一致
    '''
    corr = pearsonCroCor(data1, data2)
    # npsignalplot(corr)
    plt.figure()
    plt.plot(corr)
    # plt.show()

    index = np.argmax(corr)
    print(index)

    frame_len = 2048 * 2
    y = gcc_phat(data1[:frame_len], data2[:frame_len])
    print('ifft max ccor val: ', np.max(y))
    print('ifft delay of sample num: ', np.argmax(y))
    plt.figure()
    plt.plot(y)
    plt.title('ifft gcc')
    plt.show()

    one_frame(data[:-1, int(fs * t1):int(fs * t1) + frame_len])
def simu():
    fs = 48e3
    pos = cons_uca(0.043)
    # data_1 = cos_wave(1, 18e3, fs, 10)
    # data_1 = [0] * 100 + [1] * 10 + [0] * 100
    # data_1 = np.array(data_1)
    data_1 = get_sinusoid(17000, 350, 8, fs, 10)

    data_7 = np.tile(data_1.T, (7, 1))
    theta = np.deg2rad([[90, 0]])
    print('ground truth: ', np.rad2deg(theta))
    sd = bf.steering_plane_wave(pos, 343, theta)
    data = beamform_real(data_7, sd, fs)

    # for i in range(7):
    #     plt.subplot(4,2,i+1)
    #     plt.plot(data[i])
    #     plt.title(str(i))
    # plt.show()

    data = data.T

    t1 = 2
    t2 = 3
    data1 = data[int(fs * t1):int(fs * t2 + 500), 0].T
    data2 = data[int(fs * t1):int(fs * t2), 1].T

    # data1 = data[50:150 + 50, 0].T
    # data2 = data[50:150, 1].T

    # print(len(data2)/48000)
    # plt.plot((np.arange(48000 * t1, len(data2) + 48000 * t1))/48000, data2)
    # plt.show()

    corr = pearsonCroCor(data1, data2)
    # npsignalplot(corr)
    # plt.scatter(np.arange(len(corr)), corr)
    plt.plot(corr)
    plt.title('time domin coor')

    index = np.argmax(corr)
    print(index)

    frame_len = 2048 * 2
    y = gcc_phat(data1[:frame_len], data2[:frame_len])
    print('ifft max ccor val: ', np.max(y))
    print('ifft delay of sample num: ', np.argmax(y))
    plt.figure()
    plt.plot(y)
    plt.title('ifft gcc')
    plt.show()

    one_frame(data[int(fs * t1):int(fs * t1)+frame_len].T)



if __name__ == '__main__':
    simu()
    # crosscor()
    # one_frame()
    # split_frame()