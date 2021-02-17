import numpy as np
import matplotlib.pyplot as plt
from arlpy import bf

from IQ import load_audio_data


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


def cos_complex_wave(A, f, fs, t):
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
    y = A * np.exp(1j * 2 * np.pi * f * n * Ts)
    return y


class SoundSource:
    def __init__(self, pos: np.ndarray, A, f, t):
        self.pos = pos
        self.signal = cos_complex_wave(A, f, 48e3, t)
        self.f = f
        self.reflect_pos = None
        self.dist = []
        self.diff_dist = []

    def cal_dist(self, mic_array_pos: np.ndarray):
        self.dist.clear()
        self.diff_dist.clear()
        for pos in mic_array_pos:
            self.dist.append(np.linalg.norm(self.pos - pos))
        for dist in self.dist:
            self.diff_dist.append(dist - self.dist[0])

    def cal_reflect_pos(self, reflect_object_pos):
        self.reflect_pos = [2 * reflect_object_pos[0] - self.pos[0], self.pos[1], self.pos[2]]


def cons_ula(num, spacing):
    pos = []
    for i in range(num):
        x_i = i * spacing
        pos.append([x_i, 0, 0])
    return np.array(pos)


def cons_uca(r):
    theta = np.pi / 3
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    return np.array(pos)

def cons_sound_source(f):
    main_source = SoundSource(np.array([1, 1 * np.tan(np.deg2rad(30)), 0]), 1, f, 1)  # 45度
    other_sources = [SoundSource(np.array([-1, -1 * np.tan(np.deg2rad(100)), 0]), 1, f, 1)]
    return main_source, other_sources


def cons_reflect_object():
    return np.array([15, 5, 0])


# beamformer doa仿真
def one_source_no_wall_simu():
    c = 343
    spacing = 0.0043
    mic_array_pos = cons_ula(6, spacing)
    main_source, _ = cons_sound_source()
    main_source.cal_dist(mic_array_pos)
    # 每个麦克风接收到的信号
    mic_receive_signals = []
    for d_diff in main_source.diff_dist:
        signal = main_source.signal * np.exp(1j * 2 * np.pi * main_source.f * d_diff / c)
        mic_receive_signals.append(signal)
    mic_receive_signals = np.array(mic_receive_signals)
    # 角度估计
    sigma = []
    for angel in range(0, 360):
        two_angel = [[np.deg2rad(angel), 0]]
        sd = steering_plane_wave(mic_array_pos, c, two_angel)
        adjust = np.exp(-1j * 2 * np.pi * main_source.f * sd)
        syn_signals = mic_receive_signals * adjust.T
        # syn_signals = np.real(syn_signals)
        beamformed_signal = np.sum(syn_signals, axis=0)
        sigma.append(np.sum(abs(beamformed_signal)))
    print(np.argmax(sigma))
    plt.plot(sigma)
    plt.grid()
    plt.show()


def one_source_no_wall_simu_2():
    c = 343
    spacing = 0.03
    mic_array_pos = cons_ula(7, spacing)
    # mic_array_pos = cons_uca(spacing)
    main_source, _ = cons_sound_source()
    main_source.cal_dist(mic_array_pos)
    # 每个麦克风接收到的信号
    x_in = np.tile(main_source.signal, (7, 1))

    angel = [[np.deg2rad(20), 0]]
    sd = steering_plane_wave(mic_array_pos, c, angel)  # 时延
    adjust = np.exp(1j * 2 * np.pi * main_source.f * sd)
    adjust = adjust.T
    mic_receive_signals = x_in * adjust

    angel = [[np.deg2rad(100), 0]]
    sd = steering_plane_wave(mic_array_pos, c, angel)  # 时延
    adjust = np.exp(1j * 2 * np.pi * main_source.f * sd)
    adjust = adjust.T
    mic_receive_signals += x_in * adjust

    # angel2 = [[np.deg2rad(120), 0]]
    # sd2 = steering_plane_wave(mic_array_pos, c, angel2)  # 时延
    # adjust2 = np.exp(1j * 2 * np.pi * main_source.f * sd2)
    # adjust2 = adjust2.T
    # mic_receive_signals = mic_receive_signals + x_in * adjust2

    # adjust = np.exp(-1j * 2 * np.pi * main_source.f * sd)
    # adjust = adjust.T
    # mic_receive_signals = mic_receive_signals * adjust
    # sp = music(mic_receive_signals, mic_array_pos, 20e3, c, np.arange(0, 360), np.arange(-30, 30), 1)
    # # plt.plot(sp.reshape(-1))
    # # plt.show()
    # plt.pcolormesh(sp)
    # plt.colorbar()
    # plt.show()
    # 角度估计
    sigma = []
    for angel in range(0, 180):
        two_angel = [[np.deg2rad(angel), 0]]
        sd = steering_plane_wave(mic_array_pos, c, two_angel)
        adjust = np.exp(-1j * 2 * np.pi * main_source.f * sd)
        syn_signals = mic_receive_signals * adjust.T
        # syn_signals = np.real(syn_signals)
        beamformed_signal = np.sum(syn_signals, axis=0)
        # print(np.sum(abs(beamformed_signal)))
        sigma.append(np.sum(abs(beamformed_signal)))
    print(np.argmax(sigma))
    plt.plot(sigma)
    plt.grid()
    plt.show()

    # m = []
    # for angel in range(15, 30):
    #     two_angel = [[np.deg2rad(angel), 0]]
    #     sd = steering_plane_wave(mic_array_pos, c, two_angel)
    #     y = bf.music(mic_receive_signals, main_source.f, sd)
    #     m.append(y)
    # plt.plot(m)
    # plt.grid()
    # plt.show()


def two_dem_simu():
    nmic = 7
    spacing = 0.03
    f = 5e3
    for f in np.arange(5e3, 20e3, 3000):
        # mic_array_pos = cons_ula(nmic, spacing)
        mic_array_pos = cons_uca(spacing)
        main_source, other_sources = cons_sound_source(f)
        wall_pos = cons_reflect_object()

        main_source.cal_dist(mic_array_pos)
        main_source.cal_reflect_pos(wall_pos)

        for other_source in other_sources:
            other_source.cal_dist(mic_array_pos)
            other_source.cal_reflect_pos(wall_pos)

        c = 343
        mic_receive_signals = []
        for d_diff in main_source.diff_dist:
            signal = main_source.signal * np.exp(1j * 2 * np.pi * main_source.f * d_diff / c)
            mic_receive_signals.append(signal)
        mic_receive_signals = np.array(mic_receive_signals)

        A = 1
        for other_source in other_sources:
            for i, d_diff in enumerate(other_source.diff_dist):
                signal = other_source.signal * np.exp(1j * 2 * np.pi * main_source.f * d_diff / c)
                mic_receive_signals[i] += A * signal

        # sp = music(mic_receive_signals, mic_array_pos, 20e3, c, np.arange(0, 360), np.arange(0, 30), 2)
        # # plt.plot(sp.reshape(-1))
        # plt.pcolormesh(sp)
        # plt.show()

        # 估计
        sigma = []
        for angel in range(0, 180):
            two_angel = [[np.deg2rad(angel), 0]]
            sd = steering_plane_wave(mic_array_pos, c, two_angel)
            adjust = np.exp(-1j * 2 * np.pi * main_source.f * sd)
            syn_signals = mic_receive_signals * adjust.T
            # syn_signals = np.real(syn_signals)
            beamformed_signal = np.sum(syn_signals, axis=0)
            sigma.append(np.mean(abs(beamformed_signal)))
        print(np.argmax(sigma))
        plt.plot(sigma, label=f'f:{f}')
    plt.title(f"uca,change with f. 30°and100°,n_mic={nmic}, spacing={spacing}")
    plt.legend()
    plt.grid()
    plt.show()


def move_simu():
    n_mic = 7
    c = 343
    spacing = 0.043
    mic_array_pos = cons_ula(n_mic, spacing)
    # mic_array_pos = cons_uca(spacing)
    main_source, other_sources = cons_sound_source(20e3)

    main_source.cal_dist(mic_array_pos)
    for other_source in other_sources:
        other_source.cal_dist(mic_array_pos)

    dists = []
    # move, 10s
    v = 0.001
    for t in range(100):
        main_source.pos = np.array([main_source.pos[0] + v, main_source.pos[1], main_source.pos[2]])
        main_source.cal_dist(mic_array_pos)
        dists.append(main_source.dist.copy())

    dists = np.array(dists).T
    A = 1 / (dists ** 2)  # 衰减矩阵
    phi = 2 * np.pi * main_source.f * dists / c

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(phi[0])
    plt.title("origin phase")
    # plt.show()

    mic_receive_signals = A * np.exp(1j * phi)
    # plt.figure()
    plt.subplot(2,2,2)
    plt.plot(np.unwrap(np.angle(mic_receive_signals[0])))
    plt.title("calculate origin phase")
    # plt.show()

    A_other = 1

    for other_source in other_sources:
        d = np.reshape(other_source.dist, (n_mic, 1))
        o_phase = A_other * np.exp(1j * 2 * np.pi * other_source.f * d / c)
        mic_receive_signals += o_phase
    # plt.figure()
    plt.subplot(2,2,3)
    plt.plot(np.unwrap(np.angle(mic_receive_signals[0])))
    plt.title("mixed phase")
    # plt.show()

    # draw_circle(np.real(mic_receive_signals[0]), np.imag(mic_receive_signals[0]))

    # beamform
    angel = 30
    two_angel = [[np.deg2rad(angel), 0]]
    sd = steering_plane_wave(mic_array_pos, c, two_angel)
    adjust = np.exp(-1j * 2 * np.pi * main_source.f * sd)
    syn_signals = mic_receive_signals * adjust.T
    # syn_signals = bf.delay_and_sum(mic_receive_signals, fs=48e3, sd=sd)

    # syn_signals = np.real(syn_signals)
    beamformed_signal = np.sum(syn_signals, axis=0)

    # plt.figure()
    plt.subplot(2,2,4)
    plt.plot(np.unwrap(np.angle(beamformed_signal)))
    plt.title("beamformed phase")
    plt.show()

    draw_circle(np.real(beamformed_signal), np.imag(beamformed_signal))


index = 0


def draw_circle(I, Q, fs=48e3):
    fig, ax = plt.subplots()
    plt.grid()
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    circle, = ax.plot(0, 0, label='I/Q')
    timer = fig.canvas.new_timer(interval=100)

    def OnTimer(ax):
        global index
        speed = 1
        circle.set_ydata(Q[:index * speed])
        circle.set_xdata(I[:index * speed])
        index = index + 1
        ax.draw_artist(circle)
        ax.figure.canvas.draw()
        if index * speed > len(Q):
            print("end")
        else:
            print(f"time:{(index * speed) / fs}")

    timer.add_callback(OnTimer, ax)
    timer.start()
    plt.show()


"""
aziang = np.arange(-180, 180)
eleang = np.arange(-1, 30)
"""
def music(x: np.ndarray, pos, f, c, aziang, eleang, nsignals):
    # a = bf.covariance(x)
    R = np.dot(x, x.conj().T) / 1000
    D, V = np.linalg.eigh(R)
    idx = D.argsort()[::-1]
    eigenvals = D[idx]  # Sorted vector of eigenvalues
    eigenvects = V[:, idx]  # Eigenvectors rearranged accordingly
    noise_eigenvects = eigenvects[:, nsignals:len(eigenvects)]  # Noise eigenvectors
    # 计算迭代次数，最小化计算开销
    len_az = len(aziang)
    len_el = len(eleang)
    pattern_shape = (len_el, len_az)
    scan_az, scan_el = np.meshgrid(aziang, eleang)
    scan_angles = np.vstack((scan_az.reshape(-1, order='F'), scan_el.reshape(-1, order='F'))) # shape=(2, len_el*len_az)

    num_iter = min(len_el, len_az)
    scan_angle_block_size = max(len_el, len_az)
    scan_angle_block_index = np.arange(scan_angle_block_size)
    # scan
    pattern = np.zeros(len_az * len_el)
    for i in range(num_iter):
        cur_idx = scan_angle_block_index + i * scan_angle_block_size
        cur_angles = scan_angles[:, cur_idx]
        time_delay = bf.steering_plane_wave(pos, c, np.deg2rad(cur_angles.T)).T
        a = np.exp(-1j*2*np.pi*f*time_delay)
        # 两种方法一样
        # A = a.conj().T.dot(noise_eigenvects).dot(noise_eigenvects.conj().T).dot(a)
        # D = np.diag(A)
        D = np.sum(np.abs((a.T.dot(noise_eigenvects)))**2, 1)
        pattern[cur_idx] = 1 / D
    scan_pattern = np.sqrt(pattern).reshape(pattern_shape, order='F')
    return scan_pattern


def music_test():
    DELAY_TIME = 1
    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\0\0\0.wav', 'wav')
    data = data.T
    data = data[:7, int(fs * DELAY_TIME):]

    fc = 2e3
    c = 343
    spacing = 0.043
    mic_array_pos = cons_uca(spacing)
    plt.ion()

    for i in range(0, data.shape[1]-2048, 2048):
        print(f"time:{i/48000}")
        plt.clf()
        d = data[:, i:i+2048]
        sp = music(d, mic_array_pos, fc, c, np.arange(0, 180), np.arange(0, 80), 1)
        # plt.plot(sp.reshape(-1))
        plt.pcolormesh(sp)
        plt.pause(1)





if __name__ == '__main__':
    # music_test()
    # one_source_no_wall_simu_2()
    # z = np.random.randn(1000).view(np.complex128)
    # two_dem_simu()
    move_simu()
    # music(np.random.random((7,1000)),cons_uca(0.043),20e3,343,np.arange(-180, 181),np.arange(-1, 31), 1)

    # print(np.arange(100))
    # aziang = np.arange(-180, 180)
    # eleang = np.arange(-1, 30)
    # xx, yy = np.meshgrid(aziang, eleang)
    # scan_angles = np.vstack((xx.reshape(-1, order='F'), yy.reshape(-1, order='F')))
    # pass
    # a[0:2] = [2,3]
    # print(a)
