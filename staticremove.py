from scipy.signal import find_peaks

import numpy as np


def find_next_extreme_index(i, extremes):
    temp = extremes[extremes > i]
    if len(temp) > 0:
        return temp[0]
    else:
        return -1


def LEVD(I, Thr=0.0015):
    mean_I = np.mean(I)
    print(mean_I)
    local_maxs, _ = find_peaks(I, mean_I)
    local_mins, _ = find_peaks(-I, -mean_I)

    extremes = np.hstack((local_maxs, local_mins))
    extremes.sort()

    E = [I[local_maxs[0]]]
    last_extreme = 'max'
    static = [0]
    for i in range(1, len(I)):
        I_i = I[i]
        if i in local_maxs and last_extreme == 'max':
            if I_i > E[len(E) - 1]:
                E[len(E) - 1] = I_i
        elif i in local_mins and last_extreme == 'min':
            if I_i < E[len(E) - 1]:
                E[len(E) - 1] = I_i
        elif i in local_maxs and last_extreme == 'min':
            nei = find_next_extreme_index(i, extremes)
            if nei in local_mins:
                # 当前确实是局部最大值
                if np.abs(I_i - E[len(E) - 1]) > Thr:
                    E.append(I_i)
                    last_extreme = 'max'
        elif i in local_mins and last_extreme == 'max':
            nei = find_next_extreme_index(i, extremes)
            if nei in local_maxs:
                # 当前确实是局部最小值
                if np.abs(I_i - E[len(E) - 1]) > Thr:
                    E.append(I_i)
                    last_extreme = 'min'
        if len(E) < 2:
            static.append(0)
        else:
            static.append(0.9 * static[i - 1] + 0.1 * (E[len(E) - 2] + E[len(E) - 1]) / 2)
    # plt.figure()
    # plt.plot(I)
    # plt.plot(local_maxs, I[local_maxs], "x")
    # plt.plot(local_mins, I[local_mins], "o")
    # plt.plot(np.zeros_like(x), "--", color="gray")
    # plt.show()
    return static
