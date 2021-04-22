import numpy as np
aziang = np.arange(-180, 180)
eleang = np.arange(0, 90)
scan_az, scan_el = np.meshgrid(aziang, eleang)
scan_angles = np.vstack((scan_az.reshape(-1, order='F'), scan_el.reshape(-1, order='F')))
scan_angles = np.deg2rad(scan_angles.T)
print(scan_angles.shape)

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

def vec2theta(vec):
    vec = np.array(vec)
    r = np.linalg.norm(vec[0])
    theta = np.zeros((vec.shape[0], 2))
    theta[:, 0] = np.arctan2(vec[:, 1], vec[:, 0])
    theta[:, 1] = np.arcsin(vec[:, 2]/r)
    return theta

x = theta2vec(scan_angles)
print(x.shape)
x2 = np.rad2deg(vec2theta(x))
print(x2.shape)