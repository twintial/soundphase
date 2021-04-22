import numpy as np
def cons_uca(r):
    theta = np.pi / 3
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    return np.array(pos)

p = cons_uca(0.043)
print(p)