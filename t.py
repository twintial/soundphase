import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.ion() #interactive mode on
ax = plt.gca()
line, = ax.plot(x,y)
ax.set_ylim([-5,5])

for i in np.arange(100):
    line.set_ydata(y)
    plt.draw()
    y = y*1.01
    plt.pause(0.001)