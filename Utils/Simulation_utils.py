import numpy as np
from matplotlib import pyplot as plt

def velocity_profile(t0: float, tf: float, t: float):
    v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
    return v

if __name__ == "__main__":
    y = np.zeros((1000, ))
    for t in range(0, 1000):
        y[t] = velocity_profile(0, 1, t/1000.0)
    x = np.arange(0, 1000)
    plt.plot(x, y)
    plt.show()