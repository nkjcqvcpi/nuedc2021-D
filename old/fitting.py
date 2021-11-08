from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np


def fourier(t, a0, a1, b1, w):
    ret = a0 + a1 * np.cos(2 * np.pi * t * w) + b1 * np.sin(2 * np.pi * t * w)
    return ret


if __name__ == '__main__':
    cords = np.genfromtxt('centers_07_03-15-47.csv', delimiter=',', skip_header=1, unpack=True)
    cords = cords.T

    h = np.percentile(cords, 98, axis=0)
    l = np.percentile(cords, 3, axis=0)
    cords = cords[np.where((cords[:, 1] < h[1]) & (cords[:, 1] > l[1]) & (cords[:, 2] < h[2]) & (cords[:, 2] > l[2]))]
    cords[:, 0] *= 1e-9

    # y = savgol_filter(cords[:, 1], 3, 1, mode='nearest')
    # u = cords[370: 400, 0]
    # y = y[370: 400]
    u = cords[150:200, 0]
    y = cords[150:200, 1]
    ff = np.fft.fftfreq(len(u), (u[1] - u[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(y))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y)
    guess_offset = np.mean(y)
    guess = np.array([guess_offset, guess_amp, -guess_amp, guess_freq])

    popt, pcov = curve_fit(fourier, u, y, p0=guess)#, method='trf', jac='cs', tr_solver='lsmr', max_nfev=1000000)
    plt.plot(u, y, color='r', label="original")
    plt.plot(u, fourier(u, *popt), color='g', label="fitting")
    plt.legend()
    plt.show()
    print("参数如下：")
    print(popt)
