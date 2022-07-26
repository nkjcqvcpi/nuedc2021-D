import math
import os
import time
import numpy as np
from scipy.optimize import curve_fit

try:
    from gpiozero import LED, Buzzer
    PI = True
except ModuleNotFoundError:
    PI = False

g = 9.80665


class LaserPointer:
    bbox = None
    angle = 0
    centers, bboxs, angles = [], [], []

    def __init__(self, path, dt):
        self.lfp = open(path + 'centers_{}.csv'.format(dt), mode='w')
        self.lfp.write('time, angle, x, y, p0, p1, p2, p3\n')
        self.flag = False
        return

    def update(self, angle, bbox, t):
        self.bbox = bbox
        self.angle = float(angle)
        self.bboxs.append(self.bbox)
        self.angles.append(self.angle)
        c = np.mean(self.bbox, axis=0)
        self.centers.append((t, c[0], c[1]))
        self.lfp.write(
            '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(t, self.angle, c[0], c[1], bbox[0], bbox[1], bbox[2], bbox[3]))


class Pendulum:
    def __init__(self):
        dt = time.strftime("%d_%H-%M-%S", time.localtime())
        path = 'log/' + dt + '/'
        os.makedirs(path)
        self.pointer = LaserPointer(path, dt)
        self.pfp = open(path + 'length_{}.csv'.format(dt), mode='w')
        self.pfp.write('time, T, length, maxx, popt, pcov\n')
        self.l_cnt = 0
        self.amps = []
        self.amp: float = 0
        self.len: float = 0
        self.lengths = []
        self.thetas = []
        if PI:
            self.led = LED(21)
            self.led.off()
            self.buzzer = Buzzer(19)
        self.global_time = time.time_ns()

    @staticmethod
    def fourier(t, a0, a1, b1, w):
        ret = a0 + a1 * np.cos(2 * np.pi * t * w) + b1 * np.sin(2 * np.pi * t * w)
        return ret

    def length(self, r=5):
        ct = np.array(self.pointer.centers)
        hw = np.percentile(ct, 98, axis=0)
        lw = np.percentile(ct, 3, axis=0)
        c = ct[np.where((ct[:, 1] < hw[1]) & (ct[:, 1] > lw[1]))]
        c = np.array(c[self.l_cnt * r: self.l_cnt * r + 32])
        u = c[:, 0] * 1e-9
        y = c[:, 1]
        try:
            ff = np.fft.fftfreq(len(u), (u[1] - u[0]))
            Fyy = abs(np.fft.fft(y))
            guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
            guess_amp = np.std(y)
            guess_offset = np.mean(y)
            guess = np.array([guess_offset, guess_amp, -guess_amp, guess_freq])
        except:
            return 1, 2, 100
        try:
            popt, pcov = curve_fit(self.fourier, u, y, p0=guess)
        except:
            popt = guess
        self.amp = math.sqrt(popt[1] ** 2 + popt[2] ** 2)
        T = 1 / popt[3]
        self.len = (g * T ** 2) / (4 * math.pi ** 2)
        if isinstance(self.amp, float):
            self.amps.append(self.amp)
            self.lengths.append(self.len)
        # self.pfp.write(f'{time.time_ns() - self.global_time}, {T}, {self.len}, {self.amp}, {popt}, {pcov}\n')
        self.l_cnt += 1
        return self.len, T, self.amp

    def output(self):
        length = np.array(self.lengths).mean()
        theta = np.array(self.thetas).mean()
        return length, theta

    def update(self, angle, bbox, t, frame):
        self.pointer.update(angle, bbox, t - self.global_time)
        if frame > 32 and frame % 32 == 0:
            self.length()
