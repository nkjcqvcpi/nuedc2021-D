import math
import time
import numpy as np
from gpiozero import LED, Buzzer
from scipy.optimize import curve_fit

g = 9.80665


class LaserPointer:
    bbox = None
    angle = 0

    centers = []
    bboxs = []
    angles = []

    def __init__(self):
        self.cfp = open('../centers_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.cfp.write('time, x, y\n')
        self.bfp = open('../bbox_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.bfp.write('time, p0, p1, p2, p3\n')
        self.afp = open('../angle_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.afp.write('time, angle\n')
        self.lfp = open('../length_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.lfp.write('time, T, length, maxx, popt, pcov\n')
        self.flag = False

    def update(self, angle, bbox, t):
        self.bboxs.append(self.bbox)
        self.angles.append(self.angle)
        self.afp.write('{}, {}\n'.format(t, self.angle))
        self.bbox = bbox
        self.bfp.write('{}, {}, {}, {}, {}\n'.format(t, bbox[0], bbox[1], bbox[2], bbox[3]))
        self.angle = float(angle)
        c = np.mean(self.bbox, axis=0)
        self.centers.append((t, c[0], c[1]))
        self.cfp.write('{}, {}, {}\n'.format(t, c[0], c[1]))


class Pendulum:
    def __init__(self):
        self.pointer = LaserPointer()
        self.global_time = time.time_ns()
        self.l_cnt = 0
        self.amps = []
        self.amp: float = 0
        self.led = LED(21)
        self.buzzer = Buzzer(19)
        self.l: float = 0
        self.lengths = []
        self.thetas = []
        self.led.off()

    @staticmethod
    def fourier(t, a0, a1, b1, w):
        ret = a0 + a1 * np.cos(2 * np.pi * t * w) + b1 * np.sin(2 * np.pi * t * w)
        return ret

    def length(self, r=5):
        c = np.array(self.pointer.centers[self.l_cnt * r: self.l_cnt * r + 50])
        hw = np.percentile(c, 98, axis=0)
        lw = np.percentile(c, 3, axis=0)
        c = c[np.where((c[:, 1] < hw[1]) & (c[:, 1] > lw[1]) & (c[:, 2] < hw[2]) & (c[:, 2] > lw[2]))]
        u = c[:, 0] * 1e-9
        y = c[:, 1]
        try:
            ff = np.fft.fftfreq(len(u), (u[1] - u[0]))
            Fyy = abs(np.fft.fft(y))
            guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
            guess_amp = np.std(y)
            guess_offset = np.mean(y)
            guess = np.array([guess_offset, guess_amp, -guess_amp, guess_freq])
            popt, pcov = curve_fit(self.fourier, u, y, p0=guess)
            self.amp = math.sqrt(popt[1] ** 2 + popt[2] ** 2)
            T = 1 / popt[3]
            self.l = (g * T**2) / (4 * math.pi ** 2)
            if isinstance(self.amp, float):
                self.amps.append(self.amp)
                self.lengths.append(self.l)
            self.pointer.lfp.write('{}, {}, {}, {}, {}, {}\n'.format(time.time_ns(), T, self.l, self.amp, popt, pcov))
            self.l_cnt += 1
            return self.l, T, self.amp
        except Exception:
            pass

    def theta(self, cam_amp):
        if self.amp != 0:
            t = math.degrees(math.atan(cam_amp[0] / self.amp))
            print(t)
            self.thetas.append(t)

    def output(self):
        length = np.array(self.lengths).mean()
        theta = np.array(self.thetas).mean()
        return length, theta

    def update(self, angle, bbox, t, frame, cam_amp=None):
        self.pointer.update(angle, bbox, t - self.global_time)
        if frame > 50 and frame % 10 == 0:
            print(self.length())
            if cam_amp:
                self.theta(cam_amp)
