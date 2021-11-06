import numpy as np
import math
import time

g = 9.80665


class LaserPointer:
    x0, x1, x2, x3, y0, y1, y2, y3 = 0, 0, 0, 0, 0, 0, 0, 0
    bbox = np.array(((x0, y0), (x1, y1), (x2, y2), (x3, y3)))
    angle = 0
    refx = 0

    centers = []
    bboxs = []
    angles = []

    def __init__(self, frame_size):
        self.cfp = open('../centers_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.cfp.write('time, x, y\n')
        self.bfp = open('../bbox_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.bfp.write('time, p0, p1, p2, p3\n')
        self.afp = open('../angle_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.afp.write('time, angle\n')
        self.lfp = open('../length_{}.csv'.format(time.strftime("%d_%H-%M-%S", time.localtime())), mode='w')
        self.lfp.write('time, T, length\n')
        self.pre_c = 0
        self.refx = frame_size[0]
        self.frame = 0
        self.flag = False

    def center(self, t):
        c = np.mean(self.bbox, axis=0)
        if not self.flag:
            self.pre_c = c[0]
        if self.pre_c < self.refx < c[0]:
            self.centers.append((c, t, 1))
        elif self.pre_c > self.refx > c[0]:
            self.centers.append((c, t, -1))
        elif self.refx == c[0]:
            self.centers.append((c, t, 0))
        else:
            self.centers.append((c, t, 2))
        self.cfp.write('{}, {}, {}\n'.format(t, c[0], c[1]))
        self.pre_c = c[0]
        self.frame += 1
        return c

    def update(self, angle, bbox, t):
        if self.flag:
            np.concatenate((self.bboxs, self.bbox), axis=1)
        else:
            self.bboxs = self.bbox
            self.flag = True
        self.angles.append(self.angle)
        self.afp.write('{}, {}\n'.format(t, self.angle))
        self.bbox = bbox
        self.bfp.write('{}, {}, {}, {}, {}\n'.format(t, bbox[0], bbox[1], bbox[2], bbox[3]))
        self.angle = float(angle)
        _ = self.center(t)


class Pendulum:
    def __init__(self, frame_size):
        self.pointer = LaserPointer(frame_size)
        self.ref = (0, 0)
        self.t_s = time.time_ns()
        self.l_cnt = 1
        self.c_cnt = 0

    def calibration(self):
        cts = np.array([i[0] for i in self.pointer.centers[self.c_cnt * 100: (self.c_cnt + 1) * 100]])
        self.ref = np.mean(cts, axis=0)
        self.pointer.refx = self.ref[0]
        self.c_cnt += 1

    def length(self, r=100):
        t = []
        for i, center in enumerate(self.pointer.centers[r * self.l_cnt: r * (self.l_cnt + 1)]):
            if center[2] == -1:
                x1, t1 = self.pointer.centers[i + 1][0][0], self.pointer.centers[i + 1][1]
                t.append(self.ts(center[0][0], x1, center[1], t1))
            elif center[2] == 1:
                x0, t0 = self.pointer.centers[i - 1][0][0], self.pointer.centers[i - 1][1]
                t.append(self.ts(x0, center[0][0], t0, center[1]))
            elif center[2] == 0:
                t.append(center[1])
        T = np.diff(np.array(t)).mean() * 2e-9
        L = (g * T**2) / (4 * math.pi ** 2)
        t = time.time_ns()
        self.pointer.lfp.write('{}, {}, {}\n'.format(t, T, L))
        self.l_cnt += 1
        return L, T

    def ts(self, x0, x1, t0, t1):
        return t0 + (t1 - t0) * ((x1 - self.ref[0]) / (x1 - x0))

    def update(self, angle, bbox, t):
        self.pointer.update(angle, bbox, t - self.t_s)
