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
        self.pre_c = 0
        self.refx = frame_size[0]
        self.frame = 0
        self.flag = False

    @property
    def center(self):
        c = tuple(np.mean(self.bbox, axis=1))
        if not self.flag:
            self.pre_c = c[0]
        if self.pre_c < self.refx < c[0]:
            self.centers.append((c, time.time_ns(), 1))
        elif self.pre_c > self.refx > c[0]:
            self.centers.append((c, time.time_ns(), -1))
        elif self.refx == c[0]:
            self.centers.append((c, time.time_ns(), 0))
        else:
            self.centers.append((c, time.time_ns(), 2))
        self.pre_c = c[0]
        self.frame += 1
        return c

    def update(self, angle, bbox):
        if self.flag:
            np.concatenate((self.bboxs, self.bbox), axis=1)
        else:
            self.bboxs = self.bbox
            self.flag = True
        self.angles.append(self.angle)
        self.bbox = bbox
        self.angle = float(angle)


class Pendulum:
    def __init__(self, frame_size):
        self.pointer = LaserPointer(frame_size)
        self.ref = (0, 0)

    def calibration(self):
        self.ref = np.mean(self.pointer.centers[:12], axis=1)
        self.pointer.refx = self.ref[0]

    def length(self, r=48):
        t = []
        for i, center in enumerate(self.pointer.centers[13: r]):
            if center[2] == -1:
                x1, t1 = self.pointer.centers[i + 1][0][0], self.pointer.centers[i + 1][1]
                t.append(self.ts(center[0][0], x1, center[1], t1))
            elif center[2] == 1:
                x0, t0 = self.pointer.centers[i - 1][0][0], self.pointer.centers[i - 1][1]
                t.append(self.ts(x0, center[0][0], t0, center[1]))
            elif center[2] == 0:
                t.append(center[1])
        T = np.array(t).mean() * 2
        return g * T**2 / 4 * math.pi ** 2

    def ts(self, x0, x1, t0, t1):
        return t0 + (t1 - t0) * ((x1 - self.ref[0]) / (x1 - x0))

    def update(self, angle, bbox):
        self.pointer.update(angle, bbox)
