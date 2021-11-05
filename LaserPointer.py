import numpy as np
import math

g = 9.80665


class LaserPointer:
    x0, x1, x2, x3, y0, y1, y2, y3 = 0, 0, 0, 0, 0, 0, 0, 0
    bbox = np.array(((x0, y0), (x1, y1), (x2, y2), (x3, y3)))
    angle = 0

    @property
    def center(self):
        return np.mean(self.bbox[:, 0]), np.mean(self.bbox[:, 1])


class Pendulum:
    def __init__(self):
        self.pointer = LaserPointer()

    def calibration(self, frame):
        pass

    def length(self, T):
        return g * T**2 / 4 * math.pi ** 2