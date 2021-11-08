import numpy as np
import math

g = 9.80

cords = np.genfromtxt('centers_process.csv', delimiter=',', skip_header=1, unpack=True)
cords = cords.T
cords = np.insert(cords, 3, values=np.zeros(len(cords)), axis=1)


l_cnt = 0
c_cnt = 0
leap = 100


def center():
    direction = True
    pre_x = cords[0][1]
    for i, c in enumerate(cords):
        s = 0
        if pre_x < c[1]:
            if not direction:
                direction = True
                s = -1
        elif pre_x > c[1]:
            if direction:
                direction = False
                s = 1
        cords[i][3] = s
        pre_x = c[1]


def calibration():
    global c_cnt
    ref = np.mean(cords[c_cnt * leap: (c_cnt + 1) * leap], axis=0)
    c_cnt += 1
    return ref[1]


def length(r=leap):
    global l_cnt
    t = []
    for i, center in enumerate(cords[r * l_cnt: r * (l_cnt + 1)]):
        if center[3] == -1:
            x1, t1 = cords[i + 1][1], cords[i + 1][0]
            t.append(ts(center[1], x1, center[0], t1))
        elif center[3] == 1:
            x0, t0 = cords[i - 1][1], cords[i - 1][0]
            t.append(ts(x0, center[1], t0, center[0]))
        elif center[3] == 0:
            t.append(center[0])
    T = np.diff(np.array(t)).mean() * 2e-9
    L = (g * T ** 2) / (4 * math.pi ** 2)
    l_cnt += 1
    return L, T


def ts(x0, x1, t0, t1):
    return t0 + (t1 - t0) * ((x1 - refx) / (x1 - x0))


if __name__ == '__main__':
    refx = np.mean(cords[: leap], axis=0)[1]
    center()
    while True:
        refx = calibration()
        l, f = length()
        print(l, f)
        if np.isnan(l) or np.isnan(f):
            break
