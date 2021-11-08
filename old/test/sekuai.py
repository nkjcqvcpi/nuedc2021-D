import cv2
import numpy as np
import time
from LaserPointer import Pendulum


pd = Pendulum((640, 360))
color_dist = {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])}
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
frame_cnt = 0
while cam.isOpened():
    _, frame = cam.read()
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
    erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀粗的变细
    inRange_hsv = cv2.inRange(erode_hsv, color_dist['Lower'], color_dist['Upper'])
    cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) != 0:
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        bbox = cv2.boxPoints(rect)
        pd.update(rect[2], bbox, time.time_ns())
        if frame_cnt > 0 and frame_cnt % 100 == 0:
            pd.calibration()
        if frame_cnt > 100 and frame_cnt % 100 == 0:
            print(pd.length())
        cv2.line(frame, (int(pd.pointer.refx), 0), (int(pd.pointer.refx), 360), (0, 255, 255), 2)
        cv2.putText(frame, 'frame:{}'.format(frame_cnt), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
        cv2.drawContours(frame, [np.int0(bbox)], -1, (0, 255, 255), 2)
        frame_cnt += 1

    cv2.imshow('dect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



