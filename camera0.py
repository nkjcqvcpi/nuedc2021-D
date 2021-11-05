import socket
import cv2 as cv2
import numpy as np
import time
import picamera
from picamera.array import PiRGBArray, PiYUVArray
from LaserPointer import Pendulum


def send():
    HOST = '192.168.199.131'
    PORT = 8001
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    time.sleep(2)
    sock.send(b'1')
    print(sock.recv(1024).decode())
    sock.close()


pointer = Pendulum()


color_dist = {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])}

with picamera.PiCamera(resolution=(160, 96), framerate=30) as camera:
    time.sleep(1)
    frame_cnt = 0
    cv2.namedWindow('name')
    while True:
        with PiRGBArray(camera, size=(160, 96)) as stream:
            camera.capture(stream, format='bgr')
            frame = stream.array
            stream.truncate(0)

        gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        inRange_hsv = cv2.inRange(erode_hsv, color_dist['Lower'], color_dist['Upper'])
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) != 0:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            pointer.pointer.angle = rect[2]
            pointer.pointer.position = cv2.boxPoints(rect)
        # if frame_cnt <= 12:
        #     pointer.calibration(frame_cnt)
            cv2.drawContours(frame, [np.int0(pointer.pointer.position)], -1, (0, 255, 255), 2)

        frame_cnt += 1

        cv2.imshow('name', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()



# def py_cpu_nms(dets, thresh):
#     y1 = dets[:, 1]
#     x1 = dets[:, 0]
#     y2 = y1 + dets[:, 3]
#     x2 = x1 + dets[:, 2]
#     scores = dets[:, 4]  # bbox打分
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     # 打分从大到小排列，取index
#     order = scores.argsort()[::-1]
#     # keep为最后保留的边框
#     keep = []
#     while order.size > 0:
#         # order[0]是当前分数最大的窗口，肯定保留
#         i = order[0]
#         keep.append(i)
#         # 计算窗口i与其他所有窗口的交叠部分的面积
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         # 交/并得到iou值
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
#         inds = np.where(ovr <= thresh)[0]
#         # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
#         order = order[inds + 1]
#
#     return keep
#
#
# class Detector(object):
#     def __init__(self, name='my_video', frame_num=10, k_size=7):
#         self.name = name
#         self.nms_threshold = 0.3
#         self.time = 1/frame_num
#         self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
#
#     def catch_video(self, video_index=0, k_size=7, iterations=3, threshold=20, bias_num=1, min_area=360, show_test=True,
#                     enhance=True):
#         # video_index：摄像头索引或者视频路径
#         # k_size：中值滤波的滤波器大小
#         # iteration：腐蚀+膨胀的次数
#         # threshold：二值化阙值
#         # bias_num：计算帧差图时的帧数差
#         # min_area：目标的最小面积
#         # show_test：是否显示二值化图片
#         if not bias_num > 0:
#             raise Exception('bias_num must > 0')
#         cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类
#         if not cap.isOpened():
#             # 如果没有检测到摄像头，报错
#             raise Exception('Check if the camera is on.')
#         frame_num = 0
#         previous = []
#         while cap.isOpened():
#             catch, frame = cap.read()  # 读取每一帧图片
#             if not catch:
#                 raise Exception('Unexpected Error.')
#             if frame_num < bias_num:
#                 value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 previous.append(value)
#                 frame_num += 1
#             raw = frame.copy()
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.absdiff(gray, previous[0])
#             gray = cv2.medianBlur(gray, k_size)
#             ret, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
#             if enhance:
#                 mask = cv2.dilate(mask, self.es, iterations)
#                 mask = cv2.erode(mask, self.es, iterations)
#
#             cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#             bounds = self.nms_cnts(cnts, mask, min_area)
#             for b in bounds:
#                 cv2.drawContours(frame, [np.int0(b)], -1, (0, 255, 255), 2)
#                 # x, y, w, h = b
#                 # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.imshow(self.name, frame)  # 在window上显示图片
#             if show_test:
#                 cv2.imshow(self.name+'_frame', mask)  # 边界
#             value = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
#             previous = self.pop(previous, value)
#
#             cv2.waitKey(1)
#
#         # 释放摄像头
#         cap.release()
#         cv2.destroyAllWindows()
#
#     def nms_cnts(self, cnts, mask, min_area):
#         bounds = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
#         c = [max(c, key=cv2.contourArea) for c in cnts if cv2.contourArea(c) > min_area]
#         rect = [cv2.minAreaRect(c) for c in c]
#         bbox = [cv2.boxPoints(c) for c in rect]
#         bbox = np.array(bbox)
#         if len(bounds) == 0:
#             return []
#         scores = [self.calculate(b, mask) for b in bounds]
#         bounds = np.array(bounds)
#         scores = np.expand_dims(np.array(scores), axis=-1)
#         keep = py_cpu_nms(np.hstack([bounds, scores]), self.nms_threshold)
#         return bbox[keep]
#
#     def calculate(self, bound, mask):
#         x, y, w, h = bound
#         area = mask[y:y+h, x:x+w]
#         pos = area > 0 + 0
#         score = np.sum(pos)/(w*h)
#         return score
#
#     def pop(self, l, value):
#         l.pop(0)
#         l.append(value)
#         return l
#
#
# if __name__ == "__main__":
#     detector = Detector()
#     detector.catch_video(0, bias_num=2, iterations=3, k_size=5, show_test=True, enhance=False)
