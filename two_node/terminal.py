import time
import socket
import struct
import cv2
import numpy as np
from multiprocessing import Process
from gpiozero import Button
from LaserPointer import Pendulum


pd = Pendulum()


class Detector(object):
    def __init__(self, frame_num=10, k_size=7, stream=True, video_stream=False):
        self.nms_threshold = 0.3
        self.time = 1/frame_num
        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        self.frame_cnt = 1
        self.k_size = k_size
        self.button = Button(20)
        self.global_time = time.time()
        self.cam_amp: float = 0
        self.stream = stream
        self.video_stream = video_stream
        if self.stream:
            self.client = None
            self.Socket_Connect()

    def Socket_Connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client.connect(('192.168.0.20', 8880))
        print("Connected")

    def receive(self):
        if self.video_stream:
            buf_size, self.cam_amp = struct.unpack("qd", self.client.recv(16))
            if buf_size:
                try:
                    buf = b""  # 代表bytes类型
                    while buf_size:  # 读取每一张图片的长度
                        temp_buf = self.client.recv(buf_size)
                        buf_size -= len(temp_buf)
                        buf += temp_buf
                        data = np.frombuffer(buf, dtype='uint8')  # 按uint8转换为图像矩阵
                        cv2.imshow('cam0', cv2.imdecode(data, 1))  # 展示图片
                except Exception:
                    print('terminal error')
        else:
            self.cam_amp = struct.unpack("d", self.client.recv(8))

    def catch_video(self, iterations=3, threshold=20, bias_num=1, show_test=True, enhance=True, framerate=30):
        # video_index：摄像头索引或者视频路径
        # k_size：中值滤波的滤波器大小
        # iteration：腐蚀+膨胀的次数
        # threshold：二值化阙值
        # bias_num：计算帧差图时的帧数差
        # min_area：目标的最小面积
        # show_test：是否显示二值化图片
        cap = cv2.VideoCapture(0)  # 创建摄像头识别类
        cap.set(3, 640)
        cap.set(4, 360)
        cap.set(cv2.CAP_PROP_FPS, framerate)
        frame_num = 0
        previous = []
        while cap.isOpened() and time.time() - self.global_time < 30:
            _, frame = cap.read()  # 读取每一帧图片
            if frame_num < bias_num:
                value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                previous.append(value)
                frame_num += 1
            raw = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.absdiff(gray, previous[0])
            gray = cv2.medianBlur(gray, self.k_size)
            ret, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            if enhance:
                mask = cv2.dilate(mask, self.es, iterations)
                mask = cv2.erode(mask, self.es, iterations)

            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounds = self.nms_cnts(cnts)
            cv2.putText(frame, 'frame:{}'.format(self.frame_cnt), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
            if bounds is not None:
                cv2.drawContours(frame, [np.int0(bounds)], -1, (0, 0, 255), 2)

            if self.stream:
                receive = Process(target=self.receive())
                receive.start()
                receive.join()

            cv2.imshow('cam1', frame)  # 在window上显示图片
            if show_test:
                cv2.imshow('cam1_mask', mask)  # 边界
            value = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            previous = self.pop(previous, value)
            cv2.waitKey(1)
            if time.time() - self.global_time > 25:
                l, t = pd.output()
                print('长度为：{}，角度为：{}'.format(l, t))
                break

        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()

    def nms_cnts(self, cnts):
        bbox = None
        if len(cnts) != 0:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            bbox = cv2.boxPoints(rect)
            self.frame_cnt += 1
            pd.update(rect[2], bbox, time.time_ns(), self.frame_cnt, self.cam_amp)
        return bbox

    @staticmethod
    def pop(l, value):
        l.pop(0)
        l.append(value)
        return l


if __name__ == '__main__':
    detector = Detector(k_size=5, stream=True)
    time.sleep(5)
    detector.catch_video(bias_num=2, iterations=3, show_test=False, enhance=True)
