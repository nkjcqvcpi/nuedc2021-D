import cv2
import socket
import struct
import time
from multiprocessing import Process
import numpy as np
from LaserPointer import Pendulum


pd = Pendulum()


class Detector(object):
    def __init__(self, frame_num=10, k_size=7, stream=True, video_stream=False):
        self.nms_threshold = 0.3
        self.time = 1/frame_num
        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        self.frame_cnt = 1
        self.k_size = k_size
        self.global_time = time.time()
        self.k_size = k_size
        self.stream = stream
        self.video_stream = video_stream
        if self.stream:
            self.server, self.client = None, None
            self.Set_Socket()

    def Set_Socket(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 端口可复用
        self.server.bind(("192.168.0.20", 8880))
        self.server.listen(5)
        self.client, _ = self.server.accept()
        print("192.168.0.20 Online")

    def send(self, frame):
        if self.video_stream:
            try:
                # 按照相应的格式进行打包发送图片
                _, img_encode = cv2.imencode('.jpg', frame)  # 按格式生成图片
                img_data = np.array(img_encode).tobytes()
                self.client.send(struct.pack("qdd", len(img_data), pd.amp, pd.l) + img_data)
            except Exception:
                print('camera error')
        else:
            self.client.send(struct.pack("dd", pd.amp, pd.l))

    def catch_video(self, iterations=3, threshold=20, bias_num=1, show_test=True, enhance=True, framerate=40):
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
                send = Process(target=self.send, args=(frame,))
                send.start()
                send.join()

            cv2.imshow('cam1', frame)

            if show_test:
                cv2.imshow('cam1_mask', mask)  # 边界
            cv2.waitKey(1)
            previous = self.pop(previous, cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY))

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
            pd.update(rect[2], bbox, time.time_ns(), self.frame_cnt)
        return bbox

    @staticmethod
    def pop(l, value):
        l.pop(0)
        l.append(value)
        return l


if __name__ == "__main__":
    detector = Detector(k_size=5, stream=True, video_stream=False)
    detector.catch_video(bias_num=2, iterations=3, show_test=False, enhance=True)
