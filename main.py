import time
import math
import socket
import struct
import cv2
import numpy as np

try:
    from gpiozero import Button, LED, Buzzer
    PI = True
except ModuleNotFoundError:
    PI = False

from models.sound import Sound


class VideoPlayer(object):
    def __init__(self, video_stream=False):
        self.global_time = time.time()
        self.client0, self.client1 = None, None
        if PI:
            self.button = Button(26)
            self.buzzer = Buzzer(21)
            self.led = LED(20)
            self.led.off()
        self.thetas, self.lengths = [], []
        self.video_stream = video_stream
        self.amp0, self.amp1, self.l0, self.l1 = 0, 0, 0, 0

    def socket_connect(self):
        self.client0 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client0.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client0.connect(('192.168.0.10', 8880))
        self.client1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client1.connect(('192.168.0.20', 8880))
        print('Connected')

    def output(self) -> (float, float):
        try:
            c = np.array(self.lengths)
            hw = np.percentile(c, 98, axis=0)
            lw = np.percentile(c, 3, axis=0)
            length = c[np.where((c < hw) & (c > lw))].mean()
        except:
            try:
                lengths = [i for i in self.lengths if 0.5 <= i <= 1.5]
                length = np.array(lengths).mean()
            except:
                length = 1
        try:
            c = np.array(self.thetas)
            hw = np.percentile(c, 98, axis=0)
            lw = np.percentile(c, 3, axis=0)
            theta = c[np.where((c < hw) & (c > lw))].mean()
        except:
            try:
                thetas = [i for i in self.thetas if 0 <= i <= 90]
                theta = np.array(thetas).mean()
            except:
                theta = 45
        return length - 0.11, theta

    def show_video(self):
        while time.time() - self.global_time < 30:
            if self.video_stream:
                buf_size0, self.amp0, self.l0 = struct.unpack("!qdd", self.client0.recv(24))
                if buf_size0:
                    try:
                        buf0 = b""  # 代表bytes类型
                        while buf_size0:  # 读取每一张图片的长度
                            temp_buf0 = self.client0.recv(buf_size0)
                            buf_size0 -= len(temp_buf0)
                            buf0 += temp_buf0
                            data0 = np.frombuffer(buf0, dtype='uint8')  # 按uint8转换为图像矩阵
                            cv2.imshow('camera0', cv2.imdecode(data0, 1))  # 展示图片
                            cv2.waitKey(1)
                    except:
                        pass

                buf_size1, self.amp1, self.l1 = struct.unpack("!qdd", self.client1.recv(24))
                if buf_size1:
                    try:
                        buf1 = b""  # 代表bytes类型
                        while buf_size1:  # 读取每一张图片的长度
                            temp_buf1 = self.client1.recv(buf_size1)
                            buf_size1 -= len(temp_buf1)
                            buf1 += temp_buf1
                            data1 = np.frombuffer(buf1, dtype='uint8')  # 按uint8转换为图像矩阵
                            cv2.imshow('camera1', cv2.imdecode(data1, 1))  # 展示图片
                            cv2.waitKey(1)
                    except:
                        pass
            else:
                try:
                    self.amp0, self.l0 = struct.unpack("!dd", self.client0.recv(16))
                    self.amp1, self.l1 = struct.unpack("!dd", self.client1.recv(16))
                except:
                    pass
            if self.amp1 != 0:
                theta = math.degrees(math.atan(self.amp0 / self.amp1))
                self.thetas.append(theta)
            self.lengths.append(self.l0)
            self.lengths.append(self.l1)

            if time.time() - self.global_time > 24:
                l, t = self.output()
                print('长度为：{}，角度为：{}'.format(l, t))
                play = Sound(l, t)
                play()
                if PI:
                    self.buzzer.beep()
                    self.led.on()
                    time.sleep(2)
                    self.led.off()
                break


if __name__ == '__main__':
    player = VideoPlayer(video_stream=False)
    player.socket_connect()
    player.show_video()
