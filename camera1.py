import socket
import time

import cv2


def send():
    HOST = '192.168.199.131'
    PORT = 8001
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    time.sleep(2)
    sock.send(b'1')
    print(sock.recv(1024).decode())
    sock.close()


import numpy as np
import subprocess as sp
import atexit

(w, h) = (300, 200)
fps = 60
bytesPerFrame = w * h
videoCmd = "raspividyuv -w " + str(w) + " -h " + str(h) + " --output - --timeout 0 --framerate " + str(fps) + "--nopreview"
videoCmd = videoCmd.split()

cameraProcess = sp.Popen(videoCmd, stdout=sp.PIPE, bufsize=0)
atexit.register(cameraProcess.terminate)
rawStream = cameraProcess.stdout.read(bytesPerFrame)
cv2.namedWindow('cap', 0)
cv2.resizeWindow('cap', 300, 200)

while True:
    cameraProcess.stdout.flush()
    frame = np.fromfile(cameraProcess.stdout, count=bytesPerFrame, dtype=np.uint8)
    if frame.size != bytesPerFrame:
        print("Error: Camera stream closed unexpectedly")
        break
    frame.shape = (h, w)

    cv2.imshow('cap', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
cameraProcess.terminate()