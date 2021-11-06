import cv2
import numpy as np

knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,12))
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    fg = knn.apply(frame.copy())
    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    bw_and = cv2.bitwise_and(fg_bgr, frame)
    draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)
    draw = cv2.GaussianBlur(draw, (21, 21), 0)
    draw = cv2.threshold(draw, 10, 255, cv2.THRESH_BINARY)[1]
    draw = cv2.dilate(draw, es, iterations = 2)
    image, contours = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_index = 0
    for index, c in enumerate(image):
        if cv2.contourArea(c) > max_index:
            max_index = index
    c = image[max_index]
    rect = cv2.minAreaRect(c)
    bbox = cv2.boxPoints(rect)
    cv2.drawContours(frame, [np.int0(bbox)], -1, (0, 255, 255), 2)
    cv2.imshow("motion detection", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
      break

camera.release()
cv2.destroyAllWindows()