import cv2.cv2 as cv2

with cv2.VideoCapture(0) as cam:
    frame_cnt = 0
    while cam.isOpened():
        _, frame = cam.read()
        if frame_cnt == 100:
            ref_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gray = cv2.absdiff(gray, ref_frame)
        gray = cv2.medianBlur(gray, 7)
        ret, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, es, 3)
        mask = cv2.erode(mask, es, 3)
