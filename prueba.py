import cv2
import numpy as np


video = cv2.VideoCapture(0)
hsv_img = np.array([125, 0, 0])
lower_blue = np.array([0, 100, 20])
upper_blue = np.array([10, 255, 255])
while True:
    check, frame = video.read()
    status = 0
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # threshold clasifica las cosas y las pone o blancas o negras,el valor max, justo despues de delta_frame dice a partir de este color lo clasifico lo paso a blanco o lo dejo en negro, blanco es el 255
    thresh_frame = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

    # thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts, _) = cv2.findContours(thresh_frame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 100:
            continue
        if cv2.contourArea(contour) > 80000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 3)

    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)
    cv2.imshow(" mask", mask)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows
