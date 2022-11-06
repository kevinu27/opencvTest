# Python code for Multiple Color Detection


import numpy as np
import cv2


# Capturing video through cap
cap = cv2.VideoCapture(0)
hsv_img = np.array([125, 0, 0])
lower_blue = np.array([0, 100, 20])
upper_blue = np.array([10, 255, 255])
# Start a while loop
while True:

    ret, frame = cap.read()

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    ret, thresh1 = cv2.threshold(res, 127, 255, 0)
    gray = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
    thresh_frame = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

    (cnts, _) = cv2.findContours(thresh_frame,
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)

    # cv2.imshow("Multiple Color Detection in Real-TIme", thresh_frame)
    # cv2.imshow("res", res)
    # cv2.imshow("gray", gray)
    cv2.imshow("thresh1", thresh1)
    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
