
import argparse


import cv2
import numpy as np


image = cv2.VideoCapture(0)
hsv_img = np.array([125, 0, 0])
lower = np.array([0, 100, 20])
upper = np.array([10, 255, 255])
while True:
    check, frame = image.read()

    for contour in cnts:
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
    # show the images
    cv2.imshow("images", np.hstack([image, output]))

    # cv2.imshow("Color Frame", frame)
    cv2.imshow(" mask", mask)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
image.release()
cv2.destroyAllWindows
