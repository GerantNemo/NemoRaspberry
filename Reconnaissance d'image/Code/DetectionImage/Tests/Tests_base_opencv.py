#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

if __name__ == "__main__":

    # capture from camera at location 0
    #cap = cv2.VideoCapture(2)
    # set the width and height, and UNSUCCESSFULLY set the exposure time
    #cap.set(3, 1280)
    #cap.set(4, 1024)
    #cap.set(15, 0.1)

    all_camera_idx_available = []

    for camera_idx in range(10):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            print(f'Camera index available: {camera_idx}')
            all_camera_idx_available.append(camera_idx)
            cap.release()

    cap = cv2.VideoCapture(1)
    cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

    while True:
        ret, img = cap.read()
        cv2.imshow("input", img)
        # cv2.imshow("thresholded", imgray*thresh2)

        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()