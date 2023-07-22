import cv2 as cv
import argparse
import numpy as np
def threshold_image():


	path = "/home/nemo/Documents/OpenCvAymeric/Image/img_4_29.png"
	frame = cv.imread(path)
	frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)


	lower_red = np.array([10,50,120])
	upper_red = np.array([75,150,155])
	red_mask = cv.inRange(frame_HSV, lower_red,upper_red)
	red_pixel = np.sum(red_mask>0)

	lower_yellow = np.array([25,85,155])
	upper_yellow = np.array([75,140,255])
	yellow_mask = cv.inRange(frame_HSV,lower_yellow,upper_yellow)
	yellow_pixel = np.sum(yellow_mask>0)

	# res = cv.bitwise_and(frame,frame, mask= mask)

	cv.imshow("Image capture", frame)
	cv.imshow("Red mark", red_mask)


if __name__ == '__main__':
    threshold_image()
