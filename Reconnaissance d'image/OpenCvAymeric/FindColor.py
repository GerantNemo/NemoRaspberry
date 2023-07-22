import cv2 as cv
import argparse
import numpy as np
def threshold_image():

	while True:
		path = "/home/nemo/Documents/OpenCvAymeric/Image/img_2_3_1.png"
		frame = cv.imread(path)
		frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

		lower_red = np.array([10,50,120])
		upper_red = np.array([70,150,215])
		red_mask = cv.inRange(frame_HSV, lower_red,upper_red)
		red_pixel = np.sum(red_mask>0)

		lower_yellow = np.array([25,85,200])
		upper_yellow = np.array([70,140,255])
		yellow_mask = cv.inRange(frame_HSV,lower_yellow,upper_yellow)
		yellow_pixel = np.sum(yellow_mask>0)
		
		lower_pipe = np.array([65,205,195])
		upper_pipe = np.array([80,255,230])
		pipe_mask = cv.inRange(frame_HSV,lower_pipe,upper_pipe)
		pipe_pixel = np.sum(pipe_mask>0)
		
		if yellow_pixel>red_pixel and yellow_pixel>pipe_pixel :
			mask = yellow_mask
		elif red_pixel>yellow_pixel and red_pixel>pipe_pixel:
			mask = red_mask
		elif pipe_pixel>red_pixel and pipe_pixel>yellow_pixel:
			mask = pipe_mask

		res = cv.bitwise_and(frame,frame, mask= mask)

		cv.imshow("Image capture", frame)
		cv.imshow("Object", res)
		
		key = cv.waitKey(30)
		if key == ord('q') or key == 27:
		    break


if __name__ == '__main__':
    threshold_image()
