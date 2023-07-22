import cv2 as cv
import numpy as np
import time

from datetime import datetime

def Capture():
	cap = cv.VideoCapture(0)
	if not cap.isOpened():
		print("Cannot open camera")
		exit()
	ret,frame = capture.read()
	if not ret:
		print("Echouée")
		exit()
	now = datetime.now()
	time_format = "%Y_%m_%d_%H_%M_%S"
	time_str = now.strftime(time_format)+".jpg"
	cv.imWrite("home/nemo/Documents/OpenCvAymeric/image.jpg",frame)
	
	capture.release()
	cv.destroyAllWindows()

if __name__ == "Main":
	Capture()
		


