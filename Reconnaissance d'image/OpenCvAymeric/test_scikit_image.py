import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from skimage import measure

def scikit_contour():

    path = /home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(img, 0.8)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

if __name__ == '__main__':
    scikit_contour()
