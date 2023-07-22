import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lecture():
    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    return cv.imread(path)

def test1():

    img = lecture()
    if img is None:
        sys.exit("Could not read the image.")
    cv.imshow("Display window", img)
    k = cv.waitKey(0)
    print(img)
    #np.savetxt('save.txt', img, delimiter=",")
    #pd.DataFrame(img[0:][0:][0]).to_csv("save.csv")


    #if k == ord("s"):
    #    cv.imwrite("starry_night.png", img)

def histogramme():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')& 0xFF
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

    #cv.imshow("Display window", img)
    #k = cv.waitKey(0)

#Fonctionne bien pour demarquer les bouees
def equalization():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    equ = cv.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    cv.imwrite('equalization.jpg',res)
    #cv.imshow("Display window", img)
    #k = cv.waitKey(0)

def clahe():
    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv.imwrite('clahe_2.jpg',cl1)

def histogrammes2D():
    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    plt.imshow(hist,interpolation = 'nearest')
    plt.show()

def masquage():

    # Take each frame
    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    frame = cv.imread(path)
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_c = np.array([0,150,0])
    upper_c = np.array([150,200,50])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_c, upper_c)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(0)
    cv.destroyAllWindows()

def thresholding():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    lim_min = 200
    lim_max = 255
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret,thresh1 = cv.threshold(img,lim_min,lim_max,cv.THRESH_BINARY)
    ret,thresh2 = cv.threshold(img,lim_min,lim_max,cv.THRESH_BINARY_INV)
    ret,thresh3 = cv.threshold(img,lim_min,lim_max,cv.THRESH_TRUNC)
    ret,thresh4 = cv.threshold(img,lim_min,lim_max,cv.THRESH_TOZERO)
    ret,thresh5 = cv.threshold(img,lim_min,lim_max,cv.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

#Le coef adaptative mean thresholding detoure tres bien la bouee
def adaptative_thresholding():
    
    lim_min = 200
    lim_max = 255
    
    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.medianBlur(img,5)
    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

#Bof...
def otsu_binarization():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # global thresholding
    ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
     img, 0, th2,
     blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
     'Original Noisy Image','Histogram',"Otsu's Thresholding",
     'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()

def laplacian_derivative():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    laplacian = cv.Laplacian(img,cv.CV_64F)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()

def canny_edge_detection():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

#Fonctionne bien pour demarquer les bouees
def equalization_then_canny_edge_detection():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    equ = cv.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    edges = cv.Canny(res,0,255)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def adaptative_thresholding_then_canny_edge_detection():

    lim_min = 200
    lim_max = 255
    
    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.medianBlur(img,5)
    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    edges = cv.Canny(th2,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


#Pas top
def simple_find_contours():

    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    img = cv.imread(path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th2 = cv.adaptiveThreshold(imgray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    cv.THRESH_BINARY,11,2)
    ret, thresh = cv.threshold(th2, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0), 3)

    (x,y),radius = cv.minEnclosingCircle(contours[3])
    print(x)
    center = (int(x),int(y))
    radius = int(radius)
    cv.circle(img,center,radius,(0,255,0),2)
    plt.imshow(img)
    plt.title('Cercle'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Create a black image
    img_b = np.zeros((950,1686,3), np.uint8)
    cv.circle(img_b,center,radius,(0,255,0),2)
    cv.circle(img_b,(447,63), 63, (0,0,255), -1)
    print(center, radius)
    plt.imshow(img_b)
    plt.title('Cercle'), plt.xticks([]), plt.yticks([])
    plt.show()

#Marche tres bien...trop bien?
def HoughCircle():

    # Read image.
    
    path = "/home/nemo/Documents/OpenCvAymeric/Image/img_1_red_1.png"
    #path = "/home/aymeric/Documents/Programmation/Python/OpenCV/img/class_1/white/img_1_white_1.png"
    #path = "/home/aymeric/Documents/Programmation/Python/OpenCV/img/class_1/yellow/img_1_yellow_1.png"
    #path = "/home/aymeric/Documents/Programmation/Python/OpenCV/img/class_2/number_1/img_2_1_1.png"
    #path = "/home/aymeric/Documents/Programmation/Python/OpenCV/img/class_3/number_3/img_3_3_1.png"
    #path = "/home/aymeric/Documents/Programmation/Python/OpenCV/img/class_4/img_4_1.png"
    #path = "/home/aymeric/Documents/Programmation/Python/OpenCV/img/class_5/img_5_1.png"
    img = cv.imread(path, cv.IMREAD_COLOR)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # Convert to grayscale.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Blur using 3 * 3 kernel.
    gray_blurred = cv.blur(gray, (3, 3))

    th2 = cv.adaptiveThreshold(gray_blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    cv.THRESH_BINARY,11,2)
    
    # Apply Hough transform on the blurred image.
    detected_circles = cv.HoughCircles(th2, 
                       cv.HOUGH_GRADIENT, 1, 20, param1 = 200,
                   param2 = 30, minRadius = 20, maxRadius = 700)
    
    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
    
            # Draw the circumference of the circle.
            cv.circle(img, (a, b), r, (0, 255, 0), 2)
    
            # Draw a small circle (of radius 1) to show the center.
            cv.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv.imshow("Detected Circle", img)
            cv.waitKey(0)

    else:
        print("None were found")

if __name__ == '__main__':
    histogrammes2D()
    #test1()
    #masquage()
    #thresholding()
    #adaptative_thresholding()
    #otsu_binarization()
    #laplacian_derivative()
    #canny_edge_detection()
    #equalization_then_canny_edge_detection()
    #simple_find_contours()
    #adaptative_thresholding_then_canny_edge_detection()
    #HoughCircle()
