import cv2
import numpy as np
import pandas as pd
import math
import scipy.ndimage

#Open Camera
cap = cv2.VideoCapture(0)
cap.set(10,150)

#Analysis Wall Defects
def oriented_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0],[1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

#Start calculation
imgGray = cap.read()
with_nmsup = True
fudgefactor = 1.3
sigma = 21
kernel = 2*math.ceil(2*sigma)+1
imgGray = imgGray/float(255.0)
imgBlur = cv2.GaussianBlur(imgGray, (kernel, kernel), sigma)
imgGray = cv2.subtract(imgGray, imgBlur)

#Calculate sobel response
sobelx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(sobelx, sobely)
ang = np.arctan2(sobely, sobelx)

#Threshold
threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0

#either get edges directly
if with_nmsup is False:
    mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Camera', result)
    cv2.waitKey()

#or apply a non-maximal suppression
else:
    while True:
        success, img = cap.read()
        # non-maximal suppression
        mag = oriented_non_max_suppression(mag, ang)
        # create mask
        mag[mag > 0] = 255
        mag = mag.astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Camera",result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
