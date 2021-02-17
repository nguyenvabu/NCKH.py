import cv2
import numpy as np
import pandas as pd
import math
import scipy.ndimage

widthImg=540
heightImg =640
#Open Camera
cap = cv2.VideoCapture(0)
cap.set(10,150)

#Crop and zoom
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
    return imgCropped


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

#Analysis Wall Defects
class Analysis_Wall_Defects:
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
class Calculate(Analysis_Wall_Defects):
    def preProcessing(img):
        with_nmsup = True
        fudgefactor = 1.3
        sigma = 21
        kernel = 2 * math.ceil(2 * sigma) + 1
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgGray = imgGray / 255.0
        imgBlur = cv2.GaussianBlur(imgGray, (kernel, kernel), sigma)
        imgGray = cv2.subtract(imgGray, imgBlur)

        sobelx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=3)

        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)
        threshold = 4 * fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0
        return with_nmsup

#either get edges directly
while True:
    success, img = cap.read()
    with_nmsup = Calculate.preProcessing(img)
    mag = Analysis_Wall_Defects.oriented_non_max_suppression(mag,ang)
    if with_nmsup is False:
        mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
        kernel = np.ones((5,5),np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Camera', result)
        cv2.waitKey()

#or apply a non-maximal suppression
    else:
        # non-maximal suppression
        mag = Analysis_Wall_Defects.oriented_non_max_suppression(mag)
        # create mask
        mag[mag > 0] = 255
        mag = mag.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Camera",result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#Crop image and zoom in the picture from cam
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(widthImg,heightImg))
        imgContour = img.copy()
        imgThres = preProcessing(img)
        biggest = getContours(imgThres)
        if biggest.size !=0:
            imgWarped=getWarp(img,biggest)
            imageArray = ([imgContour, imgWarped])
            cv2.imshow("ImageWarped", imgWarped)
        else:
            imageArray = ([imgContour, img])
        stackedImages = stackImages(0.6,imageArray)
        cv2.imshow("WorkFlow", stackedImages)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break