import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)

while True:
    success, img = cap.read()
   # cv2.imshow("Video", img)