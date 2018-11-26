import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

car_cascade = cv2.CascadeClassifier('dhl.xml')

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    for (x,y,w,h) in cars:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)      

    cv2.imshow('video', image)
    key = cv2.waitKey(1) & 0xFF
 
    rawCapture.truncate(0)
 
    if key == ord("q"):
        break
cv2.destroyAllWindows()
