import sys
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from random import randint
import pygame

# settings
useGui = False if len(sys.argv) < 2 else sys.argv[1] == 'gui'
floatingAverageSize = 10
floatingAverageLowerThreshold = 1
floatingAverageUpperThreshold = 5
audioFileStartNumber = 42
audioFileEndNumber = 69

# initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# camera warmup
time.sleep(0.1)

#use trained classifier
car_cascade = cv2.CascadeClassifier('dhl.xml')

# floating average setup
index = 0
floatingAverageBuffer = [0] * floatingAverageSize
floatingAverage = 0
detected = False

# audio setup
pygame.mixer.init()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.1, 3, 0, (20, 10))

    # floating average
    inc = 1 if len(cars) > 0 else 0
    floatingAverage -= floatingAverageBuffer[index]
    floatingAverage += inc
    floatingAverageBuffer[index] = inc
    index = (index + 1) % floatingAverageSize
    print floatingAverage

    # new detection?
    if detected == False:
        if floatingAverage >= floatingAverageUpperThreshold:
            fileName = "sounds/" + str(randint(audioFileStartNumber, audioFileEndNumber)) + "_Audiospur.mp3"
            pygame.mixer.music.load(fileName)
            pygame.mixer.music.play()
            detected = True
    elif floatingAverage < floatingAverageLowerThreshold:
        detected = False

    # GUI
    if useGui:
        #to draw arectangle in each cars
        for (x,y,w,h) in cars:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

        #display the resulting frame
        cv2.imshow('video', image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
cv2.destroyAllWindows()
