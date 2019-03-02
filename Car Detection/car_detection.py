#import libraries of python opencv
#using numoy and opencv
import cv2
import numpy as np

#create VideoCapture object and read from video file
cap = cv2.VideoCapture('cars.mp4')
#use trained cars XML classifiers
car_cascade = cv2.CascadeClassifier('cars.xml')

#read until video is completed
while True:
    #capture frame by frame
    ret, frame = cap.read()
    #convert video into gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    #to draw arectangle in each cars 
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)      

    #display the resulting frame
    cv2.imshow('Car_Capture', frame)
    #press X on keyboard to exit
    if cv2.waitKey(5) & 0xFF == ord('x'):
        break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()
