from djitellopy import Tello

me = Tello()
#me.connect()
#me.streamon()
#me.get_frame_read()


import cv2 as cv
import numpy as np
import argparse

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255))
        faceROI = frame_gray[y:y+h, x:x+w]
    cv.imshow('Capture  - Face detection', frame)

def repositionDrone(cap):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    vid = (cap.get(cv.CAP_PROP_FRAME_WIDTH)/2, cap.get(cv.CAP_PROP_FRAME_HEIGHT)/2)
    center = (vid)
    for (x,y,w,h) in faces:
        center = (x+w//2, y+h//2)
    print(center)
    if (center[0] > vid[0]):
        #me.move_right(20)
        print("moving right")
    if (center[0] < vid[0]):
        #me.move_left(20)
        print("moving left")
    if (center[1] > vid[1]):
        #me.move_left(20)
        print("moving down")
    if (center[1] < vid[1]):
        #me.move_left(20)
        print("moving up")

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='D:\Program Files (x86)\OpenCV\opencv\sources\data\haarcascades/haarcascade_frontalface_alt.xml')
#parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade

face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('Error loading face cascade')
    exit(0)

#camera_device = args.camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't rexceive frame. Exiting.")
        break
    detectAndDisplay(frame)
    repositionDrone(cap)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


