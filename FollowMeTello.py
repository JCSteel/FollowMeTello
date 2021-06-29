from djitellopy import Tello
from threading import Thread
import cv2 as cv
import numpy as np
import argparse
import dlib

#me = Tello()
#me.connect()
#x = me.get_battery()
#print(x)
#me.takeoff()
#me.move_up(75)
#me.streamon()
#frame_reader = me.get_frame_read()
desiredFaceWidth = 256
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lbfmodel.dat")
#capCenter = (960/2, 720/2)
capCenter = (1280/2, 720/2)


#Detects and displays a box around faces
def detectAndDisplay(frame):
    location = 0,0,0,0
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    #For each detected face, draw a box around it.
    for (x,y,w,h) in faces:
        location = x,y,w,h
        frame = cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255))
        faceROI = frame_gray[y:y+h, x:x+w]
    cv.imshow('Capture  - Face detection', frame)
    return location

#Detects and displays the facial landmarks
#Will be necessary for pose estimation
def displayLandmarks(frame):
    #Resizing the video feed for the sake of dlib optimization
    resize = cv.resize(frame, (240,180))
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    dlibFaces = detector(gray)
    #For each detected face, draw circles on all landmarks.
    for face in dlibFaces:
      points = predictor(gray, face)
      for n in range(68):
           frame = cv.circle(frame, (points.part(n).x*4, points.part(n).y*4), 5, (50,50,255), cv.FILLED)
    cv.imshow('Capture  - Face detection', frame)


def repositionDroneX(pos):
    #Finds the center of the frame capture
    center = location[0]+location[2]//2
    #Moves the drone in the necessary direction
    if ((center > capCenter[0]+100) & (location[2] != 0)):
        #me.move_right(20)
        print("moving right")
    if ((center < capCenter[0]-100) & (location[2] != 0)):
        #me.move_left(20)
        print("moving left")

def repositionDroneY(pos):
    center = pos[1]+pos[3]//2
    if ((center < capCenter[1]-150) & (location[3] != 0)):
        #me.move_up(20)
        print("moving up")
    if ((center > capCenter[1]+75) & (location[3] != 0)):
        #me.move_down(20)
        print("moving down")

def repositionDroneZ(pos):
    if ((pos[2] > desiredFaceWidth+150) & (location[2] != 0)):
        #me.move_back(20)
        print("moving back")
    if ((pos[2] < desiredFaceWidth-150) & (location[2] != 0)):
        #me.move_forward(20)
        print("moving forward")


#Parsing the haarcascade
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='D:\Program Files (x86)\OpenCV\opencv\sources\data\haarcascades/haarcascade_frontalface_alt.xml')
args = parser.parse_args()

#Setting up other necessary variables
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()

#If we can't load the cascade, end the program.
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('Error loading face cascade')
    exit(0)

#Create a video capture. If we can't, end the program.
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    location = 0,0,0,0
    ret, frame = cap.read()
    #if not frame_reader:
    if not ret:
        print("Can't receive frame. Exiting.")
        break
    location = detectAndDisplay(frame)
    #location = detectAndDisplay(frame_reader.frame)
    #displayLandmarks(frame_reader.frame)
    #detectAndDisplay(frame)
    #displayLandmarks(frame)
    x = Thread(repositionDroneX(location))
    y = Thread(repositionDroneY(location))
    z = Thread(repositionDroneZ(location))
    x.start()
    y.start()
    z.start()

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
#me.streamoff()
cv.destroyAllWindows()


